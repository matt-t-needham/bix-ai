"""Route + tool-registration tests for the staged-write feature.

These exercise the FastAPI handlers and the tool wiring, so they need the
service deps (fastapi / httpx / forge). They are skipped automatically where
those aren't installed (e.g. a bare local checkout) and run in the container,
which is where the pre-deploy gate should execute them.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("forge")

from fastapi.testclient import TestClient  # noqa: E402

import config       # noqa: E402
import staging       # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FS_ROOT", tmp_path.resolve())
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    import main
    return TestClient(main.app), tmp_path


# ── Routes ────────────────────────────────────────────────────────────────────

def test_staging_list_empty(client):
    c, _ = client
    r = c.get("/staging")
    assert r.status_code == 200
    assert "No staged changes" in r.text


def test_count_route_resolves_before_id(client):
    # /staging/count must not be captured by /staging/{rec_id}.
    c, _ = client
    r = c.get("/staging/count")
    assert r.status_code == 200
    assert set(r.json()) >= {"pending", "total"}


def test_detail_404(client):
    c, _ = client
    assert c.get("/staging/doesnotexist").status_code == 404


def test_approve_writes_file(client):
    c, root = client
    target = root / "bix-blog" / "post.md"
    rec = staging.create(str(target), "hello body")
    r = c.post(f"/staging/{rec['id']}/approve", follow_redirects=False)
    assert r.status_code == 303
    assert target.read_text() == "hello body"
    assert staging.get(rec["id"])["status"] == "approved"


def test_reject_route(client):
    c, root = client
    rec = staging.create(str(root / "x.md"), "nope")
    r = c.post(f"/staging/{rec['id']}/reject", follow_redirects=False)
    assert r.status_code == 303
    assert staging.get(rec["id"])["status"] == "rejected"


def test_detail_escapes_content(client):
    # Staged content is untrusted — it must be HTML-escaped in the diff view.
    c, root = client
    rec = staging.create(str(root / "x.md"), "<script>alert(1)</script>")
    body = c.get(f"/staging/{rec['id']}").text
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


def test_review_route_persists_advisory(client, monkeypatch):
    c, root = client
    rec = staging.create(str(root / "x.md"), "content")

    captured = {}

    async def _fake_anthropic(model, prompt, max_tokens):
        captured["model"], captured["prompt"] = model, prompt
        return "ADVISORY: looks fine"

    import main
    monkeypatch.setattr(main, "_anthropic_text", _fake_anthropic)
    r = c.post(f"/staging/{rec['id']}/review")   # bare POST = default-model review
    assert r.status_code == 200
    assert r.json()["ok"] is True
    stored = staging.get(rec["id"])
    assert stored["claude_review"] == "ADVISORY: looks fine"
    assert stored["review_model"] == captured["model"]
    assert "ADVISORY: looks fine" in c.get(f"/staging/{rec['id']}").text


def test_review_route_model_selection_and_comments_in_prompt(client, monkeypatch):
    c, root = client
    rec = staging.create(str(root / "x.md"), "content line one")
    staging.add_comment(rec["id"], "content line one", "tighten this up")

    captured = {}

    async def _fake_anthropic(model, prompt, max_tokens):
        captured["model"], captured["prompt"] = model, prompt
        return "ok"

    import main
    monkeypatch.setattr(main, "_anthropic_text", _fake_anthropic)
    model = sorted(config._ALLOWED_CLAUDE_MODELS)[0]
    r = c.post(f"/staging/{rec['id']}/review", json={"model": model, "action": "review"})
    assert r.status_code == 200
    assert captured["model"] == model
    assert "reviewer_comments" in captured["prompt"]
    assert "tighten this up" in captured["prompt"]


def test_review_route_rejects_unknown_model(client, monkeypatch):
    c, root = client
    rec = staging.create(str(root / "x.md"), "content")
    r = c.post(f"/staging/{rec['id']}/review", json={"model": "gpt-99"})
    assert r.status_code == 400
    assert staging.get(rec["id"])["claude_review"] is None


def test_revise_updates_content_and_keeps_revision(client, monkeypatch):
    c, root = client
    rec = staging.create(str(root / "x.md"), "old body")
    staging.add_comment(rec["id"], "old body", "please improve")

    async def _fake_anthropic(model, prompt, max_tokens):
        return "- improved it\n<updated_file>\nnew body\n</updated_file>"

    import main
    monkeypatch.setattr(main, "_anthropic_text", _fake_anthropic)
    r = c.post(f"/staging/{rec['id']}/review", json={"action": "revise"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    stored = staging.get(rec["id"])
    assert stored["content"] == "new body"
    assert stored["claude_review"] == "- improved it"
    assert [rv["content"] for rv in stored["revisions"]] == ["old body"]
    assert stored["status"] == "pending"          # still gated by human approve
    assert not (root / "x.md").exists()           # live file untouched


def test_revise_without_open_comments_400(client, monkeypatch):
    c, root = client
    rec = staging.create(str(root / "x.md"), "body")
    r = c.post(f"/staging/{rec['id']}/review", json={"action": "revise"})
    assert r.status_code == 400
    assert staging.get(rec["id"])["content"] == "body"


def test_revise_unparseable_response_leaves_record(client, monkeypatch):
    c, root = client
    rec = staging.create(str(root / "x.md"), "body")
    staging.add_comment(rec["id"], "", "a comment")

    async def _fake_anthropic(model, prompt, max_tokens):
        return "no tags here"

    import main
    monkeypatch.setattr(main, "_anthropic_text", _fake_anthropic)
    r = c.post(f"/staging/{rec['id']}/review", json={"action": "revise"})
    assert r.status_code == 502
    stored = staging.get(rec["id"])
    assert stored["content"] == "body"
    assert stored["revisions"] == []


# ── Comment routes ────────────────────────────────────────────────────────────

def test_comment_add_and_resolve_routes(client):
    c, root = client
    rec = staging.create(str(root / "x.md"), "some body text")
    r = c.post(f"/staging/{rec['id']}/comments",
               json={"quote": "body text", "text": "needs work"})
    assert r.status_code == 200
    comments = r.json()["comments"]
    assert len(comments) == 1 and comments[0]["resolved"] is False
    cid = comments[0]["id"]

    r = c.post(f"/staging/{rec['id']}/comments/{cid}/resolve", json={"resolved": True})
    assert r.status_code == 200
    assert staging.get(rec["id"])["comments"][0]["resolved"] is True

    r = c.post(f"/staging/{rec['id']}/comments/{cid}/resolve", json={"resolved": False})
    assert staging.get(rec["id"])["comments"][0]["resolved"] is False


def test_comment_routes_404_and_400(client):
    c, root = client
    assert c.post("/staging/nope/comments",
                  json={"quote": "", "text": "x"}).status_code == 404
    rec = staging.create(str(root / "x.md"), "body")
    assert c.post(f"/staging/{rec['id']}/comments",
                  json={"quote": "q", "text": "   "}).status_code == 400
    assert c.post(f"/staging/{rec['id']}/comments/nocomment/resolve",
                  json={"resolved": True}).status_code == 404


# ── Chat context endpoint ─────────────────────────────────────────────────────

def test_context_endpoint(client):
    c, root = client
    rec = staging.create(str(root / "bix-blog" / "post.md"), "# Title\n\nbody")
    staging.add_comment(rec["id"], "body", "flesh this out")
    r = c.get(f"/staging/{rec['id']}/context")
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True and d["target_path"].endswith("post.md")
    assert "# Title" in d["context"]
    assert "flesh this out" in d["context"]

    assert c.get("/staging/nope/context").status_code == 404


# ── Tool registration (both wire surfaces) ────────────────────────────────────

def test_stage_write_registered_everywhere():
    import tools
    assert "stage_write" in [t["name"] for t in tools.FS_TOOLS]
    assert "stage_write" in [t["function"]["name"] for t in tools.OLLAMA_TOOLS]


def test_execute_tool_stage_write(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FS_ROOT", tmp_path.resolve())
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    import asyncio
    import tools
    out = asyncio.run(tools._execute_tool(
        "stage_write", {"target_path": str(tmp_path / "bix-blog" / "p.md"), "content": "hi"}
    ))
    assert "Staged for review" in out
    assert not (tmp_path / "bix-blog" / "p.md").exists()
    assert len(staging.list_records()) == 1
