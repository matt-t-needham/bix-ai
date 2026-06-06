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

    async def _fake_review(record):
        return "ADVISORY: looks fine"

    import main
    monkeypatch.setattr(main, "_claude_review_text", _fake_review)
    r = c.post(f"/staging/{rec['id']}/review", follow_redirects=False)
    assert r.status_code == 303
    assert staging.get(rec["id"])["claude_review"] == "ADVISORY: looks fine"
    assert "ADVISORY: looks fine" in c.get(f"/staging/{rec['id']}").text


# ── Tool registration (all three surfaces) ────────────────────────────────────

def test_stage_write_registered_everywhere():
    import tools
    assert "stage_write" in [t["name"] for t in tools.FS_TOOLS]
    assert "stage_write" in [t["function"]["name"] for t in tools.OLLAMA_TOOLS]
    assert "stage_write" in tools.FORGE_TOOLS


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
