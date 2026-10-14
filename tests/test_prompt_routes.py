"""The /prompt editor routes: the page, the JSON view, and the three writes.

Editing the prompt changes what every future turn is told, so the writing
routes carry the same Cloudflare Access check as agent approvals — that gate
is what most of these tests are about. The forms are urlencoded and parsed
with the stdlib (no python-multipart), so the body handling is pinned here too.
"""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("forge")

from fastapi.testclient import TestClient  # noqa: E402

import config        # noqa: E402
import main          # noqa: E402
import prompt_store  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    prompt_store._cache.update(key=None, text="")
    return tmp_path


@pytest.fixture
def client():
    return TestClient(main.app)


@pytest.fixture
def signed_in(monkeypatch):
    monkeypatch.setattr(main.access, "access_verified", lambda request: True)


# ── Reading ───────────────────────────────────────────────────────────────────

def test_page_renders_when_nothing_is_set(env, client):
    r = client.get("/prompt")
    assert r.status_code == 200
    assert "Custom prompt" in r.text
    assert "No earlier versions yet" in r.text


def test_page_shows_the_current_text_and_counts(env, client):
    prompt_store.save("Prefer metric units.")
    r = client.get("/prompt")
    assert "Prefer metric units." in r.text
    assert f"of {prompt_store.MAX_CHARS:,} characters" in r.text


def test_raw_route_returns_the_state(env, client):
    prompt_store.save("hello")
    body = client.get("/prompt/raw").json()
    assert body["text"] == "hello\n"
    assert body["chars"] == len("hello\n")
    assert body["max_chars"] == prompt_store.MAX_CHARS
    assert body["revisions"] == 0


def test_raw_route_is_not_captured_by_the_page_route(env, client):
    assert client.get("/prompt/raw").headers["content-type"].startswith("application/json")


def test_preview_shows_each_mode_s_own_base(env, client):
    # The local base lists tools inline; the Claude base deliberately doesn't.
    local = client.get("/prompt?preview=local").text
    api   = client.get("/prompt?preview=api").text
    assert "Available tools:" in local
    assert "Available tools:" not in api
    assert 'href="/prompt?preview=openrouter"' in local


def test_unknown_preview_mode_falls_back(env, client):
    r = client.get("/prompt?preview=../../etc/passwd")
    assert r.status_code == 200
    assert "Available tools:" in r.text          # the local default


def test_prompt_text_is_escaped_in_the_page(env, client):
    prompt_store.save("<script>alert('x')</script> & <b>bold</b>")
    body = client.get("/prompt").text
    assert "<script>alert" not in body
    assert "&lt;script&gt;alert" in body


# ── The Access gate ───────────────────────────────────────────────────────────

def test_saving_without_access_is_refused_and_writes_nothing(env, client):
    r = client.post("/prompt", data={"text": "sneaked in"}, follow_redirects=False)
    assert r.status_code == 403
    assert prompt_store.read_text() == ""


def test_reset_and_restore_need_access_too(env, client):
    prompt_store.save("keep me")
    assert client.post("/prompt/reset", follow_redirects=False).status_code == 403
    assert client.post("/prompt/restore", data={"revision_id": "abcdef12"},
                       follow_redirects=False).status_code == 403
    assert prompt_store.read_text() == "keep me\n"


def test_access_is_refused_from_a_cross_site_post(env, client, signed_in):
    # A sibling *.bix.computer page carries the Access cookie; Sec-Fetch-Site
    # is what separates it from our own page.
    r = client.post("/prompt", data={"text": "x"},
                    headers={"sec-fetch-site": "same-site"}, follow_redirects=False)
    assert r.status_code == 403
    assert prompt_store.read_text() == ""


# ── Writing ───────────────────────────────────────────────────────────────────

def test_saving_stores_the_text_and_redirects(env, client, signed_in):
    r = client.post("/prompt", data={"text": "Prefer metric units."},
                    follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/prompt?saved=1"
    assert prompt_store.read_text() == "Prefer metric units.\n"


def test_the_saved_page_says_so(env, client, signed_in):
    client.post("/prompt", data={"text": "x"}, follow_redirects=False)
    assert "Saved." in client.get("/prompt?saved=1").text


def test_an_oversize_prompt_is_refused_visibly(env, client, signed_in):
    prompt_store.save("keep me")
    r = client.post("/prompt", data={"text": "x" * (prompt_store.MAX_CHARS + 10)},
                    follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]
    assert prompt_store.read_text() == "keep me\n"      # not truncated, not lost
    assert "the cap is" in client.get(r.headers["location"]).text


def test_an_absurd_body_is_refused_before_parsing(env, client, signed_in):
    r = client.post("/prompt", data={"text": "x" * (main._PROMPT_BODY_MAX + 1)},
                    follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]
    assert prompt_store.read_text() == ""


def test_saving_an_empty_form_clears_the_prompt(env, client, signed_in):
    prompt_store.save("something")
    r = client.post("/prompt", data={"text": ""}, follow_redirects=False)
    assert r.status_code == 303
    assert prompt_store.read_text() == ""


def test_reset_clears_the_prompt(env, client, signed_in):
    prompt_store.save("something")
    r = client.post("/prompt/reset", follow_redirects=False)
    assert r.status_code == 303
    assert prompt_store.read_text() == ""


def test_restore_puts_an_earlier_version_back(env, client, signed_in):
    prompt_store.save("original")
    prompt_store.save("replacement")
    rev_id = prompt_store.revisions()[0]["id"]
    r = client.post("/prompt/restore", data={"revision_id": rev_id},
                    follow_redirects=False)
    assert r.status_code == 303
    assert prompt_store.read_text() == "original\n"


def test_restoring_an_unknown_revision_is_reported(env, client, signed_in):
    prompt_store.save("current")
    r = client.post("/prompt/restore", data={"revision_id": "deadbeef"},
                    follow_redirects=False)
    assert r.status_code == 303
    assert "error=" in r.headers["location"]
    assert prompt_store.read_text() == "current\n"
    assert "unknown revision" in client.get(r.headers["location"]).text


def test_the_restore_button_is_offered_for_each_revision(env, client, signed_in):
    prompt_store.save("original")
    prompt_store.save("replacement")
    rev_id = prompt_store.revisions()[0]["id"]
    body = client.get("/prompt").text
    assert f'value="{rev_id}"' in body
    assert 'action="/prompt/restore"' in body


# ── Containment ───────────────────────────────────────────────────────────────

def test_no_model_facing_tool_can_write_the_prompt(env):
    """The editor's POST is the only writer: the model's four write tools all
    go through workspace.py, which is confined to the todos and outbox dirs."""
    import tools
    import workspace
    assert tools.WRITE_TOOLS == {"add_todo", "add_plan", "complete_todo", "write_outbox"}
    prompt_dir = (config.DATA_DIR / "prompt").resolve()
    for writable in (workspace.todos_dir(), workspace.outbox_dir()):
        assert prompt_dir != writable.resolve()
        assert writable.resolve() not in prompt_dir.parents
