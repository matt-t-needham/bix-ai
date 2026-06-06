"""Unit tests for staging.py — pure logic, stdlib-only, no Claude/Ollama.

FS_ROOT and STAGING_DIR are pointed at tmp_path via monkeypatch so the
denylists can be exercised against the *write* class specifically.
"""
import pytest

import config
import staging


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FS_ROOT", tmp_path.resolve())
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    return tmp_path


DENIED = [
    ".env",                         # secrets
    "x.sh",                         # shell script
    "scripts/build.py",            # scripts/ dir
    "Dockerfile",                   # container config
    "docker-compose.yml",          # compose config
    ".github/workflows/ci.yml",    # CI
    "bix-ai/main.py",              # self-modification
]


@pytest.mark.parametrize("rel", DENIED)
def test_create_denied(env, rel):
    with pytest.raises(ValueError):
        staging.create(str(env / rel), "data")
    # nothing staged, nothing written
    assert staging.list_records() == []
    assert not (env / rel).exists()


ALLOWED = [
    "bix-blog/content/blog/post.md",
    "ai_graph_mode/app/page.ts",
    "bix-infra/todos/note.md",
]


@pytest.mark.parametrize("rel", ALLOWED)
def test_create_allowed_stages_without_writing(env, rel):
    rec = staging.create(str(env / rel), "hello")
    assert rec["status"] == "pending"
    assert not (env / rel).exists()                 # NOT written live
    assert staging.get(rec["id"])["content"] == "hello"


def test_outside_fs_root_denied(env, tmp_path):
    other = tmp_path.parent / "outside.md"
    with pytest.raises(ValueError):
        staging.create(str(other), "x")


def test_content_too_large(env):
    with pytest.raises(ValueError):
        staging.create(str(env / "big.md"), "x" * 200_001)


def test_approve_writes_file(env):
    target = env / "bix-blog" / "content" / "blog" / "post.md"
    rec = staging.create(str(target), "published")
    result = staging.approve(rec["id"])
    assert result["ok"] is True
    assert target.read_text() == "published"
    assert staging.get(rec["id"])["status"] == "approved"


def test_reject_discards(env):
    target = env / "ai_graph_mode" / "x.ts"
    rec = staging.create(str(target), "nope")
    result = staging.reject(rec["id"])
    assert result["ok"] is True
    assert not target.exists()
    assert staging.get(rec["id"])["status"] == "rejected"


def test_approve_revalidates_denied_target(env):
    # Hand-craft a pending record whose target is a guardrail path, bypassing
    # create()'s up-front check, to prove approve() re-validates.
    rec = staging.create(str(env / "ok.md"), "data")
    rec["target_path"] = str(env / "scripts" / "evil.py")
    staging._write_record(rec)
    result = staging.approve(rec["id"])
    assert result["ok"] is False
    assert "Re-validation failed" in result["message"]
    assert not (env / "scripts" / "evil.py").exists()


def test_double_approve_is_blocked(env):
    rec = staging.create(str(env / "bix-blog" / "post.md"), "v1")
    assert staging.approve(rec["id"])["ok"] is True
    second = staging.approve(rec["id"])
    assert second["ok"] is False
    assert "Already applied" in second["message"]


def test_cannot_approve_rejected(env):
    rec = staging.create(str(env / "a.md"), "x")
    staging.reject(rec["id"])
    assert staging.approve(rec["id"])["ok"] is False
    assert staging.get(rec["id"])["status"] == "rejected"


def test_cannot_reject_applied(env):
    rec = staging.create(str(env / "b.md"), "x")
    staging.approve(rec["id"])
    assert staging.reject(rec["id"])["ok"] is False
    assert staging.get(rec["id"])["status"] == "approved"


def test_parent_traversal_escape_denied(env):
    # FS_ROOT/../sibling resolves outside FS_ROOT and must be refused.
    with pytest.raises(ValueError):
        staging.create(str(env / ".." / "escape.md"), "x")


def test_symlink_escape_denied(env, tmp_path):
    # A symlink inside FS_ROOT pointing outside must be refused (resolve() guard).
    outside = tmp_path.parent / "outside_target_dir"
    outside.mkdir(exist_ok=True)
    link = env / "linkdir"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        staging.create(str(link / "evil.md"), "x")


def test_list_records_pending_first(env):
    r1 = staging.create(str(env / "one.md"), "1")
    r2 = staging.create(str(env / "two.md"), "2")
    staging.approve(r2["id"])                       # r2 -> approved
    recs = staging.list_records()
    assert recs[0]["id"] == r1["id"]                # pending sorts first
    assert recs[0]["status"] == "pending"


def test_record_id_is_sanitised(env):
    # A crafted id cannot escape the staging dir.
    p = staging._record_path("../../etc/passwd")
    assert p.parent == config.STAGING_DIR
    assert p.name == "etcpasswd.json"


def test_approve_read_only_target_stays_pending(env):
    # Parent is a regular file, so mkdir(parents=True) raises OSError —
    # stands in for a read-only mount / permission failure.
    blocker = env / "blocker"
    blocker.write_text("i am a file")
    rec = staging.create(str(blocker / "child.md"), "data")
    result = staging.approve(rec["id"])
    assert result["ok"] is False
    assert staging.get(rec["id"])["status"] == "pending"
