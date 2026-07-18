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
    "bix-ai/deploy.sh",            # filename rules still bite inside bix-ai
    "bix-ai/Dockerfile",
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


# ── Self-changes (bix-ai's own source → staging-tree redirect) ────────────────

def test_self_change_flags(env):
    rec = staging.create(str(env / "bix-ai" / "main.py"), "# new")
    assert rec["self_change"] is True
    assert rec["critical"] is True
    assert rec["applied_to"] is None and rec["promoted_at"] is None


def test_self_change_noncritical_file(env):
    rec = staging.create(str(env / "bix-ai" / "static" / "index.html"), "<html>")
    assert rec["self_change"] is True
    assert rec["critical"] is False


def test_non_self_change_not_flagged(env):
    rec = staging.create(str(env / "bix-blog" / "post.md"), "hi")
    assert rec["self_change"] is False
    assert rec["critical"] is False


def test_staging_clone_paths_not_flagged_critical(env):
    # bix-ai-staging is a distinct path part — exact-part match must not fire.
    rec = staging.create(str(env / "bix-ai-staging" / "main.py"), "# direct")
    assert rec["critical"] is False
    assert rec["self_change"] is False


def test_approve_self_change_writes_staging_tree_only(env):
    prod_file = env / "bix-ai" / "strategy.py"
    prod_file.parent.mkdir(parents=True)
    prod_file.write_text("# prod version")
    rec = staging.create(str(prod_file), "# proposed")
    result = staging.approve(rec["id"])
    assert result["ok"] is True
    assert prod_file.read_text() == "# prod version"          # prod untouched
    staged = env / "bix-ai-staging" / "strategy.py"
    assert staged.read_text() == "# proposed"
    rec = staging.get(rec["id"])
    assert rec["applied_to"] == str(staged)
    assert staging.current_source_path(rec) == staged


def test_approve_non_self_change_applies_in_place(env):
    target = env / "bix-blog" / "post.md"
    rec = staging.create(str(target), "content")
    assert staging.approve(rec["id"])["ok"] is True
    assert target.read_text() == "content"
    assert staging.get(rec["id"])["applied_to"] == str(target)


def test_approve_refused_when_staging_tree_symlinks_out(env, tmp_path):
    # bix-ai-staging symlinked outside FS_ROOT: the rewritten apply path must
    # be re-validated and refused; the record stays pending.
    outside = tmp_path.parent / "evil_staging_tree"
    outside.mkdir(exist_ok=True)
    (env / "bix-ai-staging").symlink_to(outside)
    rec = staging.create(str(env / "bix-ai" / "helpers.py"), "# x")
    result = staging.approve(rec["id"])
    assert result["ok"] is False
    assert "Re-validation failed" in result["message"]
    assert staging.get(rec["id"])["status"] == "pending"
    assert not (outside / "helpers.py").exists()


# ── Comments ──────────────────────────────────────────────────────────────────

def test_add_comment_and_resolve(env):
    rec = staging.create(str(env / "c.md"), "alpha beta gamma")
    rec = staging.add_comment(rec["id"], "beta", "unclear phrasing")
    assert len(rec["comments"]) == 1
    c = rec["comments"][0]
    assert c["quote"] == "beta" and c["text"] == "unclear phrasing"
    assert c["resolved"] is False and c["resolved_at"] is None

    rec = staging.set_comment_resolved(rec["id"], c["id"], True)
    assert rec["comments"][0]["resolved"] is True
    assert rec["comments"][0]["resolved_at"] is not None

    rec = staging.set_comment_resolved(rec["id"], c["id"], False)
    assert rec["comments"][0]["resolved"] is False
    assert rec["comments"][0]["resolved_at"] is None


def test_add_comment_validation(env):
    rec = staging.create(str(env / "c.md"), "body")
    with pytest.raises(ValueError):
        staging.add_comment(rec["id"], "q", "   ")
    with pytest.raises(ValueError):
        staging.add_comment(rec["id"], "q" * 2_001, "text")
    with pytest.raises(ValueError):
        staging.add_comment(rec["id"], "q", "t" * 4_001)
    assert staging.get(rec["id"])["comments"] == []


def test_comment_missing_record_or_comment(env):
    assert staging.add_comment("nope", "q", "t") is None
    rec = staging.create(str(env / "c.md"), "body")
    assert staging.set_comment_resolved(rec["id"], "nocomment", True) is None


def test_add_comment_on_legacy_record(env):
    # Records written before the comments feature lack the new keys.
    rec = staging.create(str(env / "c.md"), "body")
    for key in ("comments", "revisions", "review_model"):
        rec.pop(key, None)
    staging._write_record(rec)
    rec = staging.add_comment(rec["id"], "", "works anyway")
    assert len(rec["comments"]) == 1


# ── Revisions ─────────────────────────────────────────────────────────────────

def test_update_content_revises_pending(env):
    target = env / "r.md"
    rec = staging.create(str(target), "v1")
    result = staging.update_content(rec["id"], "v2", "claude:test-model")
    assert result["ok"] is True
    stored = staging.get(rec["id"])
    assert stored["content"] == "v2"
    assert stored["status"] == "pending"
    assert [rv["content"] for rv in stored["revisions"]] == ["v1"]
    assert stored["revisions"][0]["by"] == "claude:test-model"
    assert not target.exists()                       # live file never touched


def test_update_content_only_pending(env):
    rec = staging.create(str(env / "r.md"), "v1")
    staging.reject(rec["id"])
    result = staging.update_content(rec["id"], "v2", "x")
    assert result["ok"] is False
    assert staging.get(rec["id"])["content"] == "v1"


def test_update_content_size_guard(env):
    rec = staging.create(str(env / "r.md"), "v1")
    result = staging.update_content(rec["id"], "x" * 200_001, "x")
    assert result["ok"] is False
    assert staging.get(rec["id"])["content"] == "v1"


def test_update_content_caps_revision_history(env):
    rec = staging.create(str(env / "r.md"), "v0")
    for i in range(1, 8):
        assert staging.update_content(rec["id"], f"v{i}", "x")["ok"] is True
    revs = staging.get(rec["id"])["revisions"]
    assert len(revs) == 5                            # capped
    assert [rv["content"] for rv in revs] == ["v2", "v3", "v4", "v5", "v6"]
