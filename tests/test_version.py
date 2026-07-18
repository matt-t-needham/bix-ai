"""Tests for GET /version and the Phase-0 config surface (role/build identity)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402


@pytest.fixture
def client():
    import main
    return TestClient(main.app)


def test_version_shape(client):
    r = client.get("/version")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"role", "git_sha", "built_at", "started_at"}
    assert body["role"] == "prod"          # env default
    assert body["git_sha"] == "unknown"    # not stamped in dev/test runs
    assert body["built_at"] == "unknown"
    # started_at is a real ISO timestamp, not a placeholder
    from datetime import datetime
    datetime.fromisoformat(body["started_at"])


def test_version_reads_config_at_call_time(client, monkeypatch):
    monkeypatch.setattr(config, "BIX_ROLE", "staging")
    monkeypatch.setattr(config, "GIT_SHA", "abc1234")
    body = client.get("/version").json()
    assert body["role"] == "staging"
    assert body["git_sha"] == "abc1234"


def test_self_trees_track_monkeypatched_fs_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FS_ROOT", tmp_path)
    assert config.self_prod_tree() == tmp_path / "bix-ai"
    assert config.self_staging_tree() == tmp_path / "bix-ai-staging"


def test_self_trees_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SELF_PROD_TREE", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("SELF_STAGING_TREE", str(tmp_path / "elsewhere-staging"))
    assert config.self_prod_tree() == tmp_path / "elsewhere"
    assert config.self_staging_tree() == tmp_path / "elsewhere-staging"
