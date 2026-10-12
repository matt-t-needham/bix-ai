"""Tests for GET /version (build identity)."""
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
    assert set(body) == {"git_sha", "built_at", "started_at"}
    assert body["git_sha"] == "unknown"    # not stamped in dev/test runs
    assert body["built_at"] == "unknown"
    # started_at is a real ISO timestamp, not a placeholder
    from datetime import datetime
    datetime.fromisoformat(body["started_at"])


def test_version_reads_config_at_call_time(client, monkeypatch):
    monkeypatch.setattr(config, "GIT_SHA", "abc1234")
    body = client.get("/version").json()
    assert body["git_sha"] == "abc1234"

