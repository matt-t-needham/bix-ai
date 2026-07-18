"""BIX_ROLE gating: the staging twin is read-only — no mutating tools, no
mutating routes. Prod keeps everything."""
import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("forge")

from fastapi.testclient import TestClient  # noqa: E402

import config   # noqa: E402
import tools    # noqa: E402


# ── Registry filtering ────────────────────────────────────────────────────────

def _names_fs(entries):
    return {t["name"] for t in entries}


def _names_ollama(entries):
    return {t["function"]["name"] for t in entries}


def test_prod_registries_include_stage_write():
    assert "stage_write" in _names_fs(tools.fs_tools_for_role("prod"))
    assert "stage_write" in _names_ollama(tools.ollama_tools_for_role("prod"))
    assert "stage_write" in {t["name"] for t in tools.tool_table_for_role("prod")}


def test_staging_registries_omit_mutating_tools():
    assert "stage_write" not in _names_fs(tools.fs_tools_for_role("staging"))
    assert "stage_write" not in _names_ollama(tools.ollama_tools_for_role("staging"))
    assert "stage_write" not in {t["name"] for t in tools.tool_table_for_role("staging")}


def test_staging_registries_keep_read_tools():
    names = _names_fs(tools.fs_tools_for_role("staging"))
    assert {"list_directory", "read_file", "read_blob"} <= names


def test_prod_only_tools_filtered_for_staging():
    prod = _names_fs(tools.fs_tools_for_role("prod"))
    stag = _names_fs(tools.fs_tools_for_role("staging"))
    assert {"check_staging", "ask_staging"} <= prod
    assert not ({"check_staging", "ask_staging"} & stag)
    assert not ({"check_staging", "ask_staging"} & _names_ollama(tools.ollama_tools_for_role("staging")))


# ── Call-time refusal (defence in depth) ─────────────────────────────────────

def test_execute_tool_refuses_mutating_on_staging(monkeypatch):
    monkeypatch.setattr(config, "BIX_ROLE", "staging")
    out = asyncio.run(tools._execute_tool("stage_write", {"target_path": "/x", "content": "y"}))
    assert "read-only" in out and "staging" in out


def test_execute_prod_only_tool_refused_on_staging(monkeypatch):
    # Recursion guard: the staging twin must never query itself.
    monkeypatch.setattr(config, "BIX_ROLE", "staging")
    out = asyncio.run(tools._execute_tool("check_staging", {}))
    assert "read-only" in out


def test_check_staging_unreachable_message(monkeypatch):
    # Offline: the staging hostname never resolves in tests, so the tool must
    # return its clear "unreachable" guidance rather than raising.
    monkeypatch.setattr(config, "BIX_ROLE", "prod")
    monkeypatch.setattr(config, "STAGING_ROUTER_URL", "http://bix-test-nonexistent-host:1")
    out = asyncio.run(tools._execute_tool("check_staging", {}))
    assert "unreachable" in out and "Deploy staging" in out


def test_execute_tool_allows_read_on_staging(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "BIX_ROLE", "staging")
    monkeypatch.setattr(config, "FS_ROOT", tmp_path.resolve())
    out = asyncio.run(tools._execute_tool("read_blob", {"hash": "0" * 64}))
    assert "read-only" not in out    # executes (blob missing is fine)


# ── MCP dispatch refusal ─────────────────────────────────────────────────────

def test_mcp_write_file_refused_on_staging(monkeypatch):
    import bix_mcp
    monkeypatch.setattr(bix_mcp, "_ROLE", "staging")
    result = bix_mcp._execute("write_file", {"path": "/tmp/x", "content": "y"})
    assert result.get("isError") is True
    assert "read-only" in result["content"][0]["text"]


# ── Route gating ─────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FS_ROOT", tmp_path.resolve())
    monkeypatch.setattr(config, "STAGING_DIR", tmp_path / "staging")
    import main
    return TestClient(main.app)


MUTATING_STAGING_ROUTES = [
    "/staging/abc123/approve",
    "/staging/abc123/reject",
    "/staging/abc123/review",
]


@pytest.mark.parametrize("route", MUTATING_STAGING_ROUTES)
def test_mutating_routes_405_on_staging(client, monkeypatch, route):
    monkeypatch.setattr(config, "BIX_ROLE", "staging")
    r = client.post(route)
    assert r.status_code == 405
    assert "read-only" in r.json()["detail"]


@pytest.mark.parametrize("route", MUTATING_STAGING_ROUTES)
def test_mutating_routes_reachable_on_prod(client, monkeypatch, route):
    monkeypatch.setattr(config, "BIX_ROLE", "prod")
    r = client.post(route, follow_redirects=False)
    assert r.status_code != 405     # missing record, not role-blocked


def test_get_routes_open_on_staging(client, monkeypatch):
    monkeypatch.setattr(config, "BIX_ROLE", "staging")
    assert client.get("/staging").status_code == 200
    assert client.get("/version").json()["role"] == "staging"
