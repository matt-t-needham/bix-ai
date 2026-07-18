"""model_admin: name validation, installed-list cache, allowlist fallback.
Fully offline — _fetch_installed is monkeypatched, never a real Ollama call."""
import asyncio

import httpx
import pytest

import config
import model_admin


@pytest.fixture(autouse=True)
def fresh_cache():
    model_admin.invalidate_cache()
    yield
    model_admin.invalidate_cache()


# ── validate_model_name ───────────────────────────────────────────────────────

ACCEPT = [
    ("gemma4", ("library/gemma4", "latest")),
    ("gemma4:26b", ("library/gemma4", "26b")),
    ("qwen2.5:0.5b", ("library/qwen2.5", "0.5b")),
    ("qwen3.5:9b", ("library/qwen3.5", "9b")),
    ("llama3:8b-instruct-q4_K_M", ("library/llama3", "8b-instruct-q4_K_M")),
    ("someuser/custom-model:v1", ("someuser/custom-model", "v1")),
    ("  gemma4:26b  ", ("library/gemma4", "26b")),   # trimmed
]


@pytest.mark.parametrize("name,expected", ACCEPT)
def test_validate_accepts(name, expected):
    assert model_admin.validate_model_name(name) == expected


REJECT = [
    "",
    "   ",
    "UPPER",                          # names are lowercase
    "has space:7b",
    "http://evil.example/x",          # URL
    "../../etc/passwd",               # traversal
    ".hidden",                        # must start alnum
    ":onlytag",
    "a/b/c",                          # at most one repo segment
    "name:",                          # empty tag
    "name:tag:extra",
    "x" * 80,                         # too long
]


@pytest.mark.parametrize("name", REJECT)
def test_validate_rejects(name):
    with pytest.raises(ValueError):
        model_admin.validate_model_name(name)


# ── list_installed cache ──────────────────────────────────────────────────────

def _fake_fetch(models, calls):
    async def fetch():
        calls.append(1)
        return models
    return fetch


def test_list_installed_caches(monkeypatch):
    calls: list = []
    models = [{"name": "gemma4:26b", "size": 1, "modified_at": "", "loaded": False, "size_vram": 0}]
    monkeypatch.setattr(model_admin, "_fetch_installed", _fake_fetch(models, calls))
    assert asyncio.run(model_admin.list_installed()) == models
    assert asyncio.run(model_admin.list_installed()) == models
    assert len(calls) == 1                      # second hit served from cache

    model_admin.invalidate_cache()
    asyncio.run(model_admin.list_installed())
    assert len(calls) == 2                      # invalidate forces a refetch

    asyncio.run(model_admin.list_installed(force=True))
    assert len(calls) == 3                      # force bypasses the cache


# ── is_allowed_local_model ────────────────────────────────────────────────────

def test_allowlist_uses_installed_set(monkeypatch):
    models = [{"name": "tinyllama:latest", "size": 1, "modified_at": "", "loaded": False, "size_vram": 0}]
    monkeypatch.setattr(model_admin, "_fetch_installed", _fake_fetch(models, []))
    assert asyncio.run(model_admin.is_allowed_local_model("tinyllama:latest")) is True
    assert asyncio.run(model_admin.is_allowed_local_model("tinyllama")) is True   # :latest implied
    assert asyncio.run(model_admin.is_allowed_local_model("gemma4:26b")) is False


def test_allowlist_falls_back_when_ollama_down(monkeypatch):
    async def broken():
        raise httpx.ConnectError("no ollama")
    monkeypatch.setattr(model_admin, "_fetch_installed", broken)
    # falls back to the static config list — never blocks chat on an outage
    some_static = next(iter(config._ALLOWED_OLLAMA_MODELS))
    assert asyncio.run(model_admin.is_allowed_local_model(some_static)) is True
    assert asyncio.run(model_admin.is_allowed_local_model("not-a-model:1b")) is False
