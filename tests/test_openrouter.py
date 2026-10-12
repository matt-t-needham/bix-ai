"""openrouter.py: model-id shape check, catalogue normalisation + fallback,
cost estimate, spend caps, daily spend from the routing log. Fully offline —
_fetch_catalogue is monkeypatched, never a real HTTP call."""
import asyncio
import json

import pytest

import config
import openrouter


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def fresh_cache():
    openrouter.invalidate_cache()
    yield
    openrouter.invalidate_cache()


# ── is_valid_model_id ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("mid", [
    "qwen/qwen3-coder", "deepseek/deepseek-chat-v3-0324", "meta-llama/llama-4-maverick",
    "google/gemini-2.5-flash", "mistralai/mistral-small-3.2-24b-instruct", "qwen/qwen3-coder:free",
])
def test_valid_ids(mid):
    assert openrouter.is_valid_model_id(mid)


@pytest.mark.parametrize("mid", [
    "", "qwen3-coder", "openrouter/auto", "a/b/c", "UPPER/case", "has space/x",
    "../x/y", "qwen/", "/qwen", "x" * 70 + "/y",
])
def test_invalid_ids(mid):
    assert not openrouter.is_valid_model_id(mid)


# ── catalogue ─────────────────────────────────────────────────────────────────

RAW = [
    {"id": "qwen/qwen3-coder", "name": "Qwen3 Coder", "context_length": 262144,
     "pricing": {"prompt": "0.0000003", "completion": "0.0000012"},
     "architecture": {"output_modalities": ["text"]}, "supported_parameters": ["tools"]},
    {"id": "openrouter/auto", "name": "Auto Router", "pricing": {"prompt": "-1", "completion": "-1"}},
    {"id": "openai/gpt-4o-audio-preview", "name": "audio", "pricing": {"prompt": "0", "completion": "0"},
     "architecture": {"output_modalities": ["text", "audio"]}},
    {"id": "deepseek/deepseek-r1", "name": "R1", "pricing": {"prompt": "0.00000055", "completion": "0.00000219"},
     "reasoning": {"mandatory": True}},
    {"id": "not-a-slug", "name": "junk"},
]


def test_normalise_filters_and_prices():
    models = [m for m in (openrouter._normalise(e) for e in RAW) if m]
    ids = [m["id"] for m in models]
    assert ids == ["qwen/qwen3-coder", "deepseek/deepseek-r1"]
    q = models[0]
    assert q["prompt_usd_per_m"] == pytest.approx(0.3)
    assert q["completion_usd_per_m"] == pytest.approx(1.2)
    assert q["supports_tools"] is True
    assert q["vendor"] == "qwen"
    assert models[1]["reasoning_mandatory"] is True


def test_list_models_falls_back_to_static(monkeypatch):
    async def boom():
        raise OSError("no network")
    monkeypatch.setattr(openrouter, "_fetch_catalogue", boom)
    models, source = run(openrouter.list_models())
    assert source == "static"
    assert any(m["id"] == "qwen/qwen3-coder" for m in models)


def test_list_models_caches_live_result(monkeypatch):
    calls = {"n": 0}

    async def fake():
        calls["n"] += 1
        return [openrouter._normalise(RAW[0])]
    monkeypatch.setattr(openrouter, "_fetch_catalogue", fake)
    m1, s1 = run(openrouter.list_models())
    m2, s2 = run(openrouter.list_models())
    assert s1 == s2 == "live" and m1 == m2
    assert calls["n"] == 1
    assert run(openrouter.model_info("qwen/qwen3-coder"))["id"] == "qwen/qwen3-coder"
    assert run(openrouter.model_info("nope/nope")) is None


# ── cost + caps ───────────────────────────────────────────────────────────────

def test_estimate_cost():
    info = {"prompt_usd_per_m": 0.3, "completion_usd_per_m": 1.2}
    assert openrouter.estimate_cost_usd(info, 1_000_000, 0) == pytest.approx(0.3)
    assert openrouter.estimate_cost_usd(info, 10_000, 1_000) == pytest.approx(0.0042)
    assert openrouter.estimate_cost_usd(None, 10_000, 1_000) == 0.0


def test_check_caps(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MAX_USD_PER_REQUEST", 0.10)
    monkeypatch.setattr(config, "OPENROUTER_MAX_USD_PER_DAY", 1.00)
    assert openrouter.check_caps(0.05, 0.50) is None
    assert "per-request cap" in openrouter.check_caps(0.11, 0.0)
    assert "daily cap" in openrouter.check_caps(0.05, 0.96)
    monkeypatch.setattr(config, "OPENROUTER_MAX_USD_PER_REQUEST", 0)
    monkeypatch.setattr(config, "OPENROUTER_MAX_USD_PER_DAY", 0)
    assert openrouter.check_caps(99, 99) is None   # 0 disables


def test_spent_today_sums_only_openrouter_today(tmp_path):
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%dT10:00:00Z")
    log = tmp_path / "routing.ndjson"
    log.write_text("\n".join([
        json.dumps({"ts": today, "mode": "openrouter", "est_cost_usd": 0.25}),
        json.dumps({"ts": today, "mode": "openrouter", "est_cost_usd": 0.05}),
        json.dumps({"ts": today, "mode": "api", "est_cost_usd": 3.0}),
        json.dumps({"ts": "2000-01-01T00:00:00Z", "mode": "openrouter", "est_cost_usd": 9.0}),
        "not json",
    ]) + "\n")
    assert openrouter.spent_today_usd(log) == pytest.approx(0.30)
    assert openrouter.spent_today_usd(tmp_path / "missing.ndjson") == 0.0


def test_headers_carry_our_attribution(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
    h = openrouter.headers()
    assert h["Authorization"] == "Bearer sk-test"
    assert h["X-OpenRouter-Title"] == config.OPENROUTER_TITLE
    assert h["HTTP-Referer"] == config.OPENROUTER_REFERER
