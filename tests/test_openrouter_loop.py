"""streaming.openrouter — the OpenRouter adapter on the shared tool loop.

Pins: unconfigured → one error event; a cap breach refuses before any HTTP;
a text turn with the final usage chunk reports real tokens + a cost in the
routing record; reasoning deltas surface as `thinking` events ahead of the
answer; the attribution headers ride on the request."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
import openrouter  # noqa: E402
from streaming import openrouter as or_mod  # noqa: E402
from tests.sse_harness import FakeClient, collect, normalise, ollama_turn_lines  # noqa: E402


def run(coro):
    return asyncio.run(coro)


INFO = {"id": "qwen/qwen3-coder", "prompt_usd_per_m": 0.3, "completion_usd_per_m": 1.2}


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setattr(config, "OPENROUTER_MAX_USD_PER_REQUEST", 1.0)
    monkeypatch.setattr(config, "OPENROUTER_MAX_USD_PER_DAY", 10.0)

    async def info(model):
        return INFO if model == INFO["id"] else None
    monkeypatch.setattr(openrouter, "model_info", info)
    monkeypatch.setattr(openrouter, "spent_today_usd", lambda *a, **k: 0.0)


class RoutingRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def test_unconfigured_is_one_error(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    events = normalise(run(collect(or_mod._stream_openrouter(
        [{"role": "user", "content": "hi"}], "qwen/qwen3-coder"))))
    assert [e["event"] for e in events] == ["error"]
    assert "not configured" in events[0]["data"]["message"]


def test_cap_refusal_happens_before_any_request(configured, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_MAX_USD_PER_DAY", 0.0001)
    monkeypatch.setattr(openrouter, "spent_today_usd", lambda *a, **k: 0.0001)
    fake_client = FakeClient([])
    with patch.object(or_mod.httpx, "AsyncClient", lambda **kw: fake_client):
        events = normalise(run(collect(or_mod._stream_openrouter(
            [{"role": "user", "content": "hi"}], "qwen/qwen3-coder"))))
    assert events[-1]["event"] == "error"
    assert "daily cap" in events[-1]["data"]["message"]
    assert fake_client.stream_calls == 0


def test_text_turn_reports_real_usage_and_cost(configured):
    turn = ollama_turn_lines([
        {"choices": [{"delta": {"reasoning": "let me think"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 7}},
    ])
    fake_client = FakeClient([turn])
    routing = RoutingRecorder()
    with patch.object(or_mod.httpx, "AsyncClient", lambda **kw: fake_client), \
         patch.object(or_mod, "_write_routing_event", routing):
        chunks = run(collect(or_mod._stream_openrouter(
            [{"role": "user", "content": "hi"}], "qwen/qwen3-coder")))
    events = normalise(chunks)
    names = [e["event"] for e in events]
    assert names.index("thinking") < names.index("delta")
    assert [e["data"]["text"] for e in events if e["event"] == "delta"] == ["Hello", " world"]
    metrics = next(e["data"] for e in events if e["event"] == "metrics")
    assert metrics["input_tokens"] == 120 and metrics["output_tokens"] == 7
    assert names[-2:] == ["history", "done"]
    history = next(e["data"]["messages"] for e in events if e["event"] == "history")
    assert history[0]["role"] == "user"          # injected system prompt stripped
    assert history[-1]["content"] == [{"type": "text", "text": "Hello world"}]
    (args, kwargs), = routing.calls
    assert args == ("openrouter", "qwen/qwen3-coder")
    assert kwargs["input_tokens"] == 120 and kwargs["output_tokens"] == 7
    assert kwargs["est_cost_usd"] == pytest.approx((120 * 0.3 + 7 * 1.2) / 1e6)


def test_request_carries_attribution_headers_and_usage_opt_in(configured):
    seen = {}

    class SpyClient(FakeClient):
        def stream(self, method, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs.get("headers")
            seen["json"] = kwargs.get("json")
            return super().stream(method, url, **kwargs)

    fake_client = SpyClient([ollama_turn_lines([
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    ])])
    with patch.object(or_mod.httpx, "AsyncClient", lambda **kw: fake_client), \
         patch.object(or_mod, "_write_routing_event", RoutingRecorder()):
        run(collect(or_mod._stream_openrouter([{"role": "user", "content": "hi"}], "qwen/qwen3-coder")))
    assert seen["url"] == config.OPENROUTER_URL
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["headers"]["X-OpenRouter-Title"] == config.OPENROUTER_TITLE
    assert seen["json"]["stream_options"] == {"include_usage": True}
    assert seen["json"]["model"] == "qwen/qwen3-coder"
    assert seen["json"]["messages"][0]["role"] == "system"


def test_upstream_error_chunk_surfaces_as_error(configured):
    fake_client = FakeClient([ollama_turn_lines([
        {"error": {"message": "insufficient credits", "code": 402}},
    ])])
    with patch.object(or_mod.httpx, "AsyncClient", lambda **kw: fake_client):
        events = normalise(run(collect(or_mod._stream_openrouter(
            [{"role": "user", "content": "hi"}], "qwen/qwen3-coder"))))
    assert events[-1]["event"] == "error"
    assert "insufficient credits" in events[-1]["data"]["message"]
    assert not any(e["event"] == "done" for e in events)


def test_unknown_but_well_formed_id_is_allowed_unknown_shape_is_not(configured):
    # Unknown-but-valid: cap check runs with no price info (cost 0) and the
    # request proceeds. Malformed: refused with no HTTP.
    fake_client = FakeClient([ollama_turn_lines([
        {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
    ])])
    with patch.object(or_mod.httpx, "AsyncClient", lambda **kw: fake_client), \
         patch.object(or_mod, "_write_routing_event", RoutingRecorder()):
        ok = normalise(run(collect(or_mod._stream_openrouter(
            [{"role": "user", "content": "hi"}], "someone/new-model"))))
        bad = normalise(run(collect(or_mod._stream_openrouter(
            [{"role": "user", "content": "hi"}], "openrouter/auto"))))
    assert ok[-1]["event"] == "done"
    assert bad[-1]["event"] == "error" and fake_client.stream_calls == 1
