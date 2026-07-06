"""Tests for streaming.local_first — Ollama-first with Claude fallback."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402
from helpers import sse  # noqa: E402
from streaming import local_first  # noqa: E402


def run(coro):
    # asyncio.run: get_event_loop() breaks under Python 3.12 once any earlier
    # test has used asyncio.run (no current loop is left on the policy).
    return asyncio.run(coro)


async def _collect(agen):
    out = []
    async for chunk in agen:
        out.append(chunk)
    return out


def _events(chunks: list[str]) -> list[str]:
    return [local_first._parse_sse_event(c) for c in chunks]


def _data_for(chunks: list[str], event_name: str) -> list[dict]:
    out = []
    for c in chunks:
        if local_first._parse_sse_event(c) != event_name:
            continue
        for line in c.split("\n"):
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


def _make_ollama(*events):
    async def fake(messages, model, mode="local", tool_offload=False,
                   route_reason="", on_exhausted="best_effort", **kwargs):
        for ev, payload in events:
            yield sse(ev, payload)
    return fake


def _make_claude(*events):
    async def fake(messages, model, max_tokens, skip_preprocess, mode="api", **kwargs):
        for ev, payload in events:
            yield sse(ev, payload)
    return fake


def test_clean_forge_success_no_escalation():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from gemma4:26b…"}),
        ("delta", {"text": "hello "}),
        ("delta", {"text": "world"}),
        ("metrics", {"output_tokens": 2, "tps": 10}),
        ("done", {}),
    )
    fake_claude = _make_claude(("delta", {"text": "SHOULD_NOT_APPEAR"}))

    with patch.object(local_first, "_stream_ollama", fake_ollama), \
         patch.object(local_first, "_stream_claude", fake_claude):
        chunks = run(_collect(local_first._stream_local_first(
            [{"role": "user", "content": "hi"}],
            ollama_model="gemma4:26b", claude_model="claude-sonnet-4-6",
            max_tokens=4096, mode="auto",
        )))

    events = _events(chunks)
    assert "fallback_triggered" not in events
    assert "delta" in events and "done" in events
    deltas = _data_for(chunks, "delta")
    assert all("SHOULD_NOT_APPEAR" not in d["text"] for d in deltas)


def test_silent_escalation_before_first_delta():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from gemma4:26b…"}),
        ("error", {"message": "Forge: ToolCallError"}),
    )
    fake_claude = _make_claude(
        ("delta", {"text": "claude reply"}),
        ("done", {}),
    )

    with patch.object(local_first, "_stream_ollama", fake_ollama), \
         patch.object(local_first, "_stream_claude", fake_claude):
        chunks = run(_collect(local_first._stream_local_first(
            [{"role": "user", "content": "hi"}],
            ollama_model="gemma4:26b", claude_model="claude-sonnet-4-6",
            max_tokens=4096, mode="auto",
        )))

    events = _events(chunks)
    assert "fallback_triggered" not in events  # nothing visible to clear
    statuses = _data_for(chunks, "status")
    assert any("Switching to Claude" in s["message"] for s in statuses)
    assert any(d["text"] == "claude reply" for d in _data_for(chunks, "delta"))


def test_visible_escalation_after_delta():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from gemma4:26b…"}),
        ("delta", {"text": "partial..."}),
        ("error", {"message": "Forge: connection reset"}),
    )
    fake_claude = _make_claude(
        ("delta", {"text": "claude takeover"}),
        ("done", {}),
    )

    with patch.object(local_first, "_stream_ollama", fake_ollama), \
         patch.object(local_first, "_stream_claude", fake_claude):
        chunks = run(_collect(local_first._stream_local_first(
            [{"role": "user", "content": "hi"}],
            ollama_model="gemma4:26b", claude_model="claude-sonnet-4-6",
            max_tokens=4096, mode="auto",
        )))

    events = _events(chunks)
    assert "fallback_triggered" in events
    fb = _data_for(chunks, "fallback_triggered")
    assert fb[0]["reason"] == "ollama_error"
    fb_idx = events.index("fallback_triggered")
    after  = events[fb_idx + 1:]
    assert "status" in after
    assert "delta" in after
    # The Ollama `error` event must not leak past the escalation boundary —
    # the consumer would render an error and stop.
    assert "error" not in events


def test_claude_also_fails():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from gemma4:26b…"}),
        ("error", {"message": "Forge: ToolExecutionError"}),
    )
    fake_claude = _make_claude(
        ("error", {"message": "Claude: upstream error"}),
    )

    with patch.object(local_first, "_stream_ollama", fake_ollama), \
         patch.object(local_first, "_stream_claude", fake_claude):
        chunks = run(_collect(local_first._stream_local_first(
            [{"role": "user", "content": "hi"}],
            ollama_model="gemma4:26b", claude_model="claude-sonnet-4-6",
            max_tokens=4096, mode="auto",
        )))

    events = _events(chunks)
    assert events[-1] == "error"
    err = _data_for(chunks, "error")[-1]
    assert "Claude" in err["message"]


def test_error_reason_appears_in_switching_status():
    """The error message from Ollama must surface in the Claude-switch status."""
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from gemma4:26b…"}),
        ("error", {"message": "Forge: max_retries_per_step exceeded"}),
    )
    fake_claude = _make_claude(
        ("delta", {"text": "ok"}),
        ("done", {}),
    )

    with patch.object(local_first, "_stream_ollama", fake_ollama), \
         patch.object(local_first, "_stream_claude", fake_claude):
        chunks = run(_collect(local_first._stream_local_first(
            [{"role": "user", "content": "hi"}],
            ollama_model="gemma4:26b", claude_model="claude-sonnet-4-6",
            max_tokens=4096, mode="auto",
        )))

    statuses = _data_for(chunks, "status")
    # Find the "Switching to Claude" status — must contain the error reason.
    switch = next((s for s in statuses if "Switching to Claude" in s["message"]), None)
    assert switch is not None
    assert "max_retries_per_step exceeded" in switch["message"]


def _capture_claude(seen):
    async def fake(messages, model, max_tokens, skip_preprocess, mode="api", **kwargs):
        seen["model"] = model
        seen["route_reason"] = kwargs.get("route_reason", "")
        yield sse("delta", {"text": "ok"})
        yield sse("done", {})
    return fake


def test_size_only_claude_route_downshifts_to_cheap_model():
    # A long request with no code/prose/multi-step intent routes to Claude on
    # size alone — that should run on the cheap tier, not the selected model.
    seen = {}
    with patch.object(local_first, "_stream_claude", _capture_claude(seen)):
        chunks = run(_collect(local_first._stream_local_first(
            [{"role": "user", "content": "summarise this " + "blah " * 2000}],
            ollama_model="gemma4:26b", claude_model="claude-sonnet-4-6",
            max_tokens=4096, mode="auto",
        )))

    assert seen["model"] == config.ROUTING_CHEAP_CLAUDE_MODEL
    assert "downshift" in seen["route_reason"]
    statuses = _data_for(chunks, "status")
    assert any("auto → Haiku" in s["message"] for s in statuses)


def test_intent_claude_route_keeps_selected_model():
    seen = {}
    with patch.object(local_first, "_stream_claude", _capture_claude(seen)):
        run(_collect(local_first._stream_local_first(
            [{"role": "user", "content": "write a function that parses nginx logs"}],
            ollama_model="gemma4:26b", claude_model="claude-sonnet-4-6",
            max_tokens=4096, mode="auto",
        )))

    assert seen["model"] == "claude-sonnet-4-6"
    assert "downshift" not in seen["route_reason"]
