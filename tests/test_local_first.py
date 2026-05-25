"""Tests for streaming.local_first — Ollama-first with Claude fallback."""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers import sse  # noqa: E402
from streaming import local_first  # noqa: E402


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for chunk in agen:
        out.append(chunk)
    return out


def _events(chunks: list[str]) -> list[str]:
    """Extract event names from the SSE chunks."""
    return [local_first._parse_sse_event(c) for c in chunks]


def _data_for(chunks: list[str], event_name: str) -> list[dict]:
    """Extract parsed data payloads for events matching event_name."""
    out = []
    for c in chunks:
        if local_first._parse_sse_event(c) != event_name:
            continue
        for line in c.split("\n"):
            if line.startswith("data:"):
                out.append(json.loads(line[5:].strip()))
    return out


def _make_ollama(*events):
    async def fake(messages, model, mode="local", tool_offload=False):
        for ev, payload in events:
            yield sse(ev, payload)
    return fake


def _make_claude(*events):
    async def fake(messages, model, max_tokens, skip_preprocess, mode="api"):
        for ev, payload in events:
            yield sse(ev, payload)
    return fake


def test_clean_local_success_no_escalation():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from Ollama…"}),
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
    # No Claude content snuck in
    deltas = _data_for(chunks, "delta")
    assert all("SHOULD_NOT_APPEAR" not in d["text"] for d in deltas)


def test_silent_escalation_before_first_delta():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from Ollama…"}),
        ("error", {"message": "Ollama: HTTP 500"}),
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
    # Status switch to Claude must be present
    statuses = _data_for(chunks, "status")
    assert any("Switching to Claude" in s["message"] for s in statuses)
    # Claude stream must follow
    assert any(d["text"] == "claude reply" for d in _data_for(chunks, "delta"))


def test_visible_escalation_after_delta():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from Ollama…"}),
        ("delta", {"text": "partial..."}),
        ("error", {"message": "Ollama: connection reset"}),
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
    # fallback_triggered emitted because at least one delta was sent
    assert "fallback_triggered" in events
    fb = _data_for(chunks, "fallback_triggered")
    assert fb[0]["reason"] == "ollama_error"
    # The status switch and Claude delta both follow
    fb_idx    = events.index("fallback_triggered")
    after     = events[fb_idx + 1:]
    assert "status" in after
    assert "delta" in after
    # The 'error' event from Ollama is NOT propagated past the escalation point
    # (the consumer would have shown an error otherwise)
    assert "error" not in events


def test_claude_also_fails():
    fake_ollama = _make_ollama(
        ("status", {"stage": "streaming", "message": "Streaming from Ollama…"}),
        ("error", {"message": "Ollama: HTTP 500"}),
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
    # The final visible event must be an error so the client doesn't hang
    assert events[-1] == "error"
    err = _data_for(chunks, "error")[-1]
    assert "Claude" in err["message"]
