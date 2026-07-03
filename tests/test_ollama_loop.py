"""Tests for streaming.ollama — history/tool_result SSE events and loop governor.

Mirrors tests/test_claude_loop.py for the OpenAI-fn-shaped Ollama path. Key
difference from the Claude path: the injected OLLAMA_SYSTEM message must be
stripped from the emitted `history` before the client re-sends it, or it
duplicates via the insert(0, …) guard on the next request.
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming import ollama as ollama_mod  # noqa: E402


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
    out = []
    for c in chunks:
        for line in c.split("\n"):
            if line.startswith("event:"):
                out.append(line[6:].strip())
    return out


def _data_for(chunks: list[str], event_name: str) -> list[dict]:
    out = []
    for c in chunks:
        ev, data = None, None
        for line in c.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if ev == event_name:
            out.append(data)
    return out


def _turn_lines(chunks: list[dict]) -> list[str]:
    """Raw `data: {...}` lines as r.aiter_lines() yields them, plus [DONE]."""
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return lines


class _FakeResp:
    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return b"{}"


class _FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    def __init__(self, turns: list[list[str]]):
        self._turns = list(turns)
        self.stream_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, **kwargs):
        lines = self._turns[self.stream_calls]
        self.stream_calls += 1
        return _FakeStreamCtx(_FakeResp(lines))


class _FakeTime:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def _apply(patches):
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in patches:
        p.stop()


def test_history_strips_injected_system_message():
    turn0 = _turn_lines([
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])
    fake_client = _FakeClient([turn0])
    messages = [{"role": "user", "content": "hello"}]

    patches = _apply([patch.object(ollama_mod.httpx, "AsyncClient", lambda **kw: fake_client)])
    try:
        chunks = run(_collect(ollama_mod._stream_ollama(messages, "gemma4:26b")))
    finally:
        _stop(patches)

    history = _data_for(chunks, "history")[0]["messages"]
    assert history == messages  # injected OLLAMA_SYSTEM must not appear
    events = _events(chunks)
    assert events.index("history") < events.index("done")


def test_history_keeps_preexisting_non_injected_system_message():
    turn0 = _turn_lines([
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])
    fake_client = _FakeClient([turn0])
    messages = [
        {"role": "system", "content": "custom system prompt, not ours"},
        {"role": "user", "content": "hello"},
    ]

    patches = _apply([patch.object(ollama_mod.httpx, "AsyncClient", lambda **kw: fake_client)])
    try:
        chunks = run(_collect(ollama_mod._stream_ollama(messages, "gemma4:26b")))
    finally:
        _stop(patches)

    history = _data_for(chunks, "history")[0]["messages"]
    assert history == messages  # untouched — this system message isn't ours to strip


def test_tool_turn_emits_tool_result_and_survives_in_history():
    turn0 = _turn_lines([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    turn1 = _turn_lines([
        {"choices": [{"delta": {"content": "done"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])
    fake_client = _FakeClient([turn0, turn1])

    async def fake_execute_tool(name, tool_input):
        assert name == "read_file"
        assert tool_input == {"path": "a.txt"}
        return "FILE CONTENTS"

    messages = [{"role": "user", "content": "read a.txt"}]
    patches = _apply([
        patch.object(ollama_mod.httpx, "AsyncClient", lambda **kw: fake_client),
        patch.object(ollama_mod, "_execute_tool", fake_execute_tool),
    ])
    try:
        chunks = run(_collect(ollama_mod._stream_ollama(messages, "gemma4:26b")))
    finally:
        _stop(patches)

    events = _events(chunks)
    assert "tool_result" in events
    tr = _data_for(chunks, "tool_result")[0]
    assert tr == {"tool_use_id": "call_1", "content": "FILE CONTENTS", "is_error": False}

    history = _data_for(chunks, "history")[0]["messages"]
    assert history[0] == messages[0]  # injected system message stripped
    assert history[1]["role"] == "assistant"
    assert history[1]["tool_calls"][0]["id"] == "call_1"
    assert history[2] == {"role": "tool", "tool_call_id": "call_1", "content": "FILE CONTENTS"}
    assert events.index("history") < events.index("done")


def test_governor_token_budget_stops_before_first_request():
    fake_client = _FakeClient([])  # never consumed
    big_messages = [{"role": "user", "content": "x" * 100}]

    patches = _apply([
        patch.object(ollama_mod.httpx, "AsyncClient", lambda **kw: fake_client),
        patch.object(ollama_mod, "LOOP_MAX_TOKENS", 1),
    ])
    try:
        chunks = run(_collect(ollama_mod._stream_ollama(big_messages, "gemma4:26b")))
    finally:
        _stop(patches)

    assert fake_client.stream_calls == 0
    events = _events(chunks)
    assert events[-1] == "error"
    assert "history" not in events
    err = _data_for(chunks, "error")[-1]
    assert "budget" in err["message"].lower()


def test_governor_wall_clock_budget_stops_loop():
    turn0 = _turn_lines([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "read_file", "arguments": "{}"},
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    fake_client = _FakeClient([turn0])
    fake_time = _FakeTime()

    async def fake_execute_tool(name, tool_input):
        fake_time.t = 999.0
        return "ok"

    patches = _apply([
        patch.object(ollama_mod.httpx, "AsyncClient", lambda **kw: fake_client),
        patch.object(ollama_mod, "_execute_tool", fake_execute_tool),
        patch.object(ollama_mod, "time", fake_time),
        patch.object(ollama_mod, "LOOP_MAX_SECONDS", 1.0),
    ])
    try:
        chunks = run(_collect(ollama_mod._stream_ollama(
            [{"role": "user", "content": "go"}], "gemma4:26b",
        )))
    finally:
        _stop(patches)

    assert fake_client.stream_calls == 1
    events = _events(chunks)
    assert events[-1] == "error"
    assert "history" not in events
