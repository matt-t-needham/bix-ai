"""Tests for streaming.claude — history/tool_result SSE events and loop governor.

Phase 1 of PLAN-pi-tools.md: tool turns must survive across requests via a
`history` SSE event at clean completion, `tool_result` must fire on every tool
call (not just the pro/subprocess path), and the loop must stop on a token or
wall-clock budget breach instead of running unbounded.
"""
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming import claude as claude_mod  # noqa: E402


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


def _turn_lines(events: list[tuple[str, dict]]) -> list[str]:
    """Flatten (event, data) pairs into the raw lines r.aiter_lines() yields."""
    lines = []
    for ev, data in events:
        lines.append(f"event: {ev}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
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
    """Reused across all turns of one loop; one queued response per turn."""

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
    """Controllable monotonic clock — starts at 0, only moves when told to."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def _patches(fake_client, fake_execute_tool=None, fake_time=None):
    patches = [
        patch.object(claude_mod.httpx, "AsyncClient", lambda **kw: fake_client),
        patch.object(claude_mod, "_load_recent_memories", lambda n: []),
        patch.object(claude_mod, "_memory_system_prompt", lambda recent: None),
    ]
    if fake_execute_tool is not None:
        patches.append(patch.object(claude_mod, "_execute_tool", fake_execute_tool))
    if fake_time is not None:
        patches.append(patch.object(claude_mod, "time", fake_time))
    return patches


def _apply(patches):
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in patches:
        p.stop()


def test_history_emitted_on_clean_single_turn_completion():
    turn0 = _turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "hi"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 3}}),
    ])
    fake_client = _FakeClient([turn0])
    messages = [{"role": "user", "content": "hello"}]

    patches = _apply(_patches(fake_client))
    try:
        chunks = run(_collect(claude_mod._stream_claude(
            messages, "claude-sonnet-4-6", 4096, skip_preprocess=True,
        )))
    finally:
        _stop(patches)

    events = _events(chunks)
    assert events.index("history") < events.index("done")
    history = _data_for(chunks, "history")[0]
    assert history["messages"] == messages
    assert "tool_result" not in events
    assert "error" not in events


def test_tool_turn_emits_tool_result_and_survives_in_history():
    turn0 = _turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "name": "read_file", "id": "toolu_1"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.txt"}'}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
    ])
    turn1 = _turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 20}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "done"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
    ])
    fake_client = _FakeClient([turn0, turn1])

    async def fake_execute_tool(name, tool_input):
        assert name == "read_file"
        assert tool_input == {"path": "a.txt"}
        return "FILE CONTENTS"

    messages = [{"role": "user", "content": "read a.txt"}]
    patches = _apply(_patches(fake_client, fake_execute_tool=fake_execute_tool))
    try:
        chunks = run(_collect(claude_mod._stream_claude(
            messages, "claude-sonnet-4-6", 4096, skip_preprocess=True,
        )))
    finally:
        _stop(patches)

    events = _events(chunks)
    assert "tool_result" in events
    tr = _data_for(chunks, "tool_result")[0]
    assert tr == {"tool_use_id": "toolu_1", "content": "FILE CONTENTS", "is_error": False}

    history = _data_for(chunks, "history")[0]["messages"]
    assert history[0] == messages[0]
    assert history[1] == {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}}],
    }
    assert history[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "FILE CONTENTS"}],
    }
    assert events.index("history") < events.index("done")
    assert "error" not in events


def test_tool_result_sse_truncated_but_history_keeps_full_content():
    big = "x" * 5000
    turn0 = _turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "name": "read_log", "id": "toolu_1"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
    ])
    turn1 = _turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 20}}}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 2}}),
    ])
    fake_client = _FakeClient([turn0, turn1])

    async def fake_execute_tool(name, tool_input):
        return big

    patches = _apply(_patches(fake_client, fake_execute_tool=fake_execute_tool))
    try:
        chunks = run(_collect(claude_mod._stream_claude(
            [{"role": "user", "content": "read the log"}], "claude-sonnet-4-6", 4096,
            skip_preprocess=True,
        )))
    finally:
        _stop(patches)

    tr = _data_for(chunks, "tool_result")[0]
    assert len(tr["content"]) < len(big)
    assert tr["content"].endswith("(truncated)")

    history = _data_for(chunks, "history")[0]["messages"]
    tool_result_block = history[2]["content"][0]
    assert tool_result_block["content"] == big  # full, untruncated


def test_governor_token_budget_stops_loop_before_next_turn():
    turn0 = _turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 1000}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "name": "read_file", "id": "toolu_1"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 50}}),
    ])
    fake_client = _FakeClient([turn0])

    async def fake_execute_tool(name, tool_input):
        return "ok"

    patches = _apply(_patches(fake_client, fake_execute_tool=fake_execute_tool))
    patches.append(patch.object(claude_mod, "LOOP_MAX_TOKENS", 5))
    _apply(patches[-1:])
    try:
        chunks = run(_collect(claude_mod._stream_claude(
            [{"role": "user", "content": "go"}], "claude-sonnet-4-6", 4096,
            skip_preprocess=True,
        )))
    finally:
        _stop(patches)

    assert fake_client.stream_calls == 1  # turn 1's request never happened
    events = _events(chunks)
    assert events[-1] == "error"
    assert "history" not in events
    assert "done" not in events
    err = _data_for(chunks, "error")[-1]
    assert "budget" in err["message"].lower()


def test_governor_wall_clock_budget_stops_loop():
    turn0 = _turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 10}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "name": "read_file", "id": "toolu_1"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 5}}),
    ])
    fake_client = _FakeClient([turn0])
    fake_time = _FakeTime()

    async def fake_execute_tool(name, tool_input):
        fake_time.t = 999.0  # simulate a slow tool call blowing the wall-clock budget
        return "ok"

    patches = _apply(_patches(fake_client, fake_execute_tool=fake_execute_tool, fake_time=fake_time))
    patches.append(patch.object(claude_mod, "LOOP_MAX_SECONDS", 1.0))
    _apply(patches[-1:])
    try:
        chunks = run(_collect(claude_mod._stream_claude(
            [{"role": "user", "content": "go"}], "claude-sonnet-4-6", 4096,
            skip_preprocess=True,
        )))
    finally:
        _stop(patches)

    assert fake_client.stream_calls == 1
    events = _events(chunks)
    assert events[-1] == "error"
    assert "history" not in events
