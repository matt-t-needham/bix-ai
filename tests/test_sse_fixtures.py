"""Golden SSE-sequence tests (Phase 4 of PLAN-pi-tools.md).

Fixtures under tests/fixtures/sse/ were recorded from the pre-refactor
claude/ollama loops. Each scenario replays a mocked upstream and asserts the
loop emits the exact same normalised SSE event sequence and routing-log
records. If a fixture is missing, regenerate with:

    RECORD_SSE=1 .venv/bin/python -m pytest tests/test_sse_fixtures.py -q

Only regenerate when a behaviour change is *intended* — the whole point is
that the provider-seam refactor must not change these.
"""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from streaming import claude as claude_mod
from streaming import ollama as ollama_mod
from tests.sse_harness import (
    FakeClient, FakeTime, claude_turn_lines, collect, normalise,
    normalise_routing, ollama_turn_lines,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sse"
RECORD = bool(os.environ.get("RECORD_SSE"))


def run(coro):
    return asyncio.run(coro)


async def fake_execute_tool(name, tool_input):
    return f"RESULT[{name}:{json.dumps(tool_input, sort_keys=True)}]"


class RoutingRecorder:
    def __init__(self):
        self.calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _claude_patches(fake_client, routing, extra=()):
    ps = [
        patch.object(claude_mod.httpx, "AsyncClient", lambda **kw: fake_client),
        patch.object(claude_mod, "_execute_tool", fake_execute_tool),
        patch.object(claude_mod, "_write_routing_event", routing),
        patch.object(claude_mod, "time", FakeTime()),
        *extra,
    ]
    for p in ps:
        p.start()
    return ps


def _ollama_patches(fake_client, routing, extra=()):
    ps = [
        patch.object(ollama_mod.httpx, "AsyncClient", lambda **kw: fake_client),
        patch.object(ollama_mod, "_execute_tool", fake_execute_tool),
        patch.object(ollama_mod, "_write_routing_event", routing),
        *extra,
    ]
    for p in ps:
        p.start()
    return ps


def _stop(ps):
    for p in ps:
        p.stop()


# ── Scenario turn scripts ─────────────────────────────────────────────────────

def _claude_text_turn(text="hello there", in_tok=10, out_tok=4):
    return claude_turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": in_tok}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": out_tok}}),
    ])


def _claude_mixed_tool_turn():
    """Text block followed by a tool_use block in one assistant message."""
    return claude_turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 12}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text"}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Let me check. "}}),
        ("content_block_stop", {"index": 0}),
        ("content_block_start", {"index": 1, "content_block": {"type": "tool_use", "name": "read_file", "id": "toolu_1"}}),
        ("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"path": '}}),
        ("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '"a.txt"}'}}),
        ("content_block_stop", {"index": 1}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
    ])


def _claude_tool_turn(name="read_file", tool_id="toolu_1", args='{"path": "a.txt"}'):
    return claude_turn_lines([
        ("message_start", {"message": {"usage": {"input_tokens": 12}}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "tool_use", "name": name, "id": tool_id}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "input_json_delta", "partial_json": args}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}}),
    ])


def _ollama_text_turn(text="hi from local"):
    return ollama_turn_lines([
        {"choices": [{"delta": {"content": text}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ])


def _ollama_two_tool_turn():
    return ollama_turn_lines([
        {"choices": [{"delta": {"content": "checking"}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
        }]}}]},
        {"choices": [{"delta": {"tool_calls": [{
            "index": 1, "id": "call_2",
            "function": {"name": "list_directory", "arguments": '{"path": "/tmp"}'},
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])


# ── Scenarios: (name, runner) ─────────────────────────────────────────────────

def _run_claude(turns, messages, extra_patches=(), client_kwargs=None):
    fake_client = FakeClient(turns, **(client_kwargs or {}))
    routing = RoutingRecorder()
    ps = _claude_patches(fake_client, routing, extra_patches)
    try:
        chunks = run(collect(claude_mod._stream_claude(
            messages, "claude-sonnet-4-6", 4096, skip_preprocess=True,
        )))
    finally:
        _stop(ps)
    return {"events": normalise(chunks), "routing": normalise_routing(routing.calls)}


def _run_ollama(turns, messages, extra_patches=(), client_kwargs=None, **call_kwargs):
    fake_client = FakeClient(turns, **(client_kwargs or {}))
    routing = RoutingRecorder()
    ps = _ollama_patches(fake_client, routing, extra_patches)
    try:
        chunks = run(collect(ollama_mod._stream_ollama(
            messages, "gemma4:26b", **call_kwargs,
        )))
    finally:
        _stop(ps)
    return {"events": normalise(chunks), "routing": normalise_routing(routing.calls)}


SCENARIOS = {
    "claude_text": lambda: _run_claude(
        [_claude_text_turn()],
        [{"role": "user", "content": "hello"}],
    ),
    "claude_mixed_tool_then_text": lambda: _run_claude(
        [_claude_mixed_tool_turn(), _claude_text_turn("done", in_tok=30, out_tok=2)],
        [{"role": "user", "content": "read a.txt"}],
    ),
    "claude_token_budget_breach": lambda: _run_claude(
        [_claude_tool_turn()],
        [{"role": "user", "content": "go"}],
        extra_patches=(patch.object(claude_mod, "LOOP_MAX_TOKENS", 5),),
    ),
    "claude_max_depth": lambda: _run_claude(
        [_claude_tool_turn(tool_id=f"toolu_{i}") for i in range(10)],
        [{"role": "user", "content": "loop forever"}],
    ),
    "claude_upstream_exception": lambda: _run_claude(
        [],
        [{"role": "user", "content": "hello"}],
        client_kwargs={"raise_on_stream": RuntimeError("connection refused")},
    ),
    "ollama_text": lambda: _run_ollama(
        [_ollama_text_turn()],
        [{"role": "user", "content": "hello"}],
    ),
    "ollama_two_tools_then_text": lambda: _run_ollama(
        [_ollama_two_tool_turn(), _ollama_text_turn("done")],
        [{"role": "user", "content": "read stuff"}],
    ),
    "ollama_token_budget_breach": lambda: _run_ollama(
        [],
        [{"role": "user", "content": "x" * 100}],
        extra_patches=(patch.object(ollama_mod, "LOOP_MAX_TOKENS", 1),),
    ),
    "ollama_http_error": lambda: _run_ollama(
        [["ignored"]],
        [{"role": "user", "content": "hello"}],
        client_kwargs={"status_code": 500, "body": b'{"error": "model not found"}'},
    ),
    "ollama_model_swap_tool_offload": lambda: _run_ollama(
        [_ollama_text_turn()],
        [{"role": "user", "content": "hello"}],
        tool_offload=True,
    ),
}


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_sse_sequence_matches_fixture(name):
    got = SCENARIOS[name]()
    path = FIXTURE_DIR / f"{name}.json"
    if RECORD:
        FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(got, indent=2) + "\n")
        return
    assert path.exists(), f"missing fixture {path} — regenerate with RECORD_SSE=1"
    want = json.loads(path.read_text())
    assert got["events"] == want["events"], f"SSE sequence drifted for {name}"
    assert got["routing"] == want["routing"], f"routing records drifted for {name}"
