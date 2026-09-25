"""Visibility heartbeats for slow local models.

Two mechanisms under test:
- helpers.with_progress — idle-gap heartbeat wrapper: emits `status` SSE
  events when the wrapped stream is quiet (prompt eval, tool execution),
  passing inner items through untouched.
- OllamaProvider `working` events — throttled progress (growing byte count)
  while output accumulates invisibly: tool-call JSON, or text buffered
  during a guardrail retry. Includes the immediate retry notice.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from helpers import with_progress  # noqa: E402

from test_ollama_guardrail import (  # noqa: E402
    _collect, _FakeClient, _provider, _turn_lines, run,
)


def test_with_progress_heartbeats_during_silence():
    async def slow_gen():
        await asyncio.sleep(0.06)
        yield "one"
        await asyncio.sleep(0.06)
        yield "two"

    events = run(_collect(with_progress(slow_gen(), 0.02, label="gemma4:26b")))

    passthrough = [e for e in events if not (isinstance(e, str) and e.startswith("event: status"))]
    assert passthrough == ["one", "two"]

    first_item = events.index("one")
    before = [e for e in events[:first_item] if isinstance(e, str) and e.startswith("event: status")]
    after  = [e for e in events[first_item + 1:] if isinstance(e, str) and e.startswith("event: status")]
    assert before and "evaluating prompt" in before[0] and "gemma4:26b" in before[0]
    assert after and "still working" in after[0]


def test_with_progress_quiet_stream_passes_through_unchanged():
    async def fast_gen():
        yield "a"
        yield "b"

    events = run(_collect(with_progress(fast_gen(), 5.0)))
    assert events == ["a", "b"]


def test_provider_emits_working_events_while_tool_json_accumulates():
    fake_client = _FakeClient([
        _turn_lines([
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "read_file", "arguments": '{"path": '},
            }]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": '"a.txt"}'},
            }]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]),
    ])
    provider = _provider(fake_client)
    provider.progress_interval = 0  # defeat the wall-clock throttle

    events = run(_collect(provider.stream_turn([{"role": "user", "content": "read a.txt"}])))

    working = [e for e in events if e["kind"] == "working"]
    assert working, "expected working progress events during tool-JSON accumulation"
    assert all("writing read_file call" in e["message"] for e in working)
    turn_end = [e for e in events if e["kind"] == "turn_end"][0]
    assert turn_end["tool_use"] is True  # progress events never break acceptance


def test_retry_notice_is_visible():
    # Attempt 1: structurally-attempted tool call with unrescuable garbage
    # args -> guardrail nudges and retries. Attempt 2: clean text, accepted.
    fake_client = _FakeClient([
        _turn_lines([
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "read_file", "arguments": "$$$ not json %%%"},
            }]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]),
        _turn_lines([
            {"choices": [{"delta": {"content": "Here is a plain answer."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]),
    ])
    provider = _provider(fake_client)

    events = run(_collect(provider.stream_turn([{"role": "user", "content": "read a.txt"}])))

    assert fake_client.stream_calls == 2
    notices = [e for e in events if e["kind"] == "working" and "retrying (attempt 2)" in e["message"]]
    assert notices, "expected a visible retry notice between attempts"
