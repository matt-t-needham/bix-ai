"""Tests for OllamaProvider's guardrail layer (rescue-parsing + bounded
retry-with-nudge for genuinely garbled tool-call attempts, using
forge-guardrails' standalone rescue_tool_call/ErrorTracker/retry_nudge —
not WorkflowRunner).

Drives OllamaProvider.stream_turn(...) directly rather than through
_stream_ollama, isolating guardrail behavior from pre-pass/system-prompt
concerns. Reuses the fakes pattern from test_ollama_loop.py.

Key invariant under test: an ordinary clean text answer with no tool-call
attempt at all is ALWAYS accepted immediately, never nudged/retried — only
a structurally-attempted-but-garbled tool call goes through rescue-then-retry.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming.providers import OllamaProvider  # noqa: E402


def run(coro):
    return asyncio.run(coro)


async def _collect(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out


def _turn_lines(chunks: list[dict]) -> list[str]:
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
    """Each entry in `turns` is one HTTP round-trip's raw response lines."""
    def __init__(self, turns: list[list[str]]):
        self._turns = list(turns)
        self.stream_calls = 0
        self.requests: list[dict] = []  # captured request bodies, in call order

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, **kwargs):
        self.requests.append(kwargs.get("json", {}))
        lines = self._turns[self.stream_calls]
        self.stream_calls += 1
        return _FakeStreamCtx(_FakeResp(lines))


def _provider(fake_client, **kwargs):
    return OllamaProvider(
        model="gemma4:26b",
        tools=[{"type": "function", "function": {"name": "list_directory", "description": "", "parameters": {}}},
               {"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}],
        client_factory=lambda: fake_client,
        route=lambda *a, **kw: _noop(),
        system_sentinel="SYS",
        **kwargs,
    )


async def _noop():
    return None


def test_clean_text_accepted_no_retry():
    fake_client = _FakeClient([
        _turn_lines([
            {"choices": [{"delta": {"content": "I'm bix-ai."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]),
    ])
    provider = _provider(fake_client)

    events = run(_collect(provider.stream_turn([{"role": "user", "content": "who are you?"}])))

    assert fake_client.stream_calls == 1
    turn_end = [e for e in events if e["kind"] == "turn_end"][0]
    assert turn_end["tool_use"] is False
    assert provider.guardrail_rescues == 0
    assert provider.guardrail_retries == 0
    assert provider.guardrail_exhausted is False


def test_malformed_tool_call_json_rescued_without_retry():
    # The model nested a full {"tool":...,"args":...} envelope inside the
    # arguments string instead of using it directly — invalid JSON on its
    # own (leading prose), but forge's extract_tool_call strategy can find
    # the embedded envelope.
    garbled_args = 'Sure, calling: {"tool": "list_directory", "args": {"path": "."}} done.'
    fake_client = _FakeClient([
        _turn_lines([
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "list_directory", "arguments": garbled_args},
            }]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]),
    ])
    provider = _provider(fake_client)

    events = run(_collect(provider.stream_turn([{"role": "user", "content": "list files"}])))

    assert fake_client.stream_calls == 1
    turn_end = [e for e in events if e["kind"] == "turn_end"][0]
    assert turn_end["tool_use"] is True
    assert turn_end["tool_calls"][0]["input"] == {"path": "."}
    assert provider.guardrail_rescues == 1
    assert provider.guardrail_retries == 0


def test_unrecoverable_garble_then_retry_succeeds():
    unrescuable_args = "{not json and no embedded tool call at all"
    fake_client = _FakeClient([
        _turn_lines([
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_1",
                "function": {"name": "read_file", "arguments": unrescuable_args},
            }]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]),
        _turn_lines([
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0, "id": "call_2",
                "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
            }]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]),
    ])
    provider = _provider(fake_client)
    original_messages = [{"role": "user", "content": "read a.txt"}]

    events = run(_collect(provider.stream_turn(original_messages)))

    assert fake_client.stream_calls == 2
    turn_end = [e for e in events if e["kind"] == "turn_end"][0]
    assert turn_end["tool_use"] is True
    assert turn_end["tool_calls"][0]["input"] == {"path": "a.txt"}
    assert provider.guardrail_retries == 1
    assert provider.guardrail_rescues == 0
    assert provider.guardrail_exhausted is False

    # The nudge must land in the retry request, and only there.
    from forge import retry_nudge
    second_request_messages = fake_client.requests[1]["messages"]
    assert second_request_messages[-1] == {"role": "user", "content": retry_nudge("")}
    # The caller's original list must never be mutated by the scratch retry.
    assert original_messages == [{"role": "user", "content": "read a.txt"}]


def test_retries_exhausted_best_effort_for_local():
    unrescuable = "{still not json"
    turn = _turn_lines([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "read_file", "arguments": unrescuable},
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    fake_client = _FakeClient([turn, turn])  # max_retries=1 -> 2 total attempts
    provider = _provider(fake_client, on_exhausted="best_effort", max_retries=1)

    events = run(_collect(provider.stream_turn([{"role": "user", "content": "read a.txt"}])))

    assert fake_client.stream_calls == 2
    assert not any(e["kind"] == "provider_error" for e in events)
    turn_end = [e for e in events if e["kind"] == "turn_end"]
    assert len(turn_end) == 1
    assert provider.guardrail_exhausted is True


def test_retries_exhausted_escalates_for_auto():
    unrescuable = "{still not json"
    turn = _turn_lines([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "read_file", "arguments": unrescuable},
        }]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    fake_client = _FakeClient([turn, turn])
    provider = _provider(fake_client, on_exhausted="escalate", max_retries=1)

    events = run(_collect(provider.stream_turn([{"role": "user", "content": "read a.txt"}])))

    assert fake_client.stream_calls == 2
    assert events[-1]["kind"] == "provider_error"
    assert not any(e["kind"] == "turn_end" for e in events)
    assert provider.guardrail_exhausted is True


def test_bare_text_with_embedded_toolcall_is_rescued_not_shown_as_prose():
    prose = 'Let me check that: {"tool": "read_file", "args": {"path": "x.txt"}}'
    fake_client = _FakeClient([
        _turn_lines([
            {"choices": [{"delta": {"content": prose}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]),
    ])
    provider = _provider(fake_client)

    events = run(_collect(provider.stream_turn([{"role": "user", "content": "read x.txt"}])))

    assert fake_client.stream_calls == 1
    turn_end = [e for e in events if e["kind"] == "turn_end"][0]
    assert turn_end["tool_use"] is True
    assert turn_end["tool_calls"][0]["name"] == "read_file"
    assert turn_end["tool_calls"][0]["input"] == {"path": "x.txt"}
    assert provider.guardrail_rescues == 1
