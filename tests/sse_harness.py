"""Shared fakes and helpers for SSE golden-sequence tests (Phase 4).

The golden tests in test_sse_fixtures.py pin the exact SSE event sequences the
claude/ollama loops emit, so the provider-seam refactor can prove observable
equivalence. Time-dependent fields (elapsed_ms, ttft_ms, tps, preprocess_ms)
are normalised to 0 before comparison; everything else must match exactly.
"""
import json


class FakeResp:
    def __init__(self, lines, status_code=200, body=b"{}"):
        self.status_code = status_code
        self._lines = lines
        self._body = body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aread(self):
        return self._body


class FakeStreamCtx:
    def __init__(self, resp):
        self._resp = resp

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class FakeClient:
    """Reused across all turns of one loop; one queued response per turn."""

    def __init__(self, turns, status_code=200, body=b"{}", raise_on_stream=None):
        self._turns = list(turns)
        self._status = status_code
        self._body = body
        self._raise = raise_on_stream
        self.stream_calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, **kwargs):
        if self._raise is not None:
            raise self._raise
        lines = self._turns[self.stream_calls]
        self.stream_calls += 1
        return FakeStreamCtx(FakeResp(lines, self._status, self._body))


class FakeTime:
    """Controllable monotonic clock — starts at 0, only moves when told to."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t


def claude_turn_lines(events):
    """Anthropic-style `event:`/`data:` lines as r.aiter_lines() yields them."""
    lines = []
    for ev, data in events:
        lines.append(f"event: {ev}")
        lines.append(f"data: {json.dumps(data)}")
        lines.append("")
    return lines


def ollama_turn_lines(chunks):
    """OpenAI-style `data: {...}` lines plus the [DONE] sentinel."""
    lines = [f"data: {json.dumps(c)}" for c in chunks]
    lines.append("data: [DONE]")
    return lines


async def collect(agen):
    out = []
    async for chunk in agen:
        out.append(chunk)
    return out


_TIME_FIELDS = ("elapsed_ms", "ttft_ms", "tps", "preprocess_ms")


def normalise(chunks):
    """Parse raw SSE chunk strings into [{event, data}] with time fields zeroed."""
    out = []
    for c in chunks:
        ev, data = None, None
        for line in c.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if ev is None:
            continue
        if isinstance(data, dict):
            for f in _TIME_FIELDS:
                if f in data:
                    data[f] = 0
        out.append({"event": ev, "data": data})
    return out


def normalise_routing(calls):
    """Routing-log writer call records with time fields zeroed."""
    out = []
    for args, kwargs in calls:
        rec = {"mode": args[0], "model": args[1], **kwargs}
        for f in _TIME_FIELDS:
            if f in rec:
                rec[f] = 0
        out.append(rec)
    return out
