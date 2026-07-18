"""SSE parse helpers + SSETextCollector (used by ask_staging)."""
from helpers import SSETextCollector, parse_sse_data, parse_sse_event, sse


def test_parse_helpers_roundtrip():
    block = sse("delta", {"text": "hi"}).rstrip("\n")
    assert parse_sse_event(block) == "delta"
    assert parse_sse_data(block) == {"text": "hi"}


def test_parse_helpers_malformed():
    assert parse_sse_event(": keepalive") == ""
    assert parse_sse_data("data: not-json") == {}


def test_collector_accumulates_across_chunk_boundaries():
    stream = (
        sse("status", {"stage": "checking", "message": "…"})
        + sse("delta", {"text": "Hello "})
        + sse("delta", {"text": "world"})
        + sse("done", {})
    )
    c = SSETextCollector()
    # feed in awkward 7-byte chunks — boundaries never align with events
    for i in range(0, len(stream), 7):
        c.feed(stream[i:i + 7])
    assert c.text == "Hello world"
    assert c.errors == []
    assert c.overflow is False


def test_collector_captures_errors():
    c = SSETextCollector()
    c.feed(sse("delta", {"text": "partial"}) + sse("error", {"message": "boom"}))
    assert c.text == "partial"
    assert c.errors == ["boom"]


def test_collector_caps_text():
    c = SSETextCollector(max_chars=10)
    c.feed(sse("delta", {"text": "0123456789ABCDEF"}))
    c.feed(sse("delta", {"text": "more"}))
    assert c.text == "0123456789"
    assert c.overflow is True


def test_collector_ignores_keepalives_and_unknown_events():
    c = SSETextCollector()
    c.feed(": keepalive\n\n" + sse("metrics", {"tps": 1}) + sse("delta", {"text": "x"}))
    assert c.text == "x"
