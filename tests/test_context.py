"""context.py: per-turn context assembly for server-owned conversations.

Pins: budget cut-off keeps the newest turns verbatim, the summary accretes
(never re-summarised), a bigger budget later never re-summarises, blob
pointers folded out of the tail are re-listed, and a failed summary call
fails open (nothing dropped, stats say so)."""
import asyncio

import context
from strategy import BLOB_MARKER


def run(coro):
    return asyncio.run(coro)


def _turn(i, size=4000):
    return [{"role": "user", "content": f"q{i} " + "x" * size},
            {"role": "assistant", "content": f"a{i} " + "y" * size}]


def _conv(n=6, size=4000):
    msgs = []
    for i in range(n):
        msgs += _turn(i, size)
    return msgs


async def fake_summarise(transcript):
    return f"SUMMARY({len(transcript)})"


async def failing_summarise(transcript):
    raise RuntimeError("provider down")


def test_cut_keeps_newest_turns_within_budget():
    msgs = _conv(6)                      # each turn ≈ 2000 tokens
    assert context.cut_for_budget(msgs, 100_000) == 0
    cut = context.cut_for_budget(msgs, 5000)
    assert cut == 8                      # last two turns (4 msgs) fit
    assert context.cut_for_budget(msgs, 10) == 10   # always at least the newest turn


def test_cut_treats_tool_result_turns_as_continuations():
    msgs = [
        {"role": "user", "content": "q0"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "read_file", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "r"}]},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    assert context.turn_starts(msgs) == [0, 4]


def test_assemble_summarises_once_and_accretes():
    msgs = _conv(6)
    sent, summary, st = run(context.assemble(msgs, None, 5000, fake_summarise))
    assert st["summarised"] == 1 and st["summary_failed"] == 0
    assert summary["through"] == 8
    assert st["folded_messages"] == 8 and st["sent_messages"] == 4
    assert sent[0]["content"].startswith(context.CONTEXT_MARKER)
    assert sent[1]["content"] == context.ACK_TEXT
    assert sent[2:] == msgs[8:]          # tail byte-identical

    # Grow by one turn: the old body is carried forward, only the new turn
    # is summarised, previous SUMMARY text is still there verbatim.
    msgs2 = msgs + _turn(6)
    sent2, summary2, st2 = run(context.assemble(msgs2, summary, 5000, fake_summarise))
    assert st2["summarised"] == 1
    assert summary2["text"].startswith(summary["text"])
    assert summary2["text"].count("SUMMARY") == 2
    assert summary2["through"] == 10

    # A larger budget later (switching to a cloud model) never re-summarises
    # and sends the tail from where the summary ends.
    sent3, summary3, st3 = run(context.assemble(msgs2, summary2, 1_000_000, fake_summarise))
    assert st3["summarised"] == 0 and summary3 == summary2
    assert sent3[2:] == msgs2[10:]


def test_assemble_everything_fits_sends_verbatim_no_summary():
    msgs = _conv(2)
    sent, summary, st = run(context.assemble(msgs, None, 1_000_000, fake_summarise))
    assert sent == msgs and summary["through"] == 0 and st["folded_messages"] == 0


def test_assemble_fails_open_when_summary_fails():
    msgs = _conv(6)
    sent, summary, st = run(context.assemble(msgs, None, 5000, failing_summarise))
    assert st["summary_failed"] == 1 and st["summarised"] == 0
    assert sent == msgs                   # nothing dropped
    assert summary["through"] == 0


def test_folded_blob_pointers_are_relisted():
    h = "a" * 64
    msgs = [{"role": "user", "content": f"{BLOB_MARKER} {h}]\nexcerpt"},
            {"role": "assistant", "content": "ok " + "z" * 9000}] + _conv(3)
    sent, summary, st = run(context.assemble(msgs, None, 3000, fake_summarise))
    assert st["folded_messages"] > 0
    assert h in sent[0]["content"]


def test_transcript_renders_tool_turns_compactly():
    msgs = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "read_file", "input": {"path": "/x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "hello"}]},
    ]
    doc = context.render_transcript(msgs)
    assert "called tool read_file" in doc and "tool result, 5 chars" in doc
