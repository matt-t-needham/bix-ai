"""Conversation-tail compaction tests (Phase 5 of PLAN-pi-tools.md).

Pins: trigger accounting (blob excerpts excluded), byte-identical recent
turns, marker discipline (accretion, never re-summarising the old body),
artifact-pointer carry-forward, valid user/assistant alternation, and
fail-open behaviour.
"""
import asyncio

import pytest

import compact
import strategy


def run(coro):
    return asyncio.run(coro)


async def fake_ollama(model, messages):
    return "COMPACT SUMMARY OF OLD TURNS"


async def failing_ollama(model, messages):
    raise RuntimeError("ollama down")


def _turn(i, size=6000):
    """One user+assistant pair of ~size chars each (~size/2 tokens total)."""
    return [
        {"role": "user", "content": f"question {i} " + ("q" * size)},
        {"role": "assistant", "content": f"answer {i} " + ("a" * size)},
    ]


def _long_conv(turns=8, size=6000):
    msgs = []
    for i in range(turns):
        msgs.extend(_turn(i, size))
    return msgs


def test_short_conversation_untouched():
    msgs = _long_conv(turns=2, size=100)
    new, stats = run(compact.compact(msgs, failing_ollama))  # must not even call ollama
    assert new == msgs
    assert stats == {"compacted": 0, "folded": 0, "failed": 0}


def test_blob_excerpts_do_not_count_toward_trigger():
    pointer = f"{strategy.BLOB_MARKER} {'a' * 64}]\n" + ("x" * 200000) + "\n[end-router-blob]"
    msgs = []
    for i in range(6):
        msgs.extend([
            {"role": "user", "content": pointer},
            {"role": "assistant", "content": f"short answer {i}"},
        ])
    assert compact.narrative_tokens(msgs) < 1000
    assert not compact.should_compact(msgs)


def test_compaction_keeps_recent_turns_byte_identical():
    msgs = _long_conv(turns=8)
    assert compact.should_compact(msgs)
    new, stats = run(compact.compact(msgs, fake_ollama))
    assert stats["compacted"] == 1

    # Head: the compact pair, valid user/assistant alternation.
    assert new[0]["role"] == "user"
    assert new[0]["content"].startswith(strategy.COMPACT_MARKER)
    assert new[0]["content"].rstrip().endswith(strategy.COMPACT_END_MARKER)
    assert "COMPACT SUMMARY OF OLD TURNS" in new[0]["content"]
    assert new[1]["role"] == "assistant"

    # Tail: the last COMPACT_KEEP_TURNS user turns, byte-identical objects.
    from config import COMPACT_KEEP_TURNS
    expected_tail = msgs[-(COMPACT_KEEP_TURNS * 2):]
    assert new[2:] == expected_tail
    assert len(new) == 2 + len(expected_tail)


def test_compact_output_converges_no_retrigger():
    msgs = _long_conv(turns=8)
    once, stats = run(compact.compact(msgs, fake_ollama))
    assert stats["compacted"] == 1
    # Narrative shrank below threshold? Recent turns are large fixtures here,
    # so instead assert idempotence directly: a second pass with an unchanged
    # conversation must not fold the compact pair away or re-summarise it.
    twice, stats2 = run(compact.compact(once, failing_ollama))
    if stats2["compacted"]:
        pytest.fail("re-compacted an unchanged conversation")
    assert twice == once


def test_regrowth_accretes_old_body_verbatim_without_resummarising():
    msgs = _long_conv(turns=8)
    once, _ = run(compact.compact(msgs, fake_ollama))

    # Conversation grows: several new large turns arrive after the tail.
    grown = once + _long_conv(turns=5)

    seen_prompts = []

    async def spy_ollama(model, messages):
        seen_prompts.append(messages[-1]["content"])
        return "SECOND SUMMARY"

    again, stats = run(compact.compact(grown, spy_ollama))
    assert stats["compacted"] == 1
    body = again[0]["content"]
    # Old summary carried forward verbatim, new one accreted below it.
    assert "COMPACT SUMMARY OF OLD TURNS" in body
    assert "SECOND SUMMARY" in body
    # The old compact body was NOT shown to the model for re-summarisation.
    assert "COMPACT SUMMARY OF OLD TURNS" not in seen_prompts[0]


def test_artifact_pointers_survive_into_compact_body():
    h = "b" * 64
    pointer = f"{strategy.BLOB_MARKER} {h}]\nlogfile: excerpt\n[end-router-blob]"
    msgs = [{"role": "user", "content": pointer},
            {"role": "assistant", "content": "looked at the log"}]
    msgs += _long_conv(turns=7)
    new, stats = run(compact.compact(msgs, fake_ollama))
    assert stats["compacted"] == 1
    assert f"{strategy.BLOB_MARKER} {h}]" in new[0]["content"]
    # And the request pinner finds it in the compacted history.
    assert h in strategy.referenced_blob_hashes(new)


def test_tool_result_messages_are_not_cut_points():
    msgs = _long_conv(turns=6)
    # A tool exchange inside the conversation: assistant tool_use, then a
    # user message that is a tool_result continuation, then assistant text.
    msgs += [
        {"role": "user", "content": "check the log " + "x" * 6000},
        {"role": "assistant", "content": [
            {"type": "text", "text": "checking"},
            {"type": "tool_use", "id": "t1", "name": "read_log", "input": {"path": "/l"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "log lines " + "y" * 6000},
        ]},
        {"role": "assistant", "content": "the log says hi"},
    ]
    msgs += _long_conv(turns=2)
    new, stats = run(compact.compact(msgs, fake_ollama))
    assert stats["compacted"] == 1
    # First kept message must be a plain user turn-start, never a tool_result.
    first_kept = new[2]
    assert first_kept["role"] == "user"
    assert not isinstance(first_kept["content"], list) or not any(
        b.get("type") == "tool_result" for b in first_kept["content"] if isinstance(b, dict)
    )


def test_ollama_failure_leaves_messages_untouched():
    msgs = _long_conv(turns=8)
    new, stats = run(compact.compact(msgs, failing_ollama))
    assert new == msgs
    assert stats["compacted"] == 0
    assert stats["failed"] == 1


def test_marker_registered_with_strategy_skip_check():
    big_compact = strategy.COMPACT_MARKER + "\n" + ("x" * 30000)
    assert strategy.is_already_summarised(big_compact)
    body = {"messages": [{"role": "user", "content": big_compact}]}
    assert not strategy.has_oversized_blocks(body)
