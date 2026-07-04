"""Tests for the forge runner's end-of-run safety net (_unstreamed_tail).

The safety net emits whatever part of runner.run()'s final result wasn't
already streamed as `delta` events. Historically it only accounted for
respond-tool deltas (`_respond_text_sent`), so a reply streamed via plain
TEXT_DELTA chunks was re-emitted in full at the end — the "answer repeats
itself" quirk. These tests pin the corrected behaviour.
"""
from streaming.forge_runner import _NudgeAttemptCounter, _sampling_for, _unstreamed_tail


def test_reply_streamed_as_text_deltas_is_not_repeated():
    reply = "The capital of France is Paris."
    # TEXT_DELTA streaming never advances the respond offset.
    assert _unstreamed_tail(reply, 0, reply) == ""


def test_reply_fully_streamed_via_respond_deltas_emits_nothing():
    reply = "Staged the file for review."
    assert _unstreamed_tail(reply, len(reply), reply) == ""


def test_partially_streamed_respond_emits_only_the_tail():
    reply = "First half. Second half."
    sent = len("First half. ")
    assert _unstreamed_tail(reply, sent, "First half. ") == "Second half."


def test_non_streaming_path_emits_full_result():
    reply = "Nothing was streamed for this run."
    assert _unstreamed_tail(reply, 0, "") == reply


def test_result_embedded_in_longer_streamed_text_is_skipped():
    # e.g. the model streamed preamble text and then the respond message;
    # the result is the respond message alone.
    streamed = "Let me check that for you. The answer is 42."
    assert _unstreamed_tail("The answer is 42.", 0, streamed) == ""


def test_sampling_for_gemma4_returns_recommended_profile():
    assert _sampling_for("gemma4:26b") == {
        "temperature": 1.0, "top_p": 0.95, "top_k": 64,
    }


def test_sampling_for_non_gemma_model_returns_empty():
    assert _sampling_for("qwen3.5:9b") == {}


def test_nudge_counter_increments_on_each_nudge():
    c = _NudgeAttemptCounter()
    assert c.on_nudge() == 1
    assert c.on_nudge() == 2
    assert c.on_nudge() == 3


def test_nudge_counter_resets_on_tool_call():
    c = _NudgeAttemptCounter()
    c.on_nudge()
    c.on_nudge()
    c.on_tool_call()
    assert c.on_nudge() == 1
