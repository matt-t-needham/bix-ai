"""Attributing each assistant turn to the model that wrote it.

The behaviour these pin: a transcript goes up with every assistant turn tagged,
so the model answering now can see which replies are its own and which came
from another model — and the tags come straight back off again, because the
loop's `history` event folds the sent messages into the stored record and a
persisted tag would be re-tagged on the next turn.
"""
import json

import convstate

TURNS = [{"model": "gemma4:26b", "mode": "local", "from": 1},
         {"model": "claude-opus-4-7", "mode": "api", "from": 3}]


def _conversation():
    return [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": [{"type": "text", "text": "gemma's answer"}]},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": [{"type": "text", "text": "opus's answer"}]},
    ]


# ── Tagging ───────────────────────────────────────────────────────────────────

def test_each_turn_is_tagged_with_its_own_model():
    out = convstate.tag_messages(_conversation(), TURNS)
    assert out[1]["content"][0]["text"] == "[router-model gemma4:26b]\ngemma's answer"
    assert out[3]["content"][0]["text"] == "[router-model claude-opus-4-7]\nopus's answer"


def test_user_turns_are_left_alone():
    out = convstate.tag_messages(_conversation(), TURNS)
    assert out[0] == {"role": "user", "content": "first question"}
    assert out[2] == {"role": "user", "content": "second question"}


def test_the_input_is_not_mutated():
    msgs = _conversation()
    before = json.dumps(msgs)
    convstate.tag_messages(msgs, TURNS)
    assert json.dumps(msgs) == before


def test_a_turn_spans_every_message_until_the_next_one():
    # One turn, three messages: tool_use, tool_result, then the answer.
    msgs = [
        {"role": "user", "content": "read /x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                           "name": "read_file", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": "contents"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
    ]
    out = convstate.tag_messages(msgs, [{"model": "gemma4:26b", "from": 1}])
    # A tool-only turn gets a tag block of its own, so nothing is unattributed.
    assert out[1]["content"][0] == {"type": "text", "text": "[router-model gemma4:26b]"}
    assert out[1]["content"][1]["type"] == "tool_use"
    assert out[3]["content"][0]["text"].startswith("[router-model gemma4:26b]")


def test_plain_string_content_is_tagged_too():
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a reply"}]
    out = convstate.tag_messages(msgs, [{"model": "gemma4:26b", "from": 1}])
    assert out[1]["content"] == "[router-model gemma4:26b]\na reply"


def test_an_offset_maps_a_slice_back_onto_the_record():
    # context.assemble returns only the tail; its first message is record
    # index `offset`, so attribution has to be shifted by it.
    tail = _conversation()[2:]
    out = convstate.tag_messages(tail, TURNS, offset=2)
    assert out[1]["content"][0]["text"].startswith("[router-model claude-opus-4-7]")


def test_turns_before_the_slice_do_not_bleed_in():
    tail = _conversation()[2:]
    out = convstate.tag_messages(tail, TURNS, offset=2)
    assert "gemma4:26b" not in json.dumps(out)


# ── Stripping (the invariant that keeps the record clean) ─────────────────────

def test_tagging_round_trips_exactly():
    msgs = _conversation()
    assert convstate.strip_tags(convstate.tag_messages(msgs, TURNS)) == msgs


def test_round_trip_survives_tool_turns_and_string_content():
    msgs = [
        {"role": "user", "content": "read /x"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                           "name": "read_file", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": "contents"}]},
        {"role": "assistant", "content": "plain string answer"},
    ]
    turns = [{"model": "gemma4:26b", "from": 1}]
    assert convstate.strip_tags(convstate.tag_messages(msgs, turns)) == msgs


def test_stripping_is_idempotent_and_safe_on_untagged_messages():
    msgs = _conversation()
    assert convstate.strip_tags(msgs) == msgs
    once = convstate.strip_tags(convstate.tag_messages(msgs, TURNS))
    assert convstate.strip_tags(once) == msgs


def test_only_the_leading_tag_is_removed():
    # A reply that legitimately talks about the marker keeps its text.
    msgs = [{"role": "assistant",
             "content": "[router-model gemma4:26b]\nI see [router-model x] in the log."}]
    assert convstate.strip_tags(msgs)[0]["content"] == "I see [router-model x] in the log."


def test_a_reply_that_opens_with_a_bracket_is_untouched():
    msgs = [{"role": "assistant", "content": "[note] not one of ours"}]
    assert convstate.strip_tags(msgs) == msgs


# ── Failing open ──────────────────────────────────────────────────────────────

def test_no_turns_means_no_tags():
    msgs = _conversation()
    assert convstate.tag_messages(msgs, []) == msgs
    assert convstate.tag_messages(msgs, None) == msgs


def test_malformed_turn_entries_are_skipped():
    turns = ["nonsense", {"mode": "local"}, {"model": "m", "from": "not an int"},
             {"model": "gemma4:26b", "from": 1}]
    out = convstate.tag_messages(_conversation(), turns)
    assert out[1]["content"][0]["text"].startswith("[router-model gemma4:26b]")
    assert out[3]["content"][0]["text"].startswith("[router-model gemma4:26b]")


def test_out_of_order_turns_are_sorted():
    out = convstate.tag_messages(_conversation(), list(reversed(TURNS)))
    assert out[1]["content"][0]["text"].startswith("[router-model gemma4:26b]")
    assert out[3]["content"][0]["text"].startswith("[router-model claude-opus-4-7]")


def test_messages_of_the_wrong_shape_pass_through():
    assert convstate.tag_messages("not a list", TURNS) == "not a list"
    assert convstate.strip_tags(None) is None
    weird = ["nonsense", {"role": "assistant"}]
    assert convstate.strip_tags(convstate.tag_messages(weird, TURNS)) == weird


def test_a_message_before_the_first_recorded_turn_is_not_tagged():
    # Older records have turns only from the point the sidecar was added.
    msgs = _conversation()
    out = convstate.tag_messages(msgs, [{"model": "claude-opus-4-7", "from": 3}])
    assert out[1] == msgs[1]
    assert out[3]["content"][0]["text"].startswith("[router-model claude-opus-4-7]")
