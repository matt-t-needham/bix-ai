"""conversations.py: the server-owned chat store. CONVERSATIONS_DIR is pointed
at tmp_path, same call-time-config pattern as blobstore."""
import pytest

import config
import conversations


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    return tmp_path


def test_create_get_list_delete(env):
    rec = conversations.create("gemma4:26b", "local")
    assert conversations.is_valid_id(rec["id"])
    assert rec["messages"] == [] and rec["summary"] == {"text": "", "through": 0}
    conversations.append_user(rec, {"role": "user", "content": "Hello there\nsecond line"})
    conversations.save(rec)
    got = conversations.get(rec["id"])
    assert got["title"] == "Hello there"
    assert got["messages"][0]["content"].startswith("Hello there")
    listing = conversations.list_all()
    assert [c["id"] for c in listing] == [rec["id"]]
    assert listing[0]["messages"] == 1 and listing[0]["mode"] == "local"
    assert conversations.delete(rec["id"]) is True
    assert conversations.get(rec["id"]) is None
    assert conversations.delete(rec["id"]) is False


def test_get_rejects_bad_ids(env):
    assert conversations.get("../../etc/passwd") is None
    assert conversations.get("") is None
    assert conversations.get("x" * 12 + "y") is None


def test_append_user_merges_consecutive_user_turns(env):
    rec = conversations.create()
    conversations.append_user(rec, {"role": "user", "content": "first"})
    conversations.append_user(rec, {"role": "user", "content": "second"})
    assert len(rec["messages"]) == 1
    assert rec["messages"][0]["content"] == "first\n\nsecond"


def test_replace_tail_and_assistant_text(env):
    rec = conversations.create()
    rec["messages"] = [{"role": "user", "content": "old"}, {"role": "assistant", "content": "a"},
                       {"role": "user", "content": "new"}]
    conversations.replace_tail(rec, 2, [{"role": "user", "content": "new (pointered)"},
                                        {"role": "assistant", "content": [{"type": "text", "text": "b"}]}])
    assert len(rec["messages"]) == 4
    assert rec["messages"][2]["content"] == "new (pointered)"
    conversations.append_assistant_text(rec, "c")
    assert rec["messages"][-1] == {"role": "assistant", "content": "c"}


def test_title_from():
    assert conversations.title_from("  Why does Plex transcode?\nmore") == "Why does Plex transcode?"
    assert conversations.title_from("") == "untitled"
    assert len(conversations.title_from("x" * 200)) == conversations.TITLE_MAX


def test_list_is_newest_first(env):
    a = conversations.create()
    b = conversations.create()
    b["updated"] = "2999-01-01T00:00:00Z"
    # save() restamps updated; write the file directly to pin the order
    import json
    (config.CONVERSATIONS_DIR / f"{b['id']}.json").write_text(json.dumps(b))
    assert [c["id"] for c in conversations.list_all()][0] == b["id"]
    assert a["id"] in [c["id"] for c in conversations.list_all()]


def test_append_turn_points_at_real_assistant_output(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    rec = conversations.create("gemma4:26b", "local")
    rec["messages"] = [
        {"role": "user", "content": "q"},
        {"role": "user", "content": "merged follow-up"},
        {"role": "assistant", "content": "the answer"},
    ]
    # The index was taken before the turn ran; it snaps forward to the output.
    conversations.append_turn(rec, "gemma4:26b", "local", at_index=1)
    assert rec["turns"][0]["from"] == 2
    assert rec["messages"][rec["turns"][0]["from"]]["role"] == "assistant"


def test_append_turn_keeps_the_index_when_there_is_no_assistant_turn(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    rec = conversations.create("gemma4:26b", "local")
    rec["messages"] = [{"role": "user", "content": "q"}]
    conversations.append_turn(rec, "gemma4:26b", "local", at_index=1)
    assert rec["turns"][0]["from"] == 1
