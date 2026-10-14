"""The custom-prompt store: persistence, the size cap, and the revision ring.

Stdlib-only (prompt_store imports config and nothing else), so these run in a
bare checkout. config.DATA_DIR is monkeypatched — the module reads it at call
time, and that property is what makes this possible.
"""
import asyncio
import json

import pytest

import config
import prompt_store


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    prompt_store._cache.update(key=None, text="")
    return tmp_path


# ── Reading ───────────────────────────────────────────────────────────────────

def test_unset_prompt_reads_as_empty(data):
    assert prompt_store.read_text() == ""
    state = prompt_store.load()
    assert state["text"] == "" and state["chars"] == 0
    assert state["updated_at"] == "" and state["revisions"] == 0
    assert state["max_chars"] == prompt_store.MAX_CHARS


def test_unreadable_file_fails_open_rather_than_raising(data):
    # A directory where the file should be: every read path must still answer.
    (data / "prompt").mkdir()
    (data / "prompt" / "custom.md").mkdir()
    assert prompt_store.read_text() == ""
    assert prompt_store.load()["text"] == ""


def test_custom_text_is_the_async_accessor(data):
    prompt_store.save("Prefer metric units.")
    assert asyncio.run(prompt_store.custom_text()) == "Prefer metric units.\n"


def test_an_external_edit_is_picked_up(data):
    prompt_store.save("first")
    assert prompt_store.read_text() == "first\n"
    # Editing the file directly (a dev run without Access does exactly this).
    (data / "prompt" / "custom.md").write_text("edited by hand\n")
    assert prompt_store.read_text() == "edited by hand\n"


def test_cache_is_not_shared_between_data_dirs(tmp_path, monkeypatch):
    # Same content and size in two trees: the cache key includes the path, so
    # the second read must not serve the first tree's text.
    for name, text in (("a", "aaa"), ("b", "bbb")):
        root = tmp_path / name
        monkeypatch.setattr(config, "DATA_DIR", root)
        prompt_store.save(text)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "a")
    assert prompt_store.read_text() == "aaa\n"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "b")
    assert prompt_store.read_text() == "bbb\n"


# ── Saving ────────────────────────────────────────────────────────────────────

def test_save_round_trip_and_state(data):
    state = prompt_store.save("Prefer metric units.")
    assert state["text"] == "Prefer metric units.\n"
    assert state["chars"] == len("Prefer metric units.\n")
    assert state["updated_at"].endswith("Z")
    assert prompt_store.read_text() == "Prefer metric units.\n"


def test_save_normalises_line_endings_and_trailing_space(data):
    prompt_store.save("one\r\ntwo\r\n\n   \n")
    assert prompt_store.read_text() == "one\ntwo\n"


def test_whitespace_only_prompt_is_stored_as_empty(data):
    prompt_store.save("   \n\n  ")
    assert prompt_store.read_text() == ""


def test_oversize_prompt_is_refused_and_nothing_is_written(data):
    prompt_store.save("keep me")
    with pytest.raises(prompt_store.PromptError) as e:
        prompt_store.save("x" * (prompt_store.MAX_CHARS + 1))
    assert str(prompt_store.MAX_CHARS) in str(e.value).replace(",", "")
    assert prompt_store.read_text() == "keep me\n"   # the old text survives


def test_non_text_is_refused(data):
    with pytest.raises(prompt_store.PromptError):
        prompt_store.save(None)


def test_no_tmp_file_is_left_behind(data):
    prompt_store.save("a")
    prompt_store.save("b")
    assert not list((data / "prompt").glob("*.tmp"))


# ── Revisions ─────────────────────────────────────────────────────────────────

def test_saving_snapshots_the_previous_version(data):
    prompt_store.save("version one")
    prompt_store.save("version two")
    revs = prompt_store.revisions()
    assert [r["preview"] for r in revs] == ["version one"]
    assert revs[0]["chars"] == len("version one\n")
    assert revs[0]["saved_at"].endswith("Z")


def test_first_save_snapshots_nothing(data):
    prompt_store.save("version one")
    assert prompt_store.revisions() == []


def test_revisions_are_newest_first_and_capped(data):
    for i in range(prompt_store.REVISIONS_KEPT + 4):
        prompt_store.save(f"text {i}")
    revs = prompt_store.revisions()
    assert len(revs) == prompt_store.REVISIONS_KEPT
    newest = prompt_store.REVISIONS_KEPT + 4 - 2   # the one before the current
    assert revs[0]["preview"] == f"text {newest}"
    assert revs[-1]["preview"] == f"text {newest - prompt_store.REVISIONS_KEPT + 1}"


def test_restore_brings_back_an_earlier_version(data):
    prompt_store.save("original")
    prompt_store.save("replacement")
    rev_id = prompt_store.revisions()[0]["id"]
    state = prompt_store.restore(rev_id)
    assert state["text"] == "original\n"
    assert prompt_store.read_text() == "original\n"


def test_restore_is_itself_undoable(data):
    prompt_store.save("original")
    prompt_store.save("replacement")
    prompt_store.restore(prompt_store.revisions()[0]["id"])
    # Restoring snapshotted "replacement", so it can be brought back too.
    assert prompt_store.revisions()[0]["preview"] == "replacement"


@pytest.mark.parametrize("bad", ["", "nope", "../../etc/passwd", "ABCDEF12", "0123456"])
def test_restore_refuses_an_unknown_revision(data, bad):
    prompt_store.save("current")
    with pytest.raises(prompt_store.PromptError):
        prompt_store.restore(bad)
    assert prompt_store.read_text() == "current\n"


def test_corrupt_revisions_file_fails_open(data):
    prompt_store.save("current")
    (data / "prompt" / "revisions.json").write_text("{not json")
    assert prompt_store.revisions() == []
    assert prompt_store.load()["revisions"] == 0
    prompt_store.save("still works")           # and a save over it recovers
    assert json.loads((data / "prompt" / "revisions.json").read_text())


def test_reset_clears_the_prompt_but_keeps_the_history(data):
    prompt_store.save("some instructions")
    prompt_store.reset()
    assert prompt_store.read_text() == ""
    assert prompt_store.revisions()[0]["preview"] == "some instructions"
