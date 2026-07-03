"""Unit tests for blobstore.py — pure logic, stdlib-only, no Claude/Ollama.

DATA_DIR is pointed at tmp_path via monkeypatch, mirroring test_staging.py's
FS_ROOT/STAGING_DIR pattern, so blobstore.py's call-time `config.DATA_DIR`
read stays testable.
"""
import os

import pytest

import blobstore
import config


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_put_get_roundtrip(env):
    text = "line one\nline two\nline three"
    r = blobstore.put(text)
    assert blobstore.get(r["hash"]) == text
    assert r["lines"] == 3
    assert r["bytes"] == len(text.encode("utf-8"))


def test_put_dedup_same_hash_one_file(env):
    r1 = blobstore.put("identical content")
    r2 = blobstore.put("identical content")
    assert r1["hash"] == r2["hash"]
    assert r1["path"] == r2["path"]
    files = list((env / "blobs").glob("*.txt"))
    assert len(files) == 1


def test_get_unknown_hash_returns_none(env):
    assert blobstore.get("0" * 64) is None
    assert blobstore.stat("0" * 64) is None
    assert "No blob found" in blobstore.grep("0" * 64, "x")


def test_stat_returns_metadata(env):
    r = blobstore.put("a\nb\nc")
    s = blobstore.stat(r["hash"])
    assert s["hash"]  == r["hash"]
    assert s["lines"] == 3
    assert s["bytes"] == r["bytes"]


def test_grep_finds_pattern_with_context(env):
    before = "\n".join(f"line {i}" for i in range(5))
    after  = "\n".join(f"line {i}" for i in range(5, 10))
    text = f"{before}\nERROR: boom\n{after}"
    r = blobstore.put(text)
    result = blobstore.grep(r["hash"], "ERROR", context_lines=1)
    assert "ERROR: boom" in result
    assert "line 4" in result  # one line of context before
    assert "line 5" in result  # one line of context after
    assert "line 0" not in result  # outside the context window


def test_grep_no_match(env):
    r = blobstore.put("nothing interesting here")
    result = blobstore.grep(r["hash"], "ERROR")
    assert "No matches" in result


def test_grep_invalid_pattern_reports_error(env):
    r = blobstore.put("some text")
    result = blobstore.grep(r["hash"], "[unclosed")
    assert "Invalid pattern" in result


def test_eviction_deletes_oldest_first_and_respects_cap(env, monkeypatch):
    monkeypatch.setattr(config, "BLOB_STORE_MAX_BYTES", 25)
    r1 = blobstore.put("a" * 10)
    r2 = blobstore.put("b" * 10)
    os.utime(r1["path"], (1000, 1000))
    os.utime(r2["path"], (2000, 2000))
    r3 = blobstore.put("c" * 10)  # total=30 > 25 -> evicts oldest (r1); r3 protected as just-written

    assert blobstore.get(r1["hash"]) is None
    assert blobstore.get(r2["hash"]) == "b" * 10
    assert blobstore.get(r3["hash"]) == "c" * 10


def test_pin_protects_blob_from_eviction(env, monkeypatch):
    monkeypatch.setattr(config, "BLOB_STORE_MAX_BYTES", 25)
    r1 = blobstore.put("a" * 10)
    os.utime(r1["path"], (1000, 1000))
    blobstore.pin([r1["hash"]])
    try:
        r2 = blobstore.put("b" * 10)
        os.utime(r2["path"], (2000, 2000))
        blobstore.put("c" * 10)  # total=30 > 25; r1 is oldest but pinned -> r2 evicted instead

        assert blobstore.get(r1["hash"]) == "a" * 10
        assert blobstore.get(r2["hash"]) is None
    finally:
        blobstore.unpin([r1["hash"]])


def test_pin_is_a_refcount_not_a_flag(env, monkeypatch):
    monkeypatch.setattr(config, "BLOB_STORE_MAX_BYTES", 25)
    r1 = blobstore.put("a" * 10)
    os.utime(r1["path"], (1000, 1000))
    blobstore.pin([r1["hash"]])
    blobstore.pin([r1["hash"]])       # two concurrent requests reference it
    blobstore.unpin([r1["hash"]])     # one finishes — still pinned by the other
    try:
        r2 = blobstore.put("b" * 10)
        os.utime(r2["path"], (2000, 2000))
        blobstore.put("c" * 10)  # forces eviction; r1 still protected (refcount 1)

        assert blobstore.get(r1["hash"]) == "a" * 10
        assert blobstore.get(r2["hash"]) is None
    finally:
        blobstore.unpin([r1["hash"]])  # second request finishes — fully released
