"""Tests for the read_blob/grep_blob tool wiring in tools.py (Phase 2 of
PLAN-pi-tools.md). Exercises the _execute_tool dispatch and the FS_TOOLS /
OLLAMA_TOOLS definitions; blobstore.py itself is covered by test_blobstore.py.
"""
import asyncio

import pytest

import blobstore
import config
import tools


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def test_read_blob_whole_content(env):
    r = blobstore.put("line1\nline2\nline3")
    result = run(tools._execute_tool("read_blob", {"hash": r["hash"]}))
    assert result == "line1\nline2\nline3"


def test_read_blob_line_range(env):
    r = blobstore.put("\n".join(f"line{i}" for i in range(1, 11)))
    result = run(tools._execute_tool("read_blob", {"hash": r["hash"], "start_line": 3, "end_line": 5}))
    assert "line3" in result
    assert "line5" in result
    assert "line1" not in result
    assert "line6" not in result


def test_read_blob_start_after_end_is_an_error(env):
    r = blobstore.put("a\nb\nc")
    result = run(tools._execute_tool("read_blob", {"hash": r["hash"], "start_line": 3, "end_line": 1}))
    assert "after" in result.lower()


def test_read_blob_oversized_refuses_whole_read(env):
    big = "x" * (tools._BLOB_INLINE_MAX_BYTES + 1)
    r = blobstore.put(big)
    result = run(tools._execute_tool("read_blob", {"hash": r["hash"]}))
    assert "too large" in result.lower()
    # but an explicit range still works
    ranged = run(tools._execute_tool("read_blob", {"hash": r["hash"], "start_line": 1, "end_line": 1}))
    assert ranged.endswith("x" * (tools._BLOB_INLINE_MAX_BYTES + 1))


def test_read_blob_unknown_hash(env):
    result = run(tools._execute_tool("read_blob", {"hash": "0" * 64}))
    assert "No blob found" in result


def test_read_blob_invalid_hash_shape_rejected(env):
    result = run(tools._execute_tool("read_blob", {"hash": "not-a-hash"}))
    assert "Invalid" in result


def test_read_blob_missing_hash_rejected(env):
    result = run(tools._execute_tool("read_blob", {}))
    assert "Invalid" in result


def test_grep_blob_finds_match(env):
    r = blobstore.put("ok\nok\nERROR: disk full\nok")
    result = run(tools._execute_tool("grep_blob", {"hash": r["hash"], "pattern": "ERROR"}))
    assert "ERROR: disk full" in result


def test_grep_blob_missing_pattern(env):
    r = blobstore.put("some text")
    result = run(tools._execute_tool("grep_blob", {"hash": r["hash"]}))
    assert "No pattern" in result


def test_grep_blob_invalid_hash_shape_rejected(env):
    result = run(tools._execute_tool("grep_blob", {"hash": "short", "pattern": "x"}))
    assert "Invalid" in result


def test_fs_tools_contains_blob_tools():
    names = {t["name"] for t in tools.FS_TOOLS}
    assert "read_blob" in names
    assert "grep_blob" in names


def test_ollama_tools_derived_from_fs_tools_includes_blob_tools():
    names = {t["function"]["name"] for t in tools.OLLAMA_TOOLS}
    assert "read_blob" in names
    assert "grep_blob" in names
