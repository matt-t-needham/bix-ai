"""Phase 3 strategy tests — retrieval beats compression.

The old suite asserted paraphrase behaviour (Ollama rewrites oversized blocks
into short summaries). That pipeline is deliberately abolished: oversized
blocks are now losslessly reduced, labelled, verbatim-extracted, spilled to
the blob store, and replaced by a pointer. These tests pin that behaviour.

DATA_DIR is pointed at tmp_path (mirrors test_blobstore.py) so blobstore
spills land in an isolated store.
"""
import asyncio
import json
import re
from pathlib import Path

import pytest

import blobstore
import config
import strategy

FIXTURES = Path(__file__).parent / "fixtures"

HASH_RE = re.compile(r"\[router-blob v2 ([0-9a-f]{64})\]")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    return tmp_path


def run(coro):
    return asyncio.run(coro)


async def no_ollama(model, messages):
    raise AssertionError("Ollama must not be called for confidently-classified content")


def make_router_fake(salience_reply):
    """Fake ollama_chat: answers the classify prompt with 'prose' and the
    salience prompt with `salience_reply`."""
    async def fake(model, messages):
        system = messages[0]["content"]
        if system.startswith("You classify"):
            return "prose"
        return salience_reply
    return fake


def user_body(text):
    return {"messages": [{"role": "user", "content": text}]}


def pointer_hash(content):
    m = HASH_RE.search(content)
    assert m, f"no blob pointer in: {content[:200]!r}"
    return m.group(1)


# ── Ported guarantees from the v1 suite ───────────────────────────────────────
def test_below_threshold_is_untouched(env):
    body = user_body("hi")
    new, stats = run(strategy.preprocess(body, no_ollama))
    assert new == body
    assert stats == {"summarised": 0, "skipped": 0, "failed": 0, "spilled": 0}


def test_already_marked_v1_is_skipped(env):
    big = strategy.SUMMARY_MARKER + "\n" + ("x" * 30000)
    body = user_body(big)
    new, stats = run(strategy.preprocess(body, no_ollama))
    assert new == body
    assert stats["spilled"] == 0
    assert stats["skipped"] == 1


def test_already_marked_v2_is_skipped(env):
    big = f"{strategy.BLOB_MARKER} {'a' * 64}]\n" + ("x" * 30000)
    body = user_body(big)
    new, stats = run(strategy.preprocess(body, no_ollama))
    assert new == body
    assert stats["spilled"] == 0
    assert stats["skipped"] == 1
    assert not strategy.has_oversized_blocks(body)


# ── Lossless reduce ───────────────────────────────────────────────────────────
def test_reduce_collapses_repeats_losslessly():
    text = "start\n" + "same line\n" * 500 + "end"
    reduced = strategy.reduce_text(text)
    assert reduced == "start\nsame line\n[previous line ×500]\nend"


def test_reduce_strips_ansi():
    text = "\x1b[31mred error\x1b[0m plain"
    assert strategy.reduce_text(text) == "red error plain"


def test_reduce_is_identity_on_unique_lines():
    text = "\n".join(f"line {i}" for i in range(50))
    assert strategy.reduce_text(text) == text


# ── Logfile: the buried ERROR survives byte-for-byte ─────────────────────────
def test_logfile_error_and_traceback_survive_verbatim(env):
    text = (FIXTURES / "big_logfile.log").read_text()
    assert strategy.classify(text) == ("logfile", True)
    new, stats = run(strategy.preprocess(user_body(text), no_ollama))
    assert stats["spilled"] == 1
    content = new["messages"][0]["content"]
    # The exact ERROR line and the traceback's exception line, byte-for-byte.
    assert "ERROR worker: failed to process job 9999 (frame header 0xDEADBEEF)" in content
    assert "Traceback (most recent call last):" in content
    assert "ValueError: invalid frame header 0xdeadbeef" in content
    # The bulk did not ride along.
    assert len(content) < len(text) // 10
    assert content.lstrip().startswith(strategy.BLOB_MARKER)
    assert strategy.BLOB_END_MARKER in content


def test_pointer_hash_resolves_to_original(env):
    text = (FIXTURES / "big_logfile.log").read_text()
    new, _ = run(strategy.preprocess(user_body(text), no_ollama))
    h = pointer_hash(new["messages"][0]["content"])
    assert blobstore.get(h) == text
    # grep_blob-style retrieval reaches the buried context lines.
    assert "decode_frame(chunk)" in blobstore.grep(h, r"invalid frame header")


def test_same_input_twice_same_hash_one_blob(env):
    text = (FIXTURES / "big_logfile.log").read_text()
    new1, _ = run(strategy.preprocess(user_body(text), no_ollama))
    new2, _ = run(strategy.preprocess(user_body(text), no_ollama))
    h1 = pointer_hash(new1["messages"][0]["content"])
    h2 = pointer_hash(new2["messages"][0]["content"])
    assert h1 == h2
    assert len(list((env / "blobs").glob("*.txt"))) == 1


# ── Source / JSON extraction ──────────────────────────────────────────────────
def test_source_keeps_structure_lines(env):
    text = (FIXTURES / "big_source.py.txt").read_text()
    assert strategy.classify(text) == ("source", True)
    new, stats = run(strategy.preprocess(user_body(text), no_ollama))
    assert stats["spilled"] == 1
    content = new["messages"][0]["content"]
    assert "class Stage0:" in content
    assert "import json" in content
    assert blobstore.get(pointer_hash(content)) == text


def test_json_keeps_key_structure_with_samples(env):
    text = (FIXTURES / "big.json").read_text()
    assert strategy.classify(text) == ("json", True)
    new, stats = run(strategy.preprocess(user_body(text), no_ollama))
    assert stats["spilled"] == 1
    content = new["messages"][0]["content"]
    assert '"records"' in content
    assert '"rec-00000"' in content          # sampled first item
    assert "more items" in content           # elided remainder is flagged
    assert '"rec-00700"' not in content      # the bulk is gone
    assert blobstore.get(pointer_hash(content)) == text


# ── Prose salience: model picks ranges, code slices verbatim ─────────────────
def _prose_text():
    return "\n".join(
        f"Paragraph {i}: the quarterly review considered option {i} at length "
        "and weighed its trade-offs against the alternatives on the table."
        for i in range(1, 401)
    )


def test_prose_salience_slices_verbatim(env):
    text = _prose_text()
    new, stats = run(strategy.preprocess(user_body(text), make_router_fake("[[3, 5]]")))
    assert stats["spilled"] == 1
    content = new["messages"][0]["content"]
    original_lines = text.split("\n")
    for no in (3, 4, 5):
        assert original_lines[no - 1] in content   # byte-for-byte slice
    assert original_lines[100 - 1] not in content
    assert blobstore.get(pointer_hash(content)) == text


def test_prose_salience_failure_falls_back_to_head_tail(env):
    text = _prose_text()

    async def broken(model, messages):
        if messages[0]["content"].startswith("You classify"):
            return "prose"
        raise RuntimeError("ollama down")

    new, stats = run(strategy.preprocess(user_body(text), broken))
    assert stats["spilled"] == 1                   # fallback is not a failure
    content = new["messages"][0]["content"]
    original_lines = text.split("\n")
    assert original_lines[0] in content            # head survives
    assert original_lines[-1] in content           # tail survives
    assert "omitted" in content


def test_malformed_salience_reply_falls_back(env):
    text = _prose_text()
    new, stats = run(strategy.preprocess(user_body(text), make_router_fake("keep it all!")))
    assert stats["spilled"] == 1
    assert text.split("\n")[0] in new["messages"][0]["content"]


# ── Block-structured content and failure isolation ───────────────────────────
def test_tool_result_block_is_spilled(env):
    text = (FIXTURES / "big_logfile.log").read_text()
    body = {"messages": [{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": text}],
    }]}
    new, stats = run(strategy.preprocess(body, no_ollama))
    assert stats["spilled"] == 1
    content = new["messages"][0]["content"][0]["content"]
    assert content.lstrip().startswith(strategy.BLOB_MARKER)
    assert "ValueError: invalid frame header 0xdeadbeef" in content


def test_spill_failure_leaves_original_untouched(env, monkeypatch):
    def boom(text):
        raise OSError("disk full")
    monkeypatch.setattr(blobstore, "put", boom)
    text = (FIXTURES / "big_logfile.log").read_text()
    body = user_body(text)
    new, stats = run(strategy.preprocess(body, no_ollama))
    assert stats["failed"] == 1
    assert stats["spilled"] == 0
    assert new["messages"][0]["content"] == text   # never destroyed


# ── Pointer discovery for request-scoped pinning ─────────────────────────────
def test_referenced_blob_hashes_finds_pointers(env):
    text = (FIXTURES / "big_logfile.log").read_text()
    new, _ = run(strategy.preprocess(user_body(text), no_ollama))
    h = pointer_hash(new["messages"][0]["content"])
    assert strategy.referenced_blob_hashes(new["messages"]) == {h}
    assert strategy.referenced_blob_hashes([{"role": "user", "content": "hi"}]) == set()


# ── Classification heuristics ─────────────────────────────────────────────────
def test_classify_diff():
    diff = (
        "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n"
        "@@ -1,3 +1,4 @@\n-old\n+new\n+newer\n context\n"
        "@@ -10,2 +11,2 @@\n-a\n+b\n"
    )
    assert strategy.classify(diff) == ("diff", True)
    pairs = strategy._reduce_lines(diff)
    excerpt = strategy._extract_diff(pairs)
    assert "diff --git a/foo.py b/foo.py" in excerpt
    assert "[2 hunks]" in excerpt


def test_classify_prose_is_low_confidence():
    label, confident = strategy.classify(_prose_text())
    assert label == "prose"
    assert confident is False
