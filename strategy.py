# PRE-PASS STRATEGY (PLAN-pi-tools.md Phase 3)
# Retrieval beats compression: an oversized block is losslessly reduced,
# labelled, verbatim-extracted, spilled to the content-addressed blob store,
# and replaced in-band by a pointer + excerpt. Nothing the model might need
# is destroyed — the full original stays one read_blob/grep_blob call away.
#
# main.py only knows about preprocess(body, ollama_chat) -> (new_body, stats).
# This module stays pure logic: no imports from main.py; blobstore/config are
# leaf modules. Ollama is only ever asked to *classify* or *rank* — extraction
# is code, and extracted lines enter context byte-for-byte.

import asyncio
import json
import logging
import re

import blobstore
from config import SUMMARY_LOCAL_MODEL

log = logging.getLogger("strategy")

# --- Tunables ---
LOCAL_MODEL = SUMMARY_LOCAL_MODEL   # classifier/salience model; env-overridable via SUMMARY_LOCAL_MODEL
SPILL_THRESHOLD_TOKENS = 6000       # blocks above this get spilled to the blob store
EXCERPT_MAX_LINES = 120             # cap on the verbatim excerpt kept in-band
EXCERPT_MAX_CHARS = 12000           # ~3k tokens — half the spill threshold
LOG_CONTEXT_LINES = 2               # context lines around each kept logfile line
SALIENCE_MAX_KEEP_LINES = 80        # most lines the prose salience pass may keep
SALIENCE_INPUT_MAX_CHARS = 24000    # cap on what the local model is shown (head+tail)

# v1 markers (legacy paraphrase summaries — still skipped, never re-processed)
SUMMARY_MARKER     = "[router-summary v1]"
SUMMARY_END_MARKER = "[end-router-summary]"
# v2 markers (blob pointers). BLOB_MARKER is a prefix: full form is
# "[router-blob v2 <sha256>]".
BLOB_MARKER     = "[router-blob v2"
BLOB_END_MARKER = "[end-router-blob]"
# Conversation-tail compaction markers (compact.py). Registered here so all
# router markers live in one place and skip-checks can't miss one.
COMPACT_MARKER     = "[router-compact v1]"
COMPACT_END_MARKER = "[end-router-compact]"

_ALL_MARKERS = (SUMMARY_MARKER, SUMMARY_END_MARKER, BLOB_MARKER, BLOB_END_MARKER,
                COMPACT_MARKER, COMPACT_END_MARKER)

_BLOB_POINTER_RE = re.compile(re.escape(BLOB_MARKER) + r"\s+([0-9a-f]{64})\]")

CLASSIFY_SYSTEM = (
    "You classify a document into exactly one bucket. "
    "Reply with a single word — one of: logfile, source, json, diff, prose. "
    "No other output."
)
_VALID_LABELS = {"logfile", "source", "json", "diff", "prose", "mixed"}

SALIENCE_SYSTEM = (
    "You select which lines of a document must be kept verbatim for a future reader. "
    "Reply ONLY with a JSON array of [start_line, end_line] pairs (1-based, inclusive), "
    "ordered and non-overlapping, covering at most {max_lines} lines in total. "
    "Prefer lines carrying errors, conclusions, decisions, exact identifiers, and key facts. "
    "No prose, no explanation — just the JSON array."
)


# --- Estimation ---
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


def is_already_summarised(text: str) -> bool:
    """True for blocks already carrying a router marker (v1 summary, v2
    blob pointer, or compact summary) — never re-process our own output."""
    head = text.lstrip()
    return (head.startswith(SUMMARY_MARKER) or head.startswith(BLOB_MARKER)
            or head.startswith(COMPACT_MARKER))


def referenced_blob_hashes(messages: list) -> set[str]:
    """Every blob hash referenced by v2 pointers anywhere in `messages`.

    Used by the request dispatcher to pin those blobs against LRU eviction for
    the lifetime of the request.
    """
    found: set[str] = set()
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            found.update(_BLOB_POINTER_RE.findall(content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    found.update(_BLOB_POINTER_RE.findall(block_text(block)))
    return found


def has_oversized_blocks(body: dict) -> bool:
    """Quick O(n) scan — no I/O. True if preprocess would spill anything."""
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            if content and not is_already_summarised(content) \
                    and estimate_tokens(content) > SPILL_THRESHOLD_TOKENS:
                return True
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block_text(block)
                if text and not is_already_summarised(text) \
                        and estimate_tokens(text) > SPILL_THRESHOLD_TOKENS:
                    return True
    return False


# --- Block extraction (Anthropic content-block shape) ---
def block_text(block: dict) -> str:
    """Pull the processable text out of a content block. Returns '' if none."""
    t = block.get("type")
    if t == "text":
        return block.get("text", "") or ""
    if t == "tool_result":
        c = block.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c
                           if isinstance(b, dict) and b.get("type") == "text")
    return ""


def replace_block_text(block: dict, new_text: str) -> dict:
    """Return a new block with its text replaced by `new_text`."""
    t = block.get("type")
    if t == "text":
        return {**block, "text": new_text}
    if t == "tool_result":
        return {**block, "content": new_text}
    return block


# --- Step 1: lossless reduce (pure code, no model) ---
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _reduce_lines(text: str) -> list[tuple[int | None, str]]:
    """Strip ANSI escapes and collapse runs of identical consecutive lines.

    Returns (original_1based_line_number, line) pairs. A run of N identical
    lines keeps its first occurrence and appends a synthetic annotation row
    (lineno None) reading '[previous line ×N]' — N is the run's total count,
    so the reduction is information-lossless.
    """
    out: list[tuple[int | None, str]] = []
    run_line: str | None = None
    run_count = 0
    for i, line in enumerate(_ANSI_RE.sub("", text).split("\n"), start=1):
        if line == run_line:
            run_count += 1
            continue
        if run_count > 1:
            out.append((None, f"[previous line ×{run_count}]"))
        run_line = line
        run_count = 1
        out.append((i, line))
    if run_count > 1:
        out.append((None, f"[previous line ×{run_count}]"))
    return out


def reduce_text(text: str) -> str:
    """String view of _reduce_lines — golden-testable lossless reduction."""
    return "\n".join(line for _, line in _reduce_lines(text))


# --- Step 2: label (heuristics first, one model call only when unsure) ---
_TS_RE = re.compile(
    r"^[\[\(]?\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"      # ISO date-time
    r"|^[\[\(]?\d{2}:\d{2}:\d{2}"                     # bare time
    r"|^\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"          # syslog
)
_LOGLEVEL_RE = re.compile(r"\b(ERROR|WARN(?:ING)?|INFO|DEBUG|FATAL|CRITICAL|TRACE)\b")
_SOURCE_LINE_RE = re.compile(
    r"^\s*(#!|def |class |async def |import |from \S+ import |function |const |let "
    r"|var |#include|package |use |pub fn |fn |struct |impl |public |private )"
)
_DIFF_HEADER_RE = re.compile(r"^(diff --git |--- |\+\+\+ )")
_HUNK_RE = re.compile(r"^@@ ")


def classify(text: str) -> tuple[str, bool]:
    """Heuristic label + confidence. Labels: logfile|source|json|diff|prose|mixed."""
    stripped = text.strip()
    if stripped[:1] in ("{", "["):
        try:
            json.loads(stripped)
            return "json", True
        except (json.JSONDecodeError, RecursionError):
            pass
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return "prose", True
    sample = lines[:400]
    n = len(sample)
    hunks = sum(1 for ln in sample if _HUNK_RE.match(ln))
    diff_headers = sum(1 for ln in sample if _DIFF_HEADER_RE.match(ln))
    if hunks >= 1 and diff_headers >= 2:
        return "diff", True
    ts  = sum(1 for ln in sample if _TS_RE.match(ln)) / n
    lvl = sum(1 for ln in sample if _LOGLEVEL_RE.search(ln)) / n
    if ts > 0.5 or lvl > 0.5 or (ts > 0.2 and lvl > 0.2):
        return "logfile", True
    src      = sum(1 for ln in sample if _SOURCE_LINE_RE.match(ln)) / n
    indented = sum(1 for ln in sample if ln[:1] in (" ", "\t")) / n
    if src > 0.04 and indented > 0.2:
        return "source", True
    # Weak signals — best guess, let the model confirm.
    if ts > 0.1 or lvl > 0.15:
        return "logfile", False
    if src > 0.02:
        return "source", False
    return "prose", False


async def _classify_with_model(ollama_chat, text: str) -> str:
    out = await ollama_chat(LOCAL_MODEL, [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {"role": "user", "content": text[:4000]},
    ])
    label = out.strip().lower().split()[0].strip(".,\"'") if out.strip() else ""
    return label if label in _VALID_LABELS else "prose"


# --- Step 3: extract by type (pure code — verbatim lines only) ---
_KEEP_LOG_RE = re.compile(
    r"\b(?:ERROR|Error|FATAL|Fatal|CRITICAL|Critical|WARN(?:ING)?|Warn(?:ing)?"
    r"|Traceback|panic|PANIC)\b"
)
_TRACEBACK_CONT_RE = re.compile(r"^(\s|File |Traceback)")


def _fmt(pairs: list[tuple[int | None, str]]) -> list[str]:
    """Render (lineno, line) pairs — line numbers match the *original* blob so
    read_blob(hash, start_line, end_line) ranges line up."""
    return [f"{no:>6}: {line}" if no is not None else f"        {line}"
            for no, line in pairs]


def _head_tail(pairs: list[tuple[int | None, str]],
               head: int = 40, tail: int = 20) -> str:
    if len(pairs) <= head + tail:
        return "\n".join(_fmt(pairs))
    omitted = len(pairs) - head - tail
    return "\n".join(
        _fmt(pairs[:head])
        + [f"        … [{omitted} lines omitted — full content in blob] …"]
        + _fmt(pairs[-tail:])
    )


def _extract_logfile(pairs: list[tuple[int | None, str]]) -> str:
    hits = [i for i, (_, line) in enumerate(pairs) if _KEEP_LOG_RE.search(line)]
    if not hits:
        return _head_tail(pairs)
    shown: set[int] = set()
    for i in hits:
        shown.update(range(max(0, i - LOG_CONTEXT_LINES),
                           min(len(pairs), i + LOG_CONTEXT_LINES + 1)))
        # A Traceback header is only useful with its frames — extend through
        # the indented continuation lines to the exception message.
        if "Traceback" in pairs[i][1]:
            j = i + 1
            while j < len(pairs) and j - i <= 40 and _TRACEBACK_CONT_RE.match(pairs[j][1]):
                shown.add(j)
                j += 1
            if j < len(pairs):
                shown.add(j)  # the exception line that ends the traceback
    out: list[str] = []
    prev = None
    for i in sorted(shown):
        if prev is not None and i != prev + 1:
            out.append("        --")
        out.extend(_fmt([pairs[i]]))
        prev = i
    return "\n".join(out)


def _json_skeleton(obj, depth: int = 0):
    """Key structure with sampled values — never the full payload."""
    if depth >= 4:
        return "…"
    if isinstance(obj, dict):
        out = {k: _json_skeleton(obj[k], depth + 1) for k in list(obj)[:20]}
        if len(obj) > 20:
            out["…"] = f"(+{len(obj) - 20} more keys)"
        return out
    if isinstance(obj, list):
        if not obj:
            return []
        sampled = [_json_skeleton(obj[0], depth + 1)]
        if len(obj) > 1:
            sampled.append(f"… (+{len(obj) - 1} more items)")
        return sampled
    if isinstance(obj, str):
        return obj if len(obj) <= 80 else obj[:77] + "…"
    return obj


def _extract_json(text: str, pairs: list[tuple[int | None, str]]) -> str:
    try:
        obj = json.loads(text.strip())
    except (json.JSONDecodeError, RecursionError):
        return _head_tail(pairs)
    return json.dumps(_json_skeleton(obj), indent=2)


def _extract_source(pairs: list[tuple[int | None, str]]) -> str:
    kept = [(no, line) for no, line in pairs if _SOURCE_LINE_RE.match(line)]
    if not kept:
        return _head_tail(pairs)
    return "\n".join(_fmt(kept))


def _extract_diff(pairs: list[tuple[int | None, str]]) -> str:
    out: list[str] = []
    hunks = 0
    for no, line in pairs:
        if _DIFF_HEADER_RE.match(line):
            if hunks and out:
                out.append(f"        [{hunks} hunks]")
                hunks = 0
            out.extend(_fmt([(no, line)]))
        elif _HUNK_RE.match(line):
            hunks += 1
    if hunks:
        out.append(f"        [{hunks} hunks]")
    return "\n".join(out) if out else _head_tail(pairs)


async def _extract_prose(ollama_chat, pairs: list[tuple[int | None, str]]) -> str:
    """The one salience pass: the model returns line ranges to keep; code
    slices them out verbatim. The model never rewrites a byte. Any failure
    falls back to head+tail — never to paraphrase."""
    numbered = {no: line for no, line in pairs if no is not None}
    doc = "\n".join(f"{no}: {line}" for no, line in sorted(numbered.items()))
    if len(doc) > SALIENCE_INPUT_MAX_CHARS:
        head = doc[: SALIENCE_INPUT_MAX_CHARS * 2 // 3]
        tail = doc[-SALIENCE_INPUT_MAX_CHARS // 3:]
        doc = f"{head}\n… [middle not shown] …\n{tail}"
    try:
        out = await ollama_chat(LOCAL_MODEL, [
            {"role": "system",
             "content": SALIENCE_SYSTEM.format(max_lines=SALIENCE_MAX_KEEP_LINES)},
            {"role": "user", "content": doc},
        ])
        match = re.search(r"\[.*\]", out, re.S)
        ranges = json.loads(match.group(0)) if match else []
        keep: list[int] = []
        for r in ranges:
            start, end = int(r[0]), int(r[1])
            keep.extend(no for no in range(start, end + 1) if no in numbered)
        keep = sorted(set(keep))[:SALIENCE_MAX_KEEP_LINES]
        if not keep:
            return _head_tail(pairs)
    except Exception as e:
        log.warning("salience pass failed, falling back to head+tail: %s", e)
        return _head_tail(pairs)
    out_lines: list[str] = []
    prev = None
    for no in keep:
        if prev is not None and no != prev + 1:
            out_lines.append("        --")
        out_lines.extend(_fmt([(no, numbered[no])]))
        prev = no
    return "\n".join(out_lines)


# --- Step 4: spill + emit pointer ---
def _make_pointer(h: str, label: str, excerpt: str, lines: int, nbytes: int) -> str:
    # Strip any marker strings an injected payload might have planted inside the
    # excerpt so the begin/end pair stays an unambiguous data boundary.
    for marker in _ALL_MARKERS:
        excerpt = excerpt.replace(marker, "")
    if len(excerpt) > EXCERPT_MAX_CHARS:
        excerpt = excerpt[:EXCERPT_MAX_CHARS] + "\n        … [excerpt truncated]"
    excerpt_lines = excerpt.splitlines()
    if len(excerpt_lines) > EXCERPT_MAX_LINES:
        excerpt = "\n".join(excerpt_lines[:EXCERPT_MAX_LINES]
                            + ["        … [excerpt truncated]"])
    return (
        f"{BLOB_MARKER} {h}]\n"
        f"{label}: {excerpt}\n"
        f"Full content: {lines} lines, {nbytes} bytes — "
        f'use read_blob("{h}") or grep_blob("{h}", pattern)\n'
        f"{BLOB_END_MARKER}"
    )


async def _process_block(text: str, ollama_chat) -> str:
    """reduce → label → extract (verbatim) → spill original → pointer text."""
    pairs = _reduce_lines(text)
    label, confident = classify(text)
    if not confident:
        try:
            label = await _classify_with_model(ollama_chat, text)
        except Exception as e:
            log.warning("model classify failed, keeping heuristic %r: %s", label, e)
    if label == "json":
        excerpt = _extract_json(text, pairs)
    elif label == "diff":
        excerpt = _extract_diff(pairs)
    elif label == "logfile":
        excerpt = _extract_logfile(pairs)
    elif label == "source":
        excerpt = _extract_source(pairs)
    else:  # prose / mixed
        excerpt = await _extract_prose(ollama_chat, pairs)
    info = await asyncio.to_thread(blobstore.put, text)  # spill the ORIGINAL, byte-for-byte
    return _make_pointer(info["hash"], label, excerpt, info["lines"], info["bytes"])


# --- Main entry point ---
async def preprocess(body: dict, ollama_chat) -> tuple[dict, dict]:
    """Walk the request; spill oversized text/tool_result blocks to the blob
    store in parallel, replacing each with a verbatim excerpt + pointer.

    Returns (new_body, stats). stats keys: summarised (always 0 — kept for
    wire compatibility with the v1 paraphrase pipeline), skipped, failed,
    spilled. Failures leave the original block untouched — never destroyed.
    """
    stats = {"summarised": 0, "skipped": 0, "failed": 0, "spilled": 0}
    targets = []  # (msg_index, block_index_or_None, original_text)

    for mi, m in enumerate(body.get("messages", [])):
        content = m.get("content")
        if isinstance(content, str):
            if content and estimate_tokens(content) > SPILL_THRESHOLD_TOKENS:
                if is_already_summarised(content):
                    stats["skipped"] += 1
                else:
                    targets.append((mi, None, content))
        elif isinstance(content, list):
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                text = block_text(block)
                if not text or estimate_tokens(text) <= SPILL_THRESHOLD_TOKENS:
                    continue
                if is_already_summarised(text):
                    stats["skipped"] += 1
                else:
                    targets.append((mi, bi, text))

    if not targets:
        return body, stats

    results = await asyncio.gather(
        *[_process_block(t[2], ollama_chat) for t in targets],
        return_exceptions=True,
    )

    new_messages = [dict(m) for m in body["messages"]]
    for (mi, bi, _original), result in zip(targets, results):
        if isinstance(result, Exception):
            log.warning("spill failed (msg=%d block=%s): %s", mi, bi, result)
            stats["failed"] += 1
            continue
        if bi is None:
            new_messages[mi]["content"] = result
        else:
            blocks = list(new_messages[mi]["content"])
            blocks[bi] = replace_block_text(blocks[bi], result)
            new_messages[mi]["content"] = blocks
        stats["spilled"] += 1

    return {**body, "messages": new_messages}, stats
