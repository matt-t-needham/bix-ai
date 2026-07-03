"""Conversation-tail compaction (Phase 5 of PLAN-pi-tools.md).

Opposite treatment to artifacts: artifacts are spilled at full fidelity and
retrieved by tools (strategy.py / blobstore.py); old *narrative* turns are
summarised once the conversation grows past a threshold. The most recent
COMPACT_KEEP_TURNS user turns stay byte-identical; everything older is folded
into a `[router-compact v1]` user+assistant pair at the head of the message
list. The result rides the `history` SSE event, the client adopts it, and the
next request arrives pre-compacted — so compaction converges instead of
re-running every turn.

Discipline mirrors strategy.py:
- Pure logic: injected `ollama_chat`, leaf imports only (config, strategy).
- Marker content is never re-summarised. When compaction triggers again after
  more growth, the previous compact body is carried forward VERBATIM and the
  new summary accretes below it — the model only ever sees the newly-folded
  turns.
- Blob pointers are artifacts, not narrative: their excerpt tokens don't count
  toward the trigger, and every pointer hash folded out of the tail is
  re-listed inside the compact body so the model (and the request pinner)
  can still reach the artifact.
- Fail open: any error leaves the messages untouched.
"""
import logging
import re

from config import COMPACT_KEEP_TURNS, COMPACT_THRESHOLD_TOKENS, SUMMARY_LOCAL_MODEL
from strategy import (
    BLOB_MARKER, COMPACT_END_MARKER, COMPACT_MARKER, _ALL_MARKERS,
    block_text, estimate_tokens,
)

log = logging.getLogger("compact")

LOCAL_MODEL = SUMMARY_LOCAL_MODEL
SUMMARY_TARGET_WORDS = 300
# Cap on what the local model is shown. Sized for latency, not context: on
# this host's Vulkan GPU, 12k chars summarises in ~50-65s on qwen3.5:9b /
# gemma4:26b; 24k blows the 120s ollama_chat timeout and compaction fails open.
TRANSCRIPT_MAX_CHARS = 12000
PER_TEXT_MAX_CHARS   = 1500    # cap per message chunk inside the transcript

_BLOB_POINTER_RE = re.compile(re.escape(BLOB_MARKER) + r"\s+([0-9a-f]{64})\]")

COMPACT_SYSTEM = (
    "You compress older conversation turns for an AI assistant's context window. "
    "Summarise the transcript below to at most {target} words. Preserve decisions, "
    "conclusions, exact identifiers, file paths, numbers, and unresolved questions. "
    "Note tool calls only by what they established. Output the summary only — no "
    "preamble, no quotes."
)

ACK_TEXT = "Understood — I have the compacted context above and we're continuing from there."


# ── Message inspection helpers ────────────────────────────────────────────────

def _texts(m: dict) -> list[str]:
    content = m.get("content")
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        return [t for t in (block_text(b) for b in content if isinstance(b, dict)) if t]
    return []


def _is_compact_msg(m: dict) -> bool:
    ts = _texts(m)
    return bool(ts) and ts[0].lstrip().startswith(COMPACT_MARKER)


def _is_tool_result_msg(m: dict) -> bool:
    content = m.get("content")
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    )


def narrative_tokens(messages: list) -> int:
    """Estimated tokens of conversation narrative — blob-pointer excerpts are
    artifacts, not narrative, and don't count toward the compaction trigger."""
    total = 0
    for m in messages:
        for t in _texts(m):
            if not t.lstrip().startswith(BLOB_MARKER):
                total += estimate_tokens(t)
    return total


def _turn_start_indices(messages: list) -> list[int]:
    """Indices of real user turn-starts: user role, not a tool_result
    continuation, not our own compact message."""
    return [
        i for i, m in enumerate(messages)
        if m.get("role") == "user" and not _is_tool_result_msg(m) and not _is_compact_msg(m)
    ]


def _cut_index(messages: list) -> int | None:
    """Index of the first kept message (a user turn-start), or None if there
    is nothing worth folding."""
    starts = _turn_start_indices(messages)
    if len(starts) <= COMPACT_KEEP_TURNS:
        return None
    cut = starts[-COMPACT_KEEP_TURNS]
    head = 2 if messages and _is_compact_msg(messages[0]) else 0
    if cut <= head:
        return None  # nothing between the existing compact pair and the tail
    return cut


def should_compact(messages: list) -> bool:
    """Quick O(n) trigger check — no I/O."""
    return narrative_tokens(messages) > COMPACT_THRESHOLD_TOKENS \
        and _cut_index(messages) is not None


# ── Transcript rendering (what the local model sees) ─────────────────────────

def _clip(t: str, limit: int = PER_TEXT_MAX_CHARS) -> str:
    return t if len(t) <= limit else t[:limit] + " …[truncated]"


def _render_transcript(middle: list) -> str:
    lines = []
    for m in middle:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str):
            if content.lstrip().startswith(BLOB_MARKER):
                lines.append(f"{role}: «artifact pointer kept — see artifact list»")
            else:
                lines.append(f"{role}: {_clip(content)}")
            continue
        if isinstance(content, list):
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text":
                    t = b.get("text", "")
                    if t.lstrip().startswith(BLOB_MARKER):
                        lines.append(f"{role}: «artifact pointer kept — see artifact list»")
                    elif t:
                        lines.append(f"{role}: {_clip(t)}")
                elif btype == "tool_use":
                    lines.append(f"{role}: «called tool {b.get('name', '?')}({_clip(str(b.get('input', {})), 200)})»")
                elif btype == "tool_result":
                    t = block_text(b)
                    lines.append(f"{role}: «tool result, {len(t)} chars: {_clip(t, 200)}»")
    doc = "\n".join(lines)
    if len(doc) > TRANSCRIPT_MAX_CHARS:
        head = doc[: TRANSCRIPT_MAX_CHARS * 2 // 3]
        tail = doc[-TRANSCRIPT_MAX_CHARS // 3:]
        doc = f"{head}\n…[middle of transcript omitted]…\n{tail}"
    return doc


def _artifact_hashes(messages: list) -> list[str]:
    """Blob hashes referenced anywhere in `messages`, in first-seen order."""
    seen: list[str] = []
    for m in messages:
        for t in _texts(m):
            for h in _BLOB_POINTER_RE.findall(t):
                if h not in seen:
                    seen.append(h)
    return seen


def _strip_markers(text: str) -> str:
    for marker in _ALL_MARKERS:
        text = text.replace(marker, "")
    return text.strip()


_ARTIFACT_HEADING = "Artifacts still available"


def _old_compact_body(messages: list) -> str:
    """Previous compact summary narrative, carried forward verbatim — never
    re-summarised. The code-generated artifact section is dropped here because
    it is regenerated (hashes included) on every compaction."""
    if not (messages and _is_compact_msg(messages[0])):
        return ""
    t = _texts(messages[0])[0]
    t = t.replace(COMPACT_MARKER, "").replace(COMPACT_END_MARKER, "").strip()
    idx = t.find(_ARTIFACT_HEADING)
    if idx != -1:
        t = t[:idx].rstrip()
    return t


# ── Main entry point ──────────────────────────────────────────────────────────

async def compact(messages: list, ollama_chat) -> tuple[list, dict]:
    """Fold everything before the last COMPACT_KEEP_TURNS user turns into a
    [router-compact v1] user+assistant pair. Returns (new_messages, stats);
    stats keys: compacted (0/1), folded, failed. On any failure the original
    list is returned untouched.
    """
    stats = {"compacted": 0, "folded": 0, "failed": 0}
    if not should_compact(messages):
        return messages, stats

    cut = _cut_index(messages)
    head = 2 if _is_compact_msg(messages[0]) else 0
    middle = messages[head:cut]
    tail = messages[cut:]

    try:
        out = await ollama_chat(LOCAL_MODEL, [
            {"role": "system", "content": COMPACT_SYSTEM.format(target=SUMMARY_TARGET_WORDS)},
            {"role": "user", "content": _render_transcript(middle)},
        ])
        new_summary = _strip_markers(out)
        if not new_summary:
            raise ValueError("empty summary from local model")
    except Exception as e:
        log.warning("compaction failed, keeping full history: %s", e)
        stats["failed"] = 1
        return messages, stats

    sections = []
    old_body = _old_compact_body(messages)
    if old_body:
        sections.append(old_body)  # verbatim — accretes, never re-summarised
    sections.append(new_summary)

    hashes = _artifact_hashes(messages[:cut])
    if hashes:
        sections.append(
            f"{_ARTIFACT_HEADING} via read_blob/grep_blob:\n"
            + "\n".join(f"{BLOB_MARKER} {h}]" for h in hashes)
        )

    body = "\n\n".join(sections)
    compact_pair = [
        {"role": "user", "content": f"{COMPACT_MARKER}\n{body}\n{COMPACT_END_MARKER}"},
        {"role": "assistant", "content": ACK_TEXT},
    ]

    stats["compacted"] = 1
    stats["folded"] = head + len(middle)
    log.info("compacted folded=%d kept=%d narrative_tokens=%d",
             stats["folded"], len(tail), narrative_tokens(compact_pair + tail))
    return compact_pair + tail, stats
