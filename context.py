"""Per-turn context assembly for server-owned conversations.

Given the full canonical message list, a provider token budget, and the
conversation's rolling summary, decide what actually rides along with this
turn: the newest turns that fit the budget verbatim, with everything older
represented by one summary pair at the head. The summary lives on the
conversation record (`summary.text`, `summary.through`), accretes forward
and is never re-summarised; it is only extended when the verbatim cut-off
moves past what it already covers, and that one model call runs on the
provider that is answering (the injected `summarise` callable) — never a
background local model.

Pure logic: no I/O beyond the injected callable. Fails open — if the
summary call fails, the full history is sent and the stats say so.
"""
import logging
import re

from config import CONTEXT_SUMMARY_WORDS
from strategy import BLOB_MARKER, block_text, estimate_tokens

log = logging.getLogger("router")

CONTEXT_MARKER     = "[router-context v1]"
CONTEXT_END_MARKER = "[end-router-context]"
ACK_TEXT = "Understood — I have the earlier context above and we're continuing from there."

SUMMARY_SYSTEM = (
    "You compress older conversation turns for an AI assistant's context window. "
    "Summarise the transcript below in at most {target} words. Preserve decisions, "
    "conclusions, exact identifiers, file paths, numbers, and unresolved questions. "
    "Note tool calls only by what they established. Output the summary only — no "
    "preamble, no quotes."
)

TRANSCRIPT_MAX_CHARS = 12000
PER_TEXT_MAX_CHARS   = 1500
_BLOB_POINTER_RE = re.compile(re.escape(BLOB_MARKER) + r"\s+([0-9a-f]{64})\]")
_ARTIFACT_HEADING = "Artifacts still available via read_blob/grep_blob:"


def _texts(m: dict) -> list[str]:
    c = m.get("content")
    if isinstance(c, str):
        return [c] if c else []
    if isinstance(c, list):
        return [t for t in (block_text(b) for b in c if isinstance(b, dict)) if t]
    return []


def message_tokens(m: dict) -> int:
    return sum(estimate_tokens(t) for t in _texts(m))


def _is_tool_result_msg(m: dict) -> bool:
    c = m.get("content")
    return isinstance(c, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in c)


def turn_starts(messages: list) -> list[int]:
    """Indices of real user turn-starts (not tool_result continuations)."""
    return [i for i, m in enumerate(messages)
            if m.get("role") == "user" and not _is_tool_result_msg(m)]


def _clip(t: str, limit: int = PER_TEXT_MAX_CHARS) -> str:
    return t if len(t) <= limit else t[:limit] + " …[truncated]"


def render_transcript(messages: list) -> str:
    lines = []
    for m in messages:
        role = m.get("role", "?")
        c = m.get("content")
        blocks = [{"type": "text", "text": c}] if isinstance(c, str) else (c or [])
        for b in blocks:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                t = b.get("text", "")
                if t.lstrip().startswith(BLOB_MARKER):
                    lines.append(f"{role}: «artifact pointer kept — see artifact list»")
                elif t:
                    lines.append(f"{role}: {_clip(t)}")
            elif bt == "tool_use":
                lines.append(f"{role}: «called tool {b.get('name', '?')}({_clip(str(b.get('input', {})), 200)})»")
            elif bt == "tool_result":
                t = block_text(b)
                lines.append(f"{role}: «tool result, {len(t)} chars: {_clip(t, 200)}»")
    doc = "\n".join(lines)
    if len(doc) > TRANSCRIPT_MAX_CHARS:
        doc = f"{doc[: TRANSCRIPT_MAX_CHARS * 2 // 3]}\n…[middle omitted]…\n{doc[-TRANSCRIPT_MAX_CHARS // 3:]}"
    return doc


def _artifact_hashes(messages: list) -> list[str]:
    seen: list[str] = []
    for m in messages:
        for t in _texts(m):
            for h in _BLOB_POINTER_RE.findall(t):
                if h not in seen:
                    seen.append(h)
    return seen


def cut_for_budget(messages: list, budget_tokens: int) -> int:
    """Index of the oldest turn-start that keeps the verbatim tail within
    budget. Always keeps at least the newest turn. 0 = everything fits."""
    starts = turn_starts(messages)
    if not starts:
        return 0
    bounds = list(zip(starts, starts[1:] + [len(messages)]))   # turn i = messages[s:e]
    total = 0
    cut = starts[-1]
    for s, e in reversed(bounds):
        tok = sum(message_tokens(m) for m in messages[s:e])
        if total + tok > budget_tokens and s != starts[-1]:
            break
        total += tok
        cut = s
    return 0 if cut == starts[0] else cut


def summary_pair(summary_text: str, folded: list) -> list:
    body = summary_text.strip()
    hashes = _artifact_hashes(folded)
    if hashes:
        body += "\n\n" + _ARTIFACT_HEADING + "\n" + "\n".join(f"{BLOB_MARKER} {h}]" for h in hashes)
    return [
        {"role": "user", "content": f"{CONTEXT_MARKER}\n{body}\n{CONTEXT_END_MARKER}"},
        {"role": "assistant", "content": ACK_TEXT},
    ]


async def assemble(messages: list, summary: dict, budget_tokens: int, summarise) -> tuple[list, dict, dict]:
    """Return (messages_to_send, new_summary, stats).

    `summarise(transcript: str) -> str` is awaited only when the cut-off
    moves past `summary.through`. stats: {total_messages, sent_messages,
    folded_messages, sent_tokens, summary_tokens, summarised (0/1),
    summary_failed (0/1)}.
    """
    summary = {"text": str((summary or {}).get("text") or ""),
               "through": min(int((summary or {}).get("through") or 0), len(messages))}
    if not summary["text"]:
        summary["through"] = 0
    through = summary["through"]
    cut = cut_for_budget(messages, budget_tokens)
    stats = {"total_messages": len(messages), "summarised": 0, "summary_failed": 0}

    if cut > through:
        # Extend the summary over messages[through:cut]; the previous body
        # is carried forward verbatim and the new part accretes below it.
        try:
            new_part = (await summarise(render_transcript(messages[through:cut]))).strip()
            if not new_part:
                raise ValueError("empty summary")
            summary["text"] = f"{summary['text']}\n\n{new_part}".strip()
            summary["through"] = cut
            stats["summarised"] = 1
        except Exception as e:
            # Fail open: send the turns verbatim rather than silently dropping them.
            log.warning("context summary failed, sending full history: %s", e)
            stats["summary_failed"] = 1

    start = summary["through"]          # never drop anything the summary doesn't cover
    tail = messages[start:]
    head = summary_pair(summary["text"], messages[:start]) if start else []
    stats.update({
        "sent_messages":   len(tail),
        "folded_messages": start,
        "sent_tokens":     sum(message_tokens(m) for m in head + tail),
        "summary_tokens":  sum(message_tokens(m) for m in head),
    })
    return head + tail, summary, stats
