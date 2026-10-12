"""Server-owned conversations: one JSON file per chat under
config.CONVERSATIONS_DIR. The browser holds only a conversation id; the
server keeps the canonical message list (Anthropic-shaped, tool turns
included), the rolling context summary that context.py maintains, and the
claude-CLI session id for pro mode.

Record shape:
    {id, created, updated, title, model, mode,
     messages: [...canonical...],
     summary: {"text": str, "through": int},   # covers messages[:through]
     pro_session_id: str}

Stdlib-only, synchronous file I/O — callers run it under asyncio.to_thread.
Writes are atomic (tmp + os.replace). config.CONVERSATIONS_DIR is read at
call time so tests can monkeypatch it.
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config

log = logging.getLogger("router")

_ID_LEN = 12
TITLE_MAX = 60


def _dir() -> Path:
    d = config.CONVERSATIONS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _path(conv_id: str) -> Path:
    return _dir() / f"{conv_id}.json"


def is_valid_id(conv_id: str) -> bool:
    return isinstance(conv_id, str) and len(conv_id) == _ID_LEN and conv_id.isalnum()


def title_from(text: str) -> str:
    first = (text or "").strip().splitlines() or [""]
    t = first[0].strip()
    return (t[:TITLE_MAX].rstrip() or "untitled")


def first_user_text(messages: list) -> str:
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    return b.get("text", "")
    return ""


def create(model: str = "", mode: str = "") -> dict:
    rec = {
        "id":       uuid.uuid4().hex[:_ID_LEN],
        "created":  _now(),
        "updated":  _now(),
        "title":    "",
        "model":    model,
        "mode":     mode,
        "messages": [],
        "summary":  {"text": "", "through": 0},
        "pro_session_id": "",
    }
    save(rec)
    return rec


def save(rec: dict) -> None:
    rec["updated"] = _now()
    p = _path(rec["id"])
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, p)


def get(conv_id: str) -> dict | None:
    if not is_valid_id(conv_id):
        return None
    p = _path(conv_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        log.warning("conversation unreadable id=%s err=%s", conv_id, e)
        return None


def delete(conv_id: str) -> bool:
    if not is_valid_id(conv_id):
        return False
    p = _path(conv_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def list_all(limit: int = 50) -> list[dict]:
    """Newest first; a light summary per conversation (no messages)."""
    out = []
    for p in _dir().glob("*.json"):
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({
            "id":       rec.get("id"),
            "title":    rec.get("title") or "untitled",
            "updated":  rec.get("updated", ""),
            "model":    rec.get("model", ""),
            "mode":     rec.get("mode", ""),
            "messages": len(rec.get("messages") or []),
        })
    out.sort(key=lambda r: r["updated"], reverse=True)
    return out[:limit]


def append_user(rec: dict, message: dict) -> None:
    """Append a user turn. Two user turns in a row (the previous turn
    errored before any assistant text) are merged so the list keeps the
    alternation every backend expects."""
    msgs = rec["messages"]
    if msgs and msgs[-1].get("role") == "user":
        prev, cur = msgs[-1], message
        if isinstance(prev.get("content"), str) and isinstance(cur.get("content"), str):
            prev["content"] = prev["content"] + "\n\n" + cur["content"]
            return
    msgs.append(message)
    if not rec.get("title"):
        rec["title"] = title_from(first_user_text([message]))


def replace_tail(rec: dict, start: int, tail: list) -> None:
    """Replace messages[start:] with `tail` — what the loop's `history`
    event carries after the summary pair: the (possibly pointered) turns
    that were sent plus the new assistant/tool turns."""
    rec["messages"] = rec["messages"][:start] + list(tail)


def append_assistant_text(rec: dict, text: str) -> None:
    rec["messages"].append({"role": "assistant", "content": text})
