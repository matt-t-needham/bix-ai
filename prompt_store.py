"""The human-edited custom instructions appended to every system prompt.

One file of prose under config.DATA_DIR (a bind mount of bix-ai/data), so
editing it needs no rebuild and survives a deploy. The *immutable* half of
the prompt — identity.IDENTITY_CORE, the capability index, the doc pointers —
stays in code; this module only ever holds the part Matt writes himself, and
identity.with_custom() appends it under a labelled heading. Editing the prompt
therefore cannot delete the ground rules.

Nothing the model can reach writes here: the only write tools are
workspace.py's (todos + outbox, both confined to their own directories), so
the editor UI's human-gated POST is the single writer.

Stdlib-only, synchronous core; config.DATA_DIR is read at call time so tests
can monkeypatch it. Reads are mtime-cached and fail open — a missing or
unreadable file yields "" rather than breaking a chat.

Layout under config.DATA_DIR/prompt:
    custom.md       the current text ("" when absent)
    revisions.json  [{id, saved_at, chars, text}] newest first, last 10
"""
import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config

log = logging.getLogger("router")

MAX_CHARS      = 16_000
REVISIONS_KEPT = 10

_REV_ID_RE = re.compile(r"[0-9a-f]{8}")

# Read cache for the hot path (one lookup per chat turn): stat the file and
# re-read only when it actually changed. The path is part of the key so a
# monkeypatched DATA_DIR can never serve another tree's cached text.
_cache: dict = {"key": None, "text": ""}


class PromptError(Exception):
    """A refusal the caller should surface verbatim (mirrors WorkspaceError)."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dir() -> Path:
    return config.DATA_DIR / "prompt"


def _file() -> Path:
    return _dir() / "custom.md"


def _revisions_file() -> Path:
    return _dir() / "revisions.json"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _normalise(text: str) -> str:
    """CRLF → LF and no trailing blank space, so a round trip through a
    browser textarea doesn't churn the file or the prompt."""
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n" \
        if text.strip() else ""


def read_text() -> str:
    """The current custom instructions, "" when unset. Never raises."""
    path = _file()
    try:
        stat = path.stat()
    except OSError:
        _cache.update(key=None, text="")
        return ""
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _cache["key"] == key:
        return _cache["text"]
    try:
        text = path.read_text()
    except OSError as e:
        # Fail open: a broken prompt file must not break every chat.
        log.warning("custom prompt unreadable path=%s err=%s", path, e)
        _cache.update(key=None, text="")
        return ""
    _cache.update(key=key, text=text)
    return text


async def custom_text() -> str:
    """read_text() off the event loop — what the adapters call per request."""
    return await asyncio.to_thread(read_text)


def _load_revisions() -> list[dict]:
    try:
        data = json.loads(_revisions_file().read_text())
    except (OSError, ValueError):
        return []
    return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def load() -> dict:
    """Everything the editor page needs about the current state."""
    text = read_text()
    try:
        updated_at = datetime.fromtimestamp(
            _file().stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        updated_at = ""
    return {
        "text":       text,
        "chars":      len(text),
        "max_chars":  MAX_CHARS,
        "updated_at": updated_at,
        "revisions":  len(_load_revisions()),
    }


def revisions() -> list[dict]:
    """Newest first, without the bodies — a list for the editor page."""
    out = []
    for rev in _load_revisions():
        text = str(rev.get("text", ""))
        preview = " ".join(text.split())[:120]
        out.append({
            "id":       str(rev.get("id", "")),
            "saved_at": str(rev.get("saved_at", "")),
            "chars":    len(text),
            "preview":  preview,
        })
    return out


def save(text: str) -> dict:
    """Replace the custom instructions, snapshotting what was there.

    Raises PromptError when the text is too long — the caller surfaces that;
    an oversized prompt is never silently truncated.
    """
    if not isinstance(text, str):
        raise PromptError("prompt must be text")
    new = _normalise(text)
    if len(new) > MAX_CHARS:
        raise PromptError(
            f"prompt is {len(new):,} characters; the cap is {MAX_CHARS:,}")

    previous = read_text()
    if previous.strip():
        revs = [{
            "id":       uuid.uuid4().hex[:8],
            "saved_at": _now(),
            "text":     previous,
        }] + _load_revisions()
        _write_atomic(_revisions_file(), json.dumps(revs[:REVISIONS_KEPT], indent=1))

    _write_atomic(_file(), new)
    _cache.update(key=None, text="")   # next read_text() re-stats
    log.info("custom prompt saved chars=%d", len(new))
    return load()


def restore(rev_id: str) -> dict:
    """Make an earlier revision current (itself a save, so it's undoable)."""
    if not _REV_ID_RE.fullmatch(str(rev_id or "")):
        raise PromptError("unknown revision")
    for rev in _load_revisions():
        if rev.get("id") == rev_id:
            return save(str(rev.get("text", "")))
    raise PromptError("unknown revision")


def reset() -> dict:
    """Clear the custom instructions; IDENTITY_CORE is untouched."""
    return save("")
