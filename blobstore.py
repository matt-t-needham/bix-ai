"""Content-addressed blob store for oversized artifacts spilled out of inbound
context (PLAN-pi-tools.md Phase 2/3 — "retrieval beats compression").

Stdlib-only by contract, mirroring staging.py/fs_core.py, so bix_mcp.py could
import it without pulling in FastAPI/httpx. `config.DATA_DIR` is read at call
time (not bound at import), same as FS_ROOT/STAGING_DIR, so tests can
monkeypatch it.

Layout: DATA_DIR/blobs/<sha256>.txt — one write-once file per unique content;
identical bytes always dedup to the same file (same hash). LRU recency is
tracked via filesystem mtime, refreshed on every put/get. Eviction sweeps
oldest-mtime files first when the store exceeds config.BLOB_STORE_MAX_BYTES.

Pinning: a blob referenced by the request currently being processed must
never be evicted mid-request. pin()/unpin() maintain an in-memory refcount
guarded by a lock, and eviction's pinned-check + unlink happen under that same
lock so a pin can never lose a race against a concurrent eviction sweep.
"""
import hashlib
import re
import threading
from pathlib import Path

import config

_SUFFIX = ".txt"

_pin_lock = threading.Lock()
_pinned: dict[str, int] = {}  # hash -> refcount


def _blob_dir() -> Path:
    return config.DATA_DIR / "blobs"


def _blob_path(h: str) -> Path:
    # Hashes are hex digests from our own hashlib call — never used to build a
    # path from caller-controlled data, so no traversal surface here. Tool
    # callers (tools.py) still validate the hash shape before reaching this.
    return _blob_dir() / f"{h}{_SUFFIX}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _line_count(text: str) -> int:
    return len(text.splitlines())


def put(text: str) -> dict:
    """Write `text` if not already stored; return {hash, path, lines, bytes}.

    Write-once: a second put() of identical content is a no-op besides
    refreshing LRU recency — same bytes always resolve to the same hash.
    """
    h = _hash(text)
    p = _blob_path(h)
    data = text.encode("utf-8", errors="replace")
    if not p.exists():
        _blob_dir().mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        _evict_if_needed(protect=h)
    else:
        p.touch()
    return {"hash": h, "path": str(p), "lines": _line_count(text), "bytes": len(data)}


def get(h: str) -> str | None:
    """Return the blob's full text, or None if unknown. Refreshes LRU recency."""
    p = _blob_path(h)
    if not p.exists():
        return None
    try:
        text = p.read_text(errors="replace")
    except OSError:
        return None
    p.touch()
    return text


def stat(h: str) -> dict | None:
    """Return {hash, path, lines, bytes} without the caller handling full text."""
    p = _blob_path(h)
    if not p.exists():
        return None
    text = get(h)
    if text is None:
        return None
    return {"hash": h, "path": str(p), "lines": _line_count(text), "bytes": p.stat().st_size}


def grep(h: str, pattern: str, context_lines: int = 2) -> str:
    """Return matching lines (with surrounding context) as a formatted string."""
    text = get(h)
    if text is None:
        return f"No blob found for hash {h}"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"Invalid pattern: {e}"
    lines = text.splitlines()
    hit_idxs = [i for i, line in enumerate(lines) if rx.search(line)]
    if not hit_idxs:
        return f"No matches for pattern: {pattern}"
    context_lines = max(0, context_lines)
    shown: set[int] = set()
    for i in hit_idxs:
        shown.update(range(max(0, i - context_lines), min(len(lines), i + context_lines + 1)))
    hit_set = set(hit_idxs)
    out = []
    prev = None
    for i in sorted(shown):
        if prev is not None and i != prev + 1:
            out.append("--")
        marker = ">" if i in hit_set else " "
        out.append(f"{marker}{i + 1:>6}: {lines[i]}")
        prev = i
    return "\n".join(out)


def pin(hashes) -> None:
    """Mark blobs as in-use by the request being processed — protects them from
    eviction until unpin(). Safe to call with hashes that don't exist on disk."""
    with _pin_lock:
        for h in hashes:
            _pinned[h] = _pinned.get(h, 0) + 1


def unpin(hashes) -> None:
    with _pin_lock:
        for h in hashes:
            if h in _pinned:
                _pinned[h] -= 1
                if _pinned[h] <= 0:
                    del _pinned[h]


def _evict_if_needed(protect: str | None = None) -> None:
    """Delete oldest-mtime blobs until under config.BLOB_STORE_MAX_BYTES.

    `protect` exempts the blob just written by this call's put() (it may have
    zero pins if nothing has referenced it yet this request). Pinned blobs are
    checked and deleted under the same lock pin()/unpin() use, so a pin can
    never lose a race against this sweep.
    """
    d = _blob_dir()
    if not d.exists():
        return
    entries = []
    total = 0
    for p in d.glob(f"*{_SUFFIX}"):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        total += sz
        entries.append((p, sz))
    max_bytes = config.BLOB_STORE_MAX_BYTES
    if total <= max_bytes:
        return
    entries.sort(key=lambda e: e[0].stat().st_mtime)  # oldest first
    for p, sz in entries:
        if total <= max_bytes:
            break
        h = p.stem
        if h == protect:
            continue
        with _pin_lock:
            if h in _pinned:
                continue
            try:
                p.unlink()
                total -= sz
            except OSError:
                continue
