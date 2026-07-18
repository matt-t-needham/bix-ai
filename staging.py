"""Staged-write store: propose -> review -> apply.

A model proposes a file write via stage_write (tools.py) or the MCP write_file
tool (bix_mcp.py). The proposal lands here as a JSON record and **never** touches
the live tree. A human reviews the diff at /staging and approves — the only step
that writes to disk, always human-triggered, re-validating every guard.

Self-changes (targets inside bix-ai's own prod tree) are additionally redirected
at approve time to the staging clone via apply_path_for(); the prod tree is only
ever written by the human-gated host-side deploy runner (promote).

Stdlib-only by contract: bix_mcp.py is a stdio server that imports nothing
heavier than fs_core, and it routes through this module. So no FastAPI / httpx /
pydantic here — only json / pathlib / datetime / uuid, mirroring memory.py.

FS_ROOT / STAGING_DIR are read from `config` at call time (not bound at import)
so tests can monkeypatch them.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config
from fs_core import is_critical_path, is_denied_path, is_write_denied_path

_MAX_CONTENT_BYTES = 200_000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _record_path(rec_id: str) -> Path:
    # Guard against path traversal via a crafted id — keep alnum only.
    safe = "".join(c for c in str(rec_id) if c.isalnum())
    return config.STAGING_DIR / f"{safe}.json"


def _write_record(record: dict) -> None:
    config.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    _record_path(record["id"]).write_text(json.dumps(record, indent=2))


def validate_target(target_path: str) -> Path:
    """Resolve and check a write target. Raise ValueError(reason) if not writable.

    A write must be within FS_ROOT, not a secrets path, and not a guardrail path.
    Resolves first so symlinks can't escape the checks.
    """
    if not target_path:
        raise ValueError("no target path given")
    try:
        rp = Path(target_path).resolve()
    except Exception:
        raise ValueError(f"invalid path: {target_path}")
    try:
        rp.relative_to(config.FS_ROOT)
    except ValueError:
        raise ValueError(f"'{target_path}' is outside the allowed root ({config.FS_ROOT})")
    if is_denied_path(rp):
        raise ValueError(f"'{target_path}' is a protected (secrets) path")
    if is_write_denied_path(rp):
        raise ValueError(
            f"'{target_path}' is a protected (guardrail) path — writes here are not allowed"
        )
    return rp


def apply_path_for(rp: Path) -> Path:
    """Where an approved write is actually applied.

    Targets inside the service's own prod tree (config.self_prod_tree()) are
    redirected to the same relative path in the staging clone
    (config.self_staging_tree()); everything else applies in place. This
    process never writes the prod tree — promotion to prod is the host-side
    deploy runner's job, human-gated and independently validated.
    """
    try:
        rel = Path(rp).relative_to(config.self_prod_tree())
    except ValueError:
        return Path(rp)
    return config.self_staging_tree() / rel


def current_source_path(record: dict) -> Path:
    """The path holding the current on-disk version of a record's target — the
    file approve() writes (applied_to once applied). Diffs and 'current file'
    context must read this, not target_path, or self-change diffs would compare
    against the untouched prod tree."""
    if record.get("applied_to"):
        return Path(record["applied_to"])
    return apply_path_for(Path(record["target_path"]))


def create(target_path: str, content: str, proposed_by: str = "assistant") -> dict:
    """Validate and persist a pending write proposal. Raises ValueError if denied.

    Never writes the live target — only the staging record.
    """
    rp = validate_target(target_path)
    if len(content.encode("utf-8", errors="replace")) > _MAX_CONTENT_BYTES:
        raise ValueError(f"content too large (max {_MAX_CONTENT_BYTES // 1000} KB)")
    record = {
        "id":            uuid.uuid4().hex[:12],
        "created_at":    _now(),
        "proposed_by":   proposed_by,
        "target_path":   str(rp),
        "content":       content,
        "status":        "pending",   # pending | approved | rejected
        "critical":      is_critical_path(rp),          # guardrail-surface file — UI flag
        "self_change":   apply_path_for(rp) != rp,      # applies to the staging clone
        "applied_to":    None,   # actual path written by approve()
        "promoted_at":   None,   # set by the deploy runner when promoted to prod
        "claude_review": None,
        "review_model":  None,
        "reviewed_at":   None,
        "applied_at":    None,
        "comments":      [],   # [{id, quote, text, resolved, created_at, resolved_at}]
        "revisions":     [],   # superseded contents, oldest first: [{content, at, by}]
    }
    _write_record(record)
    return record


def get(rec_id: str) -> dict | None:
    p = _record_path(rec_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def list_records() -> list[dict]:
    """All records, pending first, newest first within each group."""
    if not config.STAGING_DIR.exists():
        return []
    out: list[dict] = []
    for f in sorted(config.STAGING_DIR.glob("*.json")):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            continue
    out.sort(key=lambda r: r.get("created_at", ""), reverse=True)        # newest first
    out.sort(key=lambda r: r.get("status") != "pending")                 # stable: pending first
    return out


def set_review(rec_id: str, review_text: str, model: str | None = None) -> dict | None:
    rec = get(rec_id)
    if not rec:
        return None
    rec["claude_review"] = review_text
    rec["review_model"]  = model
    rec["reviewed_at"]   = _now()
    _write_record(rec)
    return rec


_MAX_COMMENT_QUOTE_CHARS = 2_000
_MAX_COMMENT_TEXT_CHARS  = 4_000
_MAX_REVISIONS_KEPT      = 5


def add_comment(rec_id: str, quote: str, text: str) -> dict | None:
    """Attach a reviewer comment anchored to a verbatim quote of the content.

    Returns the updated record, or None if the record doesn't exist.
    Raises ValueError on empty comment text or oversized quote/text.
    """
    rec = get(rec_id)
    if not rec:
        return None
    quote = (quote or "").strip("\n")
    text  = (text or "").strip()
    if not text:
        raise ValueError("comment text is empty")
    if len(quote) > _MAX_COMMENT_QUOTE_CHARS:
        raise ValueError(f"quote too long (max {_MAX_COMMENT_QUOTE_CHARS} chars)")
    if len(text) > _MAX_COMMENT_TEXT_CHARS:
        raise ValueError(f"comment too long (max {_MAX_COMMENT_TEXT_CHARS} chars)")
    comment = {
        "id":          uuid.uuid4().hex[:8],
        "quote":       quote,
        "text":        text,
        "resolved":    False,
        "created_at":  _now(),
        "resolved_at": None,
    }
    rec.setdefault("comments", []).append(comment)
    _write_record(rec)
    return rec


def set_comment_resolved(rec_id: str, comment_id: str, resolved: bool) -> dict | None:
    """Mark one comment resolved/reopened. Returns the record, or None if
    either the record or the comment doesn't exist."""
    rec = get(rec_id)
    if not rec:
        return None
    for c in rec.get("comments", []):
        if c.get("id") == comment_id:
            c["resolved"]    = bool(resolved)
            c["resolved_at"] = _now() if resolved else None
            _write_record(rec)
            return rec
    return None


def update_content(rec_id: str, new_content: str, revised_by: str) -> dict:
    """Replace a pending record's proposed content, keeping the old version in
    the revision history. Never touches the live target — approve() remains
    the only privileged step. Returns {ok, message, record?} like approve().
    """
    rec = get(rec_id)
    if not rec:
        return {"ok": False, "message": "No such staged change."}
    if rec["status"] != "pending":
        return {"ok": False, "message": f"Cannot revise a {rec['status']} change.", "record": rec}
    if len(new_content.encode("utf-8", errors="replace")) > _MAX_CONTENT_BYTES:
        return {"ok": False, "message": f"Revised content too large (max {_MAX_CONTENT_BYTES // 1000} KB).", "record": rec}
    revisions = rec.setdefault("revisions", [])
    revisions.append({"content": rec["content"], "at": _now(), "by": revised_by})
    del revisions[:-_MAX_REVISIONS_KEPT]
    rec["content"] = new_content
    _write_record(rec)
    return {"ok": True, "message": "Content revised.", "record": rec}


def approve(rec_id: str) -> dict:
    """Re-validate and write the target live. The only privileged step.

    Returns {ok, message, record?}. Re-checks all guards from scratch (never
    trusts the stage-time check) and degrades gracefully if the target is not
    writable (e.g. a read-only mount), leaving the record pending.
    """
    rec = get(rec_id)
    if not rec:
        return {"ok": False, "message": "No such staged change."}
    if rec["status"] == "approved":
        return {"ok": False, "message": "Already applied.", "record": rec}
    if rec["status"] == "rejected":
        return {"ok": False, "message": "Cannot approve a rejected change.", "record": rec}

    try:
        rp = validate_target(rec["target_path"])
        ap = apply_path_for(rp)
        if ap != rp:
            # Self-change redirected to the staging clone: run the full guard
            # set on the rewritten path too (resolve/containment/denylists), so
            # e.g. a symlinked bix-ai-staging can't escape FS_ROOT.
            ap = validate_target(str(ap))
    except ValueError as e:
        return {"ok": False, "message": f"Re-validation failed: {e}", "record": rec}

    try:
        ap.parent.mkdir(parents=True, exist_ok=True)
        ap.write_text(rec["content"])
    except OSError as e:
        # Read-only mount / permission — keep the record pending, report clearly.
        return {"ok": False, "message": f"Write failed (target not writable?): {e}", "record": rec}

    rec["status"]     = "approved"
    rec["applied_at"] = _now()
    rec["applied_to"] = str(ap)
    _write_record(rec)
    return {"ok": True, "message": f"Wrote {len(rec['content'])} bytes to {ap}", "record": rec}


def reject(rec_id: str) -> dict:
    rec = get(rec_id)
    if not rec:
        return {"ok": False, "message": "No such staged change."}
    if rec["status"] == "approved":
        return {"ok": False, "message": "Already applied; cannot reject.", "record": rec}
    rec["status"]      = "rejected"
    rec["reviewed_at"] = _now()
    _write_record(rec)
    return {"ok": True, "message": "Rejected.", "record": rec}
