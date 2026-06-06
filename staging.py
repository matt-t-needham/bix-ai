"""Staged-write store: propose -> review -> apply.

A model proposes a file write via stage_write (tools.py) or the MCP write_file
tool (bix_mcp.py). The proposal lands here as a JSON record and **never** touches
the live tree. A human reviews the diff at /staging and approves — the only step
that writes to disk, always human-triggered, re-validating every guard.

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
from fs_core import is_denied_path, is_write_denied_path

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
        "claude_review": None,
        "reviewed_at":   None,
        "applied_at":    None,
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


def set_review(rec_id: str, review_text: str) -> dict | None:
    rec = get(rec_id)
    if not rec:
        return None
    rec["claude_review"] = review_text
    rec["reviewed_at"]   = _now()
    _write_record(rec)
    return rec


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
    except ValueError as e:
        return {"ok": False, "message": f"Re-validation failed: {e}", "record": rec}

    try:
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(rec["content"])
    except OSError as e:
        # Read-only mount / permission — keep the record pending, report clearly.
        return {"ok": False, "message": f"Write failed (target not writable?): {e}", "record": rec}

    rec["status"]     = "approved"
    rec["applied_at"] = _now()
    _write_record(rec)
    return {"ok": True, "message": f"Wrote {len(rec['content'])} bytes to {rp}", "record": rec}


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
