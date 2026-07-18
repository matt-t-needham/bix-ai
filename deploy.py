"""Deploy queue: file-based handoff between the container and the host-side
deploy runner (bix-infra/scripts/deploy-runner.py).

The container only ever *enqueues* requests and reads results — it has no
docker socket and never executes deploys itself. The runner (human-owned
systemd unit on the host) polls DEPLOY_DIR/queue, claims a request by
os.rename into processing/ (atomic), executes it, and writes a result JSON +
full log. Both sides write via tmp + os.replace so readers never see partial
JSON; the runner only globs *.json.

Layout under config.DEPLOY_DIR (a bind mount of bix-ai/data/deploy on the host):
    queue/<id>.json       request, written here by enqueue()
    processing/<id>.json  request, moved here by the runner while executing
    results/<id>.json     runner-written status/outcome
    logs/<id>.log         full runner output

Stdlib-only, mirrors staging.py's contract; config is read at call time so
tests can monkeypatch DEPLOY_DIR.
"""
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import config
import staging

ACTIONS = ("deploy-staging", "promote", "rollback")

# Statuses that mean the runner hasn't finished (or started) yet.
ACTIVE_STATUSES = ("queued", "running")

_LOG_TAIL_DEFAULT_LINES = 400


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_id(dep_id: str) -> str:
    return "".join(c for c in str(dep_id) if c.isalnum())


def _dirs() -> dict[str, Path]:
    root = config.DEPLOY_DIR
    return {name: root / name for name in ("queue", "processing", "results", "logs")}


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def enqueue(action: str, *, record_ids: list[str] | None = None,
            note: str = "", requested_by: str = "human") -> dict:
    """Validate and queue a deploy request. Raises ValueError if invalid.

    promote additionally requires at least one staging record id, each of
    which must exist, be approved, and be a self-change — the runner still
    re-validates everything independently (it never trusts the queue).
    """
    if action not in ACTIONS:
        raise ValueError(f"unknown action: {action!r} (expected one of {', '.join(ACTIONS)})")
    record_ids = [str(r) for r in (record_ids or [])]

    if action == "promote":
        if not record_ids:
            raise ValueError("promote requires at least one staging record id")
        for rid in record_ids:
            rec = staging.get(rid)
            if rec is None:
                raise ValueError(f"no such staging record: {rid}")
            if rec.get("status") != "approved":
                raise ValueError(f"record {rid} is {rec.get('status')}, not approved")
            if not rec.get("self_change"):
                raise ValueError(f"record {rid} is not a bix-ai self-change")

    request = {
        "id":           uuid.uuid4().hex[:12],
        "action":       action,
        "record_ids":   record_ids,
        "note":         str(note or ""),
        "requested_by": str(requested_by or "human"),
        "requested_at": _now(),
    }
    _write_json_atomic(_dirs()["queue"] / f"{request['id']}.json", request)
    return request


def get(dep_id: str) -> dict | None:
    """Request merged with the runner's result. Status defaults to 'queued'
    until the runner writes a result."""
    d = _dirs()
    safe = _safe_id(dep_id)
    request = _read_json(d["queue"] / f"{safe}.json") or \
              _read_json(d["processing"] / f"{safe}.json")
    result = _read_json(d["results"] / f"{safe}.json")
    if request is None and result is None:
        return None
    merged = {"status": "queued", **(request or {}), **(result or {})}
    merged.setdefault("id", safe)
    return merged


def list_all() -> list[dict]:
    """All deploys, active (queued/running) first, newest first within groups."""
    d = _dirs()
    ids: set[str] = set()
    for name in ("queue", "processing", "results"):
        if d[name].exists():
            ids.update(p.stem for p in d[name].glob("*.json"))
    out = [dep for dep_id in ids if (dep := get(dep_id))]
    out.sort(key=lambda r: r.get("requested_at", ""), reverse=True)          # newest first
    out.sort(key=lambda r: r.get("status") not in ACTIVE_STATUSES)           # stable: active first
    return out


def read_log(dep_id: str, lines: int = _LOG_TAIL_DEFAULT_LINES) -> str:
    """Tail of the runner's log for a deploy; empty string if none yet."""
    path = _dirs()["logs"] / f"{_safe_id(dep_id)}.log"
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])
