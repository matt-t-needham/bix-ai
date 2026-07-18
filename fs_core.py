"""
Shared filesystem primitives used by both tools.py (async FastAPI) and
bix_mcp.py (sync MCP server). Stdlib-only so bix_mcp.py can import it
without pulling in FastAPI or httpx.

Each caller is responsible for validating that a path falls within its
own allowed roots before calling list_directory or read_file here.
"""
from pathlib import Path

_DENY_NAMES    = {".env", ".git", ".ssh", ".claude", ".gnupg", "secrets"}
_DENY_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".crt", ".cer")
_DENY_KEYWORDS = ("credential", "secret", "password", "passwd", "token")

# Directories under which writing could subvert the guardrails or privilege model.
# bix-ai's own source is deliberately NOT here anymore: self-changes are allowed
# to be *staged*, and staging.approve() applies them to the separate staging
# clone (bix-ai-staging), never the running prod tree. Promotion to prod is a
# human-gated host-side runner with its own independent validation.
_WRITE_DENY_DIRS = {"scripts", ".github"}

# Guardrail/privilege surface of bix-ai itself. Writes to these are stageable
# like any self-change, but the review UI flags them so the human reviews with
# extra care. UI signal only — containment comes from the staging-tree redirect.
_CRITICAL_SELF_FILES = {
    "fs_core.py", "staging.py", "config.py", "main.py",
    "tools.py", "bix_mcp.py", "deploy.py", "requirements.txt",
}


def is_critical_path(p: Path) -> bool:
    """True for guardrail-relevant files in bix-ai's own tree. Exact part match:
    'bix-ai-staging' is a different path part, so the staging clone never flags."""
    return "bix-ai" in p.parts and p.name in _CRITICAL_SELF_FILES


def is_denied_path(p: Path) -> bool:
    """Return True if the path matches a known-secrets pattern and must not be served."""
    name  = p.name
    lower = name.lower()
    if name.startswith(".env"):
        return True
    if name.endswith(_DENY_SUFFIXES):
        return True
    if any(kw in lower for kw in _DENY_KEYWORDS):
        return True
    return any(part in _DENY_NAMES for part in p.parts)


def is_write_denied_path(p: Path) -> bool:
    """Return True if writing to this path could subvert guardrails/privilege.

    Applied *in addition to* is_denied_path (secrets) on any write. Covers the
    privilege/guardrail-subverting class: shell scripts, container/CI config,
    and the scripts//.github dirs. The filename rules apply everywhere — a
    bix-ai/deploy.sh or bix-ai/Dockerfile stays denied even though bix-ai
    source files are stageable now. Does not restrict ordinary app source or
    content elsewhere — that is the point of the staged-write feature.
    """
    lower = p.name.lower()
    if lower.endswith(".sh"):
        return True
    if lower.startswith("dockerfile"):
        return True
    if (lower.startswith("docker-compose") or lower.startswith("compose")) and \
            lower.endswith((".yml", ".yaml")):
        return True
    return any(part in _WRITE_DENY_DIRS for part in p.parts)


def list_directory(p: Path) -> str:
    """Return a formatted directory listing as a plain string.

    p must already be validated (within allowed roots, not denied) by the caller.
    """
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        lines = [
            f"[dir]  {e.name}/" if e.is_dir() else f"[file] {e.name}  ({e.stat().st_size:,} bytes)"
            for e in entries
            if not is_denied_path(e)
        ]
        return "\n".join(lines) if lines else "(empty directory)"
    except PermissionError:
        return f"Permission denied: {p}"


def read_file(p: Path, max_bytes: int = 200_000) -> str:
    """Return the text content of a file as a plain string.

    p must already be validated (within allowed roots, not denied) by the caller.
    """
    size = p.stat().st_size
    if size > max_bytes:
        return f"File too large ({size:,} bytes). Max {max_bytes // 1_000} KB."
    try:
        return p.read_text(errors="replace")
    except PermissionError:
        return f"Permission denied: {p}"
    except Exception as e:
        return f"Error reading file: {e}"
