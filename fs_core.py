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
