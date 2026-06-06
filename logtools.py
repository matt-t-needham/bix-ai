"""Log-review helpers: enumerate and tail logs for the internal apps and Steam.

Pure logic (no main.py imports), stdlib-only, mirroring steam.py / staging.py.
Reads are confined to two roots (config.INTERNAL_LOG_ROOT, config.STEAM_LOG_ROOT)
which are read into config at import; logtools reads them via `config.*` at call
time so tests can monkeypatch.

Logs can be large (Steam's console-linux.txt is several MB), so read_log tails
the file — it scans at most the last few MB and returns the last N lines — rather
than loading whole files like fs_core.read_file.
"""
import time
from pathlib import Path

import config
from fs_core import is_denied_path

_MAX_SCAN_BYTES   = 8_000_000    # tail by scanning at most the last ~8 MB
_MAX_OUTPUT_BYTES = 200_000      # cap returned text
_DEFAULT_LINES    = 200
_MAX_LINES        = 2000


def _log_roots() -> list[Path]:
    return [config.INTERNAL_LOG_ROOT, config.STEAM_LOG_ROOT]


def _fmt_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def _fmt_mtime(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def _resolve_in_roots(path_str: str) -> Path | None:
    """Resolve path_str and return it only if it lies under a known log root."""
    try:
        rp = Path(path_str).resolve()
    except Exception:
        return None
    for root in _log_roots():
        try:
            rp.relative_to(root.resolve())
            return rp
        except ValueError:
            continue
    return None


def list_sources() -> str:
    """List available log files under both roots, with size + mtime."""
    out: list[str] = []

    ir = config.INTERNAL_LOG_ROOT
    out.append(f"Internal app logs (root: {ir}):")
    if ir.is_dir():
        for e in sorted(ir.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
            if is_denied_path(e):
                continue
            if e.is_dir():
                files = sorted(
                    (f for f in e.iterdir() if f.is_file() and not is_denied_path(f)),
                    key=lambda f: -f.stat().st_mtime,
                )
                out.append(f"  {e.name}/")
                for f in files[:8]:
                    st = f.stat()
                    out.append(f"    {f}  ({_fmt_size(st.st_size)}, {_fmt_mtime(st.st_mtime)})")
                if len(files) > 8:
                    out.append(f"    … +{len(files) - 8} more in {e.name}/")
            else:
                st = e.stat()
                out.append(f"  {e}  ({_fmt_size(st.st_size)}, {_fmt_mtime(st.st_mtime)})")
    else:
        out.append("  (not available)")

    sr = config.STEAM_LOG_ROOT
    out.append("")
    out.append(f"Steam logs (root: {sr}):")
    if sr.is_dir():
        files = sorted(
            (f for f in sr.iterdir() if f.is_file() and not is_denied_path(f)),
            key=lambda f: -f.stat().st_mtime,
        )
        for f in files[:30]:
            st = f.stat()
            out.append(f"  {f}  ({_fmt_size(st.st_size)}, {_fmt_mtime(st.st_mtime)})")
        if len(files) > 30:
            out.append(f"  … +{len(files) - 30} more (most-recently-modified shown first)")
    else:
        out.append("  (not mounted)")

    return "\n".join(out)


def read_log(path: str, lines: int = _DEFAULT_LINES, contains: str | None = None) -> str:
    """Return the last `lines` lines of a log under a known root, optionally
    filtered to lines containing `contains` (case-insensitive)."""
    rp = _resolve_in_roots(path)
    if rp is None:
        roots = ", ".join(str(r) for r in _log_roots())
        return f"Access denied: '{path}' is not under a known log root ({roots})."
    if is_denied_path(rp):
        return f"Access denied: '{path}' is a protected path."
    if not rp.exists():
        return f"Log not found: {path}"
    if not rp.is_file():
        return f"Not a file: {path}"

    n = max(1, min(int(lines), _MAX_LINES))
    try:
        size = rp.stat().st_size
        with rp.open("rb") as fh:
            scanned_tail = size > _MAX_SCAN_BYTES
            if scanned_tail:
                fh.seek(size - _MAX_SCAN_BYTES)
                fh.readline()  # drop the partial first line
            data = fh.read()
    except OSError as e:
        return f"Error reading log: {e}"

    all_lines = data.decode("utf-8", errors="replace").splitlines()
    if contains:
        needle = contains.lower()
        all_lines = [ln for ln in all_lines if needle in ln.lower()]
        if not all_lines:
            return f"No lines containing {contains!r} in the scanned portion of {rp.name}."

    tail = all_lines[-n:]
    body = "\n".join(tail)
    if len(body.encode("utf-8", errors="replace")) > _MAX_OUTPUT_BYTES:
        body = body[-_MAX_OUTPUT_BYTES:]

    header = f"# {rp} — last {len(tail)} line(s)"
    if contains:
        header += f" matching {contains!r}"
    if scanned_tail:
        header += f" (scanned last {_MAX_SCAN_BYTES // 1_000_000} MB of {_fmt_size(size)})"
    return header + "\n" + body
