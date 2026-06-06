"""Project TODO reader.

Pure logic, stdlib-only (mirrors steam.py / logtools.py). Surfaces the project
TODO files so the assistant can answer "review the TODOs" with a dedicated tool
rather than guessing at directory paths. config.TODOS_DIR is read at call time so
tests can monkeypatch it.

Layout (bix-infra/todos/):
  ALL.md          — compiled all-projects view (generated)
  <project>.md    — per-project todos (bix-ai, infra, demucs, …)
  GUIDE.md        — how-to, not a todo list
"""
import config

_MAX_CHARS = 100_000
_NON_PROJECT = {"ALL", "GUIDE"}


def list_projects() -> list[str]:
    d = config.TODOS_DIR
    if not d.is_dir():
        return []
    return sorted(f.stem for f in d.glob("*.md") if f.stem not in _NON_PROJECT)


def read_todos(project: str | None = None) -> str:
    """Return the compiled TODO list, or a single project's todos if named."""
    d = config.TODOS_DIR
    if not d.is_dir():
        return f"No TODOs directory at {d}."

    if project:
        safe = "".join(c for c in project if c.isalnum() or c in "-_")
        f = d / f"{safe}.md"
        if not f.is_file():
            avail = ", ".join(list_projects()) or "none"
            return f"No TODO file for project '{project}'. Available projects: {avail}."
        return f.read_text(errors="replace")[:_MAX_CHARS]

    compiled = d / "ALL.md"
    if compiled.is_file():
        return compiled.read_text(errors="replace")[:_MAX_CHARS]

    # Fallback: concatenate per-project files if the compiled view is missing.
    parts = [f"## {p}\n{(d / f'{p}.md').read_text(errors='replace')}" for p in list_projects()]
    return ("\n\n".join(parts))[:_MAX_CHARS] if parts else "No TODOs found."
