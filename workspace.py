"""bix-ai's only write path: structured edits to bix-infra/todos/<project>.md and
free-form drafts in bix-infra/outbox/. Stdlib-only — shared by tools.py (the
app) and bix_mcp.py (pro mode's MCP server), like fs_core.

Todo files follow bix-infra/todos/GUIDE.md (## Pending / ## Plans / ## Done).
Edits only ever add lines or move a Pending item to Done — nothing existing is
rewritten or deleted. Everything written is tagged with who wrote it
(`actor`, e.g. "local:gemma4:26b") because Claude Code sessions read todos/
and act on them: an entry a model wrote must stay recognisable as one.

Writes are atomic (temp file + rename) and serialised by an flock on the
directory, which the app and the MCP subprocess share. Directories are read
from the environment at call time so tests can point them elsewhere.

Every function returns a one-line result for the model, or raises
WorkspaceError with a message to hand back to it.
"""
import fcntl
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
OUTBOX_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}\.(md|txt)$")
RESERVED_TODO_FILES = {"all", "guide"}       # ALL.md is generated; GUIDE.md is the spec
MAX_TASK = 200
MAX_FIELD = 1000
MAX_STEPS = 30
MAX_OUTBOX_BYTES = 100_000


class WorkspaceError(ValueError):
    pass


def todos_dir() -> Path:
    return Path(os.environ.get("TODOS_DIR", "/home/matt/apps/bix-infra/todos"))


def outbox_dir() -> Path:
    return Path(os.environ.get("OUTBOX_DIR", "/home/matt/apps/bix-infra/outbox"))


# ── File plumbing ─────────────────────────────────────────────────────────────

@contextmanager
def _locked(directory: Path):
    if not directory.is_dir():
        raise WorkspaceError(f"{directory} does not exist or is not writable from here")
    fd = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)                      # releases the lock


def _target(directory: Path, filename: str) -> Path:
    path = directory / filename
    if path.is_symlink():
        raise WorkspaceError(f"{filename} is a symlink — refusing to write through it")
    if path.resolve().parent != directory.resolve():
        raise WorkspaceError(f"{filename} is outside {directory}")
    return path


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o664)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _one_line(value, field: str, limit: int, required: bool = False) -> str:
    text = " ".join(str(value or "").split())           # collapse newlines/whitespace
    if required and not text:
        raise WorkspaceError(f"{field} is required")
    if len(text) > limit:
        raise WorkspaceError(f"{field} is {len(text)} chars; the limit is {limit}")
    return text


def _actor(actor: str) -> str:
    return _one_line(actor, "actor", 100) or "unknown model"


# ── Todo files ────────────────────────────────────────────────────────────────

def _todo_path(project: str) -> Path:
    project = (project or "").strip().lower()
    if project.endswith(".md"):
        project = project[:-3]
    if not PROJECT_RE.match(project) or project in RESERVED_TODO_FILES:
        raise WorkspaceError(
            f"'{project}' is not a valid project name — use the todo file's name without .md, "
            "e.g. 'bix-ai' or 'rageagain_channel' (ALL and GUIDE can't be edited)")
    return _target(todos_dir(), f"{project}.md")


def _load(path: Path, project: str) -> list[str]:
    if path.exists():
        return path.read_text(encoding="utf-8").splitlines()
    return [f"# {project}", "", "## Pending", "", "## Plans", "", "## Done", ""]


def _section(lines: list[str], name: str) -> tuple[int, int]:
    """(heading index, end index) of `## name`, adding the section if missing."""
    for i, line in enumerate(lines):
        if line.strip().lower() == f"## {name}".lower():
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
            return i, end
    lines += ["", f"## {name}", ""]
    return len(lines) - 2, len(lines)


def _append_to_section(lines: list[str], name: str, block: list[str]) -> None:
    """Add block at the end of section `name`, one blank line either side."""
    head, end = _section(lines, name)
    last = end
    while last > head + 1 and not lines[last - 1].strip():
        last -= 1
    lines[last:end] = ["", *block, ""]


def _save(path: Path, lines: list[str]) -> None:
    _atomic_write(path, "\n".join(lines).rstrip("\n") + "\n")


def add_todo(project: str, task: str, why: str = "", notes: str = "", actor: str = "") -> str:
    task = _one_line(task, "task", MAX_TASK, required=True)
    why = _one_line(why, "why", MAX_FIELD)
    notes = _one_line(notes, "notes", MAX_FIELD)
    path = _todo_path(project)
    with _locked(path.parent):
        lines = _load(path, path.stem)
        block = [f"- [ ] {task}  "]
        if why:
            block.append(f"  **Why:** {why}  ")
        if notes:
            block.append(f"  **Notes:** {notes}  ")
        block.append(f"  **Added by:** ai.Bix ({_actor(actor)}), {date.today().isoformat()}")
        _append_to_section(lines, "Pending", block)
        _save(path, lines)
    return f"Added to {path.name} under Pending: {task}"


def add_plan(project: str, title: str, goal: str, steps: list, effort: str = "",
             actor: str = "") -> str:
    title = _one_line(title, "title", MAX_TASK, required=True)
    goal = _one_line(goal, "goal", MAX_FIELD, required=True)
    effort = _one_line(effort, "effort", 20).lower()
    if effort and effort not in {"small", "medium", "large"}:
        raise WorkspaceError("effort must be small, medium or large")
    if not isinstance(steps, list) or not steps:
        raise WorkspaceError("steps must be a non-empty list of strings")
    if len(steps) > MAX_STEPS:
        raise WorkspaceError(f"{len(steps)} steps; the limit is {MAX_STEPS}")
    steps = [_one_line(s, f"step {i}", MAX_FIELD, required=True) for i, s in enumerate(steps, 1)]
    path = _todo_path(project)
    with _locked(path.parent):
        lines = _load(path, path.stem)
        block = [f"### {title}", "", f"**Goal:** {goal}  "]
        if effort:
            block.append(f"**Effort:** {effort}  ")
        block.append(f"**Added by:** ai.Bix ({_actor(actor)}), {date.today().isoformat()}")
        block.append("")
        block += [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        _append_to_section(lines, "Plans", block)
        _save(path, lines)
    return f"Added plan '{title}' to {path.name} ({len(steps)} steps)"


def complete_todo(project: str, match: str, actor: str = "") -> str:
    match = _one_line(match, "match", MAX_TASK, required=True).lower()
    path = _todo_path(project)
    with _locked(path.parent):
        if not path.exists():
            raise WorkspaceError(f"{path.name} doesn't exist")
        lines = _load(path, path.stem)
        start, end = _section(lines, "Pending")
        hits = [i for i in range(start + 1, end)
                if lines[i].startswith("- [ ] ") and match in lines[i].lower()]
        if not hits:
            raise WorkspaceError(f"No open item in {path.name}'s Pending section contains '{match}'")
        if len(hits) > 1:
            found = "; ".join(lines[i][6:].strip()[:80] for i in hits[:5])
            raise WorkspaceError(f"'{match}' matches {len(hits)} open items — be more specific: {found}")
        i = hits[0]
        j = i + 1
        while j < end and lines[j].startswith("  ") and lines[j].strip():   # the item's continuation lines
            j += 1
        task = lines[i][6:].rstrip()
        del lines[i:j]
        if i < len(lines) and not lines[i].strip() and i > 0 and not lines[i - 1].strip():
            del lines[i]                                     # don't leave a double blank line
        _append_to_section(lines, "Done", [
            f"- [x] {task} — completed {date.today().isoformat()} (marked by ai.Bix, {_actor(actor)})"])
        _save(path, lines)
    return f"Moved to Done in {path.name}: {task}"


# ── Outbox ────────────────────────────────────────────────────────────────────

def write_outbox(name: str, content: str, actor: str = "") -> str:
    name = (name or "").strip()
    if not OUTBOX_NAME_RE.match(name):
        raise WorkspaceError(
            "name must be a plain file name ending in .md or .txt (letters, digits, . _ -), e.g. 'blog-draft.md'")
    content = str(content or "")
    if not content.strip():
        raise WorkspaceError("content is empty")
    if len(content.encode("utf-8")) > MAX_OUTBOX_BYTES:
        raise WorkspaceError(f"content is over {MAX_OUTBOX_BYTES // 1000} KB")
    stamp = f"written by ai.Bix ({_actor(actor)}) at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"
    header = f"<!-- {stamp} -->" if name.endswith(".md") else f"[{stamp}]"
    directory = outbox_dir()
    path = _target(directory, name)
    with _locked(directory):
        existed = path.exists()
        _atomic_write(path, f"{header}\n{content.rstrip()}\n")
    return f"{'Overwrote' if existed else 'Wrote'} {path} ({len(content):,} chars)"
