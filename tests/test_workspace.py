"""workspace.py — bix-ai's only write path (todos/ and outbox/)."""
import os
from datetime import date

import pytest

import workspace
from workspace import WorkspaceError

TODAY = date.today().isoformat()

EXISTING = """# bix-ai

Intro line.

## Pending

- [ ] Chat history — persist conversations
  **Why:** lost on reload
  **Notes:** spike first
- [x] Old done thing — 2026-05-02

- [ ] Plan mode — present a plan first
  **Why:** fewer surprises

## Plans

### Existing plan

1. Step

## Done

- [x] Shipped — completed 2026-01-01
"""


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    todos, outbox = tmp_path / "todos", tmp_path / "outbox"
    todos.mkdir()
    outbox.mkdir()
    monkeypatch.setenv("TODOS_DIR", str(todos))
    monkeypatch.setenv("OUTBOX_DIR", str(outbox))
    (todos / "bix-ai.md").write_text(EXISTING)
    return todos, outbox


def _lines(path):
    return path.read_text().splitlines()


def _kept_in_order(before: str, after: str) -> bool:
    """Every original line is still present, in the original order."""
    it = iter(after.splitlines())
    return all(any(line == a for a in it) for line in before.splitlines())


# ── add_todo ──────────────────────────────────────────────────────────────────

def test_add_todo_appends_to_pending_and_keeps_everything(dirs):
    todos, _ = dirs
    out = workspace.add_todo("bix-ai", "Write tools", why="asked for", notes="see GUIDE",
                             actor="local:gemma4:26b")
    assert "bix-ai.md" in out
    text = (todos / "bix-ai.md").read_text()
    assert _kept_in_order(EXISTING, text)
    lines = text.splitlines()
    i = lines.index("- [ ] Write tools  ")
    assert lines[i + 1:i + 4] == [
        "  **Why:** asked for  ",
        "  **Notes:** see GUIDE  ",
        f"  **Added by:** ai.Bix (local:gemma4:26b), {TODAY}",
    ]
    assert lines.index("## Pending") < i < lines.index("## Plans")
    assert lines[i - 1] == "" and lines[i + 4] == ""       # blank line either side


def test_add_todo_creates_a_new_project_file_in_guide_shape(dirs):
    todos, _ = dirs
    workspace.add_todo("beatshare", "Add swing", actor="api:claude")
    lines = _lines(todos / "beatshare.md")
    assert lines[0] == "# beatshare"
    assert [l for l in lines if l.startswith("## ")] == ["## Pending", "## Plans", "## Done"]
    assert lines.index("## Pending") < lines.index("- [ ] Add swing  ") < lines.index("## Plans")


def test_add_todo_collapses_newlines_so_one_item_stays_one_item(dirs):
    todos, _ = dirs
    workspace.add_todo("bix-ai", "Task", why="line one\n## Done\n- [ ] injected", actor="x")
    lines = _lines(todos / "bix-ai.md")
    assert lines.count("## Done") == 1
    assert "  **Why:** line one ## Done - [ ] injected  " in lines


@pytest.mark.parametrize("project", ["ALL", "all.md", "GUIDE", "../etc", "a/b", "", "Bix AI", "x" * 50])
def test_add_todo_rejects_bad_or_reserved_project_names(dirs, project):
    with pytest.raises(WorkspaceError):
        workspace.add_todo(project, "t", actor="x")


def test_add_todo_accepts_a_trailing_md_and_any_case(dirs):
    todos, _ = dirs
    workspace.add_todo("Bix-AI.md", "Case test", actor="x")
    assert "- [ ] Case test  " in _lines(todos / "bix-ai.md")


def test_add_todo_limits(dirs):
    with pytest.raises(WorkspaceError, match="task is required"):
        workspace.add_todo("bix-ai", "   ", actor="x")
    with pytest.raises(WorkspaceError, match="limit"):
        workspace.add_todo("bix-ai", "t" * 201, actor="x")
    with pytest.raises(WorkspaceError, match="limit"):
        workspace.add_todo("bix-ai", "t", notes="n" * 1001, actor="x")


def test_refuses_to_write_through_a_symlink(dirs, tmp_path):
    todos, _ = dirs
    victim = tmp_path / "victim.md"
    victim.write_text("keep me")
    (todos / "evil.md").symlink_to(victim)
    with pytest.raises(WorkspaceError, match="symlink"):
        workspace.add_todo("evil", "t", actor="x")
    assert victim.read_text() == "keep me"


def test_missing_directory_is_an_error_not_a_silent_create(tmp_path, monkeypatch):
    monkeypatch.setenv("TODOS_DIR", str(tmp_path / "nope"))
    with pytest.raises(WorkspaceError, match="does not exist"):
        workspace.add_todo("bix-ai", "t", actor="x")
    assert not (tmp_path / "nope").exists()


# ── add_plan ──────────────────────────────────────────────────────────────────

def test_add_plan_appends_under_plans(dirs):
    todos, _ = dirs
    workspace.add_plan("bix-ai", "New plan", "Do the thing", ["First", "Second"],
                       effort="Small", actor="pro:opus")
    text = (todos / "bix-ai.md").read_text()
    assert _kept_in_order(EXISTING, text)
    lines = text.splitlines()
    i = lines.index("### New plan")
    assert lines.index("## Plans") < lines.index("### Existing plan") < i < lines.index("## Done")
    assert lines[i:i + 8] == [
        "### New plan", "", "**Goal:** Do the thing  ", "**Effort:** small  ",
        f"**Added by:** ai.Bix (pro:opus), {TODAY}", "", "1. First", "2. Second",
    ]


@pytest.mark.parametrize("kwargs,match", [
    ({"steps": []}, "non-empty"),
    ({"steps": "one step"}, "non-empty"),
    ({"steps": ["s"] * 31}, "limit"),
    ({"steps": ["ok", ""]}, "step 2 is required"),
    ({"steps": ["s"], "effort": "huge"}, "small, medium or large"),
])
def test_add_plan_validation(dirs, kwargs, match):
    with pytest.raises(WorkspaceError, match=match):
        workspace.add_plan("bix-ai", "T", "G", actor="x", **kwargs)


# ── complete_todo ─────────────────────────────────────────────────────────────

def test_complete_todo_moves_the_whole_item_to_done(dirs):
    todos, _ = dirs
    out = workspace.complete_todo("bix-ai", "chat HISTORY", actor="local:qwen")
    assert "Chat history" in out
    lines = _lines(todos / "bix-ai.md")
    pending = lines[lines.index("## Pending"):lines.index("## Plans")]
    assert not any("Chat history" in l or "lost on reload" in l or "spike first" in l for l in pending)
    assert "- [ ] Plan mode — present a plan first" in pending       # neighbours untouched
    assert "- [x] Old done thing — 2026-05-02" in pending
    done = lines[lines.index("## Done"):]
    assert done[-1] == (f"- [x] Chat history — persist conversations — completed {TODAY} "
                        "(marked by ai.Bix, local:qwen)")
    assert "- [x] Shipped — completed 2026-01-01" in done
    assert "\n\n\n" not in (todos / "bix-ai.md").read_text()


def test_complete_todo_ambiguous_or_missing(dirs):
    todos, _ = dirs
    before = (todos / "bix-ai.md").read_text()
    with pytest.raises(WorkspaceError, match="matches 2 open items"):
        workspace.complete_todo("bix-ai", "—", actor="x")
    with pytest.raises(WorkspaceError, match="No open item"):
        workspace.complete_todo("bix-ai", "nonexistent", actor="x")
    with pytest.raises(WorkspaceError, match="No open item"):
        workspace.complete_todo("bix-ai", "old done thing", actor="x")   # already ticked
    with pytest.raises(WorkspaceError, match="doesn't exist"):
        workspace.complete_todo("beatshare", "x", actor="x")
    assert (todos / "bix-ai.md").read_text() == before


# ── outbox ────────────────────────────────────────────────────────────────────

def test_write_outbox_md_and_txt_with_provenance(dirs):
    _, outbox = dirs
    assert workspace.write_outbox("draft.md", "# Hi\n\nBody", actor="api:claude").startswith("Wrote")
    lines = _lines(outbox / "draft.md")
    assert lines[0].startswith("<!-- written by ai.Bix (api:claude) at ") and lines[0].endswith(" UTC -->")
    assert lines[1:] == ["# Hi", "", "Body"]
    workspace.write_outbox("notes.txt", "plain", actor="x")
    assert _lines(outbox / "notes.txt")[0].startswith("[written by ai.Bix (x) at ")
    assert workspace.write_outbox("draft.md", "v2", actor="x").startswith("Overwrote")
    assert _lines(outbox / "draft.md")[1:] == ["v2"]


@pytest.mark.parametrize("name", ["../x.md", "a/b.md", "x.sh", "x.md.sh", ".hidden.md", "", "x" * 90 + ".md"])
def test_write_outbox_rejects_bad_names(dirs, name):
    with pytest.raises(WorkspaceError):
        workspace.write_outbox(name, "c", actor="x")


def test_write_outbox_limits(dirs):
    with pytest.raises(WorkspaceError, match="empty"):
        workspace.write_outbox("a.md", "  \n", actor="x")
    with pytest.raises(WorkspaceError, match="KB"):
        workspace.write_outbox("a.md", "x" * 100_001, actor="x")


def test_writes_are_atomic_and_leave_no_temp_files(dirs):
    todos, outbox = dirs
    workspace.add_todo("bix-ai", "t", actor="x")
    workspace.write_outbox("a.md", "c", actor="x")
    assert sorted(os.listdir(todos)) == ["bix-ai.md"]
    assert sorted(os.listdir(outbox)) == ["a.md"]
