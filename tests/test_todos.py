"""Unit tests for todos.py — pure logic, stdlib-only."""
import pytest

import config
import todos


@pytest.fixture
def todir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TODOS_DIR", tmp_path)
    (tmp_path / "ALL.md").write_text("# compiled\n- bix-ai thing\n- infra thing\n")
    (tmp_path / "bix-ai.md").write_text("# bix-ai\n- one\n")
    (tmp_path / "infra.md").write_text("# infra\n- two\n")
    (tmp_path / "GUIDE.md").write_text("how to use todos")
    return tmp_path


def test_list_projects_excludes_all_and_guide(todir):
    assert todos.list_projects() == ["bix-ai", "infra"]


def test_read_todos_default_returns_compiled(todir):
    out = todos.read_todos()
    assert "# compiled" in out and "infra thing" in out


def test_read_todos_project(todir):
    out = todos.read_todos("bix-ai")
    assert out.strip().startswith("# bix-ai")


def test_read_todos_unknown_project_lists_available(todir):
    out = todos.read_todos("nope")
    assert "No TODO file" in out and "bix-ai" in out and "infra" in out


def test_read_todos_project_name_sanitised(todir):
    # Traversal characters are stripped, so it can't escape TODOS_DIR.
    out = todos.read_todos("../../etc/passwd")
    assert "No TODO file" in out


def test_read_todos_fallback_concatenates_when_no_all(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TODOS_DIR", tmp_path)
    (tmp_path / "bix-ai.md").write_text("- a\n")
    (tmp_path / "demucs.md").write_text("- b\n")
    out = todos.read_todos()
    assert "## bix-ai" in out and "## demucs" in out


def test_read_todos_missing_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TODOS_DIR", tmp_path / "nope")
    assert "No TODOs directory" in todos.read_todos()
