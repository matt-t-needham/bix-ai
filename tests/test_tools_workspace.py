"""The workspace tools as the model sees them (tools.py wiring)."""
import asyncio

import pytest

import tools


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    todos, outbox = tmp_path / "todos", tmp_path / "outbox"
    todos.mkdir()
    outbox.mkdir()
    monkeypatch.setenv("TODOS_DIR", str(todos))
    monkeypatch.setenv("OUTBOX_DIR", str(outbox))
    return todos, outbox


def run(name, tool_input, actor="local:gemma4:26b"):
    return asyncio.run(tools._execute_tool(name, tools.with_actor(name, tool_input, actor)))


def test_write_tools_are_offered_to_every_provider():
    for wire in (tools.FS_TOOLS, [t["function"] for t in tools.OLLAMA_TOOLS]):
        assert tools.WRITE_TOOLS <= {t["name"] for t in wire}


def test_with_actor_only_touches_write_tools_and_overrides_the_model():
    assert tools.with_actor("read_file", {"path": "/x"}, "a") == {"path": "/x"}
    assert tools.with_actor("add_todo", {"task": "t", "_actor": "forged"}, "real")["_actor"] == "real"


def test_add_todo_tags_the_actor_the_adapter_injected(dirs):
    todos, _ = dirs
    out = run("add_todo", {"project": "infra", "task": "T", "_actor": "forged"})
    assert out.startswith("Added to infra.md")
    text = (todos / "infra.md").read_text()
    assert "ai.Bix (local:gemma4:26b)" in text and "forged" not in text


def test_refusals_come_back_as_not_written(dirs):
    out = run("write_outbox", {"name": "x.sh", "content": "rm -rf /"})
    assert out.startswith("Not written:")
    assert not any(dirs[1].iterdir())


def test_os_errors_are_reported_not_swallowed(dirs, monkeypatch, caplog):
    def boom(*a, **k):
        raise PermissionError("read-only file system")
    monkeypatch.setattr(tools.workspace, "write_outbox", boom)
    out = run("write_outbox", {"name": "a.md", "content": "c"})
    assert out == "Not written — the write failed: read-only file system"
    assert "workspace write failed" in caplog.text


def test_every_write_tool_runs_end_to_end(dirs):
    todos, outbox = dirs
    assert run("add_todo", {"project": "beatshare", "task": "Swing"}).startswith("Added")
    assert run("add_plan", {"project": "beatshare", "title": "P", "goal": "G",
                            "steps": ["a", "b"], "effort": "small"}).startswith("Added plan")
    assert run("complete_todo", {"project": "beatshare", "match": "swing"}).startswith("Moved to Done")
    assert run("write_outbox", {"name": "d.md", "content": "hi"}).startswith("Wrote")
    assert (outbox / "d.md").exists() and (todos / "beatshare.md").exists()


def test_pro_mode_passes_its_model_to_the_mcp_server():
    import json
    from streaming import pro
    cfg = json.loads(pro._mcp_config("claude-opus-5-5"))
    server = cfg["mcpServers"]["bix"]
    assert server["env"]["BIX_ACTOR"] == "pro:claude-opus-5-5"
    assert server["args"] == ["/app/bix_mcp.py"]            # the rest of mcp.json is kept
    assert server["env"]["FS_ROOT"] == "/home/matt/apps"
