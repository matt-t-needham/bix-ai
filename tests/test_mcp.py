"""Tests for bix_mcp.py — verifies the stdio MCP server protocol and path policy."""
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

MCP_SCRIPT = str(Path(__file__).parent.parent / "bix_mcp.py")


def _spawn(read_paths: str | None = None, fs_root: str | None = None, data_dir: str | None = None,
           extra_env: dict | None = None):
    """Start the MCP server as a subprocess with optional path overrides."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("BIX_AGENT_")}
    if read_paths is not None: env["MCP_READ_PATHS"] = read_paths
    if fs_root    is not None: env["FS_ROOT"]        = fs_root
    if data_dir   is not None: env["DATA_DIR"]       = data_dir
    env.update(extra_env or {})
    return subprocess.Popen(
        [sys.executable, MCP_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def _send(proc, req: dict) -> dict:
    """Write a JSON-RPC request and read back the response."""
    proc.stdin.write(json.dumps(req) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    return json.loads(line)


def _notify(proc, method: str) -> None:
    """Send a notification (no id, no response expected)."""
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
    proc.stdin.flush()


def _init(proc):
    """Complete the MCP handshake."""
    _send(proc, {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {}})
    _notify(proc, "notifications/initialized")


# ── Test 1: initialize ────────────────────────────────────────────────────────

def test_initialize():
    proc = _spawn()
    try:
        resp = _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "bix-mcp"
    finally:
        proc.terminate()


# ── Test 2: tools/list ────────────────────────────────────────────────────────

def test_tools_list():
    proc = _spawn()
    try:
        _init(proc)
        resp = _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = {t["name"] for t in resp["result"]["tools"]}
        assert names == {"list_directory", "read_file", "recall_memories",
                         "add_todo", "add_plan", "complete_todo", "write_outbox"}
    finally:
        proc.terminate()


# ── Test 3: read_file — valid path ────────────────────────────────────────────

def test_read_file_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "hello.txt"
        test_file.write_text("hello world")
        proc = _spawn(read_paths=tmpdir)
        try:
            _init(proc)
            resp = _send(proc, {
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "read_file", "arguments": {"path": str(test_file)}},
            })
            assert resp["result"]["content"][0]["text"] == "hello world"
            assert not resp["result"].get("isError")
        finally:
            proc.terminate()


# ── Test 4: read_file — path outside read roots ───────────────────────────────

def test_read_file_denied():
    with tempfile.TemporaryDirectory() as tmpdir:
        proc = _spawn(read_paths="/nonexistent_allowed_path")
        try:
            _init(proc)
            outside_file = Path(tmpdir) / "secret.txt"
            outside_file.write_text("secret")
            resp = _send(proc, {
                "jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "read_file", "arguments": {"path": str(outside_file)}},
            })
            assert resp["result"].get("isError") is True
            assert "Access denied" in resp["result"]["content"][0]["text"]
        finally:
            proc.terminate()


# ── Test 5: list_directory — valid path ──────────────────────────────────────

def test_list_directory_valid():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "a.txt").write_text("a")
        (Path(tmpdir) / "subdir").mkdir()
        proc = _spawn(read_paths=tmpdir)
        try:
            _init(proc)
            resp = _send(proc, {
                "jsonrpc": "2.0", "id": 5, "method": "tools/call",
                "params": {"name": "list_directory", "arguments": {"path": tmpdir}},
            })
            text = resp["result"]["content"][0]["text"]
            assert "a.txt" in text
            assert "subdir" in text
            assert not resp["result"].get("isError")
        finally:
            proc.terminate()


# ── Workspace writes ──────────────────────────────────────────────────────────

def test_write_tool_schemas_match_the_tool_table():
    """bix_mcp hand-writes its definitions (it can't import tools.py); they
    must stay identical to TOOL_TABLE's so pro mode offers the same tools."""
    import bix_mcp
    import tools
    mcp = {t["name"]: t for t in bix_mcp.TOOL_DEFS if t["name"] in tools.WRITE_TOOLS}
    table = {t["name"]: t for t in tools.TOOL_TABLE if t["name"] in tools.WRITE_TOOLS}
    assert set(mcp) == set(table) == tools.WRITE_TOOLS
    for name, t in table.items():
        assert mcp[name]["description"] == t["description"], name
        assert mcp[name]["inputSchema"] == t["input_schema"], name


def test_mcp_write_tools_tag_with_bix_actor(tmp_path):
    todos, outbox = tmp_path / "todos", tmp_path / "outbox"
    todos.mkdir()
    outbox.mkdir()
    env_extra = {"TODOS_DIR": str(todos), "OUTBOX_DIR": str(outbox), "BIX_ACTOR": "pro:claude-opus"}
    os.environ.update(env_extra)
    try:
        proc = _spawn()
    finally:
        for k in env_extra:
            del os.environ[k]
    try:
        _init(proc)
        resp = _send(proc, {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {
            "name": "add_todo", "arguments": {"project": "infra", "task": "From pro"}}})
        assert "isError" not in resp["result"], resp
        assert "**Added by:** ai.Bix (pro:claude-opus)" in (todos / "infra.md").read_text()

        resp = _send(proc, {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {
            "name": "write_outbox", "arguments": {"name": "../escape.md", "content": "x"}}})
        assert resp["result"]["isError"] is True
        assert resp["result"]["content"][0]["text"].startswith("Not written:")
        assert not (tmp_path / "escape.md").exists()
    finally:
        proc.terminate()


# ── Agent tools (Pro mode, agent chats) ───────────────────────────────────────

def test_agent_tools_are_listed_and_forwarded_to_ai_router():
    """With a run token, the agent tools come from ai-router's internal route
    and every call goes back there."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append(("GET", self.path, self.headers.get("X-Bix-Agent-Token"), None))
            self._reply({"tools": [{"name": "run_command", "description": "d", "inputSchema": {"type": "object"}}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(("POST", self.path, self.headers.get("X-Bix-Agent-Token"), body))
            self._reply({"text": "Not run: denied by matt.", "is_error": True})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    proc = _spawn(extra_env={"BIX_AGENT_TOKEN": "tok", "BIX_AGENT_URL": f"http://127.0.0.1:{server.server_port}",
                             "BIX_AGENT_TIMEOUT": "10"})
    try:
        _init(proc)
        names = {t["name"] for t in _send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]}
        assert "run_command" in names and "read_file" in names
        resp = _send(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                            "params": {"name": "run_command", "arguments": {"script": "make"}}})
        assert resp["result"]["isError"] is True
        assert resp["result"]["content"][0]["text"] == "Not run: denied by matt."
        assert calls[-1] == ("POST", "/internal/agent/tool", "tok", {"name": "run_command", "arguments": {"script": "make"}})
    finally:
        proc.terminate()
        server.shutdown()
