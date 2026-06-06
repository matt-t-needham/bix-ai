"""Tests for bix_mcp.py — verifies the stdio MCP server protocol and path policy."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MCP_SCRIPT = str(Path(__file__).parent.parent / "bix_mcp.py")


def _spawn(read_paths: str | None = None, fs_root: str | None = None, data_dir: str | None = None):
    """Start the MCP server as a subprocess with optional path overrides."""
    env = {**os.environ}
    if read_paths is not None: env["MCP_READ_PATHS"] = read_paths
    if fs_root    is not None: env["FS_ROOT"]        = fs_root
    if data_dir   is not None: env["DATA_DIR"]       = data_dir
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
        assert names == {"list_directory", "read_file", "recall_memories", "write_file"}
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


# ── Test 6: write_file — stages, does not write live ─────────────────────────

def test_write_file_stages():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as data:
        target = Path(root) / "out.txt"
        proc = _spawn(read_paths=root, fs_root=root, data_dir=data)
        try:
            _init(proc)
            resp = _send(proc, {
                "jsonrpc": "2.0", "id": 6, "method": "tools/call",
                "params": {"name": "write_file", "arguments": {"path": str(target), "content": "written!"}},
            })
            assert not resp["result"].get("isError")
            assert "Staged" in resp["result"]["content"][0]["text"]
            assert not target.exists()                                   # NOT written live
            assert list((Path(data) / "staging").glob("*.json"))         # record persisted
        finally:
            proc.terminate()


# ── Test 7: write_file — outside FS_ROOT is refused ──────────────────────────

def test_write_file_outside_root_denied():
    with tempfile.TemporaryDirectory() as root, \
         tempfile.TemporaryDirectory() as outside, \
         tempfile.TemporaryDirectory() as data:
        proc = _spawn(fs_root=root, data_dir=data)
        try:
            _init(proc)
            resp = _send(proc, {
                "jsonrpc": "2.0", "id": 7, "method": "tools/call",
                "params": {
                    "name": "write_file",
                    "arguments": {"path": str(Path(outside) / "bad.txt"), "content": "oops"},
                },
            })
            assert resp["result"].get("isError") is True
            assert "outside the allowed root" in resp["result"]["content"][0]["text"]
            assert not (Path(outside) / "bad.txt").exists()
        finally:
            proc.terminate()


# ── Test 8: write_file — guardrail path (shell script) is refused ────────────

def test_write_file_guardrail_denied():
    with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as data:
        target = Path(root) / "deploy.sh"
        proc = _spawn(fs_root=root, data_dir=data)
        try:
            _init(proc)
            resp = _send(proc, {
                "jsonrpc": "2.0", "id": 8, "method": "tools/call",
                "params": {"name": "write_file", "arguments": {"path": str(target), "content": "rm -rf /"}},
            })
            assert resp["result"].get("isError") is True
            assert "guardrail" in resp["result"]["content"][0]["text"]
            assert not target.exists()
        finally:
            proc.terminate()
