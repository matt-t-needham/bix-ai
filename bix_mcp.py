#!/usr/bin/env python3
"""Stdio MCP server for bix-ai. Exposes filesystem + memory tools with a path policy."""
import json
import os
import pathlib
import sys

READ_ROOTS  = [pathlib.Path(p).resolve() for p in os.environ.get("MCP_READ_PATHS",  "/home/matt/apps").split(":") if p]
WRITE_ROOTS = [pathlib.Path(p).resolve() for p in os.environ.get("MCP_WRITE_PATHS", "/app/data").split(":") if p]
DATA_DIR    = pathlib.Path(os.environ.get("DATA_DIR", "/app/data"))
MEM_DIR     = DATA_DIR / "memories"


def _is_under(path: pathlib.Path, roots: list) -> bool:
    try:
        resolved = path.resolve()
        return any(resolved == r or r in resolved.parents for r in roots)
    except Exception:
        return False


def _tool_error(msg: str) -> dict:
    return {"content": [{"type": "text", "text": msg}], "isError": True}


def _tool_ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


TOOL_DEFS = [
    {
        "name": "list_directory",
        "description": "List files and directories at a path. Must be within an allowed read root.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to list."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the text content of a file. Max 200 KB. Must be within an allowed read root.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to the file."}
            },
            "required": ["path"],
        },
    },
    {
        "name": "recall_memories",
        "description": (
            "Search past conversation summaries stored in memory. "
            "Use when the user says 'do you remember', 'recall', or asks about previous sessions."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for in past conversations."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "write_file",
        "description": "Write text content to a file. Must be within an allowed write root.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path":    {"type": "string", "description": "Absolute path to write."},
                "content": {"type": "string", "description": "Text content to write."},
            },
            "required": ["path", "content"],
        },
    },
]


def _execute(name: str, args: dict) -> dict:
    if name == "list_directory":
        raw = args.get("path", "")
        p = pathlib.Path(raw)
        if not _is_under(p, READ_ROOTS):
            return _tool_error(f"Access denied: '{raw}' is outside allowed read roots")
        rp = p.resolve()
        if not rp.exists():
            return _tool_error(f"Path does not exist: {raw}")
        if not rp.is_dir():
            return _tool_error(f"Not a directory: {raw}")
        try:
            entries = sorted(rp.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            lines = [
                f"[dir]  {e.name}/" if e.is_dir() else f"[file] {e.name}  ({e.stat().st_size:,} bytes)"
                for e in entries
            ]
            return _tool_ok("\n".join(lines) if lines else "(empty directory)")
        except PermissionError:
            return _tool_error(f"Permission denied: {raw}")

    elif name == "read_file":
        raw = args.get("path", "")
        p = pathlib.Path(raw)
        if not _is_under(p, READ_ROOTS):
            return _tool_error(f"Access denied: '{raw}' is outside allowed read roots")
        rp = p.resolve()
        if not rp.exists():
            return _tool_error(f"File does not exist: {raw}")
        if not rp.is_file():
            return _tool_error(f"Not a file: {raw}")
        size = rp.stat().st_size
        if size > 200_000:
            return _tool_error(f"File too large ({size:,} bytes). Max 200 KB.")
        try:
            return _tool_ok(rp.read_text(errors="replace"))
        except PermissionError:
            return _tool_error(f"Permission denied: {raw}")
        except Exception as e:
            return _tool_error(f"Error reading file: {e}")

    elif name == "recall_memories":
        query = args.get("query", "").strip()
        if not query:
            return _tool_error("No query provided.")
        MEM_DIR.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []
        for f in sorted(MEM_DIR.glob("memories-*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    entries.extend(data)
            except Exception:
                pass
        if not entries:
            return _tool_ok("No memories stored yet.")
        ql = query.lower().split()
        matches = []
        for m in reversed(entries):
            text = " ".join([m.get("title", ""), m.get("summary", ""), " ".join(m.get("tags", []))]).lower()
            if any(word in text for word in ql):
                matches.append(m)
            if len(matches) >= 3:
                break
        if not matches:
            return _tool_ok(f"No memories found matching: {query}")
        results = []
        for m in matches:
            date    = m.get("date", "")[:10]
            title   = m.get("title", "—")
            summary = m.get("summary", "")
            results.append(f"[{date}] {title}\n{summary}" if summary else f"[{date}] {title}")
        return _tool_ok("\n\n".join(results))

    elif name == "write_file":
        raw     = args.get("path", "")
        content = args.get("content", "")
        p = pathlib.Path(raw)
        if not _is_under(p, WRITE_ROOTS):
            return _tool_error(f"Access denied: '{raw}' is outside allowed write roots")
        rp = p.resolve()
        try:
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(content)
            return _tool_ok(f"Written {len(content)} bytes to {rp}")
        except PermissionError:
            return _tool_error(f"Permission denied: {raw}")
        except Exception as e:
            return _tool_error(f"Error writing file: {e}")

    return _tool_error(f"Unknown tool: {name}")


def _handle(req: dict) -> dict | None:
    method = req.get("method", "")

    if method == "initialize":
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "bix-mcp", "version": "1.0"},
        }

    if method == "notifications/initialized":
        return None  # notification — no response

    if method == "tools/list":
        return {"tools": TOOL_DEFS}

    if method == "tools/call":
        params = req.get("params", {})
        result = _execute(params.get("name", ""), params.get("arguments", {}))
        return result

    # Unknown method — return empty result rather than crashing
    return {}


def main() -> None:
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue

        req_id = req.get("id")
        try:
            result = _handle(req)
        except Exception as e:
            # Protocol errors should not crash the server
            if req_id is not None:
                print(json.dumps({
                    "jsonrpc": "2.0", "id": req_id,
                    "error": {"code": -32603, "message": str(e)},
                }), flush=True)
            continue

        # Only send a response if the request had an id (notifications have none)
        if result is not None and req_id is not None:
            print(json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}), flush=True)


if __name__ == "__main__":
    main()
