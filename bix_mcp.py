#!/usr/bin/env python3
"""Stdio MCP server for bix-ai (mode="pro"). Exposes the read-only filesystem
+ memory tools with a path policy, plus the workspace writes (todos/ and
outbox/, via workspace.py). Stdlib-only — it runs inside the `claude` CLI
subprocess, not the FastAPI app. BIX_ACTOR (set per run by streaming/pro.py)
names the model in each write's tag; the write tools' schemas mirror
tools.TOOL_TABLE (tests/test_mcp.py keeps them in step).

In an agent-mode chat, pro.py also sets BIX_AGENT_TOKEN: the agent tools are
then listed from, and every call to one forwarded to, ai-router's
/internal/agent/* routes."""
import json
import os
import pathlib
import sys
import urllib.request

import workspace
from fs_core import is_denied_path, list_directory, read_file

READ_ROOTS  = [pathlib.Path(p).resolve() for p in os.environ.get("MCP_READ_PATHS",  "/home/matt/apps").split(":") if p]
DATA_DIR    = pathlib.Path(os.environ.get("DATA_DIR", "/app/data"))
MEM_DIR     = DATA_DIR / "memories"
# Set together by streaming/pro.py, for agent-mode chats only.
AGENT_TOKEN   = os.environ.get("BIX_AGENT_TOKEN", "")
AGENT_URL     = os.environ.get("BIX_AGENT_URL", "")
AGENT_TIMEOUT = float(os.environ.get("BIX_AGENT_TIMEOUT", "0"))
_agent_tools: list[dict] = []


def _agent_request(path: str, payload: dict | None = None) -> dict:
    """This server handles one request at a time, so a call queued behind an
    open approval card waits for it; the timeout covers one card plus one command."""
    req = urllib.request.Request(
        AGENT_URL + path, data=json.dumps(payload).encode() if payload is not None else None,
        headers={"X-Bix-Agent-Token": AGENT_TOKEN, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=AGENT_TIMEOUT) as resp:
        return json.loads(resp.read())


def _list_agent_tools() -> list[dict]:
    """Fetched once from ai-router; a failure is a JSON-RPC error, not a quiet
    tool list with the agent tools missing."""
    if AGENT_TOKEN and not _agent_tools:
        _agent_tools.extend(_agent_request("/internal/agent/tools")["tools"])
    return _agent_tools


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
        "name": "add_todo",
        "description": "Add a pending task to a project's todo list (bix-infra/todos/<project>.md). Only when the user asks you to note, track or remember a task. Appends to the Pending section; never edits existing entries.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Todo file name without .md, e.g. 'bix-ai', 'infra', 'rageagain_channel'. A new name creates that project's file."
                },
                "task": {
                    "type": "string",
                    "description": "One-line task title (max 200 chars)."
                },
                "why": {
                    "type": "string",
                    "description": "Motivation or context (optional)."
                },
                "notes": {
                    "type": "string",
                    "description": "Constraints, gotchas, prior decisions (optional)."
                }
            },
            "required": [
                "project",
                "task"
            ]
        }
    },
    {
        "name": "add_plan",
        "description": "Add an implementation plan to a project's todo file (its Plans section). Only when the user asks you to write a plan down. Steps should be specific enough to carry out in a later session without re-explanation.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Todo file name without .md, e.g. 'bix-ai', 'infra', 'rageagain_channel'. A new name creates that project's file."
                },
                "title": {
                    "type": "string",
                    "description": "Plan title."
                },
                "goal": {
                    "type": "string",
                    "description": "One-line summary of what it achieves."
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "Ordered steps, one string each (max 30)."
                },
                "effort": {
                    "type": "string",
                    "enum": [
                        "small",
                        "medium",
                        "large"
                    ],
                    "description": "small = under an hour, medium = half a day, large = several sessions."
                }
            },
            "required": [
                "project",
                "title",
                "goal",
                "steps"
            ]
        }
    },
    {
        "name": "complete_todo",
        "description": "Mark one pending task done: moves it from Pending to Done with today's date. Only when the user says the work is finished. `match` is a distinctive piece of the task's title; if it matches more than one open task, nothing changes and you get the candidates back.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": "Todo file name without .md, e.g. 'bix-ai', 'infra', 'rageagain_channel'. A new name creates that project's file."
                },
                "match": {
                    "type": "string",
                    "description": "Text that appears in exactly one open task's title."
                }
            },
            "required": [
                "project",
                "match"
            ]
        }
    },
    {
        "name": "write_outbox",
        "description": "Save a draft (note, write-up, blog draft, plan) for Matt to pick up, as bix-infra/outbox/<name>. Creates the file or overwrites one of the same name. Only when the user asks for something to be saved or written out.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Plain file name ending in .md or .txt, e.g. 'blog-draft.md'."
                },
                "content": {
                    "type": "string",
                    "description": "Full file content (max 100 KB)."
                }
            },
            "required": [
                "name",
                "content"
            ]
        }
    },
]

_WORKSPACE = {
    "add_todo": lambda a, actor: workspace.add_todo(
        a.get("project", ""), a.get("task", ""), a.get("why", ""), a.get("notes", ""), actor=actor),
    "add_plan": lambda a, actor: workspace.add_plan(
        a.get("project", ""), a.get("title", ""), a.get("goal", ""), a.get("steps", []),
        a.get("effort", ""), actor=actor),
    "complete_todo": lambda a, actor: workspace.complete_todo(
        a.get("project", ""), a.get("match", ""), actor=actor),
    "write_outbox": lambda a, actor: workspace.write_outbox(
        a.get("name", ""), a.get("content", ""), actor=actor),
}


def _execute(name: str, args: dict) -> dict:
    if name == "list_directory":
        raw = args.get("path", "")
        p = pathlib.Path(raw)
        if not _is_under(p, READ_ROOTS):
            return _tool_error(f"Access denied: '{raw}' is outside allowed read roots")
        rp = p.resolve()
        if is_denied_path(rp):
            return _tool_error(f"Access denied: '{raw}' is a protected path")
        if not rp.exists():
            return _tool_error(f"Path does not exist: {raw}")
        if not rp.is_dir():
            return _tool_error(f"Not a directory: {raw}")
        return _tool_ok(list_directory(rp))

    elif name == "read_file":
        raw = args.get("path", "")
        p = pathlib.Path(raw)
        if not _is_under(p, READ_ROOTS):
            return _tool_error(f"Access denied: '{raw}' is outside allowed read roots")
        rp = p.resolve()
        if is_denied_path(rp):
            return _tool_error(f"Access denied: '{raw}' is a protected file")
        if not rp.exists():
            return _tool_error(f"File does not exist: {raw}")
        if not rp.is_file():
            return _tool_error(f"Not a file: {raw}")
        return _tool_ok(read_file(rp))

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

    elif name in _WORKSPACE:
        try:
            return _tool_ok(_WORKSPACE[name](args, os.environ.get("BIX_ACTOR", "pro")))
        except workspace.WorkspaceError as e:
            return _tool_error(f"Not written: {e}")
        except OSError as e:
            return _tool_error(f"Not written — the write failed: {e}")

    if name in {t["name"] for t in _list_agent_tools()}:
        try:
            reply = _agent_request("/internal/agent/tool", {"name": name, "arguments": args})
        except (OSError, ValueError) as e:   # URLError is an OSError
            return _tool_error(f"Not run — ai.Bix's agent route failed: {e}")
        if reply["is_error"]:
            return _tool_error(reply["text"])
        return _tool_ok(reply["text"])

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
        return {"tools": TOOL_DEFS + _list_agent_tools()}

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
