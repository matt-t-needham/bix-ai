#!/usr/bin/env python3
"""Scripted stand-in for `omp --mode rpc`, for offline self-testing omp_spike.py.

Speaks just enough of the protocol (ready, responses, set_host_tools,
host_tool_call round-trips, message/tool events, prompt_result, abort,
open_session) to exercise the client's plumbing. It is NOT omp: passing
against it says nothing about omp's real behaviour.

    python3 spike/omp/omp_spike.py --omp "python3 spike/omp/fake_omp.py" --profile ''
"""

import asyncio
import json
import re
import sys

argv = sys.argv[1:]
BAD_KEY = "invalid" in (__import__("os").environ.get("ANTHROPIC_API_KEY") or "")
BUILTINS = argv[argv.index("--tools") + 1].split(",") if "--tools" in argv else []
SESSION = argv[argv.index("--session-dir") + 1] if "--session-dir" in argv else None
_mem_file = __import__("pathlib").Path(SESSION, "fake-session.json") if SESSION else None
memory: list[str] = json.loads(_mem_file.read_text()) if _mem_file and _mem_file.exists() else []
host_tools: list[str] = []
pending: dict[str, asyncio.Future] = {}
aborted = asyncio.Event()
seq = 0


def out(frame: dict) -> None:
    sys.stdout.write(json.dumps(frame) + "\n")
    sys.stdout.flush()


def nid() -> str:
    global seq
    seq += 1
    return f"h{seq}"


async def host_call(name: str, args: dict) -> str:
    hid = nid()
    fut = asyncio.get_running_loop().create_future()
    pending[hid] = fut
    out({"type": "tool_execution_start", "toolCallId": hid, "toolName": name, "args": args})
    out({"type": "host_tool_call", "id": hid, "toolCallId": hid, "toolName": name, "arguments": args})
    res = await fut
    text = res["result"]["content"][0]["text"]
    out({"type": "tool_execution_end", "toolCallId": hid, "toolName": name,
         "result": res["result"], "isError": bool(res.get("isError"))})
    return text


def say(text: str) -> None:
    for i in range(0, len(text), 16):
        out({"type": "message_update", "message": {},
             "assistantMessageEvent": {"type": "text_delta", "delta": text[i:i + 16]}})


async def run_prompt(rid: str, msg: str) -> None:
    out({"type": "agent_start"})
    status = "completed"
    err = None
    if BAD_KEY:
        status, err = "error", {"message": "401 invalid x-api-key", "retryable": False}
    elif m := re.search(r"list_directory tool on (\S+)", msg):
        root = m.group(1)
        listing = await host_call("list_directory", {"path": root})
        note = next(n for n in listing.splitlines() if n.startswith("note"))
        body = await host_call("read_file", {"path": f"{root}/{note}"})
        word = body.split()[-1].rstrip(".")
        memory.append(f"{word}|{listing}")
        if _mem_file:
            _mem_file.parent.mkdir(parents=True, exist_ok=True)
            _mem_file.write_text(json.dumps(memory))
        say(f"The secret word is {word}; the listing had {len(listing.splitlines())} entries.")
    elif "Without calling any tools" in msg:
        say(f"From earlier: {memory[-1]}" if memory else "I have no record of that.")
    elif "Create the file" in msg:
        for name in ("write", "bash"):
            if name in BUILTINS:
                hid = nid()
                out({"type": "tool_execution_start", "toolCallId": hid, "toolName": name, "args": {}})
                out({"type": "tool_execution_end", "toolCallId": hid, "toolName": name,
                     "result": {"content": [{"type": "text", "text": "approval required; no UI"}]},
                     "isError": True})
        say("Both attempts were refused.")
    elif "essay" in msg:
        for _ in range(200):
            if aborted.is_set():
                status = "aborted"
                break
            say("Bicycles. ")
            await asyncio.sleep(0.05)
    else:
        say("hi")
    out({"type": "agent_end", "yielded": True, "messages": []})
    out({"type": "prompt_result", "id": rid, "agentInvoked": True, "status": status,
         **({"error": err} if err else {}), "sessionSettled": True})


async def main() -> None:
    out({"type": "ready", "protocolVersion": 1, "supportedProtocolVersions": [1, 2],
         "maxFrameBytes": 1048576})
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    while line := await reader.readline():
        f = json.loads(line)
        t, rid = f.get("type"), f.get("id")
        ok = lambda data=None: out({"id": rid, "type": "response", "command": t,  # noqa: E731
                                    "success": True, **({"data": data} if data is not None else {})})
        if t == "host_tool_result":
            pending.pop(rid).set_result(f)
        elif t == "get_state":
            ok({"model": None, "isStreaming": False, "isSettled": True})
        elif t == "get_available_models":
            ok([{"provider": "ollama", "id": "qwen3.5:9b"},
                {"provider": "anthropic", "id": "claude-sonnet-4-5"}])
        elif t == "set_host_tools":
            host_tools[:] = [d["name"] for d in f["tools"]]
            ok({"toolNames": host_tools})
        elif t == "open_session":
            ok({"cancelled": False, "resumed": True, "sessionId": "fake", "sessionFile": "x"})
        elif t == "abort":
            aborted.set()
            ok()
        elif t == "prompt":
            aborted.clear()
            ok()
            asyncio.create_task(run_prompt(rid, f["message"]))
        else:
            out({"id": rid, "type": "response", "command": t, "success": False,
                 "error": f"fake_omp: unsupported {t}"})


if __name__ == "__main__":
    asyncio.run(main())
