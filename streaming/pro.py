import asyncio
import json
import logging
import os
import re
import time

import strategy

from helpers import _write_routing_event, ollama_chat, sse
from identity import identity_system_prompt
from memory import _load_recent_memories, _memory_system_prompt

log = logging.getLogger("router")

_TODO_SYSTEM = (
    "You have a TODOs folder at /home/matt/apps/bix-infra/todos/ accessible via the write_file and read_file tools "
    "(MCP server: bix). Use it to keep per-project plans and task lists — one .md file per project. "
    "Read /home/matt/apps/bix-infra/todos/GUIDE.md for the file structure. "
    "When asked to plan something, read the existing project file first, then write the updated version. "
    "When asked about pending work, read the relevant file and summarise it."
)

_PRO_IDENTITY_PROMPT = identity_system_prompt(
    tool_names=["list_directory", "read_file", "recall_memories", "write_file"],
    aliases={"write_file": "stage_write"},
    doc_topics=["staging", "memory", "modes"],
)


class _SessionGone(Exception):
    """`--resume` pointed at a session the claude CLI no longer knows about."""


def _looks_session_gone(text: str) -> bool:
    return "no conversation found" in text.lower()


def _build_pro_prompt(messages: list) -> str:
    parts = []
    history = messages[:-1]
    if history:
        parts.append("Previous conversation context:")
        for m in history:
            role    = "User" if m["role"] == "user" else "Assistant"
            content = str(m.get("content", ""))
            if len(content) > 3000:
                content = content[:3000] + "\n[… truncated …]"
            parts.append(f"{role}: {content}")
        parts.append("")
    last = messages[-1]
    parts.append(str(last.get("content", "")))
    return "\n".join(parts)


async def _run_claude_once(
    prompt: str, system_prompt: str, model: str,
    resume_session_id: str, state: dict,
):
    """One `claude -p` subprocess run, yielded as SSE events.

    Mutates `state` (input_tokens / output_tokens / ttft_ms / session_id /
    failed). Raises _SessionGone when a --resume attempt fails because the CLI
    no longer has the session — the caller retries fresh with the full-history
    prompt. `state["failed"] = True` means a terminal error/quota event was
    already yielded; the caller must not emit metrics/done.
    """
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--system-prompt", system_prompt,
        "--mcp-config", "/app/mcp.json",
        "--allowedTools", "mcp__bix__*",
        "--model", model,
    ]
    if resume_session_id:
        cmd += ["--resume", resume_session_id]

    def _emit_session(sid):
        # Resumed runs may return the same id or a forked one depending on CLI
        # version — always surface whatever id the run reports; the client
        # stores the latest for the next request.
        if sid and state["session_id"] != sid:
            state["session_id"] = sid
            return sse("pro_session", {"session_id": sid})
        return None

    proc = None
    try:
        proc_env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        proc_env["HOME"] = "/home/matt"
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
            limit=2**22,  # 4MB — default 64KB is too small for large Claude JSON events
        )

        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "system" and event.get("subtype") == "init":
                emitted = _emit_session(event.get("session_id"))
                if emitted:
                    yield emitted

            elif etype == "assistant":
                msg    = event.get("message", {})
                usage  = msg.get("usage", {})
                state["input_tokens"]  += usage.get("input_tokens",  0)
                state["output_tokens"] += usage.get("output_tokens", 0)
                if state["input_tokens"] and state["ttft_ms"] is None:
                    yield sse("input_tokens", {"count": state["input_tokens"]})

                for idx, block in enumerate(msg.get("content", [])):
                    btype = block.get("type")
                    if btype == "text":
                        text = block.get("text", "")
                        if text:
                            if state["ttft_ms"] is None:
                                state["ttft_ms"] = round(
                                    (time.monotonic() - state["start"]) * 1000)
                            yield sse("delta", {"text": text})
                    elif btype == "tool_use":
                        tool_id   = block.get("id", "")
                        tool_name = block.get("name", "")
                        inp       = block.get("input", {})
                        yield sse("tool_start", {"index": idx, "name": tool_name, "id": tool_id})
                        if inp:
                            yield sse("tool_input", {"index": idx, "partial_json": json.dumps(inp)})
                        yield sse("tool_end", {"index": idx})

            elif etype == "user":
                msg = event.get("message", {})
                for block in msg.get("content", []):
                    if block.get("type") != "tool_result":
                        continue
                    tool_use_id = block.get("tool_use_id", "")
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            c.get("text", "") if isinstance(c, dict) else str(c)
                            for c in content
                        )
                    elif not isinstance(content, str):
                        content = str(content)
                    if len(content) > 4000:
                        content = content[:4000] + "\n…(truncated)"
                    yield sse("tool_result", {
                        "tool_use_id": tool_use_id,
                        "content":     content,
                        "is_error":    bool(block.get("is_error", False)),
                    })

            elif etype == "result":
                emitted = _emit_session(event.get("session_id"))
                if emitted:
                    yield emitted
                if event.get("is_error"):
                    raw_err = event.get("error") or event.get("result") or event.get("message")
                    if not raw_err and event.get("errors"):
                        # error_during_execution results carry a list under
                        # "errors" (e.g. "No conversation found with session ID: …")
                        raw_err = "; ".join(str(x) for x in event["errors"])
                    if isinstance(raw_err, dict):
                        err = raw_err.get("message") or raw_err.get("error") or str(raw_err)
                    else:
                        err = str(raw_err) if raw_err else "claude subprocess returned an error"
                    if (resume_session_id and state["output_tokens"] == 0
                            and _looks_session_gone(err)):
                        raise _SessionGone(err)
                    log.error("pro result is_error event=%s", event)
                    lower = err.lower()
                    state["failed"] = True
                    if "quota" in lower or "rate limit" in lower or "usage limit" in lower or "529" in lower:
                        yield sse("quota_exceeded", {})
                    else:
                        yield sse("error", {"message": err})
                    return

        stderr_data = await proc.stderr.read()
        await proc.wait()
        err_text = stderr_data.decode("utf-8", errors="replace")
        lower    = err_text.lower()

        if proc.returncode != 0:
            if (resume_session_id and state["output_tokens"] == 0
                    and _looks_session_gone(err_text)):
                raise _SessionGone(err_text.strip())
            log.error("pro subprocess exit=%d stderr=%s", proc.returncode, err_text[:500])
            state["failed"] = True
            if "quota" in lower or "rate limit" in lower or "usage limit" in lower or "529" in lower:
                yield sse("quota_exceeded", {})
            else:
                yield sse("error", {"message": err_text or f"claude exited {proc.returncode}"})
            return

        if state["input_tokens"] == 0 and state["output_tokens"] == 0:
            if resume_session_id and _looks_session_gone(err_text):
                raise _SessionGone(err_text.strip())
            log.error("pro subprocess produced no output exit=0 stderr=%s", err_text[:500])
            state["failed"] = True
            if "quota" in lower or "rate limit" in lower or "usage limit" in lower or "529" in lower:
                yield sse("quota_exceeded", {})
            else:
                yield sse("error", {"message": err_text.strip() or "claude subprocess produced no output"})
            return

    finally:
        if proc is not None and proc.returncode is None:
            log.info("pro subprocess cleanup pid=%d (client disconnect or generator exit)", proc.pid)
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except (asyncio.TimeoutError, Exception):
                try:
                    proc.kill()
                    await proc.wait()
                except (ProcessLookupError, Exception):
                    pass


async def _stream_pro(
    messages: list, model: str, max_tokens: int,
    mode: str = "pro", session_id: str = "",
):
    recent     = _load_recent_memories(3)
    sys_prompt = _memory_system_prompt(recent)
    log.info("pro memory loaded count=%d injected=%s", len(recent), bool(sys_prompt))

    stats         = {"summarised": 0, "skipped": 0, "failed": 0, "spilled": 0}
    preprocess_ms = 0

    req_body = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if sys_prompt:
        req_body["system"] = sys_prompt

    if strategy.has_oversized_blocks(req_body):
        yield sse("status", {"stage": "summarising", "message": "Preparing context…"})
    else:
        yield sse("status", {"stage": "checking", "message": "Checking…"})

    t0 = time.monotonic()
    try:
        req_body, stats = await strategy.preprocess(req_body, ollama_chat)
        log.info("pro preprocess spilled=%d skipped=%d failed=%d",
                 stats["spilled"], stats["skipped"], stats["failed"])
    except Exception as e:
        log.warning("pro preprocess error: %s", e)
    preprocess_ms = round((time.monotonic() - t0) * 1000)

    yield sse("preprocess", {
        "summarised":    stats["summarised"],
        "spilled":       stats["spilled"],
        "skipped":       stats["skipped"],
        "failed":        stats["failed"],
        "preprocess_ms": preprocess_ms,
    })
    yield sse("status", {"stage": "streaming", "message": "Streaming via Claude Pro…"})

    sys_parts = [_PRO_IDENTITY_PROMPT]
    if req_body.get("system"):
        sys_parts.append(req_body["system"])
    sys_parts.append(_TODO_SYSTEM)
    system_prompt = "\n\n".join(sys_parts)

    # Resuming a CLI session: the session already holds the full history
    # (including tool turns the text-only prompt replay would lose), so only
    # the newest user message is sent. Fresh runs replay text history.
    pro_messages = list(req_body["messages"])
    prompt_full  = _build_pro_prompt(pro_messages)
    resume_sid   = (session_id or "").strip()
    # CLI session ids are UUIDs; anything else (especially values starting
    # with "-") must never reach the subprocess argv.
    if resume_sid and not re.fullmatch(r"[0-9a-fA-F-]{8,64}", resume_sid):
        log.warning("pro session_id rejected (bad shape): %r", resume_sid[:80])
        resume_sid = ""
    if resume_sid and pro_messages:
        prompt = str(pro_messages[-1].get("content", ""))
    else:
        prompt = prompt_full
    log.info("pro run resume=%s msgs=%d", bool(resume_sid), len(pro_messages))

    state = {
        "input_tokens": 0, "output_tokens": 0, "ttft_ms": None,
        "session_id": resume_sid or None, "failed": False,
        "start": time.monotonic(),
    }

    try:
        try:
            async for ev in _run_claude_once(prompt, system_prompt, model,
                                             resume_sid, state):
                yield ev
        except _SessionGone as e:
            log.warning("pro resume failed (session gone), retrying fresh: %s", e)
            yield sse("status", {"stage": "streaming",
                                 "message": "Session expired — replaying history…"})
            state.update(input_tokens=0, output_tokens=0, ttft_ms=None,
                         session_id=None, failed=False)
            async for ev in _run_claude_once(prompt_full, system_prompt, model,
                                             "", state):
                yield ev
    except Exception as e:
        log.error("pro stream error: %s", e)
        yield sse("error", {"message": str(e)})
        return

    if state["failed"]:
        return

    input_tokens  = state["input_tokens"]
    output_tokens = state["output_tokens"]
    ttft_ms       = state["ttft_ms"]
    elapsed = time.monotonic() - state["start"]
    tps     = output_tokens / elapsed if elapsed > 0 else 0
    yield sse("metrics", {
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "elapsed_ms":    round(elapsed * 1000),
        "ttft_ms":       ttft_ms or 0,
        "preprocess_ms": preprocess_ms,
        "tps":           round(tps, 1),
        "summarised":    stats["summarised"],
        "spilled":       stats["spilled"],
        "skipped":       stats["skipped"],
        "failed":        stats["failed"],
    })
    log.info("pro_done model=%s in=%d out=%d ttft_ms=%d elapsed_ms=%d resumed=%s",
             model, input_tokens, output_tokens, ttft_ms or 0,
             round(elapsed * 1000), bool(resume_sid))
    await _write_routing_event(mode, model,
                               reason=f"forced:{mode}" + (" · resumed" if resume_sid else ""),
                               summarised=stats["summarised"],
                               preprocess_ms=preprocess_ms, input_tokens=input_tokens,
                               output_tokens=output_tokens, ttft_ms=ttft_ms or 0,
                               elapsed_ms=round(elapsed * 1000))
    yield sse("done", {})
