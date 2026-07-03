import asyncio
import json
import logging
import os
import time

import strategy

from helpers import _write_routing_event, ollama_chat, sse
from memory import _load_recent_memories, _memory_system_prompt

log = logging.getLogger("router")

_TODO_SYSTEM = (
    "You have a TODOs folder at /home/matt/apps/bix-infra/todos/ accessible via the write_file and read_file tools "
    "(MCP server: bix). Use it to keep per-project plans and task lists — one .md file per project. "
    "Read /home/matt/apps/bix-infra/todos/GUIDE.md for the file structure. "
    "When asked to plan something, read the existing project file first, then write the updated version. "
    "When asked about pending work, read the relevant file and summarise it."
)


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


async def _stream_pro(messages: list, model: str, max_tokens: int, mode: str = "pro"):
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

    sys_parts = []
    if req_body.get("system"):
        sys_parts.append(req_body["system"])
    sys_parts.append(_TODO_SYSTEM)
    system_prompt = "\n\n".join(sys_parts)

    prompt = _build_pro_prompt(list(req_body["messages"]))
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--system-prompt", system_prompt,
        "--mcp-config", "/app/mcp.json",
        "--allowedTools", "mcp__bix__*",
        "--model", model,
    ]

    start         = time.monotonic()
    ttft_ms       = None
    input_tokens  = 0
    output_tokens = 0

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

            if etype == "assistant":
                msg    = event.get("message", {})
                usage  = msg.get("usage", {})
                input_tokens  += usage.get("input_tokens",  0)
                output_tokens += usage.get("output_tokens", 0)
                if input_tokens and ttft_ms is None:
                    yield sse("input_tokens", {"count": input_tokens})

                for idx, block in enumerate(msg.get("content", [])):
                    btype = block.get("type")
                    if btype == "text":
                        text = block.get("text", "")
                        if text:
                            if ttft_ms is None:
                                ttft_ms = round((time.monotonic() - start) * 1000)
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
                if event.get("is_error"):
                    raw_err = event.get("error") or event.get("result") or event.get("message") or {}
                    if isinstance(raw_err, dict):
                        err = raw_err.get("message") or raw_err.get("error") or str(raw_err)
                    else:
                        err = str(raw_err) if raw_err else "claude subprocess returned an error"
                    log.error("pro result is_error event=%s", event)
                    lower = err.lower()
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
            log.error("pro subprocess exit=%d stderr=%s", proc.returncode, err_text[:500])
            if "quota" in lower or "rate limit" in lower or "usage limit" in lower or "529" in lower:
                yield sse("quota_exceeded", {})
            else:
                yield sse("error", {"message": err_text or f"claude exited {proc.returncode}"})
            return

        if input_tokens == 0 and output_tokens == 0:
            log.error("pro subprocess produced no output exit=0 stderr=%s", err_text[:500])
            if "quota" in lower or "rate limit" in lower or "usage limit" in lower or "529" in lower:
                yield sse("quota_exceeded", {})
            else:
                yield sse("error", {"message": err_text.strip() or "claude subprocess produced no output"})
            return

    except Exception as e:
        log.error("pro stream error: %s", e)
        yield sse("error", {"message": str(e)})
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

    elapsed = time.monotonic() - start
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
    log.info("pro_done model=%s in=%d out=%d ttft_ms=%d elapsed_ms=%d",
             model, input_tokens, output_tokens, ttft_ms or 0, round(elapsed * 1000))
    await _write_routing_event(mode, model, reason=f"forced:{mode}",
                               summarised=stats["summarised"],
                               preprocess_ms=preprocess_ms, input_tokens=input_tokens,
                               output_tokens=output_tokens, ttft_ms=ttft_ms or 0,
                               elapsed_ms=round(elapsed * 1000))
    yield sse("done", {})
