import json
import logging
import time

import httpx
import strategy

from config import ANTHROPIC_API_KEY, ANTHROPIC_URL
from helpers import _agg, _write_routing_event, ollama_chat, sse
from memory import _load_recent_memories, _memory_system_prompt
from tools import FS_TOOLS, _execute_tool

log = logging.getLogger("router")


async def _stream_claude(
    messages: list, model: str, max_tokens: int,
    skip_preprocess: bool, mode: str = "api",
):
    _agg["requests"] += 1
    req_body   = {"model": model, "max_tokens": max_tokens, "messages": messages}
    recent     = _load_recent_memories(3)
    sys_prompt = _memory_system_prompt(recent)
    if sys_prompt:
        req_body["system"] = sys_prompt
    log.info("memory loaded count=%d injected=%s", len(recent), bool(sys_prompt))
    stats         = {"summarised": 0, "skipped": 0, "failed": 0}
    preprocess_ms = 0

    if skip_preprocess:
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})
    else:
        if strategy.has_oversized_blocks(req_body):
            yield sse("status", {"stage": "summarising", "message": "Summarising via Ollama…"})
        else:
            yield sse("status", {"stage": "checking", "message": "Checking…"})
        t0 = time.monotonic()
        try:
            req_body, stats = await strategy.preprocess(req_body, ollama_chat)
            log.info("preprocess summarised=%d skipped=%d failed=%d",
                     stats["summarised"], stats["skipped"], stats["failed"])
        except Exception as e:
            log.warning("preprocess error: %s", e)
        preprocess_ms = round((time.monotonic() - t0) * 1000)
        _agg["summarised"]    += stats["summarised"]
        _agg["checked"]       += stats["summarised"] + stats["skipped"]
        _agg["preprocess_ms"] += preprocess_ms
        _agg["failed"]        += stats["failed"]
        yield sse("preprocess", {
            "summarised":    stats["summarised"],
            "skipped":       stats["skipped"],
            "failed":        stats["failed"],
            "preprocess_ms": preprocess_ms,
        })
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})

    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }

    start              = time.monotonic()
    ttft_ms            = None
    total_input_tokens = 0
    output_tokens      = 0
    current_messages   = list(req_body["messages"])

    for _turn in range(10):
        log.info("tool_turn turn=%d msgs=%d", _turn, len(current_messages))
        api_body = {
            **req_body,
            "messages": current_messages,
            "tools":    FS_TOOLS,
            "stream":   True,
        }
        content_blocks: dict = {}
        stop_reason = None

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", ANTHROPIC_URL, json=api_body, headers=headers) as r:
                    event_type = None
                    async for line in r.aiter_lines():
                        line = line.strip()
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            raw = line[5:].strip()
                            if not raw:
                                continue
                            try:
                                data = json.loads(raw)
                            except json.JSONDecodeError:
                                continue

                            if event_type == "message_start":
                                tok = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                                total_input_tokens += tok
                                log.info("input_tokens turn=%d count=%d total=%d",
                                         _turn, tok, total_input_tokens)
                                if _turn == 0:
                                    yield sse("input_tokens", {"count": tok})

                            elif event_type == "content_block_start":
                                idx   = data.get("index", 0)
                                block = data.get("content_block", {})
                                content_blocks[idx] = {
                                    "type": block.get("type"), "name": block.get("name", ""),
                                    "id":   block.get("id",   ""), "text": "", "input_json": "",
                                }
                                if block.get("type") == "tool_use":
                                    yield sse("tool_start", {
                                        "index": idx,
                                        "name":  block.get("name", ""),
                                        "id":    block.get("id",   ""),
                                    })

                            elif event_type == "content_block_delta":
                                idx   = data.get("index", 0)
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta["text"]
                                    if ttft_ms is None:
                                        ttft_ms = round((time.monotonic() - start) * 1000)
                                    if idx in content_blocks:
                                        content_blocks[idx]["text"] += text
                                    yield sse("delta", {"text": text})
                                elif delta.get("type") == "input_json_delta":
                                    partial = delta.get("partial_json", "")
                                    if idx in content_blocks:
                                        content_blocks[idx]["input_json"] += partial
                                    yield sse("tool_input", {"index": idx, "partial_json": partial})

                            elif event_type == "content_block_stop":
                                idx = data.get("index", 0)
                                if content_blocks.get(idx, {}).get("type") == "tool_use":
                                    yield sse("tool_end", {"index": idx})

                            elif event_type == "message_delta":
                                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                                stop_reason   = data.get("delta", {}).get("stop_reason")
                                log.info("output_tokens turn=%d count=%d stop_reason=%s",
                                         _turn, output_tokens, stop_reason)

        except Exception as e:
            log.error("upstream error: %s", e)
            yield sse("error", {"message": str(e)})
            return

        if stop_reason != "tool_use":
            elapsed = time.monotonic() - start
            tps     = output_tokens / elapsed if elapsed > 0 else 0
            yield sse("metrics", {
                "input_tokens":  total_input_tokens,
                "output_tokens": output_tokens,
                "elapsed_ms":    round(elapsed * 1000),
                "ttft_ms":       ttft_ms or 0,
                "preprocess_ms": preprocess_ms,
                "tps":           round(tps, 1),
                "summarised":    stats["summarised"],
                "skipped":       stats["skipped"],
                "failed":        stats["failed"],
            })
            log.info("chat_done model=%s in=%d out=%d ttft_ms=%d elapsed_ms=%d",
                     model, total_input_tokens, output_tokens, ttft_ms or 0, round(elapsed * 1000))
            await _write_routing_event(mode, model, summarised=stats["summarised"],
                                       preprocess_ms=preprocess_ms, input_tokens=total_input_tokens,
                                       output_tokens=output_tokens, ttft_ms=ttft_ms or 0,
                                       elapsed_ms=round(elapsed * 1000))
            yield sse("done", {})
            return

        # Build assistant turn and execute tool calls
        assistant_content = []
        for idx in sorted(content_blocks):
            b = content_blocks[idx]
            if b["type"] == "text" and b["text"]:
                assistant_content.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                try:
                    inp = json.loads(b["input_json"]) if b["input_json"] else {}
                except json.JSONDecodeError:
                    inp = {}
                assistant_content.append({
                    "type": "tool_use", "id": b["id"], "name": b["name"], "input": inp,
                })
        current_messages.append({"role": "assistant", "content": assistant_content})

        tool_results = []
        for idx in sorted(content_blocks):
            b = content_blocks[idx]
            if b["type"] != "tool_use":
                continue
            try:
                inp = json.loads(b["input_json"]) if b["input_json"] else {}
            except json.JSONDecodeError:
                inp = {}
            log.info("tool call name=%s path=%s", b["name"], inp.get("path", ""))
            yield sse("status", {"stage": "checking", "message": f"Running {b['name']}…"})
            result = await _execute_tool(b["name"], inp)
            tool_results.append({"type": "tool_result", "tool_use_id": b["id"], "content": result})

        current_messages.append({"role": "user", "content": tool_results})
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})

    yield sse("error", {"message": "Maximum tool call depth reached"})
