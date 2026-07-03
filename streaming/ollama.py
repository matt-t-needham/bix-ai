import json
import logging
import time

import httpx

from config import LOOP_MAX_SECONDS, LOOP_MAX_TOKENS, OLLAMA_TOOL_MODEL, OLLAMA_URL
from helpers import _write_routing_event, sse
from strategy import estimate_tokens
from tools import OLLAMA_TOOLS, _execute_tool

log = logging.getLogger("router")

OLLAMA_SYSTEM = (
    "You are a helpful local assistant. Answer the user's question directly. "
    "Only use tools when the question genuinely requires reading files or recalling past conversations — "
    "for general questions, conversation, or tasks you can answer from your own knowledge, just respond.\n\n"
    "Available tools:\n"
    "- list_directory(path): list files and folders within /home/matt\n"
    "- read_file(path): read a text file's contents\n"
    "- recall_memories(query): search past conversation summaries\n\n"
    "If the user explicitly asks about pending work, TODOs, logs, or past conversations, "
    "these locations may be useful (do not read them otherwise):\n"
    "- /home/matt/apps/bix-infra/todos/ALL.md — compiled pending TODOs across all projects\n"
    "- /home/matt/apps/bix-infra/todos/ — per-project TODO source files\n"
    "- /home/matt/apps/bix-infra/logs/tickets/ — daily log-review tickets (YYYY-MM-DD.md)\n"
    "- /home/matt/apps/bix-infra/logs/ — service logs\n"
    "- /home/matt/apps/bix-ai/data/ — memory and conversation data\n\n"
    "Use list_directory to explore before reading files. "
    "Use recall_memories when the user asks about previous conversations. "
    "Be concise."
)


def _accumulate_tool_call(tool_calls_map: dict, tc: dict) -> None:
    idx = tc.get("index", 0)
    if idx not in tool_calls_map:
        tool_calls_map[idx] = {"id": "", "name": "", "arguments_str": ""}
    entry = tool_calls_map[idx]
    if tc.get("id"):
        entry["id"] = tc["id"]
    fn = tc.get("function") or {}
    if fn.get("name"):
        entry["name"] = fn["name"]
    if fn.get("arguments"):
        entry["arguments_str"] += fn["arguments"]


async def _stream_ollama(messages: list, model: str, mode: str = "local", tool_offload: bool = False):
    if tool_offload and model != OLLAMA_TOOL_MODEL:
        original = model
        model = OLLAMA_TOOL_MODEL
        log.info("ollama tool_offload from=%s to=%s", original, model)
        yield sse("model_swap", {"from": original, "to": model})
    yield sse("status", {"stage": "streaming", "message": f"Streaming from Ollama ({model})…"})
    start            = time.monotonic()
    ttft_ms          = None
    output_chars     = 0
    current_messages = list(messages)
    if not current_messages or current_messages[0].get("role") != "system":
        current_messages.insert(0, {"role": "system", "content": OLLAMA_SYSTEM})

    for _turn in range(10):
        elapsed_so_far = time.monotonic() - start
        # Ollama's OpenAI-compatible endpoint doesn't report usage on stream chunks
        # here, so the token budget is estimated from the growing message payload
        # (mirrors strategy.estimate_tokens' chars/4 heuristic).
        est_tokens = estimate_tokens(json.dumps(current_messages))
        if est_tokens > LOOP_MAX_TOKENS or elapsed_so_far > LOOP_MAX_SECONDS:
            log.warning("loop budget exceeded turn=%d est_tokens=%d elapsed=%.1fs",
                        _turn, est_tokens, elapsed_so_far)
            yield sse("metrics", {
                "input_tokens":  0,
                "output_tokens": est_tokens,
                "elapsed_ms":    round(elapsed_so_far * 1000),
                "ttft_ms":       ttft_ms or 0,
                "preprocess_ms": 0,
                "tps":           0,
                "summarised":    0,
                "spilled":       0,
                "skipped":       0,
                "failed":        0,
            })
            yield sse("error", {"message": "Loop budget exceeded (token or wall-clock limit) — stopping."})
            return
        tool_calls_map: dict = {}
        finish_reason        = None
        response_text        = ""

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", OLLAMA_URL, json={
                    "model": model, "messages": current_messages,
                    "tools": OLLAMA_TOOLS, "stream": True,
                }) as r:
                    if r.status_code != 200:
                        body = await r.aread()
                        try:
                            err = json.loads(body).get("error", f"HTTP {r.status_code}")
                        except Exception:
                            err = f"HTTP {r.status_code}"
                        log.error("ollama error model=%s status=%d err=%s", model, r.status_code, err)
                        yield sse("error", {"message": f"Ollama: {err}"})
                        return
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        choice        = (data.get("choices") or [{}])[0]
                        delta         = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason") or finish_reason

                        content = delta.get("content") or ""
                        if content:
                            if ttft_ms is None:
                                ttft_ms = round((time.monotonic() - start) * 1000)
                            response_text += content
                            output_chars  += len(content)
                            yield sse("delta", {"text": content})

                        for tc in delta.get("tool_calls") or []:
                            _accumulate_tool_call(tool_calls_map, tc)

        except Exception as e:
            log.error("ollama stream error: %s", e)
            yield sse("error", {"message": str(e)})
            return

        if finish_reason != "tool_calls":
            elapsed    = time.monotonic() - start
            est_tokens = max(output_chars // 4, 1)
            yield sse("metrics", {
                "input_tokens":  0,
                "output_tokens": est_tokens,
                "elapsed_ms":    round(elapsed * 1000),
                "ttft_ms":       ttft_ms or 0,
                "preprocess_ms": 0,
                "tps":           round(est_tokens / elapsed, 1),
                "summarised":    0,
                "spilled":       0,
                "skipped":       0,
                "failed":        0,
            })
            log.info("chat_done model=%s ttft_ms=%d elapsed_ms=%d",
                     model, ttft_ms or 0, round(elapsed * 1000))
            await _write_routing_event(mode, model, output_tokens=est_tokens,
                                       ttft_ms=ttft_ms or 0, elapsed_ms=round(elapsed * 1000))
            history_messages = current_messages
            if (history_messages and history_messages[0].get("role") == "system"
                    and history_messages[0].get("content") == OLLAMA_SYSTEM):
                history_messages = history_messages[1:]
            yield sse("history", {"messages": history_messages})
            yield sse("done", {})
            return

        for idx, tc in sorted(tool_calls_map.items()):
            yield sse("tool_start", {"index": idx, "name": tc["name"], "id": tc["id"]})
            yield sse("tool_input", {"index": idx, "partial_json": tc["arguments_str"]})
            yield sse("tool_end",   {"index": idx})

        assistant_tool_calls = [
            {
                "index": idx, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments_str"]},
            }
            for idx, tc in sorted(tool_calls_map.items())
        ]
        current_messages.append({
            "role":       "assistant",
            "content":    response_text or None,
            "tool_calls": assistant_tool_calls,
        })

        for idx, tc in sorted(tool_calls_map.items()):
            try:
                inp = json.loads(tc["arguments_str"]) if tc["arguments_str"] else {}
            except json.JSONDecodeError:
                inp = {}
            log.info("ollama tool call name=%s path=%s", tc["name"], inp.get("path", ""))
            yield sse("status", {"stage": "checking", "message": f"Running {tc['name']}…"})
            t_tool = time.monotonic()
            result = await _execute_tool(tc["name"], inp)
            log.debug("tool done name=%s elapsed_ms=%d result_len=%d",
                      tc["name"], round((time.monotonic() - t_tool) * 1000), len(result))
            current_messages.append({
                "role":         "tool",
                "tool_call_id": tc["id"],
                "content":      result,
            })
            sse_content = result if len(result) <= 4000 else result[:4000] + "\n…(truncated)"
            yield sse("tool_result", {
                "tool_use_id": tc["id"], "content": sse_content, "is_error": False,
            })

        yield sse("status", {"stage": "streaming", "message": f"Streaming from Ollama ({model})…"})

    yield sse("error", {"message": "Maximum tool call depth reached"})
