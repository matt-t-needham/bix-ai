"""Ollama adapter: pre-pass + tool-offload model swap + system-prompt
injection, then delegates the agentic loop to streaming.loop.run_tool_loop
via a (guardrailed) OllamaProvider. Patch points for tests (httpx, time,
_execute_tool, budgets, routing) stay module attributes here, resolved at
call time.

Shared with mode=auto's local leg (via streaming.local_first) — mode only
picks the backend + escalation policy; pre-pass, system prompt, tool
definitions, and the turn-loop are identical regardless of caller."""
import logging
import time

import httpx

import compact
import strategy
from config import LOOP_MAX_SECONDS, LOOP_MAX_TOKENS, OLLAMA_TOOL_MODEL
from helpers import _agg, _write_routing_event, ollama_chat, sse
from identity import identity_system_prompt
from streaming.loop import run_tool_loop
from streaming.providers import OllamaProvider
from tools import OLLAMA_TOOLS, _execute_tool

log = logging.getLogger("router")

OLLAMA_SYSTEM = identity_system_prompt(
    tool_names=[t["function"]["name"] for t in OLLAMA_TOOLS],
    doc_topics=["staging", "memory", "logs", "blobs", "modes", "todos"],
)


async def _stream_ollama(
    messages: list, model: str, mode: str = "local",
    tool_offload: bool = False, skip_preprocess: bool = False,
    route_reason: str = "", on_exhausted: str = "best_effort",
):
    _agg["requests"] += 1
    if tool_offload and model != OLLAMA_TOOL_MODEL:
        original = model
        model = OLLAMA_TOOL_MODEL
        log.info("ollama tool_offload from=%s to=%s", original, model)
        yield sse("model_swap", {"from": original, "to": model})

    stats         = {"summarised": 0, "skipped": 0, "failed": 0, "spilled": 0}
    compact_stats = {"compacted": 0, "folded": 0, "failed": 0}
    preprocess_ms = 0
    current_messages = list(messages)

    if skip_preprocess:
        yield sse("status", {"stage": "streaming", "message": f"Streaming from Ollama ({model})…"})
    else:
        req_body = {"messages": current_messages}
        if strategy.has_oversized_blocks(req_body) or compact.should_compact(current_messages):
            yield sse("status", {"stage": "summarising", "message": "Preparing context…"})
        else:
            yield sse("status", {"stage": "checking", "message": "Checking…"})
        t0 = time.monotonic()
        try:
            req_body, stats = await strategy.preprocess(req_body, ollama_chat)
            log.info("preprocess spilled=%d skipped=%d failed=%d",
                     stats["spilled"], stats["skipped"], stats["failed"])
        except Exception as e:
            log.warning("preprocess error: %s", e)
        try:
            new_messages, compact_stats = await compact.compact(req_body["messages"], ollama_chat)
            if compact_stats["compacted"]:
                log.info("compact folded=%d msgs=%d->%d",
                         compact_stats["folded"], len(req_body["messages"]), len(new_messages))
                req_body = {**req_body, "messages": new_messages}
        except Exception as e:
            log.warning("compact error: %s", e)
        preprocess_ms = round((time.monotonic() - t0) * 1000)
        current_messages = req_body["messages"]
        _agg["summarised"]    += stats["summarised"]
        _agg["spilled"]       += stats["spilled"]
        _agg["checked"]       += stats["summarised"] + stats["spilled"] + stats["skipped"]
        _agg["preprocess_ms"] += preprocess_ms
        _agg["failed"]        += stats["failed"] + compact_stats["failed"]
        yield sse("preprocess", {
            "summarised":    stats["summarised"],
            "spilled":       stats["spilled"],
            "compacted":     compact_stats["compacted"],
            "skipped":       stats["skipped"],
            "failed":        stats["failed"] + compact_stats["failed"],
            "preprocess_ms": preprocess_ms,
        })
        yield sse("status", {"stage": "streaming", "message": f"Streaming from Ollama ({model})…"})

    if not current_messages or current_messages[0].get("role") != "system":
        current_messages.insert(0, {"role": "system", "content": OLLAMA_SYSTEM})

    async def _route(input_tokens, output_tokens, ttft_ms, elapsed_ms,
                      guardrail_rescues=0, guardrail_retries=0, guardrail_exhausted=False):
        await _write_routing_event(mode, model, reason=route_reason or f"forced:{mode}",
                                   summarised=stats["summarised"], preprocess_ms=preprocess_ms,
                                   output_tokens=output_tokens, ttft_ms=ttft_ms, elapsed_ms=elapsed_ms,
                                   guardrail_rescues=guardrail_rescues, guardrail_retries=guardrail_retries,
                                   guardrail_exhausted=guardrail_exhausted)

    provider = OllamaProvider(
        model=model,
        tools=OLLAMA_TOOLS,
        client_factory=lambda: httpx.AsyncClient(timeout=None),
        route=_route,
        system_sentinel=OLLAMA_SYSTEM,
        on_exhausted=on_exhausted,
    )
    async for event in run_tool_loop(
        provider, current_messages,
        execute_tool=_execute_tool,
        max_tokens_budget=LOOP_MAX_TOKENS, max_seconds=LOOP_MAX_SECONDS,
        stats=stats,
        preprocess_ms=preprocess_ms,
        clock=time.monotonic,
    ):
        yield event
