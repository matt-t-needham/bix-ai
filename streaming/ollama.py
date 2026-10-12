"""Ollama adapter: pre-pass + tool-offload model swap + system-prompt
injection, then delegates the agentic loop to streaming.loop.run_tool_loop
via a (guardrailed) OllamaProvider. Patch points for tests (httpx, time,
_execute_tool, budgets, routing) stay module attributes here, resolved at
call time.
"""
import logging
import time

import httpx

import strategy
from config import LOOP_MAX_SECONDS, LOOP_MAX_TOKENS, PROGRESS_HEARTBEAT_SECONDS
from helpers import _agg, _write_routing_event, ollama_chat, sse, with_progress
from identity import identity_system_prompt
from streaming.loop import run_tool_loop
from streaming.providers import OllamaProvider
from tools import OLLAMA_TOOLS, _execute_tool, with_actor

log = logging.getLogger("router")

OLLAMA_SYSTEM = identity_system_prompt(
    tool_names=[t["function"]["name"] for t in OLLAMA_TOOLS],
    doc_topics=["memory", "blobs", "modes", "workspace"],
)


async def _stream_ollama(
    messages: list, model: str, mode: str = "local",
    skip_preprocess: bool = False,
    route_reason: str = "", on_exhausted: str = "best_effort",
):
    _agg["requests"] += 1
    stats         = {"summarised": 0, "skipped": 0, "failed": 0, "spilled": 0}
    preprocess_ms = 0
    current_messages = list(messages)

    if skip_preprocess:
        yield sse("status", {"stage": "streaming", "message": f"Streaming from Ollama ({model})…"})
    else:
        req_body = {"messages": current_messages}
        if strategy.has_oversized_blocks(req_body):
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
        preprocess_ms = round((time.monotonic() - t0) * 1000)
        current_messages = req_body["messages"]
        _agg["summarised"]    += stats["summarised"]
        _agg["spilled"]       += stats["spilled"]
        _agg["checked"]       += stats["summarised"] + stats["spilled"] + stats["skipped"]
        _agg["preprocess_ms"] += preprocess_ms
        _agg["failed"]        += stats["failed"]
        yield sse("preprocess", {
            "summarised":    stats["summarised"],
            "spilled":       stats["spilled"],
            "compacted":     0,
            "skipped":       stats["skipped"],
            "failed":        stats["failed"],
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
    # with_progress covers the fully-silent gaps the provider can't see from
    # inside a blocked read — chiefly prompt eval, which runs minutes on this
    # host's Vulkan setup before the first chunk arrives.
    async for event in with_progress(
        run_tool_loop(
            provider, current_messages,
            # Late-bound module-global lookup (tests patch _execute_tool).
            execute_tool=lambda n, i: _execute_tool(n, with_actor(n, i, f"{mode}:{model}")),
            max_tokens_budget=LOOP_MAX_TOKENS, max_seconds=LOOP_MAX_SECONDS,
            stats=stats,
            preprocess_ms=preprocess_ms,
            clock=time.monotonic,
        ),
        PROGRESS_HEARTBEAT_SECONDS,
        label=model,
    ):
        yield event
