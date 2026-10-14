"""Anthropic adapter: memory injection + pre-pass, then delegates the agentic
loop to streaming.loop.run_tool_loop via an AnthropicProvider. Everything
patched by tests (httpx, time, _execute_tool, budgets, memory, routing) stays
a module attribute here and is resolved at call time."""
import logging
import time

import httpx

import strategy
from config import ANTHROPIC_API_KEY, LOOP_MAX_SECONDS, LOOP_MAX_TOKENS
from helpers import _agg, _write_routing_event, ollama_chat, sse
from identity import identity_system_prompt, resolved_system
from streaming.loop import run_tool_loop
from streaming.providers import AnthropicProvider
from tools import FS_TOOLS, _execute_tool, with_actor

log = logging.getLogger("router")

# The immutable half of the system prompt, fixed at import; resolved_system()
# appends the human-edited half (prompt_store.py) per request.
# No capability_index here: FS_TOOLS already sends Claude the full tool
# descriptions on the wire, so one-liners would be pure duplication.
_IDENTITY_PROMPT = identity_system_prompt(
    tool_names=[],
    doc_topics=["memory", "blobs", "modes", "workspace", "prompt"],
)


async def _stream_claude(
    messages: list, model: str, max_tokens: int,
    skip_preprocess: bool, mode: str = "api",
    route_reason: str = "", agent_session=None, state: dict | None = None,
):
    _agg["requests"] += 1
    req_body   = {"model": model, "max_tokens": max_tokens, "messages": messages}
    base = await resolved_system(_IDENTITY_PROMPT, state)
    req_body["system"] = agent_session.system(base) if agent_session else base
    stats         = {"summarised": 0, "skipped": 0, "failed": 0, "spilled": 0}
    preprocess_ms = 0
    log.debug("claude request model=%s msgs=%d skip_preprocess=%s", model, len(messages), skip_preprocess)

    if skip_preprocess:
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})
    else:
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
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})

    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }

    async def _route(input_tokens, output_tokens, ttft_ms, elapsed_ms):
        await _write_routing_event(mode, model, reason=route_reason or f"forced:{mode}",
                                   summarised=stats["summarised"],
                                   preprocess_ms=preprocess_ms, input_tokens=input_tokens,
                                   output_tokens=output_tokens, ttft_ms=ttft_ms,
                                   elapsed_ms=elapsed_ms)

    provider = AnthropicProvider(
        req_body={k: v for k, v in req_body.items() if k != "messages"},
        headers=headers,
        tools=agent_session.anthropic_tools() if agent_session else FS_TOOLS,
        client_factory=lambda: httpx.AsyncClient(timeout=None),
        route=_route,
    )
    # Late-bound module-global lookup (tests patch _execute_tool).
    execute = lambda n, i: _execute_tool(n, with_actor(n, i, f"{mode}:{model}"))  # noqa: E731
    async for event in run_tool_loop(
        provider, req_body["messages"],
        execute_tool=agent_session.executor(execute) if agent_session else execute,
        gate=agent_session,
        max_tokens_budget=agent_session.token_budget(LOOP_MAX_TOKENS) if agent_session else LOOP_MAX_TOKENS,
        max_seconds=LOOP_MAX_SECONDS,
        stats=stats, preprocess_ms=preprocess_ms,
        clock=time.monotonic,
    ):
        yield event
