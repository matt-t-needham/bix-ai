"""OpenRouter adapter (mode="openrouter"): spend-cap check, pre-pass, system
prompt, then the shared tool loop on an OpenRouterProvider. Same shape as
streaming/ollama.py — the two differ only in the provider and the cost
bookkeeping. Patch points for tests (httpx, time, _execute_tool, budgets,
openrouter module) are module attributes here, resolved at call time."""
import asyncio
import logging
import time

import httpx

import config
import openrouter
import strategy
from config import LOOP_MAX_SECONDS, LOOP_MAX_TOKENS
from helpers import _agg, _write_routing_event, ollama_chat, sse
from identity import identity_system_prompt, resolved_system
from streaming.loop import run_tool_loop
from streaming.providers import OpenRouterProvider
from tools import OLLAMA_TOOLS, _execute_tool, with_actor

log = logging.getLogger("router")

# Immutable half, fixed at import; resolved_system() appends the
# human-edited half (prompt_store.py) per request.
OPENROUTER_SYSTEM = identity_system_prompt(
    tool_names=[t["function"]["name"] for t in OLLAMA_TOOLS],
    doc_topics=["memory", "blobs", "modes", "workspace", "prompt"],
)


async def _stream_openrouter(messages: list, model: str, mode: str = "openrouter",
                             skip_preprocess: bool = False, route_reason: str = "", agent_session=None,
                             state: dict | None = None):
    _agg["requests"] += 1
    if not openrouter.configured():
        yield sse("error", {"message": "OpenRouter is not configured (OPENROUTER_API_KEY is empty)."})
        return

    stats = {"summarised": 0, "skipped": 0, "failed": 0, "spilled": 0}
    preprocess_ms = 0
    req_body = {"messages": list(messages)}
    if not skip_preprocess:
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
        _agg["spilled"]       += stats["spilled"]
        _agg["checked"]       += stats["spilled"] + stats["skipped"]
        _agg["preprocess_ms"] += preprocess_ms
        _agg["failed"]        += stats["failed"]
        yield sse("preprocess", {
            "summarised": 0, "spilled": stats["spilled"], "compacted": 0,
            "skipped": stats["skipped"], "failed": stats["failed"],
            "preprocess_ms": preprocess_ms,
        })
    current_messages = list(req_body["messages"])

    # Spend caps — checked before anything is sent. The projection is the
    # listed prompt price on the estimated request size plus a nominal
    # 1k-token reply; the daily total comes from our own routing log.
    info = await openrouter.model_info(model)
    if info is None and not openrouter.is_valid_model_id(model):
        yield sse("error", {"message": f"OpenRouter: '{model}' is not a valid model id."})
        return
    # Today's spend is read before the prompt is composed so the model can be
    # told where it stands against the daily cap; check_caps below still gates
    # the call itself, before any HTTP.
    spent  = await asyncio.to_thread(openrouter.spent_today_usd)
    base   = await resolved_system(OPENROUTER_SYSTEM, {
        **state, "spend_today": spent, "spend_cap": config.OPENROUTER_MAX_USD_PER_DAY,
    } if state else state)
    system = agent_session.system(base) if agent_session else base
    est_in = strategy.estimate_tokens(system) + sum(
        strategy.estimate_tokens(strategy.block_text(b) if isinstance(b, dict) else str(b))
        for m in current_messages
        for b in (m.get("content") if isinstance(m.get("content"), list) else [m.get("content", "")])
    )
    projected = openrouter.estimate_cost_usd(info, est_in, 1000)
    refusal   = openrouter.check_caps(projected, spent)
    if refusal:
        log.warning("openrouter cap refusal model=%s projected=%.4f spent=%.4f", model, projected, spent)
        yield sse("error", {"message": refusal})
        return

    if not current_messages or current_messages[0].get("role") != "system":
        current_messages.insert(0, {"role": "system", "content": system})
    yield sse("status", {"stage": "streaming", "message": f"Streaming from OpenRouter ({model})…"})

    async def _route(input_tokens, output_tokens, ttft_ms, elapsed_ms,
                     guardrail_rescues=0, guardrail_retries=0, guardrail_exhausted=False,
                     est_cost_usd=0.0):
        await _write_routing_event(mode, model, reason=route_reason or f"forced:{mode}",
                                   preprocess_ms=preprocess_ms,
                                   input_tokens=input_tokens, output_tokens=output_tokens,
                                   ttft_ms=ttft_ms, elapsed_ms=elapsed_ms,
                                   guardrail_rescues=guardrail_rescues,
                                   guardrail_retries=guardrail_retries,
                                   guardrail_exhausted=guardrail_exhausted,
                                   est_cost_usd=est_cost_usd)

    provider = OpenRouterProvider(
        model=model,
        tools=agent_session.ollama_tools() if agent_session else OLLAMA_TOOLS,
        client_factory=lambda: httpx.AsyncClient(timeout=None),
        route=_route,
        system_sentinel=system,
        url=config.OPENROUTER_URL,
        headers=openrouter.headers(),
        cost_fn=lambda i, o: openrouter.estimate_cost_usd(info, i, o),
    )
    execute = lambda n, i: _execute_tool(n, with_actor(n, i, f"{mode}:{model}"))  # noqa: E731
    async for event in run_tool_loop(
        provider, current_messages,
        execute_tool=agent_session.executor(execute) if agent_session else execute,
        gate=agent_session,
        max_tokens_budget=agent_session.token_budget(LOOP_MAX_TOKENS) if agent_session else LOOP_MAX_TOKENS,
        max_seconds=LOOP_MAX_SECONDS,
        stats=stats, preprocess_ms=preprocess_ms,
        clock=time.monotonic,
    ):
        yield event
