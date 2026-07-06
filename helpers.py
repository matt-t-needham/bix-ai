import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import httpx

from config import ANTHROPIC_API_KEY, CLAUDE_CREDS_PATH, MODEL_COSTS, OLLAMA_URL, ROUTING_LOG

_ROUTING_LOG_MAX = 5_000_000  # 5 MB; rotate to .1 when exceeded

log = logging.getLogger("router")

# Shared request-aggregate stats, read by /stats route
_agg: dict = {"requests": 0, "summarised": 0, "spilled": 0, "checked": 0, "preprocess_ms": 0, "failed": 0}


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def with_keepalive(gen, interval: float = 15.0):
    # Cloudflare Tunnel resets idle streams after ~100s, so emit a comment
    # heartbeat during long gaps (e.g. while a subprocess thinks between tool
    # turns). The heartbeat also forces a write that surfaces client disconnects
    # so the inner generator's finally blocks can clean up.
    aiter = gen.__aiter__()
    next_task: asyncio.Task | None = None
    sentinel = object()

    async def _safe_next() -> object:
        try:
            return await aiter.__anext__()
        except StopAsyncIteration:
            return sentinel

    try:
        while True:
            if next_task is None:
                next_task = asyncio.create_task(_safe_next())
            try:
                result = await asyncio.wait_for(asyncio.shield(next_task), timeout=interval)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            next_task = None
            if result is sentinel:
                return
            yield result
    finally:
        if next_task is not None and not next_task.done():
            next_task.cancel()
            try:
                await next_task
            except BaseException:
                pass
        try:
            await aiter.aclose()
        except Exception:
            pass


async def ollama_chat(model: str, messages: list, timeout: float = 120.0) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(OLLAMA_URL, json={
            "model": model, "messages": messages, "stream": False,
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def _est_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Advisory cost from config.MODEL_COSTS; 0.0 for local/unknown models."""
    rate_in, rate_out = MODEL_COSTS.get(model, (0.0, 0.0))
    return round((input_tokens * rate_in + output_tokens * rate_out) / 1e6, 6)


async def _write_routing_event(
    mode: str, model: str, *,
    reason: str = "",
    summarised: int = 0, preprocess_ms: int = 0,
    input_tokens: int = 0, output_tokens: int = 0,
    ttft_ms: int = 0, elapsed_ms: int = 0,
    guardrail_rescues: int = 0, guardrail_retries: int = 0,
    guardrail_exhausted: bool = False,
) -> None:
    record = json.dumps({
        "ts":                  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode":                mode,
        "model":               model,
        "reason":              reason,
        "est_cost_usd":        _est_cost_usd(model, input_tokens, output_tokens),
        "summarised":          summarised,
        "preprocess_ms":       preprocess_ms,
        "input_tokens":        input_tokens,
        "output_tokens":       output_tokens,
        "ttft_ms":             ttft_ms,
        "elapsed_ms":          elapsed_ms,
        "guardrail_rescues":   guardrail_rescues,
        "guardrail_retries":   guardrail_retries,
        "guardrail_exhausted": guardrail_exhausted,
    }) + "\n"
    try:
        def _write() -> None:
            if ROUTING_LOG.exists() and ROUTING_LOG.stat().st_size > _ROUTING_LOG_MAX:
                ROUTING_LOG.rename(ROUTING_LOG.with_suffix(".ndjson.1"))
            with open(ROUTING_LOG, "a") as f:
                f.write(record)
        await asyncio.to_thread(_write)
    except Exception as e:
        log.warning("routing log write failed: %s", e)


def _claude_session() -> dict:
    """Read Claude Code credentials for display purposes only — not used for API auth."""
    try:
        data       = json.loads(CLAUDE_CREDS_PATH.read_text())
        oauth      = data.get("claudeAiOauth", {})
        token      = oauth.get("accessToken")
        expires_ms = oauth.get("expiresAt", 0)
        valid = bool(token) and expires_ms > (time.time() * 1000 + 300_000)
        return {
            "logged_in":         valid,
            "expires_at":        expires_ms or None,
            "subscription_type": oauth.get("subscriptionType"),
        }
    except Exception:
        return {"logged_in": False, "expires_at": None, "subscription_type": None}
