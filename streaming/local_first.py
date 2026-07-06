import json
import logging

import routing
from helpers import ollama_chat, sse
from streaming.claude import _stream_claude
from streaming.ollama import _stream_ollama

log = logging.getLogger("router")


def _parse_sse_event(raw: str) -> str:
    for line in raw.split("\n"):
        if line.startswith("event:"):
            return line[6:].strip()
    return ""


def _parse_sse_data(raw: str) -> dict:
    for line in raw.split("\n"):
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except Exception:
                return {}
    return {}


async def _stream_local_first(
    messages: list, ollama_model: str, claude_model: str,
    max_tokens: int, mode: str = "auto",
):
    """Routing v2 (Phase 6): decide local vs Claude up front, then stream.

    `routing.decide` applies structural rules first (code-gen / multi-step /
    prose / long requests → Claude; small tool digestion / short chat →
    local) and one local classification for the ambiguous remainder, failing
    open to Claude. Local-routed requests run through the same
    `_stream_ollama` pipeline mode=local uses (pre-pass, system prompt, tool
    loop, guardrail rescue/retry) with `on_exhausted="escalate"` so an
    unrecoverable local failure fails over to Claude; Claude-routed requests
    skip the local attempt entirely. Every decision lands in routing.ndjson
    via the `reason` field.
    """
    decision = await routing.decide(messages, ollama_chat)
    log.info("auto route=%s rule=%s reason=%s claude_model=%s",
             decision["route"], decision["rule"], decision["reason"],
             decision.get("claude_model"))

    if decision["route"] == "claude":
        # Cost tier: size-only routes carry a cheaper-model hint (Haiku);
        # intent routes keep the user-selected model. The downshifted model is
        # what _stream_claude writes to routing.ndjson, so est_cost_usd tracks
        # what actually ran.
        target_model = decision.get("claude_model") or claude_model
        route_reason = f"{decision['rule']}: {decision['reason']}"
        label = "Claude"
        if target_model != claude_model:
            route_reason += " · downshift"
            label = "Haiku" if "haiku" in target_model else target_model
        yield sse("status", {"stage": "checking",
                             "message": f"auto → {label} ({decision['reason']})"})
        async for chunk in _stream_claude(
            messages, target_model, max_tokens,
            skip_preprocess=False, mode="auto",
            route_reason=route_reason,
        ):
            yield chunk
        return

    has_delta    = False
    escalate     = False
    error_reason = "local model error"

    async for chunk in _stream_ollama(
        messages, ollama_model, mode=mode, tool_offload=False,
        route_reason=f"{decision['rule']}: {decision['reason']}",
        on_exhausted="escalate",
    ):
        event = _parse_sse_event(chunk)
        if event == "error":
            escalate     = True
            error_reason = _parse_sse_data(chunk).get("message", error_reason)
            log.info("auto mode: ollama errored, escalating has_delta=%s reason=%s",
                     has_delta, error_reason)
            if has_delta:
                yield sse("fallback_triggered", {"reason": "ollama_error"})
            break
        if event == "delta":
            has_delta = True
        yield chunk

    if escalate:
        yield sse("status", {"stage": "streaming",
                             "message": f"Switching to Claude ({error_reason})…"})
        async for chunk in _stream_claude(
            messages, claude_model, max_tokens,
            skip_preprocess=True, mode="auto_fallback",
            route_reason=f"escalated: {error_reason}",
        ):
            yield chunk
