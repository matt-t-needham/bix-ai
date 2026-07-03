"""Ollama adapter: tool-offload model swap + system-prompt injection, then
delegates the agentic loop to streaming.loop.run_tool_loop via an
OllamaProvider. Patch points for tests (httpx, time, _execute_tool, budgets,
routing) stay module attributes here, resolved at call time."""
import logging
import time

import httpx

from config import LOOP_MAX_SECONDS, LOOP_MAX_TOKENS, OLLAMA_TOOL_MODEL
from helpers import _write_routing_event, sse
from identity import identity_system_prompt
from streaming.loop import run_tool_loop
from streaming.providers import OllamaProvider
from tools import OLLAMA_TOOLS, _execute_tool

log = logging.getLogger("router")

OLLAMA_SYSTEM = identity_system_prompt(
    tool_names=[t["function"]["name"] for t in OLLAMA_TOOLS],
    doc_topics=["staging", "memory", "logs", "blobs", "modes", "todos"],
)


async def _stream_ollama(messages: list, model: str, mode: str = "local", tool_offload: bool = False):
    if tool_offload and model != OLLAMA_TOOL_MODEL:
        original = model
        model = OLLAMA_TOOL_MODEL
        log.info("ollama tool_offload from=%s to=%s", original, model)
        yield sse("model_swap", {"from": original, "to": model})
    yield sse("status", {"stage": "streaming", "message": f"Streaming from Ollama ({model})…"})

    current_messages = list(messages)
    if not current_messages or current_messages[0].get("role") != "system":
        current_messages.insert(0, {"role": "system", "content": OLLAMA_SYSTEM})

    async def _route(input_tokens, output_tokens, ttft_ms, elapsed_ms):
        await _write_routing_event(mode, model, reason=f"forced:{mode}",
                                   output_tokens=output_tokens,
                                   ttft_ms=ttft_ms, elapsed_ms=elapsed_ms)

    provider = OllamaProvider(
        model=model,
        tools=OLLAMA_TOOLS,
        client_factory=lambda: httpx.AsyncClient(timeout=None),
        route=_route,
        system_sentinel=OLLAMA_SYSTEM,
    )
    async for event in run_tool_loop(
        provider, current_messages,
        execute_tool=_execute_tool,
        max_tokens_budget=LOOP_MAX_TOKENS, max_seconds=LOOP_MAX_SECONDS,
        stats={"summarised": 0, "spilled": 0, "skipped": 0, "failed": 0},
        preprocess_ms=0,
        clock=time.monotonic,
    ):
        yield event
