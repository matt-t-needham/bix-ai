"""Ollama adapter: tool-offload model swap + system-prompt injection, then
delegates the agentic loop to streaming.loop.run_tool_loop via an
OllamaProvider. Patch points for tests (httpx, time, _execute_tool, budgets,
routing) stay module attributes here, resolved at call time."""
import logging
import time

import httpx

from config import LOOP_MAX_SECONDS, LOOP_MAX_TOKENS, OLLAMA_TOOL_MODEL
from helpers import _write_routing_event, sse
from streaming.loop import run_tool_loop
from streaming.providers import OllamaProvider
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
        await _write_routing_event(mode, model, output_tokens=output_tokens,
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
