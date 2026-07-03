"""Forge-backed local-model runner for mode="auto".

Wraps forge-guardrails' WorkflowRunner so an Ollama model gets:
  - rescue parsing for malformed tool JSON
  - terminal-tool forcing via the synthetic `respond` tool
  - retry nudges + context compaction (TieredCompact)

Emits the same SSE event shape the frontend already handles (status / delta /
tool_start / tool_input / tool_end / metrics / done / error). On any
runner.run() exception, yields an `error` SSE so streaming/local_first can
escalate to Claude.
"""
import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone

from forge import (
    ChunkType,
    ContextManager,
    Message,
    MessageMeta,
    MessageRole,
    MessageType,
    OllamaClient,
    TieredCompact,
    Workflow,
    WorkflowRunner,
    respond_tool,
)

from config import AUTO_LOG, OLLAMA_HOST
from helpers import _write_routing_event, sse
from memory import _load_recent_memories, _memory_system_prompt
from tools import FORGE_TOOLS

log = logging.getLogger("router")

_AUTO_LOG_MAX = 5_000_000  # 5 MB; rotate to .1 when exceeded

FORGE_SYSTEM = (
    "You are a helpful local assistant with filesystem access. "
    "Answer the user's question directly. Only use tools when the question "
    "genuinely requires reading files or recalling past conversations.\n\n"
    "Available tools:\n"
    "- list_directory(path): list files and folders within /home/matt\n"
    "- read_file(path): read a text file's contents\n"
    "- recall_memories(query): search past conversation summaries\n"
    "- read_todos(project): read pending TODOs. No argument = all projects' TODOs; "
    "pass a project name for one.\n"
    "- list_log_sources(): list available logs (internal app/service logs and "
    "Steam client logs)\n"
    "- read_log(path, lines, contains): read the tail of a log, optionally filtered "
    "to a substring. Call list_log_sources first.\n"
    "- stage_write(target_path, content): propose creating/editing a file. This "
    "never writes live — it stages the change for a human to review and approve. "
    "Use it when asked to write, draft, or create a file. Tell the user the change "
    "was staged for review, not written.\n"
    "- respond(message): send your final answer to the user\n\n"
    "Prefer the dedicated tool over browsing directories. Match the request:\n"
    "- pending work / TODOs / tasks / 'what's left to do' → read_todos()\n"
    "- service or Steam logs, errors, crashes → list_log_sources() then read_log(path)\n"
    "- earlier conversations / 'do you remember' → recall_memories(query)\n"
    "Only fall back to list_directory / read_file when no dedicated tool fits.\n\n"
    "When your response is ready, call respond(message='...') with your final "
    "answer. Use other tools first if needed, then respond. Be concise."
)


def _build_forge_workflow() -> Workflow:
    tools = {**FORGE_TOOLS, "respond": respond_tool()}
    return Workflow(
        name                   = "bix-assistant",
        description            = "General-purpose assistant with filesystem access.",
        tools                  = tools,
        required_steps         = [],
        terminal_tool          = "respond",
        system_prompt_template = FORGE_SYSTEM,
    )


FORGE_WORKFLOW = _build_forge_workflow()

# Set true once after first run so we don't spam debug logs.
_logged_result_shape = False


def _convert_messages(messages: list[dict]) -> list[Message]:
    """Convert OpenAI-format prior messages to Forge Message objects.

    Tool messages and system messages are skipped — Forge rebuilds context via
    ContextManager + workflow system prompt. The memory-derived system prompt
    is injected separately by the caller.
    """
    result: list[Message] = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        if not isinstance(content, str):
            content = str(content)
        if role == "user":
            result.append(Message(MessageRole.USER, content,
                                  MessageMeta(MessageType.USER_INPUT)))
        elif role == "assistant":
            result.append(Message(MessageRole.ASSISTANT, content,
                                  MessageMeta(MessageType.TEXT_RESPONSE)))
        # tool / system roles deliberately skipped
    return result


async def _write_auto_log(event: str, **kwargs) -> None:
    """Append one ndjson record to logs/auto.ndjson. Best-effort."""
    record = {
        "ts":    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "event": event,
        **kwargs,
    }
    line = json.dumps(record) + "\n"

    def _write() -> None:
        if AUTO_LOG.exists() and AUTO_LOG.stat().st_size > _AUTO_LOG_MAX:
            AUTO_LOG.rename(AUTO_LOG.with_suffix(".ndjson.1"))
        AUTO_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUTO_LOG, "a") as f:
            f.write(line)

    try:
        await asyncio.to_thread(_write)
    except Exception as e:
        log.warning("auto log write failed: %s", e)


async def _stream_forge_runner(
    messages: list[dict], ollama_model: str, max_tokens: int,
    mode: str = "auto", route_reason: str = "",
):
    """Run Forge's WorkflowRunner against Ollama; yield SSE events.

    On unrecoverable error, yields an `error` SSE — caller (_stream_local_first)
    inspects this to decide between silent and visible Claude escalation.
    """
    loop     = asyncio.get_running_loop()
    queue    = asyncio.Queue()         # unbounded — on_message uses put_nowait
    sentinel = object()
    start    = time.monotonic()

    ttft_ms            = None
    output_chars       = 0
    # Accumulated TOOL_CALL_DELTA chunk content; we extract the respond tool's
    # `message` arg from this and stream it as text deltas.
    _respond_json_acc  = ""
    _respond_text_sent = 0

    # ── Callbacks ─────────────────────────────────────────────────────────────

    async def on_chunk(chunk):
        nonlocal ttft_ms, output_chars, _respond_json_acc, _respond_text_sent
        if chunk.type == ChunkType.TEXT_DELTA and chunk.content:
            if ttft_ms is None:
                ttft_ms = round((time.monotonic() - start) * 1000)
            output_chars += len(chunk.content)
            await queue.put(sse("delta", {"text": chunk.content}))
        elif chunk.type == ChunkType.TOOL_CALL_DELTA and chunk.content:
            # Stream the respond tool's `message` arg as text. Other tools have
            # `path` / `query` keys — only `respond` uses `message`.
            _respond_json_acc += chunk.content
            m = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)', _respond_json_acc)
            if m:
                current_text = m.group(1)
                if len(current_text) > _respond_text_sent:
                    new_chars = current_text[_respond_text_sent:]
                    if ttft_ms is None:
                        ttft_ms = round((time.monotonic() - start) * 1000)
                    output_chars      += len(new_chars)
                    _respond_text_sent = len(current_text)
                    await queue.put(sse("delta", {"text": new_chars}))
        elif chunk.type == ChunkType.RETRY:
            _respond_json_acc  = ""
            _respond_text_sent = 0
            await queue.put(sse("status", {
                "stage":   "streaming",
                "message": "Local model retrying…",
            }))

    def on_message(msg):
        if msg.metadata.type == MessageType.TOOL_CALL and msg.tool_calls:
            for i, tc in enumerate(msg.tool_calls):
                if tc.name == "respond":
                    continue  # suppress respond from the SSE tool stream
                queue.put_nowait(sse("tool_start",
                                     {"index": i, "name": tc.name, "id": tc.call_id}))
                queue.put_nowait(sse("tool_input",
                                     {"index": i, "partial_json": json.dumps(tc.args)}))
                queue.put_nowait(sse("tool_end", {"index": i}))
                loop.call_soon(
                    lambda n=tc.name, a=tc.args: asyncio.create_task(
                        _write_auto_log("tool_call", tool_name=n,
                                        tool_args_keys=list(a.keys()),
                                        model=ollama_model)
                    )
                )
        elif msg.metadata.type == MessageType.TOOL_RESULT:
            queue.put_nowait(sse("status", {
                "stage":   "streaming",
                "message": f"Streaming from {ollama_model}…",
            }))
        elif msg.metadata.type in (
            MessageType.RETRY_NUDGE,
            MessageType.STEP_NUDGE,
            MessageType.PREREQUISITE_NUDGE,
        ):
            loop.call_soon(
                lambda r=msg.content: asyncio.create_task(
                    _write_auto_log("forge_nudge", forge_reason=r[:200],
                                    model=ollama_model)
                )
            )

    # ── Runner setup ──────────────────────────────────────────────────────────

    recent     = _load_recent_memories(3)
    sys_prompt = _memory_system_prompt(recent)
    prior_msgs: list[Message] = []
    if sys_prompt:
        prior_msgs.append(Message(MessageRole.SYSTEM, sys_prompt,
                                  MessageMeta(MessageType.SYSTEM_PROMPT)))
    prior_msgs.extend(_convert_messages(messages[:-1]))

    user_message = ""
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        if isinstance(content, str):
            user_message = content
        elif isinstance(content, list):
            user_message = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )

    # NOTE: recommended_sampling=True requires the model to be in Forge's
    # MODEL_SAMPLING_DEFAULTS table. Ollama tag names (e.g. "gemma4:26b") don't
    # match the HF-card style keys Forge uses, so we let Ollama's own defaults
    # apply instead.
    client = OllamaClient(
        model    = ollama_model,
        base_url = OLLAMA_HOST,
    )
    ctx    = ContextManager(strategy=TieredCompact(keep_recent=2),
                            budget_tokens=max_tokens)
    runner = WorkflowRunner(
        client          = client,
        context_manager = ctx,
        stream          = True,
        on_chunk        = on_chunk,
        on_message      = on_message,
    )

    # ── Drive runner in background, drain queue ──────────────────────────────

    async def run_and_signal():
        nonlocal output_chars
        global _logged_result_shape
        try:
            result = await runner.run(
                FORGE_WORKFLOW, user_message,
                initial_messages=prior_msgs or None,
            )
            if not _logged_result_shape:
                log.info("forge runner.run() result type=%s",
                         type(result).__name__)
                _logged_result_shape = True
            # Safety net: if respond `message` text wasn't fully streamed via
            # TOOL_CALL_DELTA (e.g. non-streaming path), emit what's left.
            if result is not None:
                remaining = str(result)[_respond_text_sent:]
                if remaining:
                    output_chars += len(remaining)
                    await queue.put(sse("delta", {"text": remaining}))
        except Exception as e:
            log.error("forge runner error model=%s: %s", ollama_model, e)
            await _write_auto_log(
                "escalation",
                escalation_reason=str(e)[:500],
                local_elapsed_ms=round((time.monotonic() - start) * 1000),
                model=ollama_model,
            )
            await queue.put(sse("error", {"message": str(e)}))
        finally:
            await queue.put(sentinel)

    asyncio.create_task(run_and_signal())

    yield sse("status", {"stage": "streaming",
                         "message": f"Streaming from {ollama_model}…"})

    while True:
        item = await queue.get()
        if item is sentinel:
            break
        yield item

    elapsed = time.monotonic() - start
    est_out = max(output_chars // 4, 1)
    est_in  = max(
        sum(len(str(m.get("content", ""))) for m in messages) // 4, 1
    )
    tps     = round(est_out / elapsed, 1) if elapsed > 0 else 0
    yield sse("metrics", {
        "input_tokens":  est_in,
        "output_tokens": est_out,
        "elapsed_ms":    round(elapsed * 1000),
        "ttft_ms":       ttft_ms or 0,
        "preprocess_ms": 0,
        "tps":           tps,
        "summarised":    0,
        "spilled":       0,
        "skipped":       0,
        "failed":        0,
    })
    await _write_routing_event(mode, ollama_model,
                               reason=route_reason or f"forced:{mode}",
                               input_tokens=est_in, output_tokens=est_out,
                               ttft_ms=ttft_ms or 0,
                               elapsed_ms=round(elapsed * 1000))
    yield sse("done", {})
