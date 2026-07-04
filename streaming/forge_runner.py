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
from identity import identity_system_prompt
from tools import FORGE_TOOLS

log = logging.getLogger("router")

_AUTO_LOG_MAX = 5_000_000  # 5 MB; rotate to .1 when exceeded

# Forge-specific mechanics only — tool capability text comes from identity.py.
_FORGE_MECHANICS = (
    "respond(message='...') is how your reply reaches the user — every turn "
    "must end by calling it, even when you used no other tool at all.\n\n"
    "First, decide: does answering this actually require checking something on "
    "this machine? If not — general knowledge, math, casual conversation, "
    "questions about yourself — call respond(message='...') immediately with "
    "your answer. Do not call list_directory or read_file 'just in case' first.\n\n"
    "Only when the answer genuinely depends on this machine's files, logs, "
    "memory, or Steam library, use the dedicated tool for it, then respond:\n"
    "- service or Steam logs, errors, crashes → list_log_sources() then read_log(path)\n"
    "- earlier conversations / 'do you remember' → recall_memories(query)\n"
    "Only fall back to list_directory / read_file when no dedicated tool fits, "
    "and only for requests that actually require it."
)

FORGE_SYSTEM = identity_system_prompt(
    tool_names=list(FORGE_TOOLS.keys()),
    doc_topics=["staging", "memory", "logs", "blobs", "modes", "todos"],
    extra=_FORGE_MECHANICS,
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


def _unstreamed_tail(result_text: str, respond_sent: int, streamed_text: str) -> str:
    """Portion of the final runner result not already sent as `delta` events.

    Respond-tool streaming advances `respond_sent`, but plain TEXT_DELTA
    streaming does not — so a reply streamed as text would be re-emitted in
    full by the end-of-run safety net (the "answer repeats itself" quirk).
    Skip the tail when it's already contained in what was streamed.
    """
    remaining = result_text[respond_sent:]
    if remaining and remaining in streamed_text:
        return ""
    return remaining


class _NudgeAttemptCounter:
    """Tracks consecutive nudge attempts within one runner.run() call.

    forge's ErrorTracker.consecutive_retries isn't exposed to on_message —
    this mirrors it locally so _write_auto_log can record how many nudges
    fired in a row before a valid tool call landed (or the run gave up).
    """
    def __init__(self) -> None:
        self.count = 0

    def on_tool_call(self) -> None:
        self.count = 0

    def on_nudge(self) -> int:
        self.count += 1
        return self.count


# Gemma 4's recommended sampling profile (HF model card). forge's own table
# only keys this under HF-card-style strings (e.g. "gemma4:26b-a4b-it-q4_K_M"),
# never our bare Ollama tag ("gemma4:26b") — recommended_sampling=True would
# silently no-op. Hardcoded here so this survives forge renaming/removing
# that table row later instead of reverting to untuned defaults with no error.
# Source: https://huggingface.co/google/gemma-4-26b-a4b-it
_GEMMA4_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}


def _sampling_for(model: str) -> dict:
    if model.startswith("gemma4"):
        return _GEMMA4_SAMPLING
    return {}


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
    # Everything already emitted as `delta` events (both TEXT_DELTA and respond
    # deltas) — consulted by the end-of-run safety net via _unstreamed_tail.
    _streamed_text     = ""
    _nudge_attempts    = _NudgeAttemptCounter()

    # ── Callbacks ─────────────────────────────────────────────────────────────

    async def on_chunk(chunk):
        nonlocal ttft_ms, output_chars, _respond_json_acc, _respond_text_sent, _streamed_text
        if chunk.type == ChunkType.TEXT_DELTA and chunk.content:
            # This workflow's only valid way to finish a turn is the respond
            # tool call (terminal_tool="respond") — a plain-text response is
            # never accepted as-is (ResponseValidator.validate always retries
            # a bare TextResponse unless rescue_tool_call manages to parse an
            # actual tool call out of it, which ordinary prose never does).
            # So every TEXT_DELTA chunk belongs to an attempt forge is about
            # to discard and retry. Count it for latency/cost accounting
            # (it's real inference time), but don't stream it to the user —
            # forwarding it live is what produced the "multiple garbled
            # partial answers" bug: each discarded attempt's full text was
            # shown before being silently superseded by the next attempt.
            if ttft_ms is None:
                ttft_ms = round((time.monotonic() - start) * 1000)
            output_chars += len(chunk.content)
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
                    _streamed_text    += new_chars
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
            _nudge_attempts.on_tool_call()
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
            attempt = _nudge_attempts.on_nudge()
            loop.call_soon(
                lambda r=msg.content, k=msg.metadata.type.value, n=attempt: asyncio.create_task(
                    _write_auto_log("forge_nudge", forge_reason=r[:200],
                                    model=ollama_model, nudge_kind=k,
                                    attempt_count=n)
                )
            )

    # ── Runner setup ──────────────────────────────────────────────────────────

    # No ambient "past session context" injection here (unlike claude.py/pro.py)
    # — this local model treats injected memory as the current task rather than
    # optional background, regardless of how the block is worded (confirmed
    # live: a pure math question with no connection to bix-ai still triggered
    # exploring bix-ai/TODO.txt because a past-session summary said so). Memory
    # stays available on-demand via the recall_memories tool, which requires an
    # active decision to use rather than ambient context to anchor on.
    prior_msgs: list[Message] = _convert_messages(messages[:-1])

    user_message = ""
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content", "")
        if isinstance(content, str):
            user_message = content
        elif isinstance(content, list):
            user_message = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )

    # WorkflowRunner.run()'s two branches silently diverge: when
    # initial_messages is None it builds the system prompt and appends
    # user_message itself; when initial_messages is given (any prior turns
    # exist), it uses the seed *verbatim* and never injects a system prompt or
    # user_message on its own (that's on the caller, per its own docstring).
    # Building the full seed here every time — instead of only passing
    # prior_msgs and relying on that other branch for turn 1 — means every
    # turn actually sees the system prompt and the current question, not just
    # the first one.
    system_msg = Message(MessageRole.SYSTEM, FORGE_WORKFLOW.build_system_prompt(),
                         MessageMeta(MessageType.SYSTEM_PROMPT))
    user_msg   = Message(MessageRole.USER, user_message,
                         MessageMeta(MessageType.USER_INPUT))
    seed: list[Message] = [system_msg, *prior_msgs, user_msg]

    client = OllamaClient(
        model    = ollama_model,
        base_url = OLLAMA_HOST,
        **_sampling_for(ollama_model),
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
                FORGE_WORKFLOW, "",
                initial_messages=seed,
            )
            if not _logged_result_shape:
                log.info("forge runner.run() result type=%s",
                         type(result).__name__)
                _logged_result_shape = True
            # Safety net: if respond `message` text wasn't fully streamed via
            # TOOL_CALL_DELTA (e.g. non-streaming path), emit what's left.
            if result is not None:
                remaining = _unstreamed_tail(str(result), _respond_text_sent,
                                             _streamed_text)
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
