"""The one governed agentic tool loop (Phase 4 of PLAN-pi-tools.md).

run_tool_loop owns everything the claude/ollama loops used to duplicate:
turn cap, token + wall-clock budgets, tool dispatch, tool_result/history/
metrics/done SSE emission, and routing-log writes. Provider-specific wire
parsing and message shapes live behind the streaming.providers seam.

`execute_tool`, `clock`, and the budget limits are injected by the adapter
module (resolved from its globals at call time) so tests keep patching
streaming.claude / streaming.ollama attributes exactly as before.

`gate` (agent mode only) holds a tool call until matt approves it: the loop
emits `approval_required` with a fresh approval id, waits on gate.wait(), and
runs the tool only if approved.

`execute_tool` returns the result text, or (text, is_error) when the tool can
fail; is_error reaches both the UI and the model.
"""
import logging
import time

import approvals
from config import LOOP_MAX_TURNS as MAX_TURNS
from helpers import sse

log = logging.getLogger("router")


def _metrics(token_fields: dict, elapsed_ms: int, ttft_ms: int | None,
             preprocess_ms: int, stats: dict) -> dict:
    return {
        "input_tokens":  token_fields["input_tokens"],
        "output_tokens": token_fields["output_tokens"],
        "elapsed_ms":    elapsed_ms,
        "ttft_ms":       ttft_ms or 0,
        "preprocess_ms": preprocess_ms,
        "tps":           token_fields["tps"],
        "summarised":    stats["summarised"],
        "spilled":       stats["spilled"],
        "skipped":       stats["skipped"],
        "failed":        stats["failed"],
    }


async def run_tool_loop(
    provider, messages: list, *,
    execute_tool,
    max_tokens_budget: int, max_seconds: float,
    stats: dict, preprocess_ms: int = 0,
    clock=None, gate=None,
):
    clock = clock or time.monotonic
    start = clock()
    waited = 0.0
    ttft_ms = None
    current_messages = list(messages)
    tools_run: list[str] = []

    for turn in range(MAX_TURNS):
        elapsed_so_far = clock() - start - waited
        budget = provider.budget_tokens(current_messages)
        if budget > max_tokens_budget or elapsed_so_far > max_seconds:
            log.warning("loop budget exceeded provider=%s turn=%d tokens=%d elapsed=%.1fs",
                        provider.name, turn, budget, elapsed_so_far)
            yield sse("metrics", _metrics(
                provider.breach_token_fields(current_messages),
                round(elapsed_so_far * 1000), ttft_ms, preprocess_ms, stats,
            ))
            msg = "Loop budget exceeded (token or wall-clock limit) — stopping."
            if tools_run:
                msg += f" Tools already executed: {', '.join(tools_run)}."
            yield sse("error", {"message": msg})
            return

        # Per-turn governor gauge: limits should be visible before they bite.
        yield sse("budget", {
            "turn": turn + 1, "max_turns": MAX_TURNS,
            "elapsed_s": round(elapsed_so_far), "max_seconds": round(max_seconds),
            "tokens": budget, "max_tokens": max_tokens_budget,
        })

        log.info("tool_turn provider=%s turn=%d msgs=%d", provider.name, turn, len(current_messages))
        turn_end = None
        try:
            async for ev in provider.stream_turn(current_messages):
                kind = ev["kind"]
                if kind == "text_delta":
                    if ttft_ms is None:
                        ttft_ms = round((clock() - start) * 1000)
                    yield sse("delta", {"text": ev["text"]})
                elif kind == "thinking_delta":
                    yield sse("thinking", {"text": ev["text"]})
                elif kind == "input_tokens":
                    if turn == 0 and provider.emits_input_tokens_sse:
                        yield sse("input_tokens", {"count": ev["count"]})
                elif kind == "tool_start":
                    yield sse("tool_start", {"index": ev["index"], "name": ev["name"], "id": ev["id"]})
                elif kind == "tool_input_delta":
                    yield sse("tool_input", {"index": ev["index"], "partial_json": ev["partial_json"]})
                elif kind == "tool_end":
                    yield sse("tool_end", {"index": ev["index"]})
                elif kind == "working":
                    yield sse("status", {"stage": "streaming", "message": ev["message"]})
                elif kind == "provider_error":
                    yield sse("error", {"message": ev["message"]})
                    return
                elif kind == "turn_end":
                    turn_end = ev
        except Exception as e:
            log.error("upstream error provider=%s: %s", provider.name, e)
            yield sse("error", {"message": str(e)})
            return

        if (turn_end is not None and turn_end.get("truncated")
                and turn_end["tool_calls"] and not turn_end["tool_use"]):
            # Output cap hit mid-tool-call: the partial tool JSON already
            # streamed to the UI, but the call can't run — surface a real
            # error instead of a clean `done` that looks like an answer.
            log.warning("turn truncated mid-tool-call provider=%s turn=%d tools=%s",
                        provider.name, turn,
                        ",".join(tc["name"] for tc in turn_end["tool_calls"]))
            elapsed = clock() - start - waited
            yield sse("metrics", _metrics(
                provider.final_token_fields(elapsed),
                round(elapsed * 1000), ttft_ms, preprocess_ms, stats,
            ))
            yield sse("error", {"message": "Response hit the output-token limit mid-tool-call — the result is incomplete. Raise max_tokens or ask for a smaller change."})
            return

        if turn_end is None or not turn_end["tool_use"]:
            elapsed = clock() - start - waited
            token_fields = provider.final_token_fields(elapsed)
            yield sse("metrics", _metrics(
                token_fields, round(elapsed * 1000), ttft_ms, preprocess_ms, stats,
            ))
            log.info("chat_done provider=%s in=%d out=%d ttft_ms=%d elapsed_ms=%d",
                     provider.name, token_fields["input_tokens"],
                     token_fields["output_tokens"], ttft_ms or 0, round(elapsed * 1000))
            await provider.write_routing(ttft_ms or 0, round(elapsed * 1000))
            # The final reply belongs in `history`: main.py stores this list as
            # the conversation. An empty turn is left out — the Anthropic API
            # rejects an assistant message with no content.
            final = []
            provider.append_assistant_turn(final)
            if final[0]["content"]:
                current_messages.append(final[0])
            yield sse("history", {"messages": provider.history_messages(current_messages)})
            yield sse("done", {})
            return

        provider.append_assistant_turn(current_messages)

        results = []
        for tc in turn_end["tool_calls"]:
            log.info("tool call name=%s path=%s", tc["name"],
                     tc["input"].get("path") or tc["input"].get("target_path", ""))
            tools_run.append(tc["name"])
            refusal = None
            if gate is not None and gate.needs_approval(tc["name"], tc["input"]):
                approval_id = gate.open_request()
                yield sse("approval_required", {"approval_id": approval_id, "tool_use_id": tc["id"],
                                                "name": tc["name"], "input": tc["input"]})
                t_wait = clock()
                refusal = await gate.wait(approval_id)
                waited += clock() - t_wait
            t_tool = clock()
            if refusal:
                log.info("tool refused name=%s reason=%s", tc["name"], refusal)
                result, is_error = approvals.not_run(refusal), True
            else:
                yield sse("status", {"stage": "checking", "message": f"Running {tc['name']}…"})
                result, is_error = await execute_tool(tc["name"], tc["input"]), False
                if isinstance(result, tuple):
                    result, is_error = result
            log.debug("tool done name=%s elapsed_ms=%d result_len=%d",
                      tc["name"], round((clock() - t_tool) * 1000), len(result))
            results.append((tc["id"], result, is_error))
            sse_content = result if len(result) <= 4000 else result[:4000] + "\n…(truncated)"
            yield sse("tool_result", {
                "tool_use_id": tc["id"], "content": sse_content, "is_error": is_error,
            })
        provider.append_tool_results(current_messages, results)

        yield sse("status", {"stage": "streaming", "message": provider.stream_status()})

    msg = f"Maximum tool call depth reached ({MAX_TURNS} turns)."
    if tools_run:
        msg += f" Tools already executed: {', '.join(tools_run)}."
    yield sse("error", {"message": msg})
