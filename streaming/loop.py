"""The one governed agentic tool loop (Phase 4 of PLAN-pi-tools.md).

run_tool_loop owns everything the claude/ollama loops used to duplicate:
turn cap, token + wall-clock budgets, tool dispatch, tool_result/history/
metrics/done SSE emission, and routing-log writes. Provider-specific wire
parsing and message shapes live behind the streaming.providers seam.

`execute_tool`, `clock`, and the budget limits are injected by the adapter
module (resolved from its globals at call time) so tests keep patching
streaming.claude / streaming.ollama attributes exactly as before.
"""
import logging
import time

from helpers import sse

log = logging.getLogger("router")

MAX_TURNS = 10


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
    clock=None,
):
    clock = clock or time.monotonic
    start = clock()
    ttft_ms = None
    current_messages = list(messages)

    for turn in range(MAX_TURNS):
        elapsed_so_far = clock() - start
        budget = provider.budget_tokens(current_messages)
        if budget > max_tokens_budget or elapsed_so_far > max_seconds:
            log.warning("loop budget exceeded provider=%s turn=%d tokens=%d elapsed=%.1fs",
                        provider.name, turn, budget, elapsed_so_far)
            yield sse("metrics", _metrics(
                provider.breach_token_fields(current_messages),
                round(elapsed_so_far * 1000), ttft_ms, preprocess_ms, stats,
            ))
            yield sse("error", {"message": "Loop budget exceeded (token or wall-clock limit) — stopping."})
            return

        log.info("tool_turn provider=%s turn=%d msgs=%d", provider.name, turn, len(current_messages))
        turn_end = None
        try:
            async for ev in provider.stream_turn(current_messages):
                kind = ev["kind"]
                if kind == "text_delta":
                    if ttft_ms is None:
                        ttft_ms = round((clock() - start) * 1000)
                    yield sse("delta", {"text": ev["text"]})
                elif kind == "input_tokens":
                    if turn == 0 and provider.emits_input_tokens_sse:
                        yield sse("input_tokens", {"count": ev["count"]})
                elif kind == "tool_start":
                    yield sse("tool_start", {"index": ev["index"], "name": ev["name"], "id": ev["id"]})
                elif kind == "tool_input_delta":
                    yield sse("tool_input", {"index": ev["index"], "partial_json": ev["partial_json"]})
                elif kind == "tool_end":
                    yield sse("tool_end", {"index": ev["index"]})
                elif kind == "provider_error":
                    yield sse("error", {"message": ev["message"]})
                    return
                elif kind == "turn_end":
                    turn_end = ev
        except Exception as e:
            log.error("upstream error provider=%s: %s", provider.name, e)
            yield sse("error", {"message": str(e)})
            return

        if turn_end is None or not turn_end["tool_use"]:
            elapsed = clock() - start
            token_fields = provider.final_token_fields(elapsed)
            yield sse("metrics", _metrics(
                token_fields, round(elapsed * 1000), ttft_ms, preprocess_ms, stats,
            ))
            log.info("chat_done provider=%s in=%d out=%d ttft_ms=%d elapsed_ms=%d",
                     provider.name, token_fields["input_tokens"],
                     token_fields["output_tokens"], ttft_ms or 0, round(elapsed * 1000))
            await provider.write_routing(ttft_ms or 0, round(elapsed * 1000))
            yield sse("history", {"messages": provider.history_messages(current_messages)})
            yield sse("done", {})
            return

        provider.append_assistant_turn(current_messages)

        results = []
        for tc in turn_end["tool_calls"]:
            log.info("tool call name=%s path=%s", tc["name"], tc["input"].get("path", ""))
            yield sse("status", {"stage": "checking", "message": f"Running {tc['name']}…"})
            t_tool = clock()
            result = await execute_tool(tc["name"], tc["input"])
            log.debug("tool done name=%s elapsed_ms=%d result_len=%d",
                      tc["name"], round((clock() - t_tool) * 1000), len(result))
            results.append((tc["id"], result))
            sse_content = result if len(result) <= 4000 else result[:4000] + "\n…(truncated)"
            yield sse("tool_result", {
                "tool_use_id": tc["id"], "content": sse_content, "is_error": False,
            })
        provider.append_tool_results(current_messages, results)

        yield sse("status", {"stage": "streaming", "message": provider.stream_status()})

    yield sse("error", {"message": "Maximum tool call depth reached"})
