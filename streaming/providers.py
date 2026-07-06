"""Provider adapters (Phase 4 of PLAN-pi-tools.md).

Each provider translates between its backend's wire protocol and the
normalised shape consumed by streaming.loop.run_tool_loop:

Normalised events yielded by stream_turn (plain dicts, `kind` discriminated):
    input_tokens      {count}                       — usage on stream open
    text_delta        {text}
    tool_start        {index, id, name}
    tool_input_delta  {index, partial_json}
    tool_end          {index}
    provider_error    {message}                     — upstream refused; loop stops
    turn_end          {tool_use: bool, tool_calls: [{id, name, input}]}
                                                    — always last unless error

Messages stay provider-native end to end: the loop never inspects them, it
only threads them through append_assistant_turn / append_tool_results /
history_messages. Token accounting is provider state (Anthropic reports real
usage; Ollama's OpenAI endpoint doesn't, so it estimates chars/4).

Providers receive a `client_factory` and `route` callable from their adapter
module instead of importing httpx/routing themselves — that keeps the
adapter module the single patch point for tests.

OllamaProvider additionally carries a guardrail layer (rescue-parsing +
bounded retry-with-nudge for genuinely garbled tool-call attempts, using
forge-guardrails' standalone `rescue_tool_call`/`ErrorTracker`/`retry_nudge`
— NOT `WorkflowRunner`, which owns its own competing loop/prompting). An
ordinary clean text answer with no tool-call attempt at all is always
accepted immediately; only a structurally-attempted-but-malformed tool call
goes through rescue-then-retry. This is what lets mode=local and mode=auto's
local leg share one implementation with no forced terminal tool.
"""
import json
import logging

from forge import ErrorTracker, rescue_tool_call, retry_nudge

from config import ANTHROPIC_URL, OLLAMA_URL
from strategy import estimate_tokens

log = logging.getLogger("router")


def _parse_input(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


def _try_parse_strict(raw: str):
    """Like _parse_input, but distinguishes 'no args' ({}) from malformed
    JSON (None) — the guardrail layer needs to know which one happened."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _ollama_assistant_message(text: str, tool_map: dict) -> dict:
    return {
        "role":    "assistant",
        "content": text or None,
        "tool_calls": [
            {
                "index": idx, "id": tc["id"], "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments_str"]},
            }
            for idx, tc in sorted(tool_map.items())
        ],
    }


def _to_ollama_wire(messages: list) -> list:
    """Messages threaded through the loop are canonical (Anthropic) shape —
    assistant tool_use / user tool_result content blocks — same as what the
    client stores and what strategy.py/compact.py assume. Ollama's chat
    endpoint speaks OpenAI's wire shape (role="tool", assistant.tool_calls),
    so translate right at the HTTP boundary rather than letting native shape
    leak into current_messages (that used to escape via the `history` SSE
    event and break a later Claude-routed turn — "Unexpected role tool").
    Native messages (e.g. this module's own retry-nudge scratch turns) pass
    through untouched, so this is safe to apply to a mixed list."""
    wire = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role == "assistant" and isinstance(content, list):
            text = "\n".join(b["text"] for b in content if b.get("type") == "text")
            tool_calls = [
                {"index": i, "id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
                for i, b in enumerate(b for b in content if b.get("type") == "tool_use")
            ]
            wire_msg = {"role": "assistant", "content": text or None}
            if tool_calls:
                wire_msg["tool_calls"] = tool_calls
            wire.append(wire_msg)
        elif role == "user" and isinstance(content, list) and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            text = "\n".join(b["text"] for b in content
                             if isinstance(b, dict) and b.get("type") == "text")
            if text:
                wire.append({"role": "user", "content": text})
            for b in content:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    wire.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": b.get("content", "")})
        else:
            wire.append(m)
    return wire


class AnthropicProvider:
    name = "claude"
    emits_input_tokens_sse = True  # loop forwards the turn-0 count to the UI

    def __init__(self, req_body: dict, headers: dict, tools: list, client_factory, route):
        self.req_body = req_body
        self.headers = headers
        self.tools = tools
        self.client_factory = client_factory
        self._route = route  # async (input_tokens, output_tokens, ttft_ms, elapsed_ms)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.last_output_tokens = 0
        self._blocks: dict = {}
        self._turn = 0

    def stream_status(self) -> str:
        return "Streaming from Claude…"

    def budget_tokens(self, messages: list) -> int:
        return self.total_input_tokens + self.total_output_tokens

    async def stream_turn(self, messages: list):
        api_body = {**self.req_body, "messages": messages, "tools": self.tools, "stream": True}
        self._blocks = {}
        stop_reason = None
        output_tokens = 0

        async with self.client_factory() as client:
            async with client.stream("POST", ANTHROPIC_URL, json=api_body, headers=self.headers) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    try:
                        err_obj = json.loads(body).get("error", {})
                        err = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                        err = err or f"HTTP {r.status_code}"
                    except Exception:
                        err = f"HTTP {r.status_code}"
                    log.error("anthropic error status=%d err=%s", r.status_code, err)
                    yield {"kind": "provider_error", "message": f"Anthropic: {err}"}
                    return
                event_type = None
                async for line in r.aiter_lines():
                    line = line.strip()
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        raw = line[5:].strip()
                        if not raw:
                            continue
                        try:
                            data = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if event_type == "message_start":
                            tok = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                            self.total_input_tokens += tok
                            log.info("input_tokens turn=%d count=%d total=%d",
                                     self._turn, tok, self.total_input_tokens)
                            yield {"kind": "input_tokens", "count": tok}

                        elif event_type == "content_block_start":
                            idx   = data.get("index", 0)
                            block = data.get("content_block", {})
                            self._blocks[idx] = {
                                "type": block.get("type"), "name": block.get("name", ""),
                                "id":   block.get("id",   ""), "text": "", "input_json": "",
                            }
                            if block.get("type") == "tool_use":
                                yield {"kind": "tool_start", "index": idx,
                                       "id": block.get("id", ""), "name": block.get("name", "")}

                        elif event_type == "content_block_delta":
                            idx   = data.get("index", 0)
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta["text"]
                                if idx in self._blocks:
                                    self._blocks[idx]["text"] += text
                                yield {"kind": "text_delta", "text": text}
                            elif delta.get("type") == "input_json_delta":
                                partial = delta.get("partial_json", "")
                                if idx in self._blocks:
                                    self._blocks[idx]["input_json"] += partial
                                yield {"kind": "tool_input_delta", "index": idx,
                                       "partial_json": partial}

                        elif event_type == "content_block_stop":
                            idx = data.get("index", 0)
                            if self._blocks.get(idx, {}).get("type") == "tool_use":
                                yield {"kind": "tool_end", "index": idx}

                        elif event_type == "message_delta":
                            output_tokens = data.get("usage", {}).get("output_tokens", 0)
                            stop_reason   = data.get("delta", {}).get("stop_reason")
                            log.info("output_tokens turn=%d count=%d stop_reason=%s",
                                     self._turn, output_tokens, stop_reason)

                        elif event_type == "error":
                            msg = data.get("error", {}).get("message", "unknown upstream error")
                            log.error("anthropic stream error err=%s", msg)
                            yield {"kind": "provider_error", "message": f"Anthropic: {msg}"}
                            return

        self.total_output_tokens += output_tokens
        self.last_output_tokens = output_tokens
        self._turn += 1
        yield {
            "kind": "turn_end",
            "tool_use": stop_reason == "tool_use",
            "tool_calls": [
                {"id": b["id"], "name": b["name"], "input": _parse_input(b["input_json"])}
                for _, b in sorted(self._blocks.items()) if b["type"] == "tool_use"
            ],
        }

    def append_assistant_turn(self, messages: list) -> None:
        content = []
        for _, b in sorted(self._blocks.items()):
            if b["type"] == "text" and b["text"]:
                content.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                content.append({
                    "type": "tool_use", "id": b["id"], "name": b["name"],
                    "input": _parse_input(b["input_json"]),
                })
        messages.append({"role": "assistant", "content": content})

    def append_tool_results(self, messages: list, results: list) -> None:
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": content}
            for tid, content in results
        ]})

    def history_messages(self, messages: list) -> list:
        return messages

    def final_token_fields(self, elapsed: float) -> dict:
        tps = self.last_output_tokens / elapsed if elapsed > 0 else 0
        return {"input_tokens": self.total_input_tokens,
                "output_tokens": self.last_output_tokens, "tps": round(tps, 1)}

    def breach_token_fields(self, messages: list) -> dict:
        return {"input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens, "tps": 0}

    async def write_routing(self, ttft_ms: int, elapsed_ms: int) -> None:
        await self._route(self.total_input_tokens, self.last_output_tokens,
                          ttft_ms, elapsed_ms)


class OllamaProvider:
    name = "ollama"
    emits_input_tokens_sse = False  # OpenAI-compat stream chunks carry no usage

    def __init__(self, model: str, tools: list, client_factory, route,
                 system_sentinel: str, *, on_exhausted: str = "best_effort",
                 max_retries: int = 2):
        self.model = model
        self.tools = tools
        self.client_factory = client_factory
        self._route = route
        self.system_sentinel = system_sentinel  # injected system msg to strip from history
        self.on_exhausted = on_exhausted  # "best_effort" | "escalate"
        self.max_retries = max_retries
        self._tool_names = [t["function"]["name"] for t in tools]
        self.output_chars = 0
        self._tool_map: dict = {}
        self._response_text = ""
        self.guardrail_rescues = 0    # rescued a malformed/embedded tool call, no retry needed
        self.guardrail_retries = 0    # nudged and re-called the model
        self.guardrail_exhausted = False

    def stream_status(self) -> str:
        return f"Streaming from Ollama ({self.model})…"

    def budget_tokens(self, messages: list) -> int:
        # No usage on stream chunks — estimate from the growing payload
        # (mirrors strategy.estimate_tokens' chars/4 heuristic).
        return estimate_tokens(json.dumps(messages))

    def _accumulate_tool_call(self, tc: dict, tool_map: dict) -> None:
        idx = tc.get("index", 0)
        if idx not in tool_map:
            tool_map[idx] = {"id": "", "name": "", "arguments_str": ""}
        entry = tool_map[idx]
        if tc.get("id"):
            entry["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            entry["name"] = fn["name"]
        if fn.get("arguments"):
            entry["arguments_str"] += fn["arguments"]

    async def _consume_stream(self, messages: list, *, live: bool):
        """One raw HTTP round-trip. Yields text_delta only when live=True —
        a discarded retry attempt still burns real inference time (tracked
        via output_chars regardless), it just isn't shown to the user until
        it's accepted. Always ends with one attempt_end."""
        text = ""
        tool_map: dict = {}
        finish_reason = None

        async with self.client_factory() as client:
            async with client.stream("POST", OLLAMA_URL, json={
                "model": self.model, "messages": _to_ollama_wire(messages),
                "tools": self.tools, "stream": True,
            }) as r:
                if r.status_code != 200:
                    body = await r.aread()
                    try:
                        err = json.loads(body).get("error", f"HTTP {r.status_code}")
                    except Exception:
                        err = f"HTTP {r.status_code}"
                    log.error("ollama error model=%s status=%d err=%s",
                              self.model, r.status_code, err)
                    yield {"kind": "provider_error", "message": f"Ollama: {err}"}
                    return
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        data = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    choice        = (data.get("choices") or [{}])[0]
                    delta         = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason") or finish_reason

                    content = delta.get("content") or ""
                    if content:
                        text               += content
                        self.output_chars += len(content)
                        if live:
                            yield {"kind": "text_delta", "text": content}

                    for tc in delta.get("tool_calls") or []:
                        self._accumulate_tool_call(tc, tool_map)

        yield {"kind": "attempt_end", "text": text, "tool_map": tool_map,
               "finish_reason": finish_reason}

    def _classify(self, text: str, tool_map: dict, finish_reason: str | None):
        """('accept_text',) | ('accept_tools', resolved_map, rescued: bool) | ('retry', reason).

        `rescued` is explicit (not inferred from attempt number or
        finish_reason) so the caller can tell "rescue_tool_call actually
        recovered something" apart from "this was simply a clean parse on a
        retry attempt" — those are different events for telemetry purposes.

        Only a structurally-attempted-but-garbled tool call is ever retried.
        An ordinary clean text answer (no tool_calls at all, and nothing
        rescuable embedded in the prose) is always accepted immediately —
        no forcing, regardless of whether the question needed a tool.
        """
        if finish_reason == "tool_calls" and tool_map:
            resolved = {}
            any_rescued = False
            for idx, tc in tool_map.items():
                parsed = _try_parse_strict(tc["arguments_str"])
                if parsed is not None:
                    resolved[idx] = tc
                    continue
                names = [tc["name"]] if tc["name"] else self._tool_names
                rescued = rescue_tool_call(tc["arguments_str"], names)
                if rescued:
                    resolved[idx] = {**tc, "arguments_str": json.dumps(rescued[0].args)}
                    any_rescued = True
                else:
                    return ("retry", f"malformed tool-call JSON for '{tc['name'] or '?'}'")
            return ("accept_tools", resolved, any_rescued)

        if finish_reason == "tool_calls" and not tool_map:
            return ("retry", "empty tool_calls with no fragments")

        rescued = rescue_tool_call(text, self._tool_names) if text.strip() else []
        if rescued:
            synth = {
                i: {"id": f"rescued-{i}", "name": r.tool, "arguments_str": json.dumps(r.args)}
                for i, r in enumerate(rescued)
            }
            return ("accept_tools", synth, True)
        return ("accept_text",)

    async def stream_turn(self, messages: list):
        self._tool_map = {}
        self._response_text = ""
        tracker = ErrorTracker(max_retries=self.max_retries)
        attempt_messages = list(messages)  # scratch copy — nudges never touch the caller's list
        attempt_no = 0
        live = True

        while True:
            text = tool_map = finish_reason = None
            async for ev in self._consume_stream(attempt_messages, live=live):
                if ev["kind"] == "provider_error":
                    yield ev
                    return
                if ev["kind"] == "text_delta":
                    yield ev
                elif ev["kind"] == "attempt_end":
                    text, tool_map, finish_reason = ev["text"], ev["tool_map"], ev["finish_reason"]

            outcome = self._classify(text, tool_map, finish_reason)

            if outcome[0] == "accept_text":
                if not live:
                    yield {"kind": "text_delta", "text": text}
                self._response_text, self._tool_map = text, {}
                break

            if outcome[0] == "accept_tools":
                if not live and text:
                    yield {"kind": "text_delta", "text": text}
                self._response_text, self._tool_map = text, outcome[1]
                if outcome[2]:
                    self.guardrail_rescues += 1
                    log.info("ollama guardrail rescued model=%s attempt=%d finish_reason=%s",
                             self.model, attempt_no, finish_reason)
                break

            # outcome[0] == "retry"
            tracker.record_retry()
            self.guardrail_retries += 1
            log.warning("ollama guardrail nudge model=%s attempt=%d reason=%s",
                       self.model, attempt_no, outcome[1])
            if tracker.retries_exhausted:
                self.guardrail_exhausted = True
                log.warning("ollama guardrail exhausted model=%s attempts=%d policy=%s",
                           self.model, attempt_no + 1, self.on_exhausted)
                if self.on_exhausted == "escalate":
                    yield {"kind": "provider_error",
                           "message": f"Ollama guardrail exhausted after {attempt_no + 1} attempts: {outcome[1]}"}
                    return
                # best_effort: keep whatever the last attempt produced — mode=local
                # has never hard-failed on a bad response, don't start now.
                self._response_text, self._tool_map = text, tool_map
                break

            attempt_messages = attempt_messages + [
                _ollama_assistant_message(text, tool_map),
                {"role": "user", "content": retry_nudge(text)},
            ]
            attempt_no += 1
            live = False

        tool_use = bool(self._tool_map)
        if tool_use:
            # This endpoint buffers tool calls rather than streaming them, so
            # the UI's start/input/end triplet is emitted once, post-stream.
            for idx, tc in sorted(self._tool_map.items()):
                yield {"kind": "tool_start", "index": idx, "id": tc["id"], "name": tc["name"]}
                yield {"kind": "tool_input_delta", "index": idx, "partial_json": tc["arguments_str"]}
                yield {"kind": "tool_end", "index": idx}
        yield {
            "kind": "turn_end",
            "tool_use": tool_use,
            "tool_calls": [
                {"id": tc["id"], "name": tc["name"], "input": _parse_input(tc["arguments_str"])}
                for _, tc in sorted(self._tool_map.items())
            ],
        }

    def append_assistant_turn(self, messages: list) -> None:
        # Canonical (Anthropic) shape, same as AnthropicProvider — current_messages
        # is the list that later rides the `history` SSE event and may continue
        # on either backend next turn, so it must never carry Ollama-native shape.
        content = []
        if self._response_text:
            content.append({"type": "text", "text": self._response_text})
        for _, tc in sorted(self._tool_map.items()):
            content.append({
                "type": "tool_use", "id": tc["id"], "name": tc["name"],
                "input": _parse_input(tc["arguments_str"]),
            })
        messages.append({"role": "assistant", "content": content})

    def append_tool_results(self, messages: list, results: list) -> None:
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": content}
            for tid, content in results
        ]})

    def history_messages(self, messages: list) -> list:
        if (messages and messages[0].get("role") == "system"
                and messages[0].get("content") == self.system_sentinel):
            return messages[1:]
        return messages

    def final_token_fields(self, elapsed: float) -> dict:
        est = max(self.output_chars // 4, 1)
        return {"input_tokens": 0, "output_tokens": est,
                "tps": round(est / elapsed, 1)}

    def breach_token_fields(self, messages: list) -> dict:
        return {"input_tokens": 0, "output_tokens": self.budget_tokens(messages), "tps": 0}

    async def write_routing(self, ttft_ms: int, elapsed_ms: int) -> None:
        await self._route(0, max(self.output_chars // 4, 1), ttft_ms, elapsed_ms,
                          guardrail_rescues=self.guardrail_rescues,
                          guardrail_retries=self.guardrail_retries,
                          guardrail_exhausted=self.guardrail_exhausted)
