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
"""
import json
import logging

from config import ANTHROPIC_URL, OLLAMA_URL
from strategy import estimate_tokens

log = logging.getLogger("router")


def _parse_input(raw: str) -> dict:
    try:
        return json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}


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
                 system_sentinel: str):
        self.model = model
        self.tools = tools
        self.client_factory = client_factory
        self._route = route
        self.system_sentinel = system_sentinel  # injected system msg to strip from history
        self.output_chars = 0
        self._tool_map: dict = {}
        self._response_text = ""

    def stream_status(self) -> str:
        return f"Streaming from Ollama ({self.model})…"

    def budget_tokens(self, messages: list) -> int:
        # No usage on stream chunks — estimate from the growing payload
        # (mirrors strategy.estimate_tokens' chars/4 heuristic).
        return estimate_tokens(json.dumps(messages))

    def _accumulate_tool_call(self, tc: dict) -> None:
        idx = tc.get("index", 0)
        if idx not in self._tool_map:
            self._tool_map[idx] = {"id": "", "name": "", "arguments_str": ""}
        entry = self._tool_map[idx]
        if tc.get("id"):
            entry["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            entry["name"] = fn["name"]
        if fn.get("arguments"):
            entry["arguments_str"] += fn["arguments"]

    async def stream_turn(self, messages: list):
        self._tool_map = {}
        self._response_text = ""
        finish_reason = None

        async with self.client_factory() as client:
            async with client.stream("POST", OLLAMA_URL, json={
                "model": self.model, "messages": messages,
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
                        self._response_text += content
                        self.output_chars   += len(content)
                        yield {"kind": "text_delta", "text": content}

                    for tc in delta.get("tool_calls") or []:
                        self._accumulate_tool_call(tc)

        tool_use = finish_reason == "tool_calls"
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
        messages.append({
            "role":    "assistant",
            "content": self._response_text or None,
            "tool_calls": [
                {
                    "index": idx, "id": tc["id"], "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments_str"]},
                }
                for idx, tc in sorted(self._tool_map.items())
            ],
        })

    def append_tool_results(self, messages: list, results: list) -> None:
        for tid, content in results:
            messages.append({"role": "tool", "tool_call_id": tid, "content": content})

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
        await self._route(0, max(self.output_chars // 4, 1), ttft_ms, elapsed_ms)
