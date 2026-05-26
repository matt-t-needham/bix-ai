import json
import logging

from helpers import sse
from streaming.claude import _stream_claude
from streaming.forge_runner import _stream_forge_runner

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
    """Stream from Forge (Ollama-backed) first; on error, fall back to Claude.

    If at least one `delta` has been emitted before the error, emit a
    `fallback_triggered` event so the UI can clear the partial response.
    Always emit a `status` event with the error reason before the Claude
    stream starts so the user can see why the model switched.
    """
    has_delta    = False
    escalate     = False
    error_reason = "local model error"

    async for chunk in _stream_forge_runner(messages, ollama_model, max_tokens, mode=mode):
        event = _parse_sse_event(chunk)
        if event == "error":
            escalate     = True
            error_reason = _parse_sse_data(chunk).get("message", error_reason)
            log.info("auto mode: forge errored, escalating has_delta=%s reason=%s",
                     has_delta, error_reason)
            if has_delta:
                yield sse("fallback_triggered", {"reason": "forge_error"})
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
        ):
            yield chunk
