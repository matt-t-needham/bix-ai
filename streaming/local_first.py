import logging

from helpers import sse
from streaming.claude import _stream_claude
from streaming.ollama import _stream_ollama

log = logging.getLogger("router")


def _parse_sse_event(raw: str) -> str:
    for line in raw.split("\n"):
        if line.startswith("event:"):
            return line[6:].strip()
    return ""


async def _stream_local_first(
    messages: list, ollama_model: str, claude_model: str,
    max_tokens: int, mode: str = "auto",
):
    """Stream from Ollama first; on error, fall back to Claude.

    If at least one `delta` has been emitted before the error, emit a
    `fallback_triggered` event so the UI can clear the partial response.
    Always emit a `status` event before the Claude stream starts so the
    user can see the model switch.
    """
    has_delta = False
    escalate  = False

    async for chunk in _stream_ollama(messages, ollama_model, mode=mode):
        event = _parse_sse_event(chunk)
        if event == "error":
            escalate = True
            log.info("auto mode: ollama errored, escalating has_delta=%s", has_delta)
            if has_delta:
                yield sse("fallback_triggered", {"reason": "ollama_error"})
            break
        if event == "delta":
            has_delta = True
        yield chunk

    if escalate:
        yield sse("status", {"stage": "streaming", "message": "Switching to Claude…"})
        async for chunk in _stream_claude(
            messages, claude_model, max_tokens,
            skip_preprocess=True, mode="auto_fallback",
        ):
            yield chunk
