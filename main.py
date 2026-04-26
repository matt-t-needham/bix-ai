import json
import logging
import logging.handlers
import os
import time
from pathlib import Path

import httpx
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse

import strategy

# ── Logging ───────────────────────────────────────────────────────────────────
_fmt = "%(asctime)s %(levelname)s %(name)s %(message)s"
logging.basicConfig(level=logging.INFO, format=_fmt)
log = logging.getLogger("router")

_log_dir = Path("logs")
try:
    _log_dir.mkdir(exist_ok=True)
    _fh = logging.handlers.RotatingFileHandler(
        _log_dir / "app.log", maxBytes=10_000_000, backupCount=3,
    )
    _fh.setFormatter(logging.Formatter(_fmt))
    logging.getLogger().addHandler(_fh)
except Exception:
    pass

# ── Config ────────────────────────────────────────────────────────────────────
app = FastAPI()

ANTHROPIC_URL        = "https://api.anthropic.com/v1/messages"
OLLAMA_URL           = "http://host.docker.internal:11434/v1/chat/completions"
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL        = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-6")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

psutil.cpu_percent()  # prime the cpu counter


# ── Helpers ───────────────────────────────────────────────────────────────────
async def ollama_chat(model: str, messages: list, timeout: float = 120.0) -> str:
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(OLLAMA_URL, json={
            "model": model, "messages": messages, "stream": False,
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ── Stream helpers ────────────────────────────────────────────────────────────
async def _stream_ollama(messages: list, model: str):
    yield sse("status", {"stage": "streaming", "message": f"Streaming from Ollama ({model})…"})
    start = time.monotonic()
    ttft_ms = None
    output_chars = 0
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json={
                "model": model, "messages": messages, "stream": True,
            }) as r:
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
                    content = (data.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
                    if content:
                        if ttft_ms is None:
                            ttft_ms = round((time.monotonic() - start) * 1000)
                        output_chars += len(content)
                        yield sse("delta", {"text": content})

        elapsed = time.monotonic() - start
        est_tokens = max(output_chars // 4, 1)
        yield sse("metrics", {
            "input_tokens": 0,
            "output_tokens": est_tokens,
            "elapsed_ms": round(elapsed * 1000),
            "ttft_ms": ttft_ms or 0,
            "preprocess_ms": 0,
            "tps": round(est_tokens / elapsed, 1),
            "summarised": 0,
        })
        yield sse("done", {})
    except Exception as e:
        log.error("ollama stream error: %s", e)
        yield sse("error", {"message": str(e)})


async def _stream_claude(messages: list, model: str, max_tokens: int, skip_preprocess: bool):
    req_body = {"model": model, "max_tokens": max_tokens, "messages": messages}
    stats = {"summarised": 0, "skipped": 0, "failed": 0}
    preprocess_ms = 0

    if skip_preprocess:
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})
    else:
        if strategy.has_oversized_blocks(req_body):
            yield sse("status", {"stage": "summarising", "message": "Summarising via Ollama…"})
        else:
            yield sse("status", {"stage": "checking", "message": "Checking…"})
        t0 = time.monotonic()
        try:
            req_body, stats = await strategy.preprocess(req_body, ollama_chat)
            log.info("preprocess summarised=%d skipped=%d failed=%d",
                     stats["summarised"], stats["skipped"], stats["failed"])
        except Exception as e:
            log.warning("preprocess error: %s", e)
        preprocess_ms = round((time.monotonic() - t0) * 1000)
        yield sse("preprocess", {
            "summarised":    stats["summarised"],
            "skipped":       stats["skipped"],
            "failed":        stats["failed"],
            "preprocess_ms": preprocess_ms,
        })
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    start = time.monotonic()
    ttft_ms = None
    input_tokens = output_tokens = 0

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", ANTHROPIC_URL,
                json={**req_body, "stream": True},
                headers=headers,
            ) as r:
                event_type    = None
                content_blocks = {}  # index -> {"type", "name", "id"}
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
                            input_tokens = data.get("message", {}).get("usage", {}).get("input_tokens", 0)
                            yield sse("input_tokens", {"count": input_tokens})
                        elif event_type == "content_block_start":
                            idx   = data.get("index", 0)
                            block = data.get("content_block", {})
                            content_blocks[idx] = {"type": block.get("type"), "name": block.get("name", ""), "id": block.get("id", "")}
                            if block.get("type") == "tool_use":
                                yield sse("tool_start", {"index": idx, "name": block.get("name", ""), "id": block.get("id", "")})
                        elif event_type == "content_block_delta":
                            idx   = data.get("index", 0)
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                if ttft_ms is None:
                                    ttft_ms = round((time.monotonic() - start) * 1000)
                                yield sse("delta", {"text": delta["text"]})
                            elif delta.get("type") == "input_json_delta":
                                yield sse("tool_input", {"index": idx, "partial_json": delta.get("partial_json", "")})
                        elif event_type == "content_block_stop":
                            idx = data.get("index", 0)
                            if content_blocks.get(idx, {}).get("type") == "tool_use":
                                yield sse("tool_end", {"index": idx})
                        elif event_type == "message_delta":
                            output_tokens = data.get("usage", {}).get("output_tokens", 0)
                        elif event_type == "message_stop":
                            elapsed = time.monotonic() - start
                            tps = output_tokens / elapsed if elapsed > 0 else 0
                            yield sse("metrics", {
                                "input_tokens":  input_tokens,
                                "output_tokens": output_tokens,
                                "elapsed_ms":    round(elapsed * 1000),
                                "ttft_ms":       ttft_ms or 0,
                                "preprocess_ms": preprocess_ms,
                                "tps":           round(tps, 1),
                                "summarised":    stats["summarised"],
                                "skipped":       stats["skipped"],
                                "failed":        stats["failed"],
                            })
                            yield sse("done", {})
    except Exception as e:
        log.error("upstream error: %s", e)
        yield sse("error", {"message": str(e)})


# ── Summarisation helper ─────────────────────────────────────────────────────
async def _summarize(user_msg: str, assistant_msg: str, local_model: str) -> tuple[str, str]:
    """Return (summary, source). Tries Haiku first, falls back to local Ollama."""
    prompt = (
        "Give a 6-word title for this exchange. No punctuation, no quotes.\n\n"
        f"User: {user_msg[:400]}\nAssistant: {assistant_msg[:400]}"
    )
    if ANTHROPIC_API_KEY:
        try:
            headers = {
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(ANTHROPIC_URL, json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 20,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                }, headers=headers)
                r.raise_for_status()
                return r.json()["content"][0]["text"].strip(), "haiku"
        except Exception as e:
            log.warning("haiku summarize failed: %s", e)
    try:
        summary = await ollama_chat(local_model, [{"role": "user", "content": prompt}], timeout=30)
        return summary.strip(), "local"
    except Exception as e:
        log.warning("local summarize failed: %s", e)
        return "", "error"


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/system")
async def system_metrics():
    mem = psutil.virtual_memory()
    result = {
        "cpu_percent":  round(psutil.cpu_percent(interval=None), 1),
        "ram_used_gb":  round(mem.used  / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "ram_percent":  round(mem.percent, 1),
        "gpu":          None,
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://host.docker.internal:11434/api/ps")
            if r.status_code == 200:
                models = r.json().get("models") or []
                if models:
                    m = models[0]
                    result["gpu"] = {
                        "model":      m.get("name", ""),
                        "num_gpu":    m.get("num_gpu"),   # None if Ollama version omits it
                        "size_vram":  m.get("size_vram", 0),
                        "size_total": m.get("size", 0),
                    }
    except Exception:
        pass
    return result


@app.post("/summarize")
async def summarize(request: Request):
    body = await request.json()
    summary, source = await _summarize(
        body.get("user_msg", ""),
        body.get("assistant_msg", ""),
        body.get("local_model", OLLAMA_DEFAULT_MODEL),
    )
    return {"summary": summary, "source": source}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/chat")
async def chat(request: Request):
    body       = await request.json()
    messages   = body.get("messages", [])
    model      = body.get("model", DEFAULT_MODEL)
    max_tokens = body.get("max_tokens", 4096)
    mode       = body.get("mode", "auto")

    log.info("chat mode=%s model=%s msgs=%d", mode, model, len(messages))

    async def stream():
        if mode == "local":
            async for event in _stream_ollama(messages, model):
                yield event
        else:
            async for event in _stream_claude(messages, model, max_tokens, skip_preprocess=(mode == "claude")):
                yield event

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/v1/messages")
async def messages(request: Request):
    body    = await request.json()
    api_key = request.headers.get("x-api-key", "")
    version = request.headers.get("anthropic-version", "2023-06-01")
    beta    = request.headers.get("anthropic-beta")

    log.info("recv model=%s msgs=%d", body.get("model"), len(body.get("messages", [])))

    try:
        body, stats = await strategy.preprocess(body, ollama_chat)
        log.info("preprocess summarised=%d skipped=%d failed=%d",
                 stats["summarised"], stats["skipped"], stats["failed"])
    except Exception as e:
        log.warning("preprocess errored, forwarding original: %s", e)

    body = {**body, "stream": True}
    headers = {
        "x-api-key": api_key,
        "anthropic-version": version,
        "content-type": "application/json",
    }
    if beta:
        headers["anthropic-beta"] = beta

    async def upstream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", ANTHROPIC_URL, json=body, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(upstream(), media_type="text/event-stream")
