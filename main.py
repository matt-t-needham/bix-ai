import asyncio
import json
import logging
import logging.handlers
from pathlib import Path

import httpx
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, Response

# Logging must be configured before local imports so all modules inherit handlers.
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

import strategy  # noqa: E402 — after logging setup

from config import (  # noqa: E402
    ANTHROPIC_API_KEY, ANTHROPIC_URL,
    DEFAULT_MODEL, OLLAMA_DEFAULT_MODEL,
    _ALLOWED_CLAUDE_MODELS, _MAX_BODY_BYTES, _MAX_TOKENS_CAP,
)
from helpers import _agg, _claude_session, _write_routing_event  # noqa: E402
from memory import _load_all_memories, _summarize, save_memory_entry  # noqa: E402
from streaming.claude import _stream_claude                       # noqa: E402
from streaming.ollama import _stream_ollama                       # noqa: E402
from streaming.pro import _stream_pro                             # noqa: E402

psutil.cpu_percent()  # prime the psutil cpu counter

app = FastAPI()


# ── Utility routes ────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/stats")
async def router_stats():
    return dict(_agg)


@app.get("/auth/status")
async def auth_status():
    session = await asyncio.to_thread(_claude_session)
    return {
        "claude_code_session": session["logged_in"],
        "expires_at":          session["expires_at"],
        "subscription_type":   session["subscription_type"],
        "has_api_key":         bool(ANTHROPIC_API_KEY),
    }


@app.get("/system")
async def system_metrics():
    mem    = psutil.virtual_memory()
    result = {
        "cpu_percent":  round(psutil.cpu_percent(interval=None), 1),
        "ram_used_gb":  round(mem.used  / 1e9, 1),
        "ram_total_gb": round(mem.total / 1e9, 1),
        "ram_percent":  round(mem.percent, 1),
        "gpu":          {"state": "unreachable"},
    }
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get("http://host.docker.internal:11434/api/ps")
            if r.status_code == 200:
                models = r.json().get("models") or []
                if models:
                    m = models[0]
                    result["gpu"] = {
                        "state":      "loaded",
                        "model":      m.get("name", ""),
                        "num_gpu":    m.get("num_gpu"),
                        "size_vram":  m.get("size_vram", 0),
                        "size_total": m.get("size", 0),
                    }
                else:
                    result["gpu"] = {"state": "idle"}
    except Exception:
        pass
    return result


# ── Memory routes ─────────────────────────────────────────────────────────────

@app.post("/memory/save")
async def save_memory_handler(request: Request):
    body = await request.json()
    return await save_memory_entry(
        messages   = body.get("messages", []),
        model_name = body.get("model", DEFAULT_MODEL),
        in_tokens  = body.get("input_tokens", 0),
        out_tokens = body.get("output_tokens", 0),
    )


@app.get("/memory")
async def get_memory():
    all_m = await asyncio.to_thread(_load_all_memories)
    return {"entries": list(reversed(all_m)), "count": len(all_m)}


# ── Summarize route ───────────────────────────────────────────────────────────

@app.post("/summarize")
async def summarize(request: Request):
    body = await request.json()
    summary, source = await _summarize(
        body.get("user_msg", ""),
        body.get("assistant_msg", ""),
        body.get("local_model", OLLAMA_DEFAULT_MODEL),
    )
    return {"summary": summary, "source": source}


# ── Static ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse("static/index.html")


# ── Chat route ────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(request: Request):
    raw = await request.body()
    if len(raw) > _MAX_BODY_BYTES:
        return Response(
            status_code=413,
            content=json.dumps({"error": "Request body too large"}),
            media_type="application/json",
        )
    body       = json.loads(raw)
    messages   = body.get("messages", [])
    model      = body.get("model", DEFAULT_MODEL)
    max_tokens = min(body.get("max_tokens", 4096), _MAX_TOKENS_CAP)
    mode       = body.get("mode", "pro")

    if mode in ("api", "pro") and model not in _ALLOWED_CLAUDE_MODELS:
        return Response(
            status_code=400,
            content=json.dumps({"error": f"Model not permitted: {model}"}),
            media_type="application/json",
        )

    log.info("chat mode=%s model=%s msgs=%d", mode, model, len(messages))

    async def stream():
        if mode == "local":
            async for event in _stream_ollama(messages, model, mode=mode):
                yield event
        elif mode == "api":
            async for event in _stream_claude(messages, model, max_tokens, skip_preprocess=False, mode=mode):
                yield event
        else:
            async for event in _stream_pro(messages, model, max_tokens, mode=mode):
                yield event

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── API proxy route ───────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def v1_messages(request: Request):
    raw = await request.body()
    if len(raw) > _MAX_BODY_BYTES:
        return Response(
            status_code=413,
            content=json.dumps({"type": "error", "error": {
                "type": "invalid_request_error", "message": "Request body too large"}}),
            media_type="application/json",
        )
    body  = json.loads(raw)
    model = body.get("model", "")
    if model not in _ALLOWED_CLAUDE_MODELS:
        return Response(
            status_code=400,
            content=json.dumps({"type": "error", "error": {
                "type": "invalid_request_error", "message": f"Model not permitted: {model}"}}),
            media_type="application/json",
        )
    if body.get("max_tokens", 0) > _MAX_TOKENS_CAP:
        body = {**body, "max_tokens": _MAX_TOKENS_CAP}
    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
        **{k: v for k, v in request.headers.items()
           if k.lower().startswith("anthropic-beta")},
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(ANTHROPIC_URL, json=body, headers=headers)
    try:
        usage = r.json().get("usage", {})
        await _write_routing_event("v1_proxy", model,
                                   input_tokens=usage.get("input_tokens", 0),
                                   output_tokens=usage.get("output_tokens", 0))
    except Exception as e:
        log.warning("v1_proxy routing log failed: %s", e)
        await _write_routing_event("v1_proxy", model)
    return Response(
        content    = r.content,
        status_code= r.status_code,
        media_type = r.headers.get("content-type", "application/json"),
    )
