import asyncio
import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any

import httpx
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel

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
from config import CONV_DIR                                          # noqa: E402
from memory import _load_all_memories, _summarize, save_memory_entry  # noqa: E402
from streaming.claude import _stream_claude                       # noqa: E402
from streaming.ollama import _stream_ollama                       # noqa: E402
from streaming.pro import _stream_pro                             # noqa: E402

psutil.cpu_percent()  # prime the psutil cpu counter

app = FastAPI()

# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages:     list[dict[str, Any]]
    model:        str = DEFAULT_MODEL
    max_tokens:   int = 4096
    mode:         str = "pro"
    tool_offload: bool = False

class MemorySaveRequest(BaseModel):
    messages:      list[dict[str, Any]]
    model:         str = DEFAULT_MODEL
    input_tokens:  int = 0
    output_tokens: int = 0

class SummarizeRequest(BaseModel):
    user_msg:      str = ""
    assistant_msg: str = ""
    local_model:   str = OLLAMA_DEFAULT_MODEL

# ── GPU response cache (3 s TTL) ──────────────────────────────────────────────
_gpu_cache: dict = {"data": None, "ts": 0.0}
_GPU_TTL = 3.0


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
    now = time.monotonic()
    if _gpu_cache["data"] is not None and now - _gpu_cache["ts"] < _GPU_TTL:
        result["gpu"] = _gpu_cache["data"]
    else:
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
        _gpu_cache["data"] = result["gpu"]
        _gpu_cache["ts"]   = now
    return result


# ── Memory routes ─────────────────────────────────────────────────────────────

@app.post("/memory/save")
async def save_memory_handler(body: MemorySaveRequest):
    return await save_memory_entry(
        messages   = body.messages,
        model_name = body.model,
        in_tokens  = body.input_tokens,
        out_tokens = body.output_tokens,
    )


@app.get("/memory")
async def get_memory():
    all_m = await asyncio.to_thread(_load_all_memories)
    return {"entries": list(reversed(all_m)), "count": len(all_m)}


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def _render_memory_html(entry: dict, convo: dict | None) -> str:
    title = _esc(entry.get("title") or "—")
    date  = _esc((entry.get("date") or "")[:19].replace("T", " "))
    model = _esc(entry.get("model") or "—")
    in_t  = entry.get("input_tokens") or 0
    out_t = entry.get("output_tokens") or 0
    tags  = " ".join(
        f'<span class="tag">{_esc(t)}</span>' for t in (entry.get("tags") or [])
    )
    summary = _esc(entry.get("summary") or "")

    msgs_html = ""
    if convo and isinstance(convo.get("messages"), list):
        for m in convo["messages"]:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict):
                        if c.get("type") == "text":
                            parts.append(c.get("text", ""))
                        elif c.get("type") == "tool_use":
                            parts.append(f"[tool_use: {c.get('name','?')}({json.dumps(c.get('input', {}))})]")
                        elif c.get("type") == "tool_result":
                            tc = c.get("content", "")
                            if isinstance(tc, list):
                                tc = "\n".join(p.get("text", "") for p in tc if isinstance(p, dict))
                            parts.append(f"[tool_result] {tc}")
                content = "\n".join(parts)
            msgs_html += (
                f'<div class="msg msg-{_esc(role)}">'
                f'<div class="role">{_esc(role)}</div>'
                f'<div class="body">{_esc(content)}</div>'
                f'</div>'
            )
    elif convo is None:
        msgs_html = '<div class="note">Conversation file unavailable — summary only.</div>'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{title} — memory</title>
<style>
:root {{
  --bg:#1e1e2e; --surface0:#313244; --surface1:#45475a;
  --text:#cdd6f4; --subtext:#a6adc8;
  --blue:#89b4fa; --mauve:#cba6f7; --green:#a6e3a1;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
body {{
  font-family:system-ui,-apple-system,sans-serif; background:var(--bg);
  color:var(--text); padding:32px 24px; max-width:880px; margin:0 auto;
  line-height:1.5;
}}
.hdr {{ border-bottom:1px solid var(--surface1); padding-bottom:14px; margin-bottom:18px; }}
.hdr h1 {{ font-size:1.3rem; font-weight:600; color:var(--text); margin-bottom:6px; }}
.meta {{ font-size:.78rem; color:var(--subtext); display:flex; gap:14px; flex-wrap:wrap; font-variant-numeric:tabular-nums; }}
.tag {{ background:var(--surface1); color:var(--subtext); padding:1px 7px; border-radius:3px; font-size:.7rem; margin-right:3px; }}
.summary {{ background:var(--surface0); border-left:2px solid var(--mauve); padding:10px 14px; margin:14px 0; font-size:.86rem; color:var(--subtext); border-radius:0 4px 4px 0; }}
.msg {{ margin:14px 0; padding:10px 14px; border-radius:6px; background:var(--surface0); }}
.msg-user {{ border-left:2px solid var(--mauve); }}
.msg-assistant {{ border-left:2px solid var(--blue); }}
.msg-tool, .msg-system {{ border-left:2px solid var(--surface1); opacity:.85; }}
.role {{ font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; color:var(--subtext); margin-bottom:5px; }}
.body {{ font-size:.85rem; white-space:pre-wrap; word-wrap:break-word; color:var(--text); }}
.note {{ color:var(--subtext); font-size:.85rem; font-style:italic; padding:14px 0; }}
</style></head>
<body>
<div class="hdr">
  <h1>{title}</h1>
  <div class="meta">
    <span>{date}</span>
    <span>{model}</span>
    <span>in: {in_t:,}</span>
    <span>out: {out_t:,}</span>
    {('<span>' + tags + '</span>') if tags else ''}
  </div>
</div>
{f'<div class="summary">{summary}</div>' if summary else ''}
{msgs_html}
</body></html>"""


@app.get("/memory/{entry_id}")
async def get_memory_entry(entry_id: str):
    all_m = await asyncio.to_thread(_load_all_memories)
    entry = next((m for m in all_m if m.get("id") == entry_id), None)
    if not entry:
        return Response(
            "<h1>Memory not found</h1>",
            status_code=404, media_type="text/html",
        )

    convo = None
    fname = entry.get("file")
    if fname:
        conv_path = CONV_DIR / fname
        if conv_path.exists():
            try:
                convo = json.loads(conv_path.read_text())
            except (OSError, ValueError) as e:
                log.warning("memory conv read failed id=%s err=%s", entry_id, e)

    html = _render_memory_html(entry, convo)
    return Response(html, media_type="text/html; charset=utf-8")


# ── Summarize route ───────────────────────────────────────────────────────────

@app.post("/summarize")
async def summarize(body: SummarizeRequest):
    summary, source = await _summarize(body.user_msg, body.assistant_msg, body.local_model)
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
    try:
        body = ChatRequest.model_validate_json(raw)
    except Exception as e:
        return Response(
            status_code=400,
            content=json.dumps({"error": f"Invalid request: {e}"}),
            media_type="application/json",
        )
    messages     = body.messages
    model        = body.model
    max_tokens   = min(body.max_tokens, _MAX_TOKENS_CAP)
    mode         = body.mode
    tool_offload = body.tool_offload

    if mode in ("api", "pro") and model not in _ALLOWED_CLAUDE_MODELS:
        return Response(
            status_code=400,
            content=json.dumps({"error": f"Model not permitted: {model}"}),
            media_type="application/json",
        )

    log.info("chat mode=%s model=%s msgs=%d", mode, model, len(messages))

    async def stream():
        if mode == "local":
            async for event in _stream_ollama(messages, model, mode=mode, tool_offload=tool_offload):
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
