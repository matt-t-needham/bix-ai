import asyncio
import json
import logging
import logging.handlers
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import psutil
from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse, JSONResponse, StreamingResponse, RedirectResponse, Response,
)
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

import blobstore  # noqa: E402 — after logging setup
import model_admin  # noqa: E402 — after logging setup
import routing_dash  # noqa: E402 — after logging setup
import strategy  # noqa: E402 — after logging setup

import config  # noqa: E402
from config import (  # noqa: E402
    ANTHROPIC_API_KEY, ANTHROPIC_URL, BIX_PROXY_SECRET,
    DEFAULT_MODEL, OLLAMA_DEFAULT_MODEL, OLLAMA_HOST, ROUTING_LOG,
    _ALLOWED_ANTHROPIC_BETAS, _ALLOWED_CLAUDE_MODELS,
    _MAX_BODY_BYTES, _MAX_TOKENS_CAP,
)
from helpers import _agg, _claude_session, _write_routing_event, with_keepalive  # noqa: E402
from config import CONV_DIR                                          # noqa: E402
from memory import _load_all_memories, save_memory_entry  # noqa: E402
from streaming.claude import _stream_claude                       # noqa: E402
from streaming.local_first import _stream_local_first             # noqa: E402
from streaming.ollama import _stream_ollama                       # noqa: E402
from streaming.pro import _stream_pro                             # noqa: E402

psutil.cpu_percent()  # prime the psutil cpu counter

app = FastAPI()

# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages:     list[dict[str, Any]]
    model:        str = DEFAULT_MODEL
    max_tokens:   int = _MAX_TOKENS_CAP
    mode:         str = "auto"
    tool_offload: bool = False
    # mode="pro" only: claude-CLI session to resume (from the pro_session SSE
    # event). Preserves tool turns across requests; empty = fresh session.
    session_id:   str = ""

class MemorySaveRequest(BaseModel):
    messages:      list[dict[str, Any]]
    model:         str = DEFAULT_MODEL
    input_tokens:  int = 0
    output_tokens: int = 0

# ── GPU response cache (3 s TTL) ──────────────────────────────────────────────
_gpu_cache: dict = {"data": None, "ts": 0.0}
_GPU_TTL = 3.0


# ── Utility routes ────────────────────────────────────────────────────────────

@app.get("/healthz")
async def healthz():
    return {"ok": True}


_STARTED_AT = datetime.now(timezone.utc).isoformat(timespec="seconds")


@app.get("/version")
async def version():
    # config attrs read at call time so tests can monkeypatch GIT_SHA etc.
    return {
        "git_sha":    config.GIT_SHA,
        "built_at":   config.BUILT_AT,
        "started_at": _STARTED_AT,
    }


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
                r = await client.get(f"{OLLAMA_HOST}/api/ps")
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


# ── Routing dashboard ─────────────────────────────────────────────────────────

@app.get("/routing")
async def routing_dashboard():
    events = await asyncio.to_thread(routing_dash.load_events, ROUTING_LOG)
    agg = routing_dash.aggregate(events)
    return Response(routing_dash.render_html(agg), media_type="text/html; charset=utf-8")


# ── Blob store housekeeping ───────────────────────────────────────────────────

# Shared stylesheet for the small server-rendered pages (Catppuccin Mocha).
_PAGE_CSS = """
:root {
  --bg:#1e1e2e; --surface0:#313244; --surface1:#45475a;
  --text:#cdd6f4; --subtext:#a6adc8;
  --blue:#89b4fa; --mauve:#cba6f7; --green:#a6e3a1; --red:#f38ba8; --yellow:#f9e2af;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:system-ui,-apple-system,sans-serif; background:var(--bg);
  color:var(--text); padding:24px 16px; max-width:980px; margin:0 auto; line-height:1.5;
}
a { color:var(--blue); text-decoration:none; }
a:hover { text-decoration:underline; }
.hdr { border-bottom:1px solid var(--surface1); padding-bottom:14px; margin-bottom:18px; }
.hdr h1 { font-size:1.3rem; font-weight:600; }
.meta { font-size:.78rem; color:var(--subtext); display:flex; gap:14px; flex-wrap:wrap; font-variant-numeric:tabular-nums; }
.row { display:block; padding:12px 14px; margin:8px 0; background:var(--surface0); border-radius:6px; border-left:2px solid var(--surface1); }
.row .path { color:var(--text); font-size:.9rem; overflow-wrap:anywhere; }
.row .sub { color:var(--subtext); font-size:.74rem; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }
.badge { font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; padding:1px 7px; border-radius:3px; background:var(--surface1); color:var(--subtext); }
.badge.pending { color:var(--yellow); }
.actions { display:flex; gap:10px; margin:18px 0; flex-wrap:wrap; align-items:center; }
.actions button, .btn-reject { font:inherit; font-size:.85rem; padding:7px 16px; border:none; border-radius:5px; cursor:pointer; color:var(--bg); min-height:36px; }
.btn-reject { background:var(--red); }
.note { color:var(--subtext); font-size:.85rem; font-style:italic; padding:10px 0; }
"""

_BLOB_HASH_RE = re.compile(r"[0-9a-f]{64}")


def _render_blob_list(blobs: list[dict]) -> str:
    total = sum(b["bytes"] for b in blobs)
    cap = config.BLOB_STORE_MAX_BYTES
    pct = min(total / cap * 100, 100) if cap else 0
    bar_cls = "crit" if pct > 85 else "warn" if pct > 70 else ""
    if not blobs:
        body = '<div class="note">Blob store is empty.</div>'
    else:
        rows = []
        for b in blobs:
            age = datetime.fromtimestamp(b["mtime"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            pin_badge = ' <span class="badge pending">pinned</span>' if b["pinned"] else ""
            action = (
                '<span class="sub">in use</span>' if b["pinned"] else
                f'<form method="post" action="/blobs/{_esc(b["hash"])}/delete" '
                f'style="display:inline"><button class="btn-reject btn-sm">Delete</button></form>'
            )
            rows.append(
                f'<div class="row">'
                f'<div class="path"><code>{_esc(b["hash"][:16])}…</code>'
                f' · {b["bytes"]:,} B{pin_badge} <span style="float:right">{action}</span></div>'
                f'<div class="sub">{_esc(age)} UTC · {_esc(b["preview"])}</div>'
                f'</div>'
            )
        body = "\n".join(rows)
    purge = (
        '<div class="actions"><form method="post" action="/blobs/purge">'
        '<button class="btn-reject">Purge all unpinned</button></form></div>'
        if blobs else ""
    )
    extra_css = """
.bar-wrap { background:var(--surface1); border-radius:4px; height:8px; margin:10px 0 4px; }
.bar-fill { background:var(--green); height:100%; border-radius:4px; min-width:2px; }
.bar-fill.warn { background:var(--yellow); } .bar-fill.crit { background:var(--red); }
.btn-sm { font-size:.72rem !important; padding:3px 10px !important; }
code { font-size:.82em; }
"""
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<title>Blob store</title><style>{_PAGE_CSS}{extra_css}</style></head><body>'
        f'<div class="hdr"><h1>Blob store</h1>'
        f'<div class="meta"><span>{len(blobs)} blob(s)</span>'
        f'<span>{total:,} of {cap:,} bytes ({pct:.1f}%)</span>'
        f'<span><a href="/">← chat</a></span></div>'
        f'<div class="bar-wrap"><div class="bar-fill {bar_cls}" style="width:{max(pct, 0.5):.1f}%"></div></div>'
        f'</div>{purge}{body}</body></html>'
    )


@app.get("/blobs")
async def blob_list():
    blobs = await asyncio.to_thread(blobstore.list_blobs)
    return Response(_render_blob_list(blobs), media_type="text/html; charset=utf-8")


@app.post("/blobs/purge")
async def blob_purge():
    result = await asyncio.to_thread(blobstore.purge_unpinned)
    log.info("blob purge deleted=%d freed=%d", result["deleted"], result["freed_bytes"])
    return RedirectResponse("/blobs", status_code=303)


@app.post("/blobs/{blob_hash}/delete")
async def blob_delete(blob_hash: str):
    if not _BLOB_HASH_RE.fullmatch(blob_hash):
        return Response("<h1>Bad blob hash</h1>", status_code=400, media_type="text/html")
    ok = await asyncio.to_thread(blobstore.delete, blob_hash)
    log.info("blob delete hash=%s ok=%s", blob_hash[:16], ok)
    return RedirectResponse("/blobs", status_code=303)


# ── Ollama model management ───────────────────────────────────────────────────

class ModelNameRequest(BaseModel):
    # POST body (not a path param) because model names contain ':' and '/'.
    name: str


@app.get("/models/ollama")
async def models_ollama():
    try:
        models = await model_admin.list_installed()
    except (httpx.HTTPError, OSError, ValueError) as e:
        log.warning("model list failed: %s", e)
        return JSONResponse({"ok": False, "message": f"Ollama unreachable: {e}"},
                            status_code=502)
    return {"ok": True, "models": models}


@app.post("/models/ollama/pull")
async def models_ollama_pull(body: ModelNameRequest):
    try:
        repo, tag = model_admin.validate_model_name(body.name)
    except ValueError as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=400)
    # Manifest pre-check: typos fail fast with a clear message instead of an
    # Ollama pull error minutes in.
    try:
        exists = await model_admin.registry_manifest_exists(repo, tag)
    except Exception as e:
        log.warning("registry check failed for %s:%s: %s", repo, tag, e)
        return JSONResponse({"ok": False, "message": f"registry check failed: {e}"},
                            status_code=502)
    if not exists:
        return JSONResponse(
            {"ok": False,
             "message": f"'{body.name}' not found in the Ollama registry — "
                        "browse ollama.com/library for available models"},
            status_code=404)
    log.info("model pull start name=%s", body.name)
    return StreamingResponse(with_keepalive(model_admin.pull_events(body.name)),
                             media_type="text/event-stream")


@app.post("/models/ollama/delete")
async def models_ollama_delete(body: ModelNameRequest):
    try:
        model_admin.validate_model_name(body.name)
    except ValueError as e:
        return JSONResponse({"ok": False, "message": str(e)}, status_code=400)
    try:
        ok, message = await model_admin.delete_model(body.name)
    except (httpx.HTTPError, OSError) as e:
        return JSONResponse({"ok": False, "message": f"Ollama unreachable: {e}"},
                            status_code=502)
    log.info("model delete name=%s ok=%s", body.name, ok)
    return JSONResponse({"ok": ok, "message": message}, status_code=200 if ok else 502)


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

    if mode in ("api", "pro", "auto") and model not in _ALLOWED_CLAUDE_MODELS:
        return Response(
            status_code=400,
            content=json.dumps({"error": f"Model not permitted: {model}"}),
            media_type="application/json",
        )
    # Dynamic allowlist: any installed Ollama model (static-config fallback
    # inside is_allowed_local_model when Ollama is down).
    if mode == "local" and not await model_admin.is_allowed_local_model(model):
        return Response(
            status_code=400,
            content=json.dumps({"error": f"Model not permitted: {model}"}),
            media_type="application/json",
        )

    log.info("chat mode=%s model=%s msgs=%d", mode, model, len(messages))

    async def stream():
        # Blobs referenced by this request must survive LRU eviction until the
        # stream finishes (a concurrent request's spill could otherwise evict
        # them mid-loop). Blobs spilled *during* this request are mtime-newest,
        # so LRU never reaches them within the request.
        pinned = strategy.referenced_blob_hashes(messages)
        blobstore.pin(pinned)
        try:
            if mode == "local":
                async for event in _stream_ollama(messages, model, mode=mode, tool_offload=tool_offload):
                    yield event
            elif mode == "api":
                async for event in _stream_claude(messages, model, max_tokens, skip_preprocess=False, mode=mode):
                    yield event
            elif mode == "auto":
                async for event in _stream_local_first(
                    messages, OLLAMA_DEFAULT_MODEL, model, max_tokens, mode=mode,
                ):
                    yield event
            else:
                async for event in _stream_pro(messages, model, max_tokens, mode=mode,
                                               session_id=body.session_id):
                    yield event
        finally:
            blobstore.unpin(pinned)

    return StreamingResponse(with_keepalive(stream()), media_type="text/event-stream")


# ── API proxy route ───────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def v1_messages(request: Request):
    # Defence-in-depth: require a shared secret as x-api-key, then substitute
    # the real ANTHROPIC_API_KEY. Cloudflare Access is the primary gate; this
    # ensures the API key cannot be drained even if the loopback port is ever
    # reachable on LAN or Access misroutes.
    if BIX_PROXY_SECRET:
        client_key = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(client_key, BIX_PROXY_SECRET):
            log.warning("v1_proxy auth failed remote=%s", request.client.host if request.client else "?")
            return Response(
                status_code=401,
                content=json.dumps({"type": "error", "error": {
                    "type": "authentication_error", "message": "Unauthorized"}}),
                media_type="application/json",
            )

    raw = await request.body()
    if len(raw) > _MAX_BODY_BYTES:
        return Response(
            status_code=413,
            content=json.dumps({"type": "error", "error": {
                "type": "invalid_request_error", "message": "Request body too large"}}),
            media_type="application/json",
        )
    try:
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
    except (ValueError, json.JSONDecodeError):
        return Response(
            status_code=400,
            content=json.dumps({"type": "error", "error": {
                "type": "invalid_request_error", "message": "Invalid JSON body"}}),
            media_type="application/json",
        )
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

    # Filter anthropic-beta values against the server-side allowlist. Each
    # client header value may itself be a comma-separated list per Anthropic's API.
    forwarded_betas = []
    for k, v in request.headers.items():
        if k.lower() != "anthropic-beta":
            continue
        for beta in (s.strip() for s in v.split(",")):
            if beta and beta in _ALLOWED_ANTHROPIC_BETAS:
                forwarded_betas.append(beta)
    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    if forwarded_betas:
        headers["anthropic-beta"] = ",".join(forwarded_betas)

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
