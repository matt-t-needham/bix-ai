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
from fastapi.staticfiles import StaticFiles
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
import context  # noqa: E402 — after logging setup
import conversations  # noqa: E402 — after logging setup
import model_admin  # noqa: E402 — after logging setup
import openrouter  # noqa: E402 — after logging setup
import strategy  # noqa: E402 — after logging setup

import config  # noqa: E402
from config import (  # noqa: E402
    ANTHROPIC_API_KEY, ANTHROPIC_URL, BIX_PROXY_SECRET,
    DEFAULT_MODEL, OLLAMA_HOST,
    _ALLOWED_ANTHROPIC_BETAS, _ALLOWED_CLAUDE_MODELS,
    _MAX_BODY_BYTES, _MAX_TOKENS_CAP,
)
from helpers import (  # noqa: E402
    _agg, _claude_session, _write_routing_event, anthropic_text, ollama_chat,
    parse_sse_data, parse_sse_event, sse, with_keepalive,
)
from config import CONV_DIR                                          # noqa: E402
from memory import _load_all_memories, save_memory_entry  # noqa: E402
from streaming.claude import _stream_claude                       # noqa: E402
from streaming.ollama import _stream_ollama                       # noqa: E402
from streaming.openrouter import _stream_openrouter               # noqa: E402
from streaming.pro import _stream_pro                             # noqa: E402

psutil.cpu_percent()  # prime the psutil cpu counter

app = FastAPI()

# ── Request models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    # With conversation_id set, `messages` is only the NEW user turn(s); the
    # server owns the history (conversations.py) and decides how much of it
    # rides along (context.py). "new" creates a conversation; the id comes
    # back on the `conversation` SSE event. Empty = stateless legacy call
    # where `messages` is the whole transcript.
    messages:        list[dict[str, Any]]
    conversation_id: str = ""
    model:           str = DEFAULT_MODEL
    max_tokens:      int = _MAX_TOKENS_CAP
    mode:            str = "local"
    # mode="pro" only: claude-CLI session to resume (from the pro_session SSE
    # event). Preserves tool turns across requests; empty = fresh session.
    # With a conversation_id the server remembers it too.
    session_id:      str = ""

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


# Pinned third-party JS/CSS/fonts (Markdown, sanitiser, math) — see
# static/vendor/README.md. Only this directory is served, not static/ itself.
app.mount("/static/vendor", StaticFiles(directory=Path(__file__).parent / "static" / "vendor"),
          name="vendor")


# ── Conversations ─────────────────────────────────────────────────────────────

@app.get("/conversations")
async def conversations_list():
    return {"conversations": await asyncio.to_thread(conversations.list_all)}


@app.get("/conversations/{conv_id}")
async def conversation_get(conv_id: str):
    rec = await asyncio.to_thread(conversations.get, conv_id)
    if rec is None:
        return JSONResponse({"ok": False, "message": "not found"}, status_code=404)
    return rec


@app.delete("/conversations/{conv_id}")
async def conversation_delete(conv_id: str):
    ok = await asyncio.to_thread(conversations.delete, conv_id)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


@app.post("/conversations/{conv_id}/remember")
async def conversation_remember(conv_id: str):
    """Explicit memory save for one conversation (replaces the old per-turn
    background save). One local-model call for the summary."""
    rec = await asyncio.to_thread(conversations.get, conv_id)
    if rec is None:
        return JSONResponse({"ok": False, "message": "not found"}, status_code=404)
    if len(rec.get("messages") or []) < 2:
        return JSONResponse({"ok": False, "message": "nothing to remember yet"}, status_code=400)
    return await save_memory_entry(
        messages=rec["messages"], model_name=rec.get("model", ""), in_tokens=0, out_tokens=0,
    )


# ── OpenRouter ────────────────────────────────────────────────────────────────

@app.get("/models/openrouter")
async def models_openrouter():
    if not openrouter.configured():
        return JSONResponse({"ok": False, "message": "OPENROUTER_API_KEY not set"}, status_code=503)
    models, source = await openrouter.list_models()
    spent = await asyncio.to_thread(openrouter.spent_today_usd)
    return {
        "ok": True, "source": source, "models": models,
        "default": config.OPENROUTER_DEFAULT_MODEL,
        "spend": {
            "today_usd":       spent,
            "day_cap_usd":     config.OPENROUTER_MAX_USD_PER_DAY,
            "request_cap_usd": config.OPENROUTER_MAX_USD_PER_REQUEST,
        },
    }


@app.get("/openrouter/usage")
async def openrouter_usage():
    spent = await asyncio.to_thread(openrouter.spent_today_usd)
    return {
        "today_usd":       spent,
        "day_cap_usd":     config.OPENROUTER_MAX_USD_PER_DAY,
        "request_cap_usd": config.OPENROUTER_MAX_USD_PER_REQUEST,
        "key":             await openrouter.key_usage(),
    }


# ── Chat route ────────────────────────────────────────────────────────────────

def _context_budget(mode: str) -> int:
    if mode == "local":
        return config.CONTEXT_BUDGET_LOCAL
    if mode == "openrouter":
        return config.CONTEXT_BUDGET_OPENROUTER
    return config.CONTEXT_BUDGET_CLAUDE


def _summariser(mode: str, model: str):
    """The one model call context assembly may make — on the provider that's
    answering. Pro has no cheap one-shot path, so it borrows the API key when
    present and otherwise fails open (full history is sent)."""
    system = context.SUMMARY_SYSTEM.format(target=config.CONTEXT_SUMMARY_WORDS)

    async def _local(transcript: str) -> str:
        return await ollama_chat(model, [{"role": "system", "content": system},
                                         {"role": "user", "content": transcript}])

    async def _or(transcript: str) -> str:
        return await openrouter.chat_text(model, [{"role": "system", "content": system},
                                                  {"role": "user", "content": transcript}])

    async def _claude(transcript: str) -> str:
        if not config.ANTHROPIC_API_KEY:
            raise RuntimeError("no ANTHROPIC_API_KEY for context summary")
        return await anthropic_text(model, [{"role": "user", "content": transcript}], system=system)

    return {"local": _local, "openrouter": _or}.get(mode, _claude)


def _json_error(status: int, message: str) -> Response:
    return Response(status_code=status, content=json.dumps({"error": message}),
                    media_type="application/json")


@app.post("/chat")
async def chat(request: Request):
    raw = await request.body()
    if len(raw) > _MAX_BODY_BYTES:
        return _json_error(413, "Request body too large")
    try:
        body = ChatRequest.model_validate_json(raw)
    except Exception as e:
        return _json_error(400, f"Invalid request: {e}")
    model        = body.model
    max_tokens   = min(body.max_tokens, _MAX_TOKENS_CAP)
    mode         = body.mode

    if mode in ("api", "pro") and model not in _ALLOWED_CLAUDE_MODELS:
        return _json_error(400, f"Model not permitted: {model}")
    # Dynamic allowlist: any installed Ollama model (static-config fallback
    # inside is_allowed_local_model when Ollama is down).
    if mode == "local" and not await model_admin.is_allowed_local_model(model):
        return _json_error(400, f"Model not permitted: {model}")
    if mode == "openrouter":
        if not openrouter.configured():
            return _json_error(503, "OpenRouter is not configured")
        if not openrouter.is_valid_model_id(model):
            return _json_error(400, f"Model not permitted: {model}")

    # ── Server-owned history ──
    conv = None
    if body.conversation_id:
        if not body.messages or body.messages[-1].get("role") != "user":
            return _json_error(400, "conversation turns must end with a user message")
        if body.conversation_id == "new":
            conv = await asyncio.to_thread(conversations.create, model, mode)
        else:
            conv = await asyncio.to_thread(conversations.get, body.conversation_id)
            if conv is None:
                return _json_error(404, f"Unknown conversation: {body.conversation_id}")
        for m in body.messages:
            conversations.append_user(conv, m)
        conv["model"], conv["mode"] = model, mode
        await asyncio.to_thread(conversations.save, conv)
        session_id = body.session_id or conv.get("pro_session_id", "")
    else:
        session_id = body.session_id

    log.info("chat mode=%s model=%s conv=%s msgs=%d", mode, model,
             conv["id"] if conv else "-", len(body.messages))

    async def stream():
        nonlocal conv
        if conv is not None:
            yield sse("conversation", {"id": conv["id"], "title": conv["title"]})
            # Pro resumes its own CLI session with just the newest message;
            # everything else gets the budgeted tail + rolling summary.
            if mode == "pro" and session_id:
                messages = [conv["messages"][-1]]
                ctx_stats = {"total_messages": len(conv["messages"]), "sent_messages": 1,
                             "folded_messages": len(conv["messages"]) - 1,
                             "sent_tokens": context.message_tokens(conv["messages"][-1]),
                             "summary_tokens": 0, "summarised": 0, "summary_failed": 0,
                             "resumed_session": 1}
                sent_base = len(conv["messages"]) - 1
            else:
                messages, new_summary, ctx_stats = await context.assemble(
                    conv["messages"], conv.get("summary"), _context_budget(mode),
                    _summariser(mode, model))
                if new_summary != conv.get("summary"):
                    conv["summary"] = new_summary
                    await asyncio.to_thread(conversations.save, conv)
                sent_base = ctx_stats["folded_messages"]
            yield sse("context", ctx_stats)
        else:
            messages = body.messages
            sent_base = 0

        # Blobs referenced by this request must survive LRU eviction until the
        # stream finishes (a concurrent request's spill could otherwise evict
        # them mid-loop). Blobs spilled *during* this request are mtime-newest,
        # so LRU never reaches them within the request.
        pinned = strategy.referenced_blob_hashes(messages)
        blobstore.pin(pinned)
        head_len = len(messages) - (len(conv["messages"]) - sent_base) if conv else 0
        reply_text = ""
        history_applied = False
        error_msg = ""
        try:
            if mode == "local":
                gen = _stream_ollama(messages, model, mode=mode)
            elif mode == "api":
                gen = _stream_claude(messages, model, max_tokens, skip_preprocess=False, mode=mode)
            elif mode == "openrouter":
                gen = _stream_openrouter(messages, model, mode=mode)
            else:
                gen = _stream_pro(messages, model, max_tokens, mode=mode, session_id=session_id)
            async for event in gen:
                if conv is not None:
                    ev = parse_sse_event(event)
                    if ev == "delta":
                        reply_text += str(parse_sse_data(event).get("text", ""))
                    elif ev == "history":
                        # Canonical transformed list: summary pair + sent tail
                        # (post-preprocess) + new assistant/tool turns.
                        conversations.replace_tail(
                            conv, sent_base, parse_sse_data(event).get("messages", [])[head_len:])
                        history_applied = True
                    elif ev == "pro_session":
                        conv["pro_session_id"] = parse_sse_data(event).get("session_id", "") or conv.get("pro_session_id", "")
                    elif ev in ("error", "quota_exceeded"):
                        error_msg = parse_sse_data(event).get("message", "") or ev
                yield event
        finally:
            blobstore.unpin(pinned)
            if conv is not None:
                if not history_applied:
                    # Pro (no history event) or an errored turn: keep what we
                    # got so the transcript stays alternating and honest.
                    text = reply_text or (f"(no reply — {error_msg})" if error_msg else "")
                    if text:
                        conversations.append_assistant_text(conv, text)
                await asyncio.to_thread(conversations.save, conv)

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
