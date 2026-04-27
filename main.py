import asyncio
import json
import logging
import logging.handlers
import os
import re
import time
import uuid
from datetime import datetime, timezone
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
FS_ROOT              = Path(os.environ.get("FS_ROOT", "/home/matt")).resolve()
DATA_DIR             = Path(os.environ.get("DATA_DIR", "/app/data"))
ENTRIES_PER_FILE     = 200

psutil.cpu_percent()  # prime the cpu counter

_agg = {"requests": 0, "summarised": 0, "checked": 0, "preprocess_ms": 0, "failed": 0}


# ── Filesystem tools ──────────────────────────────────────────────────────────
def _safe_path(path: str) -> Path | None:
    """Return resolved Path if within FS_ROOT, else None."""
    try:
        p = Path(path).resolve()
        p.relative_to(FS_ROOT)   # raises ValueError if outside root
        return p
    except (ValueError, Exception):
        return None


# ── Memory helpers ────────────────────────────────────────────────────────────
MEM_DIR  = DATA_DIR / "memories"
CONV_DIR = DATA_DIR / "convos"


def _active_memory_file() -> Path:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(MEM_DIR.glob("memories-*.json"))
    if not files:
        return MEM_DIR / "memories-001.json"
    latest = files[-1]
    try:
        entries = json.loads(latest.read_text())
        if isinstance(entries, list) and len(entries) < ENTRIES_PER_FILE:
            return latest
    except Exception:
        pass
    num = int(latest.stem.rsplit("-", 1)[-1]) + 1
    return MEM_DIR / f"memories-{num:03d}.json"


def _load_all_memories() -> list[dict]:
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    for f in sorted(MEM_DIR.glob("memories-*.json")):
        try:
            data = json.loads(f.read_text())
            if isinstance(data, list):
                entries.extend(data)
        except Exception:
            pass
    return entries


def _load_recent_memories(n: int = 3) -> list[dict]:
    all_m = _load_all_memories()
    return all_m[-n:] if len(all_m) >= n else all_m


def _append_memory(entry: dict):
    f = _active_memory_file()
    try:
        existing = json.loads(f.read_text()) if f.exists() else []
    except Exception:
        existing = []
    existing.append(entry)
    f.write_text(json.dumps(existing, indent=2))


def _is_similar_memory(a: dict, b: dict) -> bool:
    stop = {
        'a','an','the','in','on','at','to','for','of','and','or','is','was',
        'with','from','be','been','have','has','this','that','they','are','were',
    }
    def kw(text: str) -> set[str]:
        return {w.lower() for w in re.findall(r'[a-zA-Z]{4,}', text) if w.lower() not in stop}
    ta = kw(f"{a.get('title','')} {a.get('summary','')}")
    tb = kw(f"{b.get('title','')} {b.get('summary','')}")
    if len(ta) < 3 or len(tb) < 3:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) > 0.55


def _consolidate_active_file():
    f = _active_memory_file()
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    if not f.exists():
        return
    try:
        entries = json.loads(f.read_text())
        if not isinstance(entries, list) or len(entries) < 4:
            return
        merged = 0
        i = 0
        while i < len(entries) - 1:
            j = i + 1
            while j < min(i + 8, len(entries)):
                if _is_similar_memory(entries[i], entries[j]):
                    combined = (
                        f"{entries[j].get('summary') or entries[j].get('title', '')}"
                        f"; also: {entries[i].get('summary') or entries[i].get('title', '')}"
                    )
                    entries[j] = {
                        **entries[j],
                        "summary": combined,
                        "tags": list(set(entries[i].get("tags", []) + entries[j].get("tags", []))),
                    }
                    entries.pop(i)
                    merged += 1
                    break
                j += 1
            else:
                i += 1
        if merged > 0:
            f.write_text(json.dumps(entries, indent=2))
            log.info("memory consolidation merged=%d", merged)
    except Exception as e:
        log.warning("memory consolidation error: %s", e)


def _extract_tags(text: str) -> list[str]:
    stop = {
        'have','that','this','with','from','they','will','been','were','their',
        'what','when','then','also','just','some','more','into','which','there',
    }
    words = re.findall(r'[a-zA-Z][a-zA-Z0-9_-]*', text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if len(w) >= 4 and w not in stop:
            freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:5]]


def _memory_system_prompt(memories: list[dict]) -> str | None:
    if not memories:
        return None
    lines = []
    for m in memories:
        date    = m.get("date", "")[:10]
        summary = m.get("summary") or m.get("title", "")
        if summary:
            lines.append(f"[{date}] {summary}")
    if not lines:
        return None
    return (
        "# Past session context\n"
        + "\n".join(lines)
        + "\n\nUse the recall_memories tool when asked about earlier conversations."
    )


async def _execute_tool(name: str, tool_input: dict) -> str:
    if name == "list_directory":
        path = tool_input.get("path", str(FS_ROOT))
        p = _safe_path(path)
        if p is None:
            return f"Access denied: '{path}' is outside the allowed root ({FS_ROOT})"
        if not p.exists():
            return f"Path does not exist: {path}"
        if not p.is_dir():
            return f"Not a directory: {path}"
        try:
            entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
            lines = []
            for e in entries:
                if e.is_dir():
                    lines.append(f"[dir]  {e.name}/")
                else:
                    lines.append(f"[file] {e.name}  ({e.stat().st_size:,} bytes)")
            return "\n".join(lines) if lines else "(empty directory)"
        except PermissionError:
            return f"Permission denied: {path}"

    elif name == "read_file":
        path = tool_input.get("path", "")
        p = _safe_path(path)
        if p is None:
            return f"Access denied: '{path}' is outside the allowed root ({FS_ROOT})"
        if not p.exists():
            return f"File does not exist: {path}"
        if not p.is_file():
            return f"Not a file: {path}"
        size = p.stat().st_size
        if size > 200_000:
            return f"File too large ({size:,} bytes). Max 200 KB."
        try:
            return p.read_text(errors="replace")
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error reading file: {e}"

    elif name == "recall_memories":
        query = tool_input.get("query", "").strip()
        if not query:
            return "No query provided."
        all_m = _load_all_memories()
        if not all_m:
            return "No memories stored yet."
        ql = query.lower().split()
        matches = []
        for m in reversed(all_m):
            text = " ".join([
                m.get("title", ""),
                m.get("summary", ""),
                " ".join(m.get("tags", [])),
            ]).lower()
            if any(word in text for word in ql):
                matches.append(m)
            if len(matches) >= 3:
                break
        if not matches:
            return f"No memories found matching: {query}"
        results = []
        for m in matches:
            date  = m.get("date", "")[:10]
            title = m.get("title", "—")
            # Try to load and extract from the full conversation file
            conv_file = m.get("file")
            if conv_file:
                try:
                    conv_path = CONV_DIR / conv_file
                    conv_data = json.loads(conv_path.read_text())
                    msgs      = conv_data.get("messages", [])
                    context   = "\n".join(
                        f"{msg['role']}: {str(msg.get('content',''))[:400]}"
                        for msg in msgs
                    )
                    extract_prompt = (
                        f"From this conversation, extract what's relevant to: {query}\n\n"
                        f"{context}"
                    )
                    relevant = await ollama_chat(
                        OLLAMA_DEFAULT_MODEL,
                        [{"role": "user", "content": extract_prompt}],
                        timeout=20.0,
                    )
                    results.append(f"[{date}] {title}\n{relevant.strip()}")
                    continue
                except Exception as e:
                    log.warning("recall load/extract failed for %s: %s", conv_file, e)
            # Fallback to stored summary
            summary = m.get("summary", "")
            results.append(f"[{date}] {title}\n{summary}" if summary else f"[{date}] {title}")
        return "\n\n".join(results)

    return f"Unknown tool: {name}"


FS_TOOLS = [
    {
        "name": "list_directory",
        "description": (
            f"List files and directories at a path. "
            f"The accessible root is {FS_ROOT}. "
            "Use this to explore before reading files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Absolute path to list. Must be within {FS_ROOT}.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the text contents of a file. Limited to 200 KB.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": f"Absolute path to the file. Must be within {FS_ROOT}.",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "recall_memories",
        "description": (
            "Search past conversation summaries stored in memory. "
            "Use this when the user says 'do you remember', 'recall', "
            "'what did we work on', or asks about previous sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or topic to search for in past conversations.",
                }
            },
            "required": ["query"],
        },
    },
]


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
    _agg["requests"] += 1
    req_body = {"model": model, "max_tokens": max_tokens, "messages": messages}
    recent = _load_recent_memories(3)
    sys_prompt = _memory_system_prompt(recent)
    if sys_prompt:
        req_body["system"] = sys_prompt
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
        _agg["summarised"]    += stats["summarised"]
        _agg["checked"]       += stats["summarised"] + stats["skipped"]
        _agg["preprocess_ms"] += preprocess_ms
        _agg["failed"]        += stats["failed"]
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

    start              = time.monotonic()
    ttft_ms            = None
    total_input_tokens = 0
    output_tokens      = 0
    current_messages   = list(req_body["messages"])

    for _turn in range(10):  # cap at 10 tool-use turns
        api_body = {
            **req_body,
            "messages": current_messages,
            "tools":    FS_TOOLS,
            "stream":   True,
        }
        content_blocks = {}  # index -> {type, name, id, text, input_json}
        stop_reason    = None

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", ANTHROPIC_URL, json=api_body, headers=headers) as r:
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
                                total_input_tokens += tok
                                if _turn == 0:
                                    yield sse("input_tokens", {"count": tok})

                            elif event_type == "content_block_start":
                                idx   = data.get("index", 0)
                                block = data.get("content_block", {})
                                content_blocks[idx] = {
                                    "type": block.get("type"), "name": block.get("name", ""),
                                    "id": block.get("id", ""), "text": "", "input_json": "",
                                }
                                if block.get("type") == "tool_use":
                                    yield sse("tool_start", {
                                        "index": idx,
                                        "name":  block.get("name", ""),
                                        "id":    block.get("id", ""),
                                    })

                            elif event_type == "content_block_delta":
                                idx   = data.get("index", 0)
                                delta = data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta["text"]
                                    if ttft_ms is None:
                                        ttft_ms = round((time.monotonic() - start) * 1000)
                                    if idx in content_blocks:
                                        content_blocks[idx]["text"] += text
                                    yield sse("delta", {"text": text})
                                elif delta.get("type") == "input_json_delta":
                                    partial = delta.get("partial_json", "")
                                    if idx in content_blocks:
                                        content_blocks[idx]["input_json"] += partial
                                    yield sse("tool_input", {"index": idx, "partial_json": partial})

                            elif event_type == "content_block_stop":
                                idx = data.get("index", 0)
                                if content_blocks.get(idx, {}).get("type") == "tool_use":
                                    yield sse("tool_end", {"index": idx})

                            elif event_type == "message_delta":
                                output_tokens = data.get("usage", {}).get("output_tokens", 0)
                                stop_reason   = data.get("delta", {}).get("stop_reason")

        except Exception as e:
            log.error("upstream error: %s", e)
            yield sse("error", {"message": str(e)})
            return

        if stop_reason != "tool_use":
            elapsed = time.monotonic() - start
            tps = output_tokens / elapsed if elapsed > 0 else 0
            yield sse("metrics", {
                "input_tokens":  total_input_tokens,
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
            return

        # Build the assistant turn content (text + tool_use blocks)
        assistant_content = []
        for idx in sorted(content_blocks):
            b = content_blocks[idx]
            if b["type"] == "text" and b["text"]:
                assistant_content.append({"type": "text", "text": b["text"]})
            elif b["type"] == "tool_use":
                try:
                    inp = json.loads(b["input_json"]) if b["input_json"] else {}
                except json.JSONDecodeError:
                    inp = {}
                assistant_content.append({
                    "type": "tool_use", "id": b["id"], "name": b["name"], "input": inp,
                })
        current_messages.append({"role": "assistant", "content": assistant_content})

        # Execute each tool call and collect results
        tool_results = []
        for idx in sorted(content_blocks):
            b = content_blocks[idx]
            if b["type"] != "tool_use":
                continue
            try:
                inp = json.loads(b["input_json"]) if b["input_json"] else {}
            except json.JSONDecodeError:
                inp = {}
            log.info("tool call name=%s path=%s", b["name"], inp.get("path", ""))
            yield sse("status", {"stage": "checking", "message": f"Running {b['name']}…"})
            result = await _execute_tool(b["name"], inp)
            tool_results.append({"type": "tool_result", "tool_use_id": b["id"], "content": result})

        current_messages.append({"role": "user", "content": tool_results})
        yield sse("status", {"stage": "streaming", "message": "Streaming from Claude…"})

    yield sse("error", {"message": "Maximum tool call depth reached"})


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


async def _generate_memory_summary(user_msg: str, assistant_msg: str) -> str:
    prompt = (
        "Summarize this conversation in 2-3 sentences. "
        "Focus on what was asked, what was decided or built, and any key technical details.\n\n"
        f"User: {user_msg[:500]}\nAssistant: {assistant_msg[:500]}"
    )
    try:
        result = await ollama_chat(
            OLLAMA_DEFAULT_MODEL,
            [{"role": "user", "content": prompt}],
            timeout=20.0,
        )
        return result.strip()
    except Exception as e:
        log.warning("memory summary generation failed: %s", e)
        return assistant_msg[:200].strip()


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
    result["gpu"] = {"state": "unreachable"}
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
        pass  # stays "unreachable"
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


@app.get("/stats")
async def router_stats():
    return dict(_agg)


@app.post("/memory/save")
async def save_memory(request: Request):
    body       = await request.json()
    messages   = body.get("messages", [])
    model_name = body.get("model", DEFAULT_MODEL)
    in_tokens  = body.get("input_tokens", 0)

    user_msg = next(
        (str(m.get("content", "")) for m in messages if m.get("role") == "user"), ""
    )
    assistant_msg = next(
        (str(m.get("content", "")) for m in reversed(messages) if m.get("role") == "assistant"), ""
    )

    title, _ = await _summarize(user_msg, assistant_msg, OLLAMA_DEFAULT_MODEL)
    if not title:
        title = user_msg[:60]

    recent = _load_recent_memories(10)
    if any(r.get("title", "").lower() == title.lower() for r in recent):
        return {"ok": True, "skipped": True}

    summary = await _generate_memory_summary(user_msg, assistant_msg)
    tags    = _extract_tags(f"{user_msg} {assistant_msg}")
    now     = datetime.now(timezone.utc)
    now_str = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    uid     = uuid.uuid4().hex[:8]

    # Save full conversation to its own file
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    conv_filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{uid}.json"
    conv_path     = CONV_DIR / conv_filename
    try:
        conv_path.write_text(json.dumps({
            "id": uid, "date": now_str, "model": model_name,
            "input_tokens": in_tokens, "messages": messages,
        }, indent=2))
    except Exception as e:
        log.warning("conversation file write failed: %s", e)
        conv_filename = None

    entry = {
        "id":           f"{now_str}-{uid}",
        "date":         now_str,
        "title":        title,
        "summary":      summary,
        "model":        model_name,
        "input_tokens": in_tokens,
        "tags":         tags,
        "file":         conv_filename,
    }

    _append_memory(entry)
    log.info("memory saved title=%r tags=%s file=%s", title, tags, conv_filename)

    all_m = _load_all_memories()
    if len(all_m) > 0 and len(all_m) % 10 == 0:
        _consolidate_active_file()

    return {"ok": True, "id": entry["id"]}


@app.get("/memory")
async def get_memory():
    all_m = _load_all_memories()
    return {"entries": list(reversed(all_m)), "count": len(all_m)}


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
    api_key = (
      request.headers.get("x-api-key")
      or request.headers.get("authorization", "").removeprefix("Bearer ")
      or ""
    )
    version = request.headers.get("anthropic-version", "2023-06-01")
    beta = request.headers.get("anthropic-beta") or request.query_params.get("beta")

    log.info("recv model=%s msgs=%d", body.get("model"), len(body.get("messages", [])))

    stats = {"summarised": 0, "skipped": 0, "failed": 0}
    if strategy.has_oversized_blocks(body):
        try:
            body, stats = await strategy.preprocess(body, ollama_chat)
            log.info("preprocess summarised=%d skipped=%d failed=%d",
          stats["summarised"],stats["skipped"], stats["failed"])
        except Exception as e:
            log.warning("preprocess errored, forwarding original: %s", e)

        _agg["requests"]   += 1
        _agg["summarised"] += stats["summarised"]
        _agg["checked"]    += stats["summarised"] + stats["skipped"]
        _agg["failed"]     += stats["failed"]

    stream_requested = body.get("stream", False)
    body = {**body, "stream": stream_requested}

    headers = {
        "x-api-key": api_key,
        "anthropic-version": version,
        "content-type": "application/json",
    }
    if beta:
        headers["anthropic-beta"] = beta

    if stream_requested:
        async def upstream():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", ANTHROPIC_URL, json=body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

        return StreamingResponse(upstream(), media_type="text/event-stream")

    else:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(ANTHROPIC_URL, json=body, headers=headers)
            from fastapi.responses import Response
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type="application/json",
            )
