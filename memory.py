import asyncio
import json
import logging
import re
import uuid
from datetime import datetime, timezone

from config import CONV_DIR, ENTRIES_PER_FILE, MEM_DIR, OLLAMA_DEFAULT_MODEL
from helpers import ollama_chat

log = logging.getLogger("router")


# ── Memory I/O ────────────────────────────────────────────────────────────────

def _active_memory_file():
    MEM_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(MEM_DIR.glob("memories-*.json"))
    if not files:
        return MEM_DIR / "memories-001.json"
    latest = files[-1]
    try:
        entries = json.loads(latest.read_text())
        if isinstance(entries, list) and len(entries) < ENTRIES_PER_FILE:
            return latest
    except Exception as e:
        log.warning("memory file unreadable path=%s err=%s", latest, e)
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
        except Exception as e:
            log.warning("memory file unreadable path=%s err=%s", f, e)
    return entries


def _load_recent_memories(n: int = 3) -> list[dict]:
    all_m = _load_all_memories()
    return all_m[-n:] if len(all_m) >= n else all_m


def _append_memory(entry: dict) -> None:
    f = _active_memory_file()
    try:
        existing = json.loads(f.read_text()) if f.exists() else []
    except Exception as e:
        log.warning("memory file unreadable path=%s err=%s", f, e)
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


def _consolidate_active_file() -> None:
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


# ── Summarization ─────────────────────────────────────────────────────────────

async def _summarize(user_msg: str, assistant_msg: str, local_model: str) -> tuple[str, str]:
    """Return (6-word title, source) using local Ollama."""
    prompt = (
        "Give a 6-word title for this exchange. No punctuation, no quotes.\n\n"
        f"User: {user_msg[:400]}\nAssistant: {assistant_msg[:400]}"
    )
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
        log.warning("memory summary generation failed: %s: %s", type(e).__name__, e)
        return assistant_msg[:200].strip()


# ── Save pipeline ─────────────────────────────────────────────────────────────

async def save_memory_entry(
    messages: list[dict],
    model_name: str,
    in_tokens: int,
    out_tokens: int,
) -> dict:
    """Full memory-save pipeline. Returns the JSON response dict for the route."""
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

    CONV_DIR.mkdir(parents=True, exist_ok=True)
    conv_filename = f"{now.strftime('%Y%m%d-%H%M%S')}-{uid}.json"
    conv_path     = CONV_DIR / conv_filename
    try:
        conv_path.write_text(json.dumps({
            "id": uid, "date": now_str, "model": model_name,
            "input_tokens": in_tokens, "output_tokens": out_tokens,
            "messages": messages,
        }, indent=2))
    except Exception as e:
        log.warning("conversation file write failed: %s", e)
        conv_filename = None

    entry = {
        "id":            f"{now_str}-{uid}",
        "date":          now_str,
        "title":         title,
        "summary":       summary,
        "model":         model_name,
        "input_tokens":  in_tokens,
        "output_tokens": out_tokens,
        "tags":          tags,
        "file":          conv_filename,
    }

    await asyncio.to_thread(_append_memory, entry)
    log.info("memory saved title=%r tags=%s file=%s", title, tags, conv_filename)

    all_m = await asyncio.to_thread(_load_all_memories)
    if len(all_m) > 0 and len(all_m) % 10 == 0:
        await asyncio.to_thread(_consolidate_active_file)

    return {"ok": True, "id": entry["id"]}
