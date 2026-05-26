# PRE-SUMMARISATION STRATEGY
# This file decides what gets compressed and how. main.py only knows
# about preprocess(body, ollama_chat) -> (new_body, stats).

import asyncio
import logging

log = logging.getLogger("strategy")

# --- Tunables ---
LOCAL_MODEL = "gemma4:e2b"          # summariser; gemma4:e2b is the faster fallback
SUMMARY_THRESHOLD_TOKENS = 6000     # blocks above this get summarised
SUMMARY_TARGET_WORDS = 300          # rough budget for the summary
SUMMARY_MARKER = "[router-summary v1]"        # opening sentinel; also used by is_already_summarised
SUMMARY_END_MARKER = "[end-router-summary]"   # closing sentinel — defines an unambiguous data boundary

SUMMARY_SYSTEM = (
    "You are a compression layer for an AI coding assistant. "
    "Summarise the following content to at most {target} words. "
    "Preserve VERBATIM: error codes, file paths, line numbers, stack traces, "
    "timestamps, exact identifiers, command names. Drop repetition and noise. "
    "If the content is structured (JSON, logs, diffs), keep the structural shape. "
    "Output the summary only — no preamble, no apology, no surrounding quotes."
)


# --- Estimation ---
def estimate_tokens(text: str) -> int:
    """Rough estimate: 1 token ≈ 4 chars."""
    return len(text) // 4


def is_already_summarised(text: str) -> bool:
    return text.lstrip().startswith(SUMMARY_MARKER)


def has_oversized_blocks(body: dict) -> bool:
    """Quick O(n) scan — no I/O. True if preprocess would send anything to Ollama."""
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, str):
            if content and not is_already_summarised(content) \
                    and estimate_tokens(content) > SUMMARY_THRESHOLD_TOKENS:
                return True
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                text = block_text(block)
                if text and not is_already_summarised(text) \
                        and estimate_tokens(text) > SUMMARY_THRESHOLD_TOKENS:
                    return True
    return False


# --- Block extraction (Anthropic content-block shape) ---
def block_text(block: dict) -> str:
    """Pull the summarisable text out of a content block. Returns '' if none."""
    t = block.get("type")
    if t == "text":
        return block.get("text", "") or ""
    if t == "tool_result":
        c = block.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return "".join(b.get("text", "") for b in c
                           if isinstance(b, dict) and b.get("type") == "text")
    return ""


def replace_block_text(block: dict, new_text: str) -> dict:
    """Return a new block with its text replaced by `new_text`."""
    t = block.get("type")
    if t == "text":
        return {**block, "text": new_text}
    if t == "tool_result":
        return {**block, "content": new_text}
    return block


# --- Summarisation ---
async def summarise(ollama_chat, text: str) -> str:
    sys_prompt = SUMMARY_SYSTEM.format(target=SUMMARY_TARGET_WORDS)
    out = await ollama_chat(LOCAL_MODEL, [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": text},
    ])
    # Strip any markers an injected payload might have planted, then wrap in our own.
    # The begin/end pair gives downstream consumers an unambiguous data boundary
    # so instructions inside the summary can be treated as data, not commands.
    body = out.strip().replace(SUMMARY_MARKER, "").replace(SUMMARY_END_MARKER, "").strip()
    return f"{SUMMARY_MARKER}\n{body}\n{SUMMARY_END_MARKER}"


# --- Main entry point ---
async def preprocess(body: dict, ollama_chat) -> tuple[dict, dict]:
    """Walk the request, summarise oversized text/tool_result blocks in parallel.

    Returns (new_body, stats). stats keys: summarised, skipped, failed.
    """
    stats = {"summarised": 0, "skipped": 0, "failed": 0}
    targets = []  # (msg_index, block_index_or_None, original_text)

    for mi, m in enumerate(body.get("messages", [])):
        content = m.get("content")
        if isinstance(content, str):
            if content and not is_already_summarised(content) \
                    and estimate_tokens(content) > SUMMARY_THRESHOLD_TOKENS:
                targets.append((mi, None, content))
        elif isinstance(content, list):
            for bi, block in enumerate(content):
                if not isinstance(block, dict):
                    continue
                text = block_text(block)
                if not text or is_already_summarised(text):
                    continue
                if estimate_tokens(text) > SUMMARY_THRESHOLD_TOKENS:
                    targets.append((mi, bi, text))

    if not targets:
        return body, stats

    results = await asyncio.gather(
        *[summarise(ollama_chat, t[2]) for t in targets],
        return_exceptions=True,
    )

    new_messages = [dict(m) for m in body["messages"]]
    for (mi, bi, _original), result in zip(targets, results):
        if isinstance(result, Exception):
            log.warning("summary failed (msg=%d block=%s): %s", mi, bi, result)
            stats["failed"] += 1
            continue
        if bi is None:
            new_messages[mi]["content"] = result
        else:
            blocks = list(new_messages[mi]["content"])
            blocks[bi] = replace_block_text(blocks[bi], result)
            new_messages[mi]["content"] = blocks
        stats["summarised"] += 1

    return {**body, "messages": new_messages}, stats
