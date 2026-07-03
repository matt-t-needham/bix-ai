"""Routing v2 (Phase 6 of PLAN-pi-tools.md) — decides local vs Claude for
mode="auto" *before* any model runs, instead of forge-first-escalate-on-error.

Decision order makes misrouting hard→Claude-work to a local model impossible
by construction:

1. Claude signals run FIRST. Code generation/editing, multi-step task shapes,
   user-facing prose deliverables, long requests, and large contexts route to
   Claude before any local rule or classifier is consulted.
2. Local structural rules run second, and only match narrow, cheap shapes:
   digestion of small tool results, and short conversational queries in small
   contexts. (Pre-pass internals — strategy.py / compact.py — are "always
   local" structurally: they call ollama_chat directly and never route.)
3. The ambiguous remainder gets ONE cheap local classification. Only an
   affirmative EASY routes local; HARD, garbage output, or any error
   fails open to Claude.

Pure logic, same injected-`ollama_chat` pattern as strategy.py/compact.py.
Every decision carries a `rule` + `reason` that the caller writes into
routing.ndjson, plus a `claude_model` cost-tier hint — size-only Claude routes
(long-request / large-context with no intent signal anywhere in the
conversation) suggest config.ROUTING_CHEAP_CLAUDE_MODEL; intent routes and
classifier fail-opens keep the user-selected model (None).
"""
import logging
import re

import config
from config import SUMMARY_LOCAL_MODEL
from strategy import block_text, estimate_tokens

log = logging.getLogger("routing")

LOCAL_MODEL = SUMMARY_LOCAL_MODEL

# --- Tunables ---
LONG_REQUEST_TOKENS   = 400    # last user turn above this → Claude
LARGE_CONTEXT_TOKENS  = 8000   # whole conversation above this → Claude (local ctx is small)
SHORT_CHAT_WORDS      = 40     # last user turn at/below this may route local
SMALL_CONTEXT_TOKENS  = 4000   # ...if the whole conversation is also below this
TOOL_DIGEST_MAX_TOKENS = 1500  # tool results below this are cheap to digest locally

# --- Claude signals (checked first) ---
_CODE_FENCE_RE  = re.compile(r"```")
_CODE_INTENT_RE = re.compile(
    r"\b(write|implement|refactor|debug|fix|patch|build|create|generate)\b"
    r".{0,60}\b(code|script|function|class|test|bug|module|component|regex|query|program)\b",
    re.IGNORECASE | re.DOTALL,
)
_PROSE_INTENT_RE = re.compile(
    r"\b(write|draft|compose|rewrite)\b"
    r".{0,40}\b(email|blog|post|article|letter|readme|documentation|docs|essay|report|announcement)\b",
    re.IGNORECASE | re.DOTALL,
)
_MULTISTEP_RE = re.compile(
    r"step[- ]by[- ]step|\bthen\b.+\bthen\b|^\s*1[.)].*\n\s*2[.)]",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)

CLASSIFY_SYSTEM = (
    "You route requests between a small local 9B model and a large frontier model. "
    "Reply with exactly one word. EASY if the small local model can handle this "
    "reliably: casual conversation, a simple factual answer, a short summary, a "
    "simple lookup. HARD for everything else: writing or editing code, debugging, "
    "multi-step tasks, precise or long-form writing, anything high-stakes. "
    "If unsure, reply HARD."
)


def _text_of(m: dict) -> str:
    content = m.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(t for t in (block_text(b) for b in content if isinstance(b, dict)) if t)
    return ""


def _last_user_text(messages: list) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return _text_of(m)
    return ""


def _total_tokens(messages: list) -> int:
    return sum(estimate_tokens(_text_of(m)) for m in messages)


def _tool_result_tokens(m: dict) -> int | None:
    """Token estimate of a tool-result message (Anthropic tool_result blocks
    or OpenAI tool role), or None if the message isn't one."""
    if m.get("role") == "tool":
        return estimate_tokens(_text_of(m))
    content = m.get("content")
    if isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "tool_result" for b in content
    ):
        return estimate_tokens(_text_of(m))
    return None


def _decision(route: str, rule: str, reason: str,
              claude_model: str | None = None) -> dict:
    """`claude_model` is a cost-tier hint for claude-routed decisions: a
    cheaper model to use instead of the user-selected one, or None to keep
    the selection. Local-routed decisions always carry None."""
    return {"route": route, "rule": rule, "reason": reason,
            "claude_model": claude_model}


def _has_intent_signal(text: str) -> bool:
    return bool(
        _CODE_FENCE_RE.search(text)
        or _CODE_INTENT_RE.search(text)
        or _PROSE_INTENT_RE.search(text)
        or _MULTISTEP_RE.search(text)
    )


def _cheap_model_if_no_intent(messages: list) -> str | None:
    """Cost tier for size-only Claude routes (long-request / large-context).

    Those rules fire on volume, not difficulty — digesting a big tool result
    or continuing a long plain conversation is easy Claude work. Downshift to
    config.ROUTING_CHEAP_CLAUDE_MODEL unless *any* turn in the conversation
    (not just the last) shows code/prose/multi-step intent — a mid-task "yes
    do that" must not demote an ongoing coding conversation. Read at call time
    so tests can monkeypatch config (same pattern as FS_ROOT/STAGING_DIR)."""
    cheap = config.ROUTING_CHEAP_CLAUDE_MODEL
    if not cheap:
        return None
    if any(_has_intent_signal(_text_of(m)) for m in messages):
        return None
    return cheap


def decide_structural(messages: list) -> dict | None:
    """Code-only rules. Returns a decision or None (ambiguous)."""
    last = _last_user_text(messages)
    total = _total_tokens(messages)

    # Claude signals first — a request matching any of these can never reach
    # the local rules or the classifier.
    if _CODE_FENCE_RE.search(last):
        return _decision("claude", "code-blocks", "request contains code blocks")
    if _CODE_INTENT_RE.search(last):
        return _decision("claude", "code-gen", "code generation/editing intent")
    if _PROSE_INTENT_RE.search(last):
        return _decision("claude", "prose", "user-facing prose deliverable")
    if _MULTISTEP_RE.search(last):
        return _decision("claude", "multi-step", "multi-step task shape")
    if estimate_tokens(last) > LONG_REQUEST_TOKENS:
        return _decision("claude", "long-request",
                         f"last turn ~{estimate_tokens(last)} tokens",
                         claude_model=_cheap_model_if_no_intent(messages))
    if total > LARGE_CONTEXT_TOKENS:
        return _decision("claude", "large-context",
                         f"conversation ~{total} tokens",
                         claude_model=_cheap_model_if_no_intent(messages))

    # Local structural rules.
    tool_toks = _tool_result_tokens(messages[-1]) if messages else None
    if tool_toks is not None and tool_toks <= TOOL_DIGEST_MAX_TOKENS:
        return _decision("local", "tool-digest",
                         f"small tool result (~{tool_toks} tokens)")
    if last and len(last.split()) <= SHORT_CHAT_WORDS and total < SMALL_CONTEXT_TOKENS:
        return _decision("local", "short-chat", "short conversational query")

    return None


async def decide(messages: list, ollama_chat) -> dict:
    """Full decision: structural rules, then one local classification for the
    ambiguous remainder. Fail-open-to-Claude: only an affirmative EASY from
    the classifier routes local."""
    structural = decide_structural(messages)
    if structural is not None:
        return structural

    try:
        out = await ollama_chat(LOCAL_MODEL, [
            {"role": "system", "content": CLASSIFY_SYSTEM},
            {"role": "user", "content": _last_user_text(messages)[:2000]},
        ])
        word = out.strip().upper().split()[0].strip(".,!\"'") if out.strip() else ""
    except Exception as e:
        log.warning("routing classifier failed, failing open to Claude: %s", e)
        return _decision("claude", "classifier-error", f"fail-open: {e}")

    if word == "EASY":
        return _decision("local", "classifier", "classified EASY by local model")
    return _decision("claude", "classifier",
                     f"classified {word or 'unparseable'} — fail-open to Claude")
