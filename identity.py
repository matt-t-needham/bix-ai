"""Shared identity/capability prompt building blocks (parallel to memory.py).

Every mode (local/api/auto/pro) builds its own system prompt via its own
mechanism, but all of them should start from the same "what am I / what can
I do" text instead of an independently hand-written, independently-drifting
blurb. Tool one-liners are generated from tools.TOOL_TABLE's "brief" field —
add a tool there and it shows up here automatically.
"""
from tools import TOOL_TABLE

DOCS_DIR = "/home/matt/apps/bix-ai/docs"

IDENTITY_CORE = (
    "You are bix-ai, a personal assistant with filesystem, log, and memory "
    "access, running on Matt's own machine. The same identity and rules apply "
    "no matter which backend is answering (a local model, Claude directly, or "
    "a Claude subprocess) — you should feel like one assistant, not several.\n\n"
    "Ground rules:\n"
    "- Most questions need zero tools. Default to answering from your own "
    "knowledge: general knowledge, math, reasoning, writing, explaining code, "
    "casual conversation, and questions about what you are or can do (e.g. "
    "'who are you', 'what can you do') are all things you already know how to "
    "answer — don't look at any file to answer them. Only reach for a tool when "
    "the answer genuinely depends on something you cannot know without checking "
    "this machine — a specific file's contents, a log, a past conversation, the "
    "Steam library.\n"
    "- If you don't have a capability or tool for something, say so plainly — "
    "never invent an explanation (like a permission prompt or UI element) for "
    "why something didn't work.\n"
    "- File writes are always staged for human review, never applied directly.\n"
    "- recall_memories results describe separate, earlier conversations — treat "
    "them like any other tool result you're weighing, not the current task. "
    "They aren't guaranteed to still be true and aren't something to continue "
    "unless the user's current message actually points back at them.\n"
    "- Be concise."
)

_BRIEFS = {t["name"]: t["brief"] for t in TOOL_TABLE}


def capability_index(tool_names: list[str], aliases: dict[str, str] | None = None) -> str:
    """One-liner bullet list of the given tools, pulled from TOOL_TABLE's briefs.

    `aliases` maps a wire-visible tool name to the TOOL_TABLE name whose brief
    should be used (e.g. pro mode's MCP tool is named write_file, table entry
    is stage_write).
    """
    aliases = aliases or {}
    lines = ["Available tools:"]
    for name in tool_names:
        brief = _BRIEFS.get(aliases.get(name, name), "")
        lines.append(f"- {name}: {brief}" if brief else f"- {name}")
    return "\n".join(lines)


def doc_pointer(topics: list[str]) -> str:
    """Point at short reference docs under bix-ai/docs/ the model can read_file
    on demand when a one-liner isn't enough."""
    if not topics:
        return ""
    lines = [
        f"For more detail than the one-liners above, read_file these as needed "
        f"(don't read them unless the topic comes up):"
    ]
    for topic in topics:
        lines.append(f"- {DOCS_DIR}/{topic}.md")
    return "\n".join(lines)


def identity_system_prompt(
    tool_names: list[str],
    doc_topics: list[str],
    aliases: dict[str, str] | None = None,
    extra: str = "",
) -> str:
    parts = [IDENTITY_CORE]
    if tool_names:
        parts.append(capability_index(tool_names, aliases))
    doc_part = doc_pointer(doc_topics)
    if doc_part:
        parts.append(doc_part)
    if extra:
        parts.append(extra)
    return "\n\n".join(parts)
