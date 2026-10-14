"""Tagging each assistant turn with the model that wrote it.

Matt switches models mid-conversation and re-asks questions to compare answers.
Without attribution the new model reads the whole transcript as its own work and
tends to agree with it. So every assistant turn that rides along is prefixed
with the model that produced it:

    [router-model gemma4:26b]
    Use functools.lru_cache on the function…

The model is told the convention once, in identity.runtime_state_block, and
reads the rest off the transcript itself — no server-side assertion about what
happened earlier, which could only ever name one model and couldn't point at
what was actually said.

**Tags must never be stored.** The loop's `history` event carries the sent
messages back, and main.py folds them into the conversation record, so a tag
added on the way out would be persisted and then tagged again next turn. That
is what strip_tags is for, and `strip_tags(tag_messages(m, t)) == m` is the
invariant its tests pin. The marker follows the house style (`[router-blob v2
…]`, `[router-context v1]`) precisely so it is unambiguous to strip.

Attribution comes from the record's `turns` sidecar (conversations.append_turn),
never from the message dicts: those ride to the provider, where an unknown key
is a 400. Pure logic, stdlib-only, no imports from main.py. Fails open — an
unreadable record or a missing sidecar just means nothing gets tagged.
"""
import re

TAG_PREFIX = "router-model"

_TAG_RE = re.compile(r"^\[" + TAG_PREFIX + r" [^\]\n]*\]\n?")


def tag_for(model: str) -> str:
    return f"[{TAG_PREFIX} {model}]"


def models_by_index(turns) -> list[tuple[int, str]]:
    """(message index, model) for each recorded turn, oldest first.

    Each entry's "from" is where that turn's assistant output starts, so a
    message belongs to the last entry at or before its index.
    """
    out = []
    for t in turns or []:
        if not isinstance(t, dict):
            continue
        model = str(t.get("model") or "")
        start = t.get("from")
        if model and isinstance(start, int):
            out.append((start, model))
    out.sort()
    return out


def model_at(spans: list[tuple[int, str]], index: int) -> str:
    """The model that produced the message at `index`, "" when unknown."""
    found = ""
    for start, model in spans:
        if start <= index:
            found = model
        else:
            break
    return found


def _tag_content(content, tag: str):
    """Put `tag` on the first text block, or add one when the turn is all tool
    calls, so every assistant turn carries attribution."""
    if isinstance(content, str):
        return f"{tag}\n{content}"
    if not isinstance(content, list):
        return content
    blocks = list(content)
    for i, b in enumerate(blocks):
        if isinstance(b, dict) and b.get("type") == "text":
            blocks[i] = {**b, "text": f"{tag}\n{b.get('text', '')}"}
            return blocks
    return [{"type": "text", "text": tag}, *blocks]


def _strip_content(content):
    if isinstance(content, str):
        return _TAG_RE.sub("", content, count=1)
    if not isinstance(content, list):
        return content
    blocks = []
    stripped = False
    for b in content:
        if not stripped and isinstance(b, dict) and b.get("type") == "text":
            text = _TAG_RE.sub("", b.get("text", ""), count=1)
            stripped = text != b.get("text", "")
            if stripped and not text:
                continue          # a block that was only a tag: drop it again
            blocks.append({**b, "text": text} if stripped else b)
            continue
        blocks.append(b)
    return blocks


def tag_messages(messages, turns, offset: int = 0) -> list:
    """A copy of `messages` with every assistant turn attributed.

    `offset` is the record index of messages[0]: callers tag the verbatim tail
    that context.assemble returned, which starts partway into the record (the
    older turns have folded into the summary at its head). Nothing is mutated —
    the record keeps clean content, and only the copy sent out carries tags.
    """
    if not isinstance(messages, list):
        return messages
    spans = models_by_index(turns)
    if not spans:
        return list(messages)
    out = []
    for j, m in enumerate(messages):
        i = j + offset
        if not isinstance(m, dict) or m.get("role") != "assistant":
            out.append(m)
            continue
        model = model_at(spans, i)
        if not model:
            out.append(m)
            continue
        content = m.get("content")
        tagged  = _tag_content(content, tag_for(model))
        # Unchanged means there was nothing taggable (no content, odd shape):
        # keep the original object so the round trip stays exact.
        out.append(m if tagged is content else {**m, "content": tagged})
    return out


def strip_tags(messages) -> list:
    """Undo tag_messages, so nothing tagged is ever written to the record."""
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            out.append(m)
            continue
        content = _strip_content(m.get("content"))
        out.append(m if content is m.get("content") else {**m, "content": content})
    return out
