# Memory

Two separate mechanisms share the name "memory":

## Auto-injected past-session context

Before your first reply in `api`/`pro` modes, up to 3 recent conversation
summaries are loaded and folded into the system prompt automatically — you
don't have to ask for this, it's already there if present. `local` mode does
not currently get this injection (a known gap, not a bug you need to explain
away).

## `recall_memories(query)`

Searches *all* stored conversation summaries (not just the recent few
auto-injected), keyword-matched against title/summary/tags. Use it when the
user says "do you remember", "recall", "what did we work on", or asks about a
specific past session by topic — the auto-injected context alone may not cover
it if the relevant conversation was a while ago.
