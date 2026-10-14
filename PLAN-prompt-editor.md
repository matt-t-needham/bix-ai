# PLAN — editable custom prompt + model-change self-awareness

Two linked features:

1. **A place to edit bix-ai's custom prompt** — persisted outside the image, no
   rebuild, no deploy, human-edited only. **Built (2026-10-14).**
2. **Pre-determined knowledge of its own current state** — so the agent knows
   which model/backend is answering *right now*, notices that an earlier turn
   was answered by a *different* model, and recognises when Matt is deliberately
   re-asking the same question on another model to compare answers.
   **Built (2026-10-14).**

All four phases have shipped. What follows is the record of what was built and
why, not a to-do list.

The plan was originally written against pre-overhaul code; everything below was
re-derived against the 2026-10 overhaul (`PLAN-overhaul.md`) before being built.
What changed: `staging.py`/`deploy.py`/`routing.py`/
`compact.py`/`local_first.py` and `BIX_ROLE` are gone, `auto` mode with them;
the modes are now `local`/`api`/`openrouter`/`pro`, each with its own adapter;
`access.allowed` (Cloudflare Access) replaced `_require_prod()` as the gate on
mutating routes; and **the server owns conversation history**
(`conversations.py`), which removes the wire change phase 4 originally needed.

## Phase 1 — `prompt_store.py` + call-time composition ✅

- `prompt_store.py`: `data/prompt/custom.md` + a `revisions.json` ring of 10,
  atomic writes, 16 KB cap raising `PromptError`, mtime-cached reads that fail
  open to `""`, `custom_text()` as the async accessor.
- `identity.py`: `custom_section()` / `with_custom()` (pure) and
  `resolved_system(base)` (async, reads the store). `IDENTITY_CORE` is always
  prepended, so editing the prompt cannot delete the ground rules.
- All four adapters stopped caching the composed prompt at import. Each now does
  `base = await resolved_system(BASE)` and keeps its existing
  `agent_session.system(base)` wrap. In `streaming/ollama.py` and
  `streaming/openrouter.py` the provider's `system_sentinel` is that same
  resolved string, so the prompt still never leaks into the `history` event.
- `handoff.ReviewSession.system()` discards the base outright, so a fresh-eyes
  reviewer never inherits Matt's chat instructions. Pinned by a test.
- Tests: `tests/test_prompt_store.py` (23), `tests/test_identity_prompt.py` (18),
  including two that drive `_stream_claude`/`_stream_ollama` with a fake httpx
  client and assert the custom text is on the wire.

## Phase 2 — the editor: `prompt_ui.py` + `/prompt` routes ✅

- `GET /prompt` (page), `GET /prompt/raw` (JSON), `POST /prompt`,
  `POST /prompt/reset`, `POST /prompt/restore`. The three writes require
  `access.allowed`; reading is open like `/blobs`. No `/prompt/{x}` route
  exists, so the `/staging/count` ordering pitfall can't recur.
- Form bodies are parsed with `urllib.parse` in `_form_fields` — fastapi's
  `Form()` would have added a `python-multipart` dependency for three forms.
- The page: textarea with a character counter (the only JS), Save, Clear, a
  per-mode preview of the fully composed prompt with Matt's half brightened,
  and the revision list with Restore.
- `docs/prompt.md` is new and registered in all four adapters' `doc_topics`, so
  the model can explain its own prompt instead of inventing a mechanism.
- Tests: `tests/test_prompt_routes.py` (20) — the Access gate (including the
  same-site case), the size cap surfacing visibly instead of truncating,
  escaping of hostile prompt text, and the containment check that no
  model-facing write tool can reach the prompt file.

## Phase 3 — the per-turn state block ✅

`identity.runtime_state_block(st)` renders a `## This turn` block, appended to
every system prompt between the base and the custom instructions:

```
- You are claude-opus-4-7, answering through the Anthropic API, billed per token.
- Mode Matt picked: api
- Sandbox (agent) tools: on                        [only when on]
- Spent on OpenRouter today: $0.42 of $5.00        [openrouter only]
- This service: git 5b6795b, built …, up 3h02m
- Every assistant turn … is tagged … like `[router-model gemma4:26b]` …   [only when tagged]
```

- It is a **renderer only**: every value comes from `main._runtime_state`, which
  keeps `identity.py` pure and lets the editor preview it. `identity.service_line`
  is shared by the live path and the preview so the two can't drift.
- Composition still happens in exactly one place — `resolved_system(base, state)`
  — as Phase 1 set up. The adapters each gained one `state=None` kwarg.
- `streaming/openrouter.py` reads today's spend *before* composing the prompt, so
  the model is told what's left against the cap; `check_caps` still gates the
  call itself before any HTTP.

## Phase 4 — per-turn model attribution ✅

**Design note (changed after review).** The first cut had the server *assert* what
had happened — "Earlier turns were answered by gemma4:26b", "Matt has asked this
same question earlier". That was wrong twice over: it could only ever name one
model however many had answered, and it pointed at an earlier answer without
saying which turn or what was in it. It also put the server in the business of
detecting intent (a `difflib` repeat check) when the model can see the facts for
itself. Replaced with: tag the turns, state the convention once.

- `conversations.append_turn` keeps the `turns` sidecar — `{model, mode, at, from}`,
  `from` being the message index where that turn's assistant output starts. Beside
  `messages`, never on them: `replace_tail` rewrites those wholesale and they ride
  to the provider, where an unknown key on a message is a 400.
- `convstate.tag_messages` prefixes each assistant turn in the sent copy with
  `[router-model <name>]`; `convstate.strip_tags` removes them before the `history`
  tail is folded back. **The round trip is the invariant** — a tag that reached the
  record would be re-tagged next turn, so `strip_tags(tag_messages(m, t)) == m` is
  tested directly, including tool turns and string content.
- Tagging happens *after* `context.assemble`, on the verbatim tail only, with an
  `offset` back onto record indices — so the rolling summary is still built from
  untagged text and the `[router-context v1]` head is untouched.
- The prompt's tagging rules appear only when the transcript actually carries tags
  (`state["tagged"]`), so a model is never told to look for markers that aren't there.
- UI: `renderStored` used to label **every** assistant bubble with the
  conversation's *latest* model, so reopening a mixed chat misattributed earlier
  answers. It now resolves each message's model through `turnAt(rec, i)` and draws
  a thin `.model-switch` rule where the model changes, coloured by the existing
  `indicatorClass` mapping (yellow local / mauve Claude / blue OpenRouter). The
  tags themselves never reach the UI — it has the sidecar instead.

**Tests**: `tests/test_convstate.py` (17, built around the round trip),
state-block cases in `tests/test_identity_prompt.py`, and in
`tests/test_chat_conversations.py` that an earlier turn goes up tagged, that
nothing tagged is ever written back, and that a first turn is neither tagged nor
told about tags.

## Open question

Scope of the editable prompt: **one global custom prompt** (what's built) versus
per-mode or per-model overrides. Global is the right v1 — a single prompt is what
makes "ask a different model the same question" a fair comparison, and
conditional instructions can now be written as prose ("when running on a local
model, …"), since the state block tells the model which it is. Per-model overrides are a
later addition to `prompt_store`, not a redesign.
