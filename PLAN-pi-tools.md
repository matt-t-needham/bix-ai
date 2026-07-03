# PLAN — bix-ai: from "pre-summariser" to "efficient cross-platform task runner"

This plan is written for an implementer working directly on the machine, with the
real Ollama models available, and **no access to the conversation that produced it**.
It was reconciled against the actual code on 2026-07-02. Where it conflicts with
CLAUDE.md, **this document is right and CLAUDE.md is stale** (Phase 0 fixes that).

## Read this first: what already EXISTS (do not rebuild)

The repo is much further along than CLAUDE.md's "three concerns" description:

| Capability | Where it lives | Status |
|---|---|---|
| Agentic tool loop, iteration cap 10 | `streaming/claude.py:70` (`for _turn in range(10)`, exits on `stop_reason != "tool_use"`), `streaming/ollama.py:62` (same shape, `finish_reason != "tool_calls"`), `streaming/forge_runner.py` (forge `WorkflowRunner` for mode="auto") | **EXISTS** |
| Cross-turn metrics aggregation | `streaming/claude.py:98-104` — `total_input_tokens` summed across turns, `input_tokens` SSE only on turn 0, `metrics` emitted once at loop exit | **EXISTS** |
| Tool execution dispatcher | `tools.py:_execute_tool` — 8 tools: `list_directory`, `read_file`, `stage_write`, `list_steam_games`, `read_todos`, `list_log_sources`, `read_log`, `recall_memories` | **EXISTS** |
| Gated writes (propose → review → apply) | `staging.py` + review UI routes in `main.py` (`/staging…`), re-validates every guard at approve time, human-only apply | **EXISTS** |
| Path security | `fs_core.py` — `is_denied_path` (secrets: `.env*`, keys, credential keywords), `is_write_denied_path` (scripts, docker/CI config, bix-ai's own source), symlink-resolving `validate_target` in `staging.py` | **EXISTS** |
| Local-first routing with Claude fallback | `streaming/local_first.py` — forge/Ollama first, escalates to `_stream_claude` on error, `fallback_triggered` SSE clears partial UI output | **EXISTS** |
| Pre-summarisation pipeline | `strategy.py` — `preprocess(body, ollama_chat)`, threshold **6000** tokens (not 2000 as CLAUDE.md claims), `[router-summary v1]`…`[end-router-summary]` markers, parallel `asyncio.gather` | **EXISTS** (Phase 3 reworks it) |
| Memory system | `memory.py` + `recall_memories` tool + `/memory` routes; conversations persisted under `DATA_DIR/convos` | **EXISTS** |
| MCP server for mode="pro" | `bix_mcp.py`, driven by `claude` CLI subprocess in `streaming/pro.py` | **EXISTS** |
| SSE keepalive + disconnect surfacing | `helpers.py:with_keepalive` (Cloudflare Tunnel resets idle streams ~100s) | **EXISTS** |
| Docker test gate | `Dockerfile` — pytest runs in a build stage; build fails if tests fail | **EXISTS** |
| Display/payload split in the UI | `static/index.html` — `textSegments` (rendering) is already separate from `convHistory` (request payload) | **EXISTS** (Phase 1 exploits this) |

**Three parallel loop implementations** exist (claude / ollama / forge_runner) and
**three tool-definition formats** (`FS_TOOLS` Anthropic-shape, `OLLAMA_TOOLS`
OpenAI-fn-shape, `FORGE_TOOLS` forge `ToolDef`) — all in `tools.py`, all wrapping the
same `_execute_tool`. This duplication is real but deliberate for now; Phase 4
consolidates it. **Do not consolidate early** — user-visible gaps come first.

## The actual gaps (what this plan builds)

1. **Tool turns are lost between requests.** `static/index.html:1615` does
   `convHistory.push({ role: 'assistant', content: fullText })` — text only. The
   server-side loop's `current_messages` (assistant `tool_use` blocks + `tool_result`
   turns, built at `streaming/claude.py:175-208`) is discarded when the request ends.
   On the next turn Claude sees a conversation with its tool evidence amputated.
2. **Artifact compression is lossy and irreversible.** `strategy.py` paraphrases
   oversized blocks (>6000 est. tokens) via `gemma4:e2b` into ≤300 words. A pasted
   4k-line logfile becomes prose; the original is gone from the model's reach forever.
3. **No blob store.** Pasted content lives only inline in the request; there is
   nothing for retrieval tools to reach back into.
4. **Governor is partial.** Iteration cap exists; cumulative token budget and
   wall-clock limit do not.
5. **`tool_result` SSE inconsistency.** `streaming/pro.py:160` emits `tool_result`
   events (UI handles them at `static/index.html:1582`); `streaming/claude.py` and
   `streaming/ollama.py` never do — the UI shows tool calls but not their results.
6. **No conversation-tail compaction** for the api path (forge's `TieredCompact`
   covers mode="auto" only) — blocked on gap 1.
7. **Routing is error-driven only.** Escalation on failure exists; cost/task-aware
   routing does not.
8. **CLAUDE.md is stale** and will actively mislead any implementer (wrong threshold,
   wrong file inventory, no mention of `streaming/`, `tools.py`, staging, forge).
9. **`OLLAMA_URL` is hardcoded** (`config.py:5`, `host.docker.internal:11434`) — the
   one config value not env-overridable; breaks dev outside Docker.

## Core principles (non-negotiable)

1. **Retrieval beats compression.** Artifacts are kept at full fidelity and reached
   via tools. Nothing the model might need is destroyed in-band.
2. **The pre-pass widens, never narrows destructively.** Tune for recall; everything
   not surfaced stays one tool-call away.
3. **Verbatim, not paraphrase.** The local model selects what to keep; extracted
   lines enter context byte-for-byte.
4. **Model for classification, code for extraction.** Heuristics/regex/AST do the
   keeping; Ollama only picks buckets or ranks salience.
5. **Summarise conversation, retrieve artifacts.** Opposite treatments; never mix.
6. **Every phase ships with chat working.** The Docker test gate enforces green tests
   on every build; keep it that way.
7. **Writes stay gated.** All disk writes go through `staging.py`. Never bypass it,
   never auto-apply. `is_write_denied_path` protections (including bix-ai's own
   source) must never be weakened by this work.

## Decision A — RESOLVED: A1 (client adopts transformed history)

The server is stateless per-request; `convHistory` in the browser is the source of
truth. Chosen fix: at clean loop completion the server emits a **`history` SSE event**
carrying the canonical transformed `messages` array; the client adopts it and sends it
on the next request. Rules:

- **Adopt only on clean completion** (event precedes `done`). On `error`/disconnect no
  `history` is emitted; the client keeps what it had. Idempotent server transforms
  (content-addressed blobs, marker tags) make resending originals a cheap no-op.
- **Display ≠ payload.** The UI's existing `textSegments`/`convHistory` split already
  provides this; `history` adoption only changes what goes into `convHistory`.
- **A2 growth path** (server-side sessions) stays open: the canonical format is
  server-defined, so the server can later start retaining what it already emits. Do
  not build sessions now.

## How to test on this machine

- Unit: `pytest -q` (8 test files under `tests/`; `config.FS_ROOT`/`STAGING_DIR` are
  read at call time specifically so tests can monkeypatch them — keep that property).
- Integration: `uvicorn main:app --reload --port 8000` with
  `OLLAMA_URL=http://localhost:11434/v1/chat/completions` (after Phase 0) and real
  models. `curl -N -X POST localhost:8000/chat -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"mode":"local","model":"gemma4:26b"}'`
  to eyeball SSE.
- The UI is served via `FileResponse` — live from disk under uvicorn; container
  rebuild only needed for deploys (which re-runs the test gate).
- SSE protocol contract: `status`, `preprocess`, `input_tokens`, `delta`,
  `tool_start`, `tool_input`, `tool_end`, `tool_result` (pro only, until Phase 1),
  `model_swap`, `fallback_triggered`, `quota_exceeded`, `metrics`, `done`, `error` —
  both sides must move together.

---

## Phase 0 — groundwork (small, safe, do first) — ✅ SHIPPED (commit `4332ad0`)

**Current:** CLAUDE.md describes a 3-file project with a 2000-token threshold and four
tests; reality is 12 modules + `streaming/` package, threshold 6000, 8 test files.
`OLLAMA_URL` hardcoded at `config.py:5`.

**Delta:**
1. Rewrite CLAUDE.md's project-overview, gotchas, and test sections to match reality
   (module inventory, the three stream paths + loop, staging, tools, forge dependency,
   threshold 6000). An implementer reads CLAUDE.md first; stale docs poison everything.
2. `config.py`: `OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://host.docker.internal:11434/v1/chat/completions")`.
   Also derive the bare host URL used by `main.py:117` (`/api/ps` GPU probe) and
   `streaming/forge_runner.py:251` (`OllamaClient(base_url=…)`) from the same env var —
   those are two more hardcoded copies of the host.
3. Move `strategy.py`'s hardcoded `LOCAL_MODEL = "gemma4:e2b"` to `config.py`
   (env-overridable) — it must stay a summariser-independent knob.

**Done when:** `pytest -q` green; a chat via uvicorn with `OLLAMA_URL` pointed at
`localhost:11434` works end-to-end without Docker.

## Phase 1 — history sync + loop hardening (the unlock) — ✅ SHIPPED (commit `64b332f`)

**Current:** `streaming/claude.py` already builds the canonical transformed history in
`current_messages` (preprocessed input + assistant tool_use turns + tool_result turns)
and throws it away. The UI already handles `tool_result` events (pro path only).
Governor = iteration cap only.

**Target:** tool turns survive across requests; all paths emit `tool_result`; loop has
budgets.

**Delta:**
1. `streaming/claude.py`: at loop exit (the `stop_reason != "tool_use"` branch, before
   `done`), emit `sse("history", {"messages": current_messages})`. Note
   `current_messages` starts from `req_body["messages"]` — i.e. *post-preprocess* —
   so summarised/pointered blocks ride along for free.
2. `streaming/ollama.py`: same, emitting its `current_messages` (OpenAI-shape tool
   turns are fine — this path's history goes back to Ollama, not Anthropic). Strip the
   injected system message (index 0, `OLLAMA_SYSTEM`) before emitting so it isn't
   duplicated next turn by the `insert(0, …)` at `streaming/ollama.py:59-60`.
3. `streaming/claude.py`: emit `sse("tool_result", {"tool_use_id": …, "content": …,
   "is_error": false})` after each `_execute_tool` (mirror `streaming/pro.py:160-164`,
   including the 4000-char truncation for the *SSE event only* — the full result still
   goes into `current_messages`). Same in `streaming/ollama.py`.
4. Governor: add cumulative-token and wall-clock checks to both loops (env-tunable,
   e.g. `LOOP_MAX_TOKENS`, `LOOP_MAX_SECONDS`, defaults generous). On breach: emit
   `error` + `metrics` (partial) and stop — never silently truncate.
5. `static/index.html`: on `history` event, replace `convHistory` wholesale with the
   payload. Remove the `convHistory.push({role:'assistant'…})` at :1615 **only when a
   `history` event arrived this request**; keep it as fallback otherwise (pro path
   won't emit `history` initially — subprocess owns its loop). Render `tool_result`
   in the existing `.sub-bubble` pattern (handler exists at :1582).
6. `ChatRequest.messages` already accepts `list[dict[str, Any]]` — block-structured
   content passes validation today; `strategy.py` already walks both string and
   block-list content. Verify with a test, not a refactor.

**Watch out:** `_render_memory_html` (`main.py:183-189`) already renders
`tool_use`/`tool_result` blocks — memory saves of block-structured `convHistory` will
work, but add a test. Mode="auto" (forge) reconstructs context its own way
(`_convert_messages` skips tool roles) — leave it; escalation to Claude passes original
`messages` and is covered by the claude-path `history` event.

**Done when:** a two-turn conversation where turn 1 uses `read_log` shows turn 2's
request containing the tool_use/tool_result turns (verify via server log
`chat mode=… msgs=N` — message count grows); tests cover history emission, governor
breach, and tool_result parity.

## Phase 2 — blob store + retrieval tools — ✅ SHIPPED (commit `462bb17`)

**Current:** nothing reaches back into pasted content; `read_file`/`read_log` reach
real files under `FS_ROOT` but a pasted blob isn't a file. `DATA_DIR = /app/data`
already holds `memories/`, `convos/`, `staging/`.

**Target:** oversized inbound blocks are spilled to a content-addressed store and
retrievable by tools.

**Delta:**
1. New `blobstore.py` (stdlib-only, like `staging.py`/`fs_core.py`): `put(text) ->
   {hash, path, lines, bytes}` (sha256, write-once — identical bytes dedup to the same
   file under `DATA_DIR/blobs/`), `get(hash)`, `grep(hash, pattern, context_lines)`,
   `stat(hash)`. Size-capped LRU eviction; a blob referenced by the *current request's
   messages* is pinned; eviction must never race a pin.
2. Two new tools in `tools.py` (all three formats, wrapping `_execute_tool`):
   `read_blob(hash, start_line, end_line)` and `grep_blob(hash, pattern,
   context_lines)`. Blob tools take **hashes, not paths** — no traversal surface.
3. **Verify on the prod host** that `/app/data` is a compose volume (memories already
   persist across deploys, so it almost certainly is — but confirm before shipping
   blobs; an ephemeral blob dir dangles every pointer on every deploy).
4. Keep `_MAX_BODY_BYTES` (1 MB) as-is; blobs address token cost, not transport cost.

**Done when:** `put`→`grep_blob` round-trips in tests; dedup verified (same bytes,
same hash, one file); pinning test passes; tools callable by both Claude and Ollama
paths end-to-end against a real model.

## Phase 3 — pre-pass rework of `strategy.py` (replaces lossy summarisation of artifacts) — ✅ SHIPPED (2026-07-02)

*Implementation notes:* delta shipped as written, plus: request-scoped blob pinning
wired at `main.py`'s chat dispatcher via `strategy.referenced_blob_hashes` (the Phase 2
pin primitives were unwired until blobs actually entered messages); `/app/data`
verified as a compose bind mount of `bix-ai/data` (Phase 2 checklist item);
`EXCERPT_MAX_CHARS=12000` so a head+tail fallback excerpt never self-truncates.
E2E verified: 310 KB logfile paste → 2,676 input tokens, verbatim ERROR+traceback
delivered, Claude grep_blob'd the hash for context unprompted.

**Current:** `preprocess` paraphrases any >6000-token block into ≤300 words via
`gemma4:e2b`. `SUMMARY_SYSTEM` already *tries* to preserve error codes/paths verbatim —
the instinct is right but paraphrase can't be trusted with needles.

**Target:** per oversized block: lossless-reduce → label → extract verbatim → spill
original to blob store → emit excerpt + pointer.

**Delta (all inside `strategy.py`; keep `preprocess(body, ollama_chat) -> (new_body,
stats)` signature and purity — blobstore is injected or imported as a leaf, never
`main.py`):**
1. **Reduce (no model):** collapse consecutive duplicate lines (`[previous line ×N]`),
   strip ANSI. Pure function, golden-tested.
2. **Label:** heuristics first (JSON-parse attempt, timestamp-prefix ratio, `import`/
   shebang density, diff markers) → `logfile | source | json | diff | prose | mixed`;
   fall back to one `ollama_chat` call only when heuristics are low-confidence.
3. **Extract by type (code, verbatim):** logfile → `ERROR|WARN|FATAL|Traceback|panic`
   lines + N context lines; json → key structure + sampled values; source → def/class/
   import lines (regex is fine; AST only if cheap); diff → file headers + hunk counts;
   prose/mixed → the *one* remaining `ollama_chat` salience pass, which returns **line
   ranges to keep**, and code slices them out verbatim (the model never rewrites).
4. **Emit:** replace block text with
   `[router-blob v2 <hash>]\n<label>: <verbatim excerpt>\nFull content: <lines> lines, <bytes> bytes — use read_blob("<hash>") or grep_blob("<hash>", pattern)\n[end-router-blob]`
   and spill the original via `blobstore.put`. Marker discipline mirrors v1: blocks
   already carrying either marker are skipped (`is_already_summarised` grows a v2
   check). Same bytes resent by an un-adopted client → same hash → skip. Deterministic.
5. Keep `asyncio.gather` parallelism and the `stats` dict; add `spilled` count (wire
   through to the `preprocess` SSE event and the UI's stats line at
   `static/index.html:1046-1047`).
6. **Latency budget:** steps 1/3/4 are pure code. Only step 2's fallback and the prose
   case may call Ollama. Emit the existing `status stage=summarising` message (rename
   text to "Preparing context…") so cold-start model loads are visible, not mysterious.

**Tests:** `tests/test_strategy.py`'s four tests assert paraphrase behaviour that this
phase **deliberately abolishes — rewrite them**. New fixtures under `tests/fixtures/`:
a ~4k-line logfile with one buried `ERROR` + traceback, a big source file, a huge JSON.
Assert: the ERROR line survives byte-for-byte; pointer hash resolves via blobstore;
reduction is lossless on a repeats-heavy fixture; same input twice → same hash, one
blob; below-threshold and already-marked blocks untouched (port those two cases from
the old tests).

**Done when:** pasting the logfile fixture into a real chat yields Claude receiving the
verbatim error + pointer, and Claude can `grep_blob` the rest — demonstrated end-to-end
on the machine.

## Phase 4 — provider seam consolidation (refactor, after value has shipped) — ✅ SHIPPED (2026-07-02)

*Implementation notes:* delta shipped as written — `streaming/loop.py` (one governed
loop), `streaming/providers.py` (Anthropic/Ollama adapters yielding normalised
events), tools defined once in `tools.py:TOOL_TABLE` with all three formats
generated (forge via pydantic `create_model`, still lazy). 10 golden SSE-sequence
fixtures (`tests/fixtures/sse/`) recorded pre-refactor replay identically post-
refactor; routing records pinned too. One honest deviation: total line count of
`streaming/` + `tools.py` is roughly flat (-394 in rewritten files, +428 in the two
new seam modules) — the duplication became interface, it didn't vanish. Adapters
inject `execute_tool`/`clock`/budgets resolved from their module globals at call
time so every pre-existing test patch target still works unchanged.

**Current:** `streaming/claude.py` and `streaming/ollama.py` are structurally parallel
~200-line loops (stream-parse → accumulate blocks/tool-calls → execute → append →
repeat), duplicating loop control, SSE emission, metrics, and routing-log writes. Tool
definitions triplicated in `tools.py`.

**Target:** one governed loop; per-provider adapters that translate to/from a
normalised event/message shape; a single tool table generating all three formats.

**Delta:**
1. Single source of truth for tools: a list of `{name, description, input_schema,
   handler}`; generate `FS_TOOLS` / `OLLAMA_TOOLS` / `FORGE_TOOLS` from it (the
   forge branch keeps its lazy `try: import forge` guard — tests must pass without
   forge installed, as today).
2. `providers.py`: `AnthropicProvider` / `OllamaProvider` each yielding normalised
   events (`text_delta`, `tool_call_start/delta/end`, `usage`, `stop_reason` mapped to
   a common enum) and accepting/producing a common message shape at the loop boundary
   (adapter converts to Anthropic content-blocks or OpenAI tool_calls at the wire).
3. `loop.py`: the one governed loop (cap/budgets from Phase 1), taking a provider +
   tool registry, yielding the SSE events. `claude.py`/`ollama.py` shrink to adapters.
   `pro.py` (subprocess) and `forge_runner.py` (owns its own loop) stay as-is.
4. Behaviour must be observably identical: same SSE sequences (assert with recorded
   fixtures), same routing-log records, same test suite green.

**Done when:** both refactored paths pass the SSE-sequence fixtures; `tools.py` defines
each tool exactly once; line count of `streaming/` drops meaningfully.

## Phase 5 — conversation-tail compaction (needs Phase 1 live) — ✅ SHIPPED (2026-07-02)

*Implementation notes:* `compact.py`, wired after the spill pass in
`streaming/claude.py` (api path only, as scoped by gap 6). Cut points are plain
user turn-starts (tool_result continuations excluded) so alternation stays valid;
the fold produces a `[router-compact v1]` user msg + assistant ack pair. On
re-growth the old compact body accretes verbatim — the model never re-sees it.
Blob hashes folded out of the tail are re-listed in the compact body so pinning
and retrieval keep working. `compacted` rides the `preprocess` SSE event (not
`metrics`, which is pinned by the Phase 4 golden fixtures). E2E on this machine:
23 msgs / 31,077 input tokens → compacted to 7 msgs / 7,764 tokens, tail
byte-identical; resent adopted history logged `msgs=9` and did not re-compact.
**Environment gotcha found during E2E:** `gemma4:e2b` (the SUMMARY_LOCAL_MODEL
default) no longer loads on this host — compose now overrides to `gemma4:26b`,
and `TRANSCRIPT_MAX_CHARS=12000` keeps the summarise call inside the 120s
ollama_chat timeout (~50-90s observed).

Old turns (narrative, not artifacts) summarised once the *conversation* exceeds a
threshold; recent K turns kept verbatim; result rides the `history` event so the
client adopts it and it converges (never re-summarised — `[router-compact v1]` marker).
Artifacts are untouched — they were pointered in Phase 3. Trigger: total estimated
tokens of `convHistory` minus blob excerpts > threshold (env-tunable). Implement as a
`strategy.py`-adjacent pure module with the same injected-`ollama_chat` pattern.

**Done when:** a long session compacts once, the next request arrives pre-compacted
(verify via msgs count in server log), recent turns byte-identical.

## Phase 6 — routing v2 + cost surfaces — ✅ SHIPPED (2026-07-03)

*Implementation notes:* `routing.py` — pure logic, injected `ollama_chat` like
strategy/compact. The misroute guarantee is by construction via decision order:
Claude signals (code fences/intent, prose deliverable, multi-step shape, long
request, large context) are checked before any local rule; local structural
rules match only narrow cheap shapes (small tool-result digestion, short chat
in small context); the ambiguous remainder gets one local classification where
only an affirmative EASY routes local — HARD, garbage output, or classifier
error all fail open to Claude (all tested, incl. "classifier always says EASY
still can't pull code-gen local"). Wired into `_stream_local_first`:
claude-routed requests skip the local attempt entirely; local-routed keep the
forge-first flow with error escalation. Every `routing.ndjson` record now
carries `reason` (`forced:<mode>` on non-auto paths) and `est_cost_usd`
(advisory `MODEL_COSTS` in `config.py`, mirrored by the UI's `MODEL_RATES`);
the UI stats line shows per-request Est. cost, and local/unknown models are
never billed (removed the old sonnet-rate fallback). Golden SSE routing
records updated for the `reason` field. Bonus fix found during E2E:
`AnthropicProvider` swallowed non-200 responses — a 401 produced a clean
zero-token `history`+`done` — it now yields `provider_error` on bad status and
on mid-stream `event: error` frames (2 new tests). E2E on this machine:
short chat → gemma4:26b (`short-chat`); "write a function…" → Haiku
(`code-gen`, $0.0031 logged); ambiguous strategy question → classifier HARD →
Claude (fail-open). Note for dev runs: the API key lives in
`bix-infra/.env` as `BIX_AI_API_KEY` (compose maps it to `ANTHROPIC_API_KEY`),
not `/home/matt/apps/.env` as the root CLAUDE.md used to claim.

**Current:** mode="auto" = forge-first, escalate on error. No cost/task awareness.

**Delta:** structural rules first (pre-pass internals → always local; tool-result
digestion below size threshold → local; code-gen / multi-step reasoning / user-facing
prose → Claude); ambiguous remainder → one cheap local classification with
**fail-open-to-Claude**; every decision logged to the existing `routing.ndjson` with a
`reason` field; surface per-request provider + est. cost in the UI stats line
(`.hi-stats` grid; model table with advisory per-token costs added to `config.py`).
Colour semantics stay: yellow = local, blue = streaming, green = done, red = error,
mauve = brand.

**Done when:** routing decisions are inspectable in `routing.ndjson` and the UI, and
misrouting hard→local is impossible by construction (fail-open path tested).

---

## Sequencing and the one rule

Phase 0 → 1 → 2 → 3 ship user-visible value in order of leverage; 4 is the paid-down
refactor once behaviour is pinned by tests; 5–6 build on 1–3. Every phase: tests green
(`pytest -q`), chat verified against real Ollama + Claude on this machine, then deploy
(`docker compose build ai-router` re-runs the gate).

The one rule if anything here conflicts with reality: **read the code first, trust the
code, update this plan** — it was written from a survey on 2026-07-02 and the repo
moves fast.
