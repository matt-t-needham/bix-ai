# CLAUDE.md — bix-ai

## Project overview

A FastAPI chat router: one mobile-first chat UI at `ai.bix.computer` in front of
three providers — local Ollama models, Claude on the Pro subscription (via the
`claude` CLI), and OpenRouter on a personal key with server-side spend caps. The
server owns conversation history (`conversations.py`) and decides per turn how
much of it rides along (`context.py`). Models get a read-only agentic tool loop
(filesystem reads, memory recall, blob retrieval); oversized pastes are spilled to
a content-addressed blob store and replaced in-band with a verbatim excerpt +
retrieval pointer.

**Overhaul (2026-10):** `PLAN-overhaul.md` is the plan; `docs/request-pipeline.md`
is the pre-overhaul audit of what ran on one turn. The self-modification/staging/
deploy machinery, Steam, log tools, auto-mode routing, conversation compaction and
every background per-turn model call were removed — don't reintroduce them.

Module map:

| Module | Role |
|---|---|
| `main.py` | FastAPI app — all routes (`/chat`, `/memory*`, `/models/ollama*`, `/system`, `/auth/status`, `/version`, `/blobs*`, `/v1/messages` proxy), SSE streaming dispatch |
| `config.py` | All tunables/env vars — `OLLAMA_HOST` is the single source for every Ollama host reference |
| `strategy.py` | Pre-pass pipeline (Phase 3 shape): per oversized block (>6000 est. tokens) it losslessly reduces (ANSI strip + duplicate-line collapse), labels via heuristics (`logfile\|source\|json\|diff\|prose`, one Ollama call only when unsure), extracts salient lines **verbatim** by type (code does the slicing; the model never rewrites a byte), spills the original to `blobstore.py`, and replaces the block with `[router-blob v2 <hash>]` pointer + excerpt. Pure logic, `preprocess(body, ollama_chat) -> (new_body, stats)`, no imports from `main.py`. Ollama is used for *classification/ranking only* — never paraphrase. The v1 paraphrase pipeline is abolished; `[router-summary v1]` blocks are still recognised and skipped |
| `tools.py` | Single tool registry: `TOOL_TABLE` defines each tool exactly once as `{name, description, input_schema, handler, brief}`; `FS_TOOLS` (Anthropic shape) and `OLLAMA_TOOLS` (OpenAI fn shape) are both generated from it. Five read-only tools: `list_directory`, `read_file`, `recall_memories`, `read_blob`, `grep_blob`. There is no write tool. Add a tool = add one table entry |
| `model_admin.py` | Ollama model management: free-text pulls validated by name regex + a registry manifest pre-check (typos 400 fast), NDJSON→SSE pull translation (`pull_progress`/`pull_done` events on `POST /models/ollama/pull`, wrapped in `with_keepalive`), delete, and the dynamic mode=local allowlist (`is_allowed_local_model` = any installed model, 30 s cached `/api/tags`; falls back to the static `_ALLOWED_OLLAMA_MODELS` when Ollama is down) |
| `openrouter.py` | OpenRouter catalogue (`GET /api/v1/models`, 5-min cache, audio-output models and `openrouter/*` meta-routers dropped, static fallback), model-id shape check (`is_valid_model_id` — any well-formed `vendor/model[:variant]` is sent as typed), per-model cost estimate from listed prices, spend caps (`check_caps`: per-request projection + today's total from `routing.ndjson`), `key_usage()` for the budget pill, `chat_text()` one-shot completion. Attribution headers are ours (`OPENROUTER_REFERER`/`OPENROUTER_TITLE`), not bix-graph's |
| `conversations.py` | Server-owned chats: one JSON per conversation under `config.CONVERSATIONS_DIR` with the canonical (Anthropic-shaped) message list, the rolling context summary (`{text, through}`), and pro mode's CLI session id. Stdlib-only, atomic writes, config read at call time. `append_user` merges consecutive user turns so an errored turn never breaks alternation |
| `context.py` | Per-turn context assembly: newest turns verbatim up to a per-provider token budget (`CONTEXT_BUDGET_LOCAL/OPENROUTER/CLAUDE`), everything older as one `[router-context v1]` summary pair at the head. The summary accretes on the conversation record and is never re-summarised; it's only extended when the cut-off moves, and that one call runs via the injected `summarise` on the provider that's answering — never a background local model. Fails open (full history, `summary_failed=1`). Replaced the old compaction pass |
| `fs_core.py` | Read-only path security — `is_denied_path` (secrets), `list_directory`, `read_file`. Stdlib-only, shared with `bix_mcp.py` |
| `blobstore.py` | Content-addressed store for oversized artifacts spilled out of context (`put`/`get`/`grep`/`stat`, sha256-keyed, write-once, dedup). LRU eviction over `config.BLOB_STORE_MAX_BYTES`, pin/unpin protects blobs referenced by the in-flight request. Backs the `read_blob`/`grep_blob` tools. `strategy.py` spills oversized blocks into it automatically; `main.py`'s chat dispatcher pins every hash referenced by the in-flight request (`strategy.referenced_blob_hashes`) for the stream's lifetime. `/app/data` is a compose bind mount of `bix-ai/data`, so blobs persist across deploys. Housekeeping UI at `GET /blobs` (styled by `main._PAGE_CSS`) backed by `list_blobs`/`delete`/`purge_unpinned` — all pin-aware under the same lock as eviction |
| `memory.py` | Memory persistence (`/memory` routes, `recall_memories` tool); snapshots under `DATA_DIR/convos`. Saving is explicit: `POST /conversations/{id}/remember` (the drawer's "Remember this chat" button) runs `save_memory_entry` once over the whole conversation — one local-model call for the summary. Nothing saves in the background any more |
| `bix_mcp.py` | Stdlib-only MCP server exposed to the `claude` CLI subprocess for mode="pro" (`list_directory`, `read_file`, `recall_memories`) |
| `helpers.py` | SSE helpers (`sse()`, `with_keepalive`, `parse_sse_event`/`parse_sse_data`, `SSETextCollector` — a chunk-boundary-safe delta/error accumulator for consuming our own stream), routing-log writer (`routing.ndjson` — the per-request usage/cost log, one line per turn), Ollama chat helper |
| `streaming/loop.py` | **The one governed agentic loop** (`run_tool_loop`): turn cap (10), token + wall-clock budgets, tool dispatch, tool_result/metrics/history/done SSE emission, routing-log write. Takes a provider + injected `execute_tool`/`clock`/budget args (resolved from the adapter module's globals at call time, so tests patch the adapter as before). Shared by `mode="api"` and `mode="local"` — `mode` only picks the backend |
| `streaming/providers.py` | `AnthropicProvider` / `OllamaProvider` / `OpenRouterProvider` (an `OllamaProvider` pointed at OpenRouter: bearer + attribution headers, `stream_options.include_usage` for real token counts, `reasoning`/`reasoning_content` deltas → `thinking_delta`, one guardrail retry max because retries cost money) — translate each backend's wire protocol to normalised events (`text_delta`, `tool_start/…`, `turn_end`, `provider_error`) and own provider-native message shapes (append_assistant_turn / append_tool_results / history_messages) and token accounting (real usage vs chars/4 estimate). `OllamaProvider` additionally carries a guardrail layer — rescue-parsing + bounded retry-with-nudge for garbled tool-call attempts, using forge-guardrails' standalone `rescue_tool_call`/`ErrorTracker`/`retry_nudge` (not `WorkflowRunner`) — `on_exhausted` stays "best_effort" (the "escalate" policy is kept in the provider but has no caller since auto mode went) |
| `streaming/claude.py` | Anthropic adapter: memory injection + pre-pass, then delegates to `run_tool_loop` with an `AnthropicProvider` |
| `streaming/ollama.py` | Ollama adapter: pre-pass (`strategy.preprocess`, same as claude.py) + tool-offload model swap + `OLLAMA_SYSTEM` injection, then delegates to `run_tool_loop` with a guardrailed `OllamaProvider` |
| `streaming/openrouter.py` | OpenRouter adapter (mode="openrouter"): configured check, pre-pass, **spend-cap check before any HTTP** (projected cost from listed prices + today's log total), system prompt, then `run_tool_loop` on an `OpenRouterProvider`. Routing records carry the real `est_cost_usd` |
| `streaming/pro.py` | mode="pro" — drives the `claude` CLI subprocess over MCP. Keeps tool-turn history across requests via CLI sessions: emits `pro_session` SSE with the run's session id; the client echoes it back as `session_id` on the next `/chat` and the server runs `claude --resume <id>` with only the newest user message (falls back to a fresh full-history run when the session is gone; "No conversation found" arrives via the result event's `errors` array) |
| `static/index.html` | Single-file, mobile-first chat UI. No build step. Vanilla JS + CSS. `PROVIDERS` drives the model-picker sheet (Local / Claude / OpenRouter — the OpenRouter section has a search box over the live catalogue and accepts a typed `vendor/model` id); the stats drawer is a full-screen sheet under 900 px and a side panel above. The browser holds only a `conversationId` (in `localStorage` `bixai-prefs`, with model prefs): each `/chat` sends the new message + `conversation_id` and the server owns the transcript; the Chats section lists/opens/deletes them and `renderStored` re-renders a stored record (tool turns as collapsed groups). `thinking` events render as a collapsed block above the answer; `context` events feed the request card |

Four `mode` values on `POST /chat`: `local` (direct to Ollama, the default), `api`
(direct to Claude, per-token), `openrouter` (OpenAI-compatible, personal key, spend
caps), and everything else falls through to `pro` (subprocess `claude` CLI with MCP
tools; subscription billing). `auto` mode and its router are gone.

**Conversations.** With `conversation_id` set (`"new"` to create), `messages` is only
the new user turn; the server appends it, runs `context.assemble` (pro with a live
CLI session sends just the newest message instead), streams, and folds the loop's
`history` tail back into the record. Without it the call is stateless (whole
transcript in `messages`) — tests use that path. `/conversations` (list), `GET`/
`DELETE /conversations/{id}`, `POST /conversations/{id}/remember`.

`PLAN-pi-tools.md` is the completed previous roadmap (Phases 0–7), kept for history.

## How to run

```bash
# Dev (outside Docker) — point OLLAMA_HOST at your local Ollama, not the Docker DNS name
export OLLAMA_HOST=http://localhost:11434
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Tests
pip install pytest==8.3.4   # not in requirements.txt — only installed in the Docker test stage
pytest -q

# Production rebuild (from repo root)
docker compose build ai-router && docker compose up -d ai-router
docker logs apps-ai-router-1 -f
```

The API key lives in `/home/matt/apps/bix-infra/.env` as `BIX_AI_API_KEY`; the compose
file maps it to `ANTHROPIC_API_KEY` inside the container. For dev runs outside Docker,
export it yourself: `export ANTHROPIC_API_KEY=$(grep BIX_AI_API_KEY /home/matt/apps/bix-infra/.env | cut -d= -f2)`.
(There is no `/home/matt/apps/.env` — older docs claiming that are stale.) Ollama must be running with `OLLAMA_HOST=0.0.0.0` (systemd override) so
the container can reach it — in Docker this resolves via `host.docker.internal`, which
is `config.OLLAMA_HOST`'s default. Override `OLLAMA_HOST` when running outside Docker.

The Docker build has a **hard test gate**: the `test` stage runs `pytest -q` and the
`runtime` stage only exists via `COPY --from=test`, so a failing suite fails the build.
pytest itself never ships in the runtime image.

## SSE event protocol

`POST /chat` streams these events (not all paths emit all events — see notes):

| Event | Payload | Notes |
|-------|---------|-------|
| `conversation` | `{id, title}` | first event when `conversation_id` was sent — the client adopts the id (a `"new"` request gets its freshly minted one) |
| `context` | `{total_messages, sent_messages, folded_messages, sent_tokens, summary_tokens, summarised, summary_failed[, resumed_session]}` | second event with a conversation — what history actually went up this turn (`context.assemble` stats; `summarised=1` means the rolling summary was extended now) |
| `status` | `{stage, message}` | checking / summarising / streaming (the `summarising` stage's message reads "Preparing context…" — it covers the whole pre-pass, not just model calls) |
| `preprocess` | `{summarised, spilled, compacted, skipped, failed, preprocess_ms}` | fires after `strategy.preprocess`, before the model stream. `spilled` = blocks pointered to the blob store; `summarised` and `compacted` are always 0 now (kept for wire compatibility — history folding is reported on `context` instead). Don't add fields to `metrics`: it would drift the golden SSE fixtures |
| `input_tokens` | `{count}` | fires on `message_start` (turn 0 only across a tool loop) |
| `budget` | `{turn, max_turns, elapsed_s, max_seconds, tokens, max_tokens}` | fires at the start of every tool-loop turn (claude/ollama paths) — per-turn governor gauge, rendered next to the status label |
| `delta` | `{text}` | streaming text chunk |
| `thinking` | `{text}` | OpenRouter reasoning models only — reasoning tokens, streamed before the answer; the UI shows them collapsed |
| `tool_start` | `{index, name, id}` | tool call begins |
| `tool_input` | `{index, partial_json}` | streaming tool input |
| `tool_end` | `{index}` | tool call complete |
| `tool_result` | `{tool_use_id, content, is_error}` | emitted by all three loop paths (`claude.py`, `ollama.py`, `pro.py`); content truncated to 4000 chars for the SSE event only — the full result still goes into `history` |
| `history` | `{messages}` | fires **once, at clean loop completion, before `done`** — the canonical transformed message list (assistant `tool_use` + `tool_result` turns included). `claude.py`/`ollama.py` only; `pro.py` doesn't emit it (subprocess owns its own loop) — the UI keeps a text-only `convHistory` fallback for that path. Never emitted on `error`/disconnect |
| `pro_session` | `{session_id}` | pro path only — the claude-CLI session id for this run (resume may keep or fork the id; the client always adopts the latest). Echoed back as `session_id` on the next `/chat` so tool turns survive across requests; any non-pro turn clears it client-side |
| `model_swap` | `{from, to}` | mode="local" tool-offload swap to `OLLAMA_TOOL_MODEL` |
| `quota_exceeded` | — | upstream quota error |
| `metrics` | `{input_tokens, output_tokens, elapsed_ms, ttft_ms, preprocess_ms, tps, summarised, spilled, skipped, failed}` | fires once per stream — either at loop exit (aggregated across all tool-loop turns) or on a governor budget breach (partial values, no `history`/`done` follow) |
| `done` | `{}` | stream complete |
| `error` | `{message}` | upstream or internal error, **or** a governor breach (`LOOP_MAX_TOKENS`/`LOOP_MAX_SECONDS` in `config.py`) |

The UI JS handles all of these. Do not reorder or rename without updating both sides.

## Frontend — design standards

**Mobile first.** Design at 390 px, then let it widen; this is the convention for
every bix.computer site. Rules the chat UI follows and new pages should too:
`100dvh` + `overflow: hidden` on body with scrolling inside the content region;
tap targets ≥ 44 px (`--tap`); inputs at `font-size: 16px` so iOS doesn't zoom;
`env(safe-area-inset-*)` on anything pinned to an edge; one breakpoint at 900 px
for a side panel, below it sheets not sidebars; no horizontal page scroll at 360 px.

The UI uses the **Catppuccin Mocha** palette via CSS variables. Do not introduce new colours or override the palette for one-off styling; use the existing variables:

```css
--bg        #1e1e2e   /* page background */
--surface0  #313244   /* cards, drawer */
--surface1  #45475a   /* borders, inactive bars */
--text      #cdd6f4   /* body text */
--subtext   #a6adc8   /* labels, secondary text */
--yellow    #f9e2af   /* local/Ollama accent */
--blue      #89b4fa   /* streaming, live values */
--green     #a6e3a1   /* done state */
--red       #f38ba8   /* errors, cancel */
--mauve     #cba6f7   /* brand, user bubbles, focus rings */
```

### Typography

**Avoid:**
- Inter as the hero typeface — the UI uses `system-ui, -apple-system, sans-serif` intentionally
- Space Grotesk, Geist, or Instrument Serif as go-to pairings
- Serif italic accent words as a stylistic trick within an otherwise-sans context

**Do instead:**
- Stick to the system font stack; vary scale and weight meaningfully
- Use `font-variant-numeric: tabular-nums` on all metric values so they don't jump width while updating

### Color

**Avoid:**
- Adding a new accent colour for a new feature — map to an existing palette variable
- Medium-grey body text that barely passes WCAG AA — use `--text` or `--subtext` only
- Gradient backgrounds, coloured glows, or large box-shadows in brand colours

**Do instead:**
- Use colour to carry meaning that's already established: yellow = local model, blue = live/streaming, green = success, red = error, mauve = brand/user action
- When adding a new status, map it to one of these — don't add a new colour

### Layout

**Avoid:**
- Identical cards with icon-top / heading / body — vary density
- Sidebar nav items with emoji icons prepended (the drawer uses text labels)
- All-caps labels as the only typographic hierarchy — the UI already uses `.d-title` for that; don't proliferate the pattern
- Glassmorphism (`backdrop-filter: blur()`) — not used anywhere, don't introduce it

**Do instead:**
- Let content hierarchy determine structure; match existing spacing (the drawer uses `gap: 9px` sections, `gap: 20px` between sections)
- New drawer sections should use `.d-section` + `.d-title` to stay visually consistent
- New stat displays should use `.hi-stat` / `.hi-stats` grid unless there's a strong reason not to

### Component patterns

- The stats grid is `display: grid; grid-template-columns: 1fr 1fr` — use `.hi-stats` / `.hi-stat` for any two-column key/value display
- Progress bars use `.bar-wrap` + `.bar-fill` (optionally `.warn` or `.crit`) — don't reinvent this
- Collapsible sections use the sub-bubble pattern (`.sub-bubble` + `.sub-bubble-hdr` + `.sub-bubble-body`) — reuse for any expandable detail block

---

## Backend — code standards

Imported and adapted from the shared project standards:

- **Fail gracefully; surface every failure; never patch over one — even temporarily.**
  A limit breach or upstream failure must never masquerade as success (no clean
  `done` after a truncated tool call) and should degrade the turn, not the whole
  stream: report what happened, what already ran, and leave partial work visible.
  Governor caps (`LOOP_MAX_TURNS`/`SECONDS`/`TOKENS`) are runaway backstops sized
  so legitimate work never hits them; the per-turn `budget` SSE event exists so
  limits are visible *before* they bite. "Temporarily" ignoring an error (a bare
  `except: pass` to ship something, a TODO to surface it later) is how failures
  get permanently invisible — surface it now or don't merge it.
- **Never swallow errors silently.** Log via `log.error()` or `log.warning()`, then emit `sse("error", {"message": str(e)})` so the client knows. Don't return a 200 with a silent failure.
- **Never hide preprocessing failures.** If `strategy.preprocess()` raises, log the error and forward the original body — don't silently drop messages.
- **Never interpolate variables into log messages with `%s` and then use f-strings** — pick one style per call site. The codebase uses `log.info("msg key=%s", val)` style throughout; keep it consistent.
- **Prefer `async` throughout.** All route handlers and helpers are async. Don't introduce sync blocking calls (file I/O, `requests`, `time.sleep`) on the event loop.
- **httpx only.** All outbound HTTP (Anthropic, Ollama) uses `httpx.AsyncClient`. Don't add `requests` or `aiohttp`.
- **Don't catch exceptions broadly then continue as if nothing happened.** The `except Exception: pass` pattern in `system_metrics()` is intentional (GPU stats are best-effort). Elsewhere, handle or re-raise with context.
- **Strategy is pure logic.** `strategy.py` takes a body dict and an `ollama_chat` callable — it has no imports from `main.py`. Importing leaf modules like `config.py` is fine; keep it decoupled from `main.py` and the FastAPI app so it stays unit-testable in isolation.
- **There is no write path.** bix-ai never writes to the mounted tree — not from a
  tool, not from a route. `fs_core.is_denied_path` (secrets) must never be weakened.
  The only disk writes are the service's own data under `DATA_DIR` (memories,
  conversations, blobs) and `logs/`.
- **One knob per external host.** Ollama's host is `config.OLLAMA_HOST` everywhere — don't reintroduce a hardcoded `host.docker.internal` at a new call site; derive from `OLLAMA_HOST`/`OLLAMA_URL`.

## Gotchas

- **Container rebuild required for any change.** `static/index.html` is `COPY`-ed at build time — there is no volume mount for it. Always rebuild after editing frontend or backend files.
- **`OPENROUTER_API_KEY` is read from the environment** (put it in `bix-infra/.env` and map it in the compose file like the Anthropic key). Without it `mode=openrouter` returns 503 and the picker says "not configured". The spend caps default to $0.50/request and $5/day (`OPENROUTER_MAX_USD_PER_*`); 0 disables a cap. Today's total is computed from `routing.ndjson`, so deleting the log resets it.
- **`ANTHROPIC_API_KEY` comes from `bix-infra/.env` (as `BIX_AI_API_KEY`)**, not from a `.env` inside `bix-ai/`. If Claude requests fail, check the key is present in the running container: `docker exec apps-ai-router-1 env | grep ANTHROPIC`. Symptom of a missing/bad key used to be a silent zero-token "done" — since Phase 6 a 401 surfaces as an `error` SSE event.
- **Ollama unreachable from container** unless `OLLAMA_HOST=0.0.0.0` is set in the Ollama systemd override (yes, this is a different `OLLAMA_HOST` — Ollama's own bind-address env var, not this repo's `config.OLLAMA_HOST` client-side knob; same name, opposite side of the connection). The container reaches Ollama via `host.docker.internal:11434` by default; running outside Docker, set this repo's `OLLAMA_HOST=http://localhost:11434`.
- **AMD GPU (RX 6600M / gfx1032) uses the Vulkan backend, not ROCm.** Ollama's bundled rocBLAS has never shipped gfx1032 kernels (checked through v0.24.0 — preset includes gfx1030 but not gfx1032). The old `HSA_OVERRIDE_GFX_VERSION=10.3.0` spoof made the GPU pretend to be gfx1030 so it could borrow those kernels — it worked silently until Ollama 0.21 tightened GPU discovery, after which the spoofed device hangs the ROCm probe for 30s every cold start and Ollama falls back to CPU. Required env vars in the systemd override (`/etc/systemd/system/ollama.service.d/override.conf`):
    - `OLLAMA_VULKAN=1` — enable the Vulkan backend
    - `OLLAMA_LLM_LIBRARY=vulkan` — skip the ROCm probe entirely (saves the 30s timeout)
    - Do NOT set `HSA_OVERRIDE_GFX_VERSION` or any `*_VISIBLE_DEVICES` envs — they trigger an "override visible devices" warning and don't help.

    Symptom of regression: `curl localhost:11434/api/ps` shows `size_vram: 0` after loading a model, and `journalctl -u ollama` shows `failure during GPU discovery` or `inference compute id=cpu library=cpu` at startup. Expected healthy state: startup log line `inference compute id=gpu0 library=vulkan ...` and `size_vram > 0` after model load. Vulkan delivers ~70-90% of theoretical ROCm perf on RDNA2 for chat workloads — fine for this use case.
- **`gemma4:e2b` fails to load on this host** (llama runner exit status 2, any prompt size, even with nothing else resident — broken since the Vulkan backend switch). It is still the *code default* for `SUMMARY_LOCAL_MODEL`, so the pre-pass classify/salience fallback silently fails open unless `SUMMARY_LOCAL_MODEL` is overridden. (Context summaries don't use it — they run on the answering provider.) The compose file sets `SUMMARY_LOCAL_MODEL=gemma4:26b` for the container; export it yourself for dev runs. Fix or re-pull e2b to get the fast summariser back.
- **Ollama unloads models** after ~5 minutes idle. The GPU section in the sidebar shows "idle" when this happens — that's expected.
- **SSE golden fixtures:** `tests/fixtures/sse/*.json` pin the exact event sequences the claude/ollama loops emit (recorded pre-Phase-4-refactor). If `tests/test_sse_fixtures.py` fails, the SSE contract drifted — that's usually a bug, not a fixture-refresh situation. Only regenerate (`RECORD_SSE=1 pytest tests/test_sse_fixtures.py`) when the change is intended, and update the UI to match.
- **Tests:** `cd bix-ai && pytest -q` — 21 test files under `tests/`. Fully offline: all httpx clients and `ollama_chat` callables are faked, so the suite never hits Anthropic or Ollama (no token cost; ~0.5s runtime). A `.venv/` exists in `bix-ai/` (gitignored) with requirements + pytest installed — use `.venv/bin/python -m pytest -q` for local runs; the system Python has neither httpx nor fastapi. Strategy fixtures live under `tests/fixtures/` (regenerable — a ~4k-line logfile with one buried ERROR+traceback, a big source file, a huge JSON). `config.FS_ROOT`/`DATA_DIR`/`CONVERSATIONS_DIR` are read at call time specifically so tests can monkeypatch them — keep that property. `pytest` is not in `requirements.txt` (it's installed only in the Docker test stage); install it separately for local runs.

## Model management events

`POST /models/ollama/pull` only (not `/chat`): `pull_progress {status, digest?,
total?, completed?}`, `pull_done {name}`, `error`. Ollama's pull stream has long
quiet phases — keep the `with_keepalive` wrapper or Cloudflare kills the stream
at ~100 s idle. `GET /version` reports `{git_sha, built_at, started_at}`; the
Dockerfile stamps `GIT_SHA`/`BUILT_AT` via build args **at the end of the runtime
stage** — moving those ARGs earlier busts the pip/test layer cache on every build.

## When Claude makes a repeat mistake

Add a short rule here describing the mistake and the fix. Prune periodically.
