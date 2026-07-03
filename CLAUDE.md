# CLAUDE.md — bix-ai

## Project overview

A FastAPI service that acts as an efficient cross-platform task runner: it proxies
Anthropic API calls, routes between local Ollama models and Claude, gives models an
agentic tool loop (filesystem reads, gated writes, logs, memory, Steam library, TODOs),
spills oversized context blocks to a content-addressed blob store (replacing them
in-band with a verbatim excerpt + retrieval pointer), and serves a single-file
dark-theme chat UI at `ai.bix.computer`.

This is not a 3-file project. Module map:

| Module | Role |
|---|---|
| `main.py` | FastAPI app — all routes, SSE streaming dispatch, system metrics, staging review UI routes |
| `config.py` | All tunables/env vars — `OLLAMA_HOST` is the single source for every Ollama host reference |
| `strategy.py` | Pre-pass pipeline (Phase 3 shape): per oversized block (>6000 est. tokens) it losslessly reduces (ANSI strip + duplicate-line collapse), labels via heuristics (`logfile\|source\|json\|diff\|prose`, one Ollama call only when unsure), extracts salient lines **verbatim** by type (code does the slicing; the model never rewrites a byte), spills the original to `blobstore.py`, and replaces the block with `[router-blob v2 <hash>]` pointer + excerpt. Pure logic, `preprocess(body, ollama_chat) -> (new_body, stats)`, no imports from `main.py`. Ollama is used for *classification/ranking only* — never paraphrase. The v1 paraphrase pipeline is abolished; `[router-summary v1]` blocks are still recognised and skipped |
| `tools.py` | Tool dispatcher (`_execute_tool`) plus three parallel tool-definition tables: `FS_TOOLS` (Anthropic shape), `OLLAMA_TOOLS` (OpenAI fn shape), `FORGE_TOOLS` (forge `ToolDef`) — deliberate duplication, not yet consolidated |
| `fs_core.py` | Path security — `is_denied_path` (secrets), `is_write_denied_path` (scripts/CI/bix-ai's own source) |
| `staging.py` | Gated writes: propose → human review → apply. Re-validates every guard at approve time. Review routes live in `main.py` under `/staging…` |
| `blobstore.py` | Content-addressed store for oversized artifacts spilled out of context (`put`/`get`/`grep`/`stat`, sha256-keyed, write-once, dedup). LRU eviction over `config.BLOB_STORE_MAX_BYTES`, pin/unpin protects blobs referenced by the in-flight request. Backs the `read_blob`/`grep_blob` tools. `strategy.py` spills oversized blocks into it automatically; `main.py`'s chat dispatcher pins every hash referenced by the in-flight request (`strategy.referenced_blob_hashes`) for the stream's lifetime. `/app/data` is a compose bind mount of `bix-ai/data`, so blobs persist across deploys |
| `memory.py` | Memory persistence (`/memory` routes, `recall_memories` tool); conversations under `DATA_DIR/convos` |
| `bix_mcp.py` | MCP server exposed to the `claude` CLI subprocess for mode="pro" |
| `steam.py`, `logtools.py`, `todos.py` | Backing implementations for the `list_steam_games`, `list_log_sources`/`read_log`, and `read_todos` tools |
| `helpers.py` | SSE helpers (`sse()`, `with_keepalive`), routing-log writer, Ollama chat helper |
| `streaming/claude.py` | Agentic loop against Anthropic — `for _turn in range(10)`, exits on `stop_reason != "tool_use"` |
| `streaming/ollama.py` | Same loop shape against Ollama's OpenAI-compatible endpoint, exits on `finish_reason != "tool_calls"` |
| `streaming/forge_runner.py` | Wraps forge-guardrails' `WorkflowRunner` for mode="auto" (local-first, its own loop/compaction) |
| `streaming/local_first.py` | mode="auto" orchestration — tries forge first, escalates to `_stream_claude` on error, emits `fallback_triggered` |
| `streaming/pro.py` | mode="pro" — drives the `claude` CLI subprocess over MCP |
| `static/index.html` | Single-file chat UI. No build step. Vanilla JS + CSS. Keeps `textSegments` (render) separate from `convHistory` (request payload) |

Four `mode` values on `POST /chat`: `local` (direct to Ollama), `api` (direct to
Claude), `auto` (forge-first local, escalates to Claude on error), and everything else
falls through to `pro` (subprocess `claude` CLI with MCP tools).

See `PLAN-pi-tools.md` for the active roadmap and the gaps it's closing (tool-turn
history currently doesn't survive across requests; artifact compression is lossy;
no blob store yet). Trust that plan over this file if they ever disagree — re-derive
this file from the code, the plan says so explicitly.

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

`ANTHROPIC_API_KEY` is read from `/home/matt/apps/.env` (Docker Compose picks it up
automatically). Ollama must be running with `OLLAMA_HOST=0.0.0.0` (systemd override) so
the container can reach it — in Docker this resolves via `host.docker.internal`, which
is `config.OLLAMA_HOST`'s default. Override `OLLAMA_HOST` when running outside Docker.

The Docker build has a **hard test gate**: the `test` stage runs `pytest -q` and the
`runtime` stage only exists via `COPY --from=test`, so a failing suite fails the build.
pytest itself never ships in the runtime image.

## SSE event protocol

`POST /chat` streams these events (not all paths emit all events — see notes):

| Event | Payload | Notes |
|-------|---------|-------|
| `status` | `{stage, message}` | checking / summarising / streaming (the `summarising` stage's message reads "Preparing context…" — it covers the whole pre-pass, not just model calls) |
| `preprocess` | `{summarised, spilled, skipped, failed, preprocess_ms}` | fires after `strategy.preprocess`, before the model stream. `spilled` = blocks pointered to the blob store; `summarised` is always 0 now (kept for wire compatibility with v1) |
| `input_tokens` | `{count}` | fires on `message_start` (turn 0 only across a tool loop) |
| `delta` | `{text}` | streaming text chunk |
| `tool_start` | `{index, name, id}` | tool call begins |
| `tool_input` | `{index, partial_json}` | streaming tool input |
| `tool_end` | `{index}` | tool call complete |
| `tool_result` | `{tool_use_id, content, is_error}` | emitted by all three loop paths (`claude.py`, `ollama.py`, `pro.py`); content truncated to 4000 chars for the SSE event only — the full result still goes into `history` |
| `history` | `{messages}` | fires **once, at clean loop completion, before `done`** — the canonical transformed message list (assistant `tool_use` + `tool_result` turns included). `claude.py`/`ollama.py` only; `pro.py` doesn't emit it (subprocess owns its own loop) — the UI keeps a text-only `convHistory` fallback for that path. Never emitted on `error`/disconnect |
| `model_swap` | — | mode="auto" escalation/model change |
| `fallback_triggered` | — | mode="auto" forge→Claude escalation; clears partial local output in the UI |
| `quota_exceeded` | — | upstream quota error |
| `metrics` | `{input_tokens, output_tokens, elapsed_ms, ttft_ms, preprocess_ms, tps, summarised, spilled, skipped, failed}` | fires once per stream — either at loop exit (aggregated across all tool-loop turns) or on a governor budget breach (partial values, no `history`/`done` follow) |
| `done` | `{}` | stream complete |
| `error` | `{message}` | upstream or internal error, **or** a governor breach (`LOOP_MAX_TOKENS`/`LOOP_MAX_SECONDS` in `config.py`) |

The UI JS handles all of these. Do not reorder or rename without updating both sides.

## Frontend — design standards

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

- **Never swallow errors silently.** Log via `log.error()` or `log.warning()`, then emit `sse("error", {"message": str(e)})` so the client knows. Don't return a 200 with a silent failure.
- **Never hide preprocessing failures.** If `strategy.preprocess()` raises, log the error and forward the original body — don't silently drop messages.
- **Never interpolate variables into log messages with `%s` and then use f-strings** — pick one style per call site. The codebase uses `log.info("msg key=%s", val)` style throughout; keep it consistent.
- **Prefer `async` throughout.** All route handlers and helpers are async. Don't introduce sync blocking calls (file I/O, `requests`, `time.sleep`) on the event loop.
- **httpx only.** All outbound HTTP (Anthropic, Ollama) uses `httpx.AsyncClient`. Don't add `requests` or `aiohttp`.
- **Don't catch exceptions broadly then continue as if nothing happened.** The `except Exception: pass` pattern in `system_metrics()` is intentional (GPU stats are best-effort). Elsewhere, handle or re-raise with context.
- **Strategy is pure logic.** `strategy.py` takes a body dict and an `ollama_chat` callable — it has no imports from `main.py`. Importing leaf modules like `config.py` is fine; keep it decoupled from `main.py` and the FastAPI app so it stays unit-testable in isolation.
- **All disk writes go through `staging.py`.** Never bypass the propose → review → apply flow, never auto-apply. `is_write_denied_path` protections (including bix-ai's own source) must never be weakened.
- **One knob per external host.** Ollama's host is `config.OLLAMA_HOST` everywhere — don't reintroduce a hardcoded `host.docker.internal` at a new call site; derive from `OLLAMA_HOST`/`OLLAMA_URL`.

## Gotchas

- **Container rebuild required for any change.** `static/index.html` is `COPY`-ed at build time — there is no volume mount for it. Always rebuild after editing frontend or backend files.
- **`ANTHROPIC_API_KEY` comes from `/home/matt/apps/.env`**, not from `.env` inside `bix-ai/`. If Claude requests fail, check the key is present in the running container: `docker exec apps-ai-router-1 env | grep ANTHROPIC`.
- **Ollama unreachable from container** unless `OLLAMA_HOST=0.0.0.0` is set in the Ollama systemd override (yes, this is a different `OLLAMA_HOST` — Ollama's own bind-address env var, not this repo's `config.OLLAMA_HOST` client-side knob; same name, opposite side of the connection). The container reaches Ollama via `host.docker.internal:11434` by default; running outside Docker, set this repo's `OLLAMA_HOST=http://localhost:11434`.
- **AMD GPU (RX 6600M / gfx1032) uses the Vulkan backend, not ROCm.** Ollama's bundled rocBLAS has never shipped gfx1032 kernels (checked through v0.24.0 — preset includes gfx1030 but not gfx1032). The old `HSA_OVERRIDE_GFX_VERSION=10.3.0` spoof made the GPU pretend to be gfx1030 so it could borrow those kernels — it worked silently until Ollama 0.21 tightened GPU discovery, after which the spoofed device hangs the ROCm probe for 30s every cold start and Ollama falls back to CPU. Required env vars in the systemd override (`/etc/systemd/system/ollama.service.d/override.conf`):
    - `OLLAMA_VULKAN=1` — enable the Vulkan backend
    - `OLLAMA_LLM_LIBRARY=vulkan` — skip the ROCm probe entirely (saves the 30s timeout)
    - Do NOT set `HSA_OVERRIDE_GFX_VERSION` or any `*_VISIBLE_DEVICES` envs — they trigger an "override visible devices" warning and don't help.

    Symptom of regression: `curl localhost:11434/api/ps` shows `size_vram: 0` after loading a model, and `journalctl -u ollama` shows `failure during GPU discovery` or `inference compute id=cpu library=cpu` at startup. Expected healthy state: startup log line `inference compute id=gpu0 library=vulkan ...` and `size_vram > 0` after model load. Vulkan delivers ~70-90% of theoretical ROCm perf on RDNA2 for chat workloads — fine for this use case.
- **Ollama unloads models** after ~5 minutes idle. The GPU section in the sidebar shows "idle" when this happens — that's expected.
- **Tests:** `cd bix-ai && pytest -q` — 14 test files under `tests/`, 120 tests as of Phase 3. A `.venv/` exists in `bix-ai/` (gitignored) with requirements + pytest installed — use `.venv/bin/python -m pytest -q` for local runs; the system Python has neither httpx nor fastapi. Strategy fixtures live under `tests/fixtures/` (regenerable — a ~4k-line logfile with one buried ERROR+traceback, a big source file, a huge JSON). `config.FS_ROOT`/`STAGING_DIR` are read at call time specifically so tests can monkeypatch them — keep that property. `pytest` is not in `requirements.txt` (it's installed only in the Docker test stage); install it separately for local runs.

## When Claude makes a repeat mistake

Add a short rule here describing the mistake and the fix. Prune periodically.
