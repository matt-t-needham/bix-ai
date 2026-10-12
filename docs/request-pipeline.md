# What happens on one chat turn (inventory, as of 2026-10-12, pre-overhaul)

Written to answer "what processes run and what stats come out of a single
request" before the overhaul. Everything here describes `main` as it was
before PR 1; `PLAN-overhaul.md` says what stayed and what went (auto mode,
staging, Steam, logs, `/summarize` and the memory title call are already gone).

Legend: **[model]** = an LLM call, **[disk]** = a disk write, **[poll]** = a
repeating fetch while the stats drawer is open.

## 1. Client, before the request

1. If the page was opened as `/?staged=<id>`, the staged-change context is
   fetched and prepended to the first message. *(staging — being removed)*
2. The user message is pushed to `convHistory`.
3. `POST /chat` is sent with the **entire** `convHistory` every time — the
   server is stateless; the browser is the only place the conversation lives.
   Reloading the page loses it.

## 2. Server, `POST /chat`

1. Body size check (413 over `_MAX_BODY_BYTES`), schema validation.
2. Model allowlist. Claude: static set in `config.py`. Local: any installed
   model — one `GET /api/tags` to Ollama, cached 30 s.
3. Blob hashes referenced by the conversation are pinned for the stream's
   lifetime (so LRU eviction can't pull them mid-loop).
4. Dispatch on `mode`:

| mode | path | notes |
|---|---|---|
| `local` | `streaming/ollama.py` | direct to Ollama |
| `api` | `streaming/claude.py` | Claude, per-token billing via `ANTHROPIC_API_KEY` |
| `auto` | `streaming/local_first.py` → one of the above | routing decision first |
| anything else (`pro`) | `streaming/pro.py` | `claude` CLI subprocess, subscription billing |

## 3. `mode=auto` only: routing decision

`routing.decide()` runs structural rules over the conversation (code fences,
prose-deliverable words, multi-step shape, request length, context size). If
nothing matches it makes **[model]** one Ollama classification call with
`SUMMARY_LOCAL_MODEL` (`gemma4:e2b` by default — which doesn't load on this
host, so this fails open to Claude). The decision is logged to
`routing.ndjson` with a `reason` and possibly a cheaper-Claude-model hint.

## 4. Pre-pass (`local`, `api`, `auto`, `pro` — all four)

Runs before the model sees anything. Emits `status` → `preprocess` → `status`.

1. **Spill** (`strategy.preprocess`). For every message block over ~6000
   estimated tokens: strip ANSI, collapse duplicate lines, classify the block
   (`logfile|source|json|diff|prose` — heuristics, **[model]** one Ollama call
   only when unsure), extract salient lines verbatim, write the original to
   the blob store **[disk]**, replace the block in-band with a
   `[router-blob v2 <hash>]` pointer + excerpt. Stats: `spilled`, `skipped`,
   `failed` (`summarised` is always 0 — v1 wire-compat).
2. **Compact** (`compact.compact`). If narrative tokens exceed
   `COMPACT_THRESHOLD_TOKENS` (20k), turns older than the last 3 user turns
   are folded into one `[router-compact v1]` pair via **[model]** one Ollama
   call with `SUMMARY_LOCAL_MODEL`. Stat: `compacted`.
3. `preprocess_ms` is measured over both and added to the server-wide `_agg`
   counters (served by `GET /stats`).

In practice: for a normal-sized chat both steps are no-ops that cost a few
milliseconds of regex. They only bite on huge pastes or very long sessions,
and when they do bite the summariser model usually isn't loadable.

## 5. System prompt

`identity.py` builds one prompt per mode: identity core + ground rules +
(local only) a one-line capability index of every tool + a pointer to
`docs/*.md` topics the model may `read_file`. The Claude path sends the full
tool schemas on the wire instead of the index. Pro mode adds a TODO-folder
blurb on top.

## 6. The tool loop (`local`, `api`, `auto`)

`streaming/loop.py:run_tool_loop`, shared by all three. Up to 10 turns.

Per turn:
- `budget` SSE: turn / elapsed / tokens against the governor caps
  (`LOOP_MAX_TURNS`/`SECONDS`/`TOKENS`).
- **[model]** one streamed provider call (`AnthropicProvider` or
  `OllamaProvider`), translated to `delta`, `tool_start`, `tool_input`,
  `tool_end`, `input_tokens` (turn 0 only).
- If the model called tools: each runs via `tools._execute_tool`, then
  `tool_result` SSE (content capped at 4000 chars on the wire, full in
  history). Tools: `list_directory`, `read_file`, `stage_write`,
  `read_deploy`, `list_deploys`, `list_steam_games`, `list_log_sources`,
  `read_log`, `recall_memories`, `read_blob`, `grep_blob`, `check_staging`,
  `ask_staging`. (Only the first two, `recall_memories` and the blob pair
  survive the overhaul.)
- Ollama only: guardrail layer rescues garbled tool JSON and retries with a
  nudge; counts go into the routing log.
- Local only: `with_progress` heartbeat so a multi-minute prompt-eval on the
  Vulkan backend doesn't look dead. All paths: `with_keepalive` so
  Cloudflare doesn't cut the stream at ~100 s idle.

At loop exit:
- `metrics` SSE (see §9).
- `history` SSE — the canonical message list including tool turns. The
  client replaces `convHistory` with it wholesale.
- `done`.
- **[disk]** one line appended to `routing.ndjson` (mode, model, reason,
  tokens, timings, `est_cost_usd` from `config.MODEL_COSTS`).

`mode=auto`: if the local leg errors, `fallback_triggered` clears the partial
output and the same request re-runs on Claude with `skip_preprocess=True`.

## 7. `mode=pro` instead of §6

Pre-pass (§4) still runs. Then the history is flattened to text (each old
message truncated to 3000 chars) — or, if the client echoed a `session_id`,
only the newest message is sent with `claude --resume`. **[model]** one
`claude -p` subprocess over MCP (`bix_mcp.py`, tools `list_directory`,
`read_file`, `recall_memories`, `write_file`). Its `stream-json` output is
translated to the same `delta`/`tool_*` events plus `pro_session`. No
`history` event; the client falls back to a text-only history push. Quota
exhaustion arrives as `quota_exceeded` and the UI offers "Use API credits",
which flips the next request to `mode=api`.

## 8. Client, after `done` — the hidden cost

Two background requests fire on **every** completed turn:

1. `POST /summarize` — **[model]** one Ollama call for a 6-word title for the
   history card. Hardcoded `gemma4:e2b` unless in local mode. On this host
   that model doesn't load, so it waits, fails, and the card shows nothing.
2. `POST /memory/save` — **[model]** one Ollama call (`gemma4:26b`) for a
   title, dedup against the last 10 titles, **[model]** a second
   `gemma4:26b` call for a 2–3 sentence summary, keyword tagging, **[disk]**
   write the whole conversation to `data/convos/*.json`, **[disk]** append to
   the memory file, and every 10th entry a consolidation pass.

So a one-line Claude reply is followed by up to three local model calls and
two file writes, and the 26B model gets loaded into VRAM whether or not you
are using a local model. The memory file also accretes one entry per *turn*
(not per conversation), deduped only by exact title match.

## 9. Stats produced

### On the wire (SSE)
`status`, `preprocess {summarised, spilled, compacted, skipped, failed,
preprocess_ms}`, `input_tokens`, `budget {turn, max_turns, elapsed_s,
max_seconds, tokens, max_tokens}`, `metrics {input_tokens, output_tokens,
elapsed_ms, ttft_ms, preprocess_ms, tps, summarised, spilled, skipped,
failed}`, plus `model_swap`, `fallback_triggered`, `quota_exceeded`,
`pro_session`, `error`.

### Server-side aggregates
- `GET /stats` — process-lifetime counters: requests, summarised, spilled,
  checked, preprocess_ms, failed.
- `GET /system` — CPU %, RAM used/total/%, GPU `{state, size_vram,
  size_total, num_gpu}` from Ollama `/api/ps` (3 s cache).
- `GET /auth/status` — Claude CLI login state + token expiry, API key present.
- `GET /version`, `GET /memory`, `GET /models/ollama`, `GET /staging/count`,
  `GET /deploys/count`.
- `routing.ndjson` → `GET /routing` dashboard (local-vs-Claude split, daily
  spend, per-model/mode/rule tables).
- Blob store → `GET /blobs` housekeeping page.

### Where the UI shows them
- Header: status dot + label, governor gauge (`turn 1/10 · 3s/1200s · 2k/1M tok`).
- Under each assistant bubble: `in N · out N · TTFT Nms · N tok/s · Nms total
  [· prep Nms] [· N spilled]`.
- Drawer (all collapsed by default), **[poll]** `/system` + `/stats` every 2 s
  while open:
  - *AI Router*: auth state, build sha/role, Requests/Spilled/Checked/Prep,
    avg TTFT per Claude model.
  - *System*: CPU, RAM, VRAM bars; GPU offload layers.
  - *Model*: picker (Claude + installed Ollama), tool-offload toggle + help.
  - *Current Request*: status, mode·model, In/Out/TTFT/Tok/s/Total/Prep/
    Est. cost, Compressed/Spilled/Compacted/Passed/Failed.
  - *Memory*: count + list linking to `/memory/{id}`.
  - *Models*: installed Ollama models, pull with progress, delete.
  - *History*: last 12 requests — time, model chain, mode tag, the
    `/summarize` title, Tokens/Cost/TTFT/Tok/s/Total/Compressed/Spilled/
    Compacted; session cost total.
  - Links out: Internal Metrics, Staged writes, Deploys, Routing dashboard,
    Blob store.

### What's missing
- Nothing says how much *history* went up with the request (messages, tokens,
  whether anything was folded/pointered, what the model could actually see).
- Cost is per-request estimates only; there is no running daily/monthly
  figure against a budget, and nothing for a pay-as-you-go provider.
- TTFT/tok/s are shown for cloud models where they mean nothing to you, and
  the local-only GPU view is buried two clicks deep.
