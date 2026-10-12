# PLAN — bix-ai overhaul (2026-10)

Why: the app isn't used. It's slow, the UI is desktop-only, the stats are
both too many and the wrong ones, and most of what's in the repo was built
when bix-ai was the hub for everything — those pieces have since moved into
their own services. `docs/request-pipeline.md` is the audit this plan was
written against. This plan wins over `CLAUDE.md` where they disagree.

## Goal

A chat app I open on my phone. Three providers in one picker, switchable
mid-conversation without losing the thread:

| Provider | Backend | Billing |
|---|---|---|
| **Local** | Ollama via the existing tool loop | free |
| **Claude** | `claude` CLI subprocess (`mode=pro`) | Pro subscription; `mode=api` kept as the quota fallback |
| **OpenRouter** | OpenAI-compatible chat completions via the existing `OllamaProvider` shape | pay-as-you-go on my personal key, budget-capped server-side |

The server owns the conversation history (not the browser) and decides per
turn how much of it to send, so the local model isn't drowned and the
cloud models aren't billed for the whole transcript every message.

## Principles

- Chat is the product. Everything else is a drawer.
- Mobile first: design at 390 px, then let it widen. This is the convention
  for every bix.computer site from here on (see "Mobile rules" below).
- One model call per turn. Nothing runs a second model in the background
  after a reply unless the user can see it in the stats and it's doing
  something they asked for.
- Stats are per-provider. Local shows GPU/TTFT/tok/s. Cloud shows tokens,
  cost, and budget. Both show what history went up.
- Keep the backend code standards from `CLAUDE.md` (fail loudly, async,
  httpx only, pure-logic modules with injected callables).

## PR 1 — mobile-first UI + cleanup  *(this branch)*

### UI (`static/index.html`, still a single file, no build step)
- Layout: full-height chat column; header is one row (title, status, drawer
  button); composer pinned to the bottom with `env(safe-area-inset-bottom)`;
  messages scroll independently; no horizontal overflow at 360 px.
- Provider/model picker lives in the composer: a chip showing the current
  provider + model opens a bottom sheet with one section per provider.
  Switching keeps the conversation.
- Drawer: a full-screen sheet under 900 px, a 360 px side panel above. Same
  content volume as today, regrouped:
  - *This request* — status, provider·model, tokens, cost (cloud) or
    TTFT/tok-s (local), what history went up (count + tokens, filled in by
    PR 2), preprocess only when it actually did something.
  - *History* — last N turns, no background summary call: the card shows the
    first line of the user message. Session cost total.
  - *System* — CPU/RAM/VRAM/GPU offload (local-relevant; polled only while
    the drawer is open, as today).
  - *Local models* — installed list, pull, delete.
  - *Memory* — count + list.
  - *About* — auth state, build, links (Internal Metrics, Routing, Blobs).
- Drop from the UI: `auto` mode and its routing banner, `/summarize`
  background call, staged-context loading, Staged-writes / Deploys links,
  tool-offload help text (toggle stays, local only).
- Mobile rules (reusable across bix.computer):
  - `100dvh`, `overflow: hidden` on body; scroll inside the content region.
  - Tap targets ≥ 44 px; inputs `font-size: 16px` so iOS doesn't zoom.
  - `env(safe-area-inset-*)` on anything pinned to an edge.
  - Breakpoint at 900 px for a side panel; below it, sheets not sidebars.
  - System font stack, Catppuccin Mocha tokens, no new colours.

### Backend pruning (same PR, separate commits)
Remove the retired hub features so the service is just a chat router:
- Staging / self-modification: `staging.py`, `staging_ui.py`, `deploy.py`,
  `/staging*`, `/deploys*`, `/staging-service/version`, the `stage_write` /
  `read_deploy` / `list_deploys` / `check_staging` / `ask_staging` tools, the
  `BIX_ROLE` staging-twin split, pro-mode `write_file`. No file writes remain
  anywhere — `fs_core.is_write_denied_path` and friends go with them.
- Steam (`steam.py`, `list_steam_games`) and logs (`logtools.py`,
  `list_log_sources`, `read_log`) — they live in their own services now.
- `/summarize` route and `memory._summarize` title generation; memory save
  moves to PR 2 where it becomes an explicit, per-conversation action.
- Tests and docs for all of the above. The SSE golden fixtures must not
  change.

Also removed in PR 1: `auto` mode and its router (`routing.py`,
`local_first.py`, `routing_dash.py`, `/routing`). The per-turn
`routing.ndjson` usage log stays — it's the cost record PR 2 reads.
Not touched in PR 1 (PR 2 decides their fate): `strategy.py`, `compact.py`,
`blobstore.py`.

## PR 2 — OpenRouter + server-owned history

- `streaming/openrouter.py`: OpenAI-compatible provider, matching how
  bix-graph's `OpenRouterProvider` (`ai_graph_mode/lib/ai-providers.ts`)
  does it, minus the npm SDK:
  - `POST https://openrouter.ai/api/v1/chat/completions` with `stream: true`
    and `stream_options: {include_usage: true}` (always — we want usage for
    the cost figure, bix-graph only sends it for o-series).
  - Headers: `Authorization: Bearer $OPENROUTER_API_KEY` (from
    `bix-infra/.env`, never sent to the browser), `HTTP-Referer:
    https://ai.bix.computer`, `X-OpenRouter-Title: bix-ai` — our own
    identity, not BixRouter's, so spend is attributed correctly.
  - One model ID per request. No `models: [...]`/`route: fallback`.
  - Stream deltas: `content` → `delta`; `reasoning_content` → a new
    `thinking` SSE event (rendered collapsed, like a tool group); ignore
    `images[]` for now.
  - Model list: `GET /api/v1/models`, cached 5 min, drop audio-output and
    `openrouter/auto`-style meta-routers; derive vendor, price tier and
    "always reasons" (`reasoning.mandatory`) from the listing — don't
    guess from `/o1`/`/o3` substrings. A small static fallback list when
    the fetch fails. Free-text IDs accepted behind the same slug-shape
    check bix-graph uses (`isValidModel`), since it's my own key.
  - Temperature: leave the model default unless the user sets one.
  - Per-request and per-day USD caps enforced server-side before the call
    (`config.OPENROUTER_MAX_USD_PER_REQUEST` / `_PER_DAY`); the day total
    comes from our own routing log. OpenRouter's `GET /api/v1/auth/key`
    (`usage`, `limit`, `limit_remaining`) feeds the budget pill in the UI.
- `conversations.py`: server-side conversation store under `DATA_DIR`.
  `POST /chat` takes a `conversation_id` + the new message only. The client
  no longer round-trips the transcript.
- Context assembly per turn, by provider: a token budget (small for local,
  larger for cloud) filled newest-first with verbatim turns; older turns
  fall into a rolling summary that is regenerated only when the cut-off
  moves (one model call, on the provider that's answering — never a
  background local call). Memory recall stays a tool. The `metrics` event
  grows `history_messages` / `history_tokens` / `history_folded` so the
  drawer can show what went up.
- Decide, with numbers from PR 1 usage, whether `strategy.py`'s spill pass
  survives. Default position: spill stays (only fires on huge pastes);
  compaction is replaced by the per-provider context assembly above.

## Open questions

- Whether the Claude CLI path can report tokens reliably enough for the cost
  figure, or whether subscription turns just show "subscription".
