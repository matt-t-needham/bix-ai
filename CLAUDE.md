# CLAUDE.md — bix-ai

## Project overview

A FastAPI service that proxies Anthropic API calls, pre-summarises large context blocks via a local Ollama model to reduce token costs, and serves a single-file dark-theme chat UI at `ai.bix.computer`.

Three concerns live here:
- **`main.py`** — FastAPI app. All routes, SSE streaming helpers, system metrics, summarisation endpoint.
- **`strategy.py`** — Pre-summarisation pipeline. Walks messages, compresses blocks >2000 estimated tokens via Ollama in parallel with `asyncio.gather`. Tags summaries with `[router-summary v1]` to prevent re-summarisation.
- **`static/index.html`** — Single-file chat UI. No build step. Vanilla JS + CSS.

## How to run

```bash
# Dev (outside Docker)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Production rebuild (from repo root)
docker compose build ai-router && docker compose up -d ai-router
docker logs apps-ai-router-1 -f
```

`ANTHROPIC_API_KEY` is read from `/home/matt/apps/.env` (Docker Compose picks it up automatically). Ollama must be running on the host with `OLLAMA_HOST=0.0.0.0`.

## SSE event protocol

`POST /chat` streams these events in order:

| Event | Payload | Notes |
|-------|---------|-------|
| `status` | `{stage, message}` | checking / summarising / streaming |
| `preprocess` | `{summarised, skipped, failed, preprocess_ms}` | fires after pipeline, before Claude stream |
| `input_tokens` | `{count}` | fires on Anthropic `message_start` |
| `delta` | `{text}` | streaming text chunk |
| `tool_start` | `{index, name, id}` | tool call begins |
| `tool_input` | `{index, partial_json}` | streaming tool input |
| `tool_end` | `{index}` | tool call complete |
| `metrics` | `{input_tokens, output_tokens, elapsed_ms, ttft_ms, preprocess_ms, tps, summarised, skipped, failed}` | fires on `message_stop` |
| `done` | `{}` | stream complete |
| `error` | `{message}` | upstream or internal error |

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
- **Strategy is pure logic.** `strategy.py` takes a body dict and an `ollama_chat` callable — it has no imports from `main.py`. Keep it that way so it can be unit-tested in isolation.

## Gotchas

- **Container rebuild required for any change.** `static/index.html` is `COPY`-ed at build time — there is no volume mount for it. Always rebuild after editing frontend or backend files.
- **`ANTHROPIC_API_KEY` comes from `/home/matt/apps/.env`**, not from `.env` inside `bix-ai/`. If Claude requests fail, check the key is present in the running container: `docker exec apps-ai-router-1 env | grep ANTHROPIC`.
- **Ollama unreachable from container** unless `OLLAMA_HOST=0.0.0.0` is set in the Ollama systemd override. The container reaches Ollama via `host.docker.internal:11434`.
- **AMD GPU (RX 6600M / gfx1032) uses the Vulkan backend, not ROCm.** Ollama's bundled rocBLAS has never shipped gfx1032 kernels (checked through v0.24.0 — preset includes gfx1030 but not gfx1032). The old `HSA_OVERRIDE_GFX_VERSION=10.3.0` spoof made the GPU pretend to be gfx1030 so it could borrow those kernels — it worked silently until Ollama 0.21 tightened GPU discovery, after which the spoofed device hangs the ROCm probe for 30s every cold start and Ollama falls back to CPU. Required env vars in the systemd override (`/etc/systemd/system/ollama.service.d/override.conf`):
    - `OLLAMA_VULKAN=1` — enable the Vulkan backend
    - `OLLAMA_LLM_LIBRARY=vulkan` — skip the ROCm probe entirely (saves the 30s timeout)
    - Do NOT set `HSA_OVERRIDE_GFX_VERSION` or any `*_VISIBLE_DEVICES` envs — they trigger an "override visible devices" warning and don't help.

    Symptom of regression: `curl localhost:11434/api/ps` shows `size_vram: 0` after loading a model, and `journalctl -u ollama` shows `failure during GPU discovery` or `inference compute id=cpu library=cpu` at startup. Expected healthy state: startup log line `inference compute id=gpu0 library=vulkan ...` and `size_vram > 0` after model load. Vulkan delivers ~70-90% of theoretical ROCm perf on RDNA2 for chat workloads — fine for this use case.
- **Ollama unloads models** after ~5 minutes idle. The GPU section in the sidebar shows "idle" when this happens — that's expected.
- **`strategy.py` tests:** `cd bix-ai && pytest tests/` — four tests covering below-threshold, above-threshold, already-summarised, and tool_result blocks.

## When Claude makes a repeat mistake

Add a short rule here describing the mistake and the fix. Prune periodically.
