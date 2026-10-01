# PLAN — spike: back bix-ai's agent loop with oh-my-pi (omp)

Written 2026-10-01 for running **on the PC itself** (HX100G: real Ollama on
Vulkan, the real Anthropic key). The cloud session that wrote it couldn't
install omp (its network policy blocks npm, omp.sh and GitHub releases), so
**nothing below has run against real omp yet**. The spike client was only
self-tested against `spike/omp/fake_omp.py`, a scripted stand-in. Treat every
claim about omp's behaviour as "per its docs at commit `95acb04`" until the
spike confirms it.

Upstream: <https://github.com/can1357/oh-my-pi>. MIT licence, a fork of Mario
Zechner's Pi, TypeScript + Bun + a Rust native core. Very active (commits
daily), so pin a version.

## The question this answers

Should `streaming/loop.py`, `streaming/providers.py`, `streaming/claude.py`,
`streaming/ollama.py`, `streaming/pro.py`, `bix_mcp.py` and `compact.py`
(~1,850 lines, the hardest-to-maintain part of the repo) be replaced by omp
running as a subprocess in RPC mode? bix-ai would keep FastAPI, the UI/SSE
contract, `staging.py`/`fs_core.py`, `strategy.py`/`blobstore.py` and
`routing.py`.

The spike is a **go/no-go gate**. It does not change the running service.

## What the spike does NOT touch

- The bix-ai container, its compose config, or the prod tree. The spike runs
  from the repo checkout with plain `python3` (stdlib only).
- Your real omp config. It uses `--profile bixspike`, an isolated omp profile,
  plus `--config spike/omp/spike-config.yml`.
- The filesystem, beyond a throwaway temp dir. Built-in tools are off
  (`--no-tools`), the host tools are jailed to `<tmp>/sandbox`, and
  `stage_write` is a stub that only records calls.

## Phase 0 — preflight (≈15 min)

```bash
# 1. Install omp. Any one of these:
curl -fsSL https://omp.sh/install | sh
#   or: bun install -g @oh-my-pi/pi-coding-agent   (needs bun ≥ 1.3.14)
omp --version                       # record this in the results

# 2. Ollama: version + does it serve the OpenAI *Responses* API?
#    omp's implicit ollama provider speaks openai-responses, not /v1/chat/completions.
ollama --version
curl -s localhost:11434/v1/responses -H 'content-type: application/json' \
  -d '{"model":"qwen3.5:9b","input":"say ok"}' | head -c 400; echo
#    404 / "not found" → Ollama too old for implicit discovery. Either upgrade
#    Ollama, or add a models.yml for the bixspike profile pointing the ollama
#    provider at /v1 with `api: openai-completions` (docs/models.md upstream).

# 3. Credentials + host. Run from the bix-ai repo root.
export ANTHROPIC_API_KEY=$(grep BIX_AI_API_KEY /home/matt/apps/bix-infra/.env | cut -d= -f2)
export OLLAMA_HOST=http://localhost:11434
unset CLAUDE_CONFIG_DIR             # the script strips it from omp's env anyway

# 4. Plumbing self-test (no omp, no models). Should print 7× PASS in ~5 s.
python3 spike/omp/omp_spike.py --omp "python3 spike/omp/fake_omp.py" --profile ''
```

## Phase 1 — run the spike (≈10–30 min; local models are slow)

```bash
python3 spike/omp/omp_spike.py \
  --claude-model anthropic/claude-sonnet-4-5 \
  --local-model  ollama/qwen3.5:9b
# Subsets while iterating:  --only S1,S3     Keep temp dir:  --keep
# Also try the tool model the router uses, and the 26b:
python3 spike/omp/omp_spike.py --only S4 --local-model ollama/gemma4:26b
```

Each run writes `spike/omp/results/<timestamp>.json` (gitignored). **Paste
that file back into a Claude session** for the write-up.

| Check | What it proves | Pass criterion |
|---|---|---|
| **S1** handshake | omp starts in RPC mode under the locked-down flags; model discovery finds Ollama (and Anthropic) | `ready` frame; `ollama/*` models listed |
| **S3** Claude round-trip | host tools work end to end: bix-ai registers tools via `set_host_tools`, omp calls back with `host_tool_call`, bix-ai answers | `list_directory` + `read_file` both called via the host; answer contains the secret word; **no non-host tool ran** |
| **S4** local round-trip | the same, on Ollama. This replaces the forge-guardrails rescue/retry layer, so it's the riskiest check | same as S3. Also record `ttft_s`/`elapsed_s` against today's mode=local |
| **S5** history, same process | tool evidence survives into the next prompt with no client-side `history` juggling (Decision A1 becomes unnecessary) | follow-up answers from memory, with zero tool calls |
| **S6** history after restart | `--session-dir` + `open_session` resume works across process restarts, which a container restart or worker recycle would require | `resumed: true`, answers from memory |
| **S7** fail-closed | **the write-containment invariant**: built-in `write`+`bash` are enabled, but with `always-ask` and `--no-ui` nobody can approve them | canary file never appears; every attempted call errors. `INCONCLUSIVE` if the model refuses to try, so re-run or rephrase |
| **S8** abort | the UI's cancel button can map to RPC `abort` | `status: aborted` within 10 s of the abort |
| **S9** provider failure | a bad key surfaces as `status: error`, never a clean completion (CLAUDE.md: "never masquerade as success") | `status: error` |

Optional: `--bix-tools list_log_sources,read_log` also wires in the **real**
`tools.py` handlers. It needs the repo `.venv` (`.venv/bin/python
spike/omp/omp_spike.py …`) plus `FS_ROOT`/`DATA_DIR` env. The real
`stage_write` is never wired.

### Also note by hand while it runs

- `htop` during S4: does omp's own CPU/RAM matter next to Ollama? (omp ≈ Bun +
  native addon.)
- Does `journalctl -u ollama` show the expected model and `library=vulkan`?
- Under `~/.omp/` (the `bixspike` profile dir): what did omp write to disk (sessions, blobs,
  caches)? That sizes the container volume later.

## Things the docs already told us (design inputs, verify in Phase 1)

1. **Host tools default to approval tier `exec`.** The RPC host-tool adapter
   declares no tier, so under `always-ask` every bix-ai tool would prompt, and
   with no UI the prompt fails. Fix: allow each host tool **by name** in
   `tools.approval` (done in `spike-config.yml`). Everything not on that list
   stays fail-closed. This is the shape production should use.
2. **Host tools default to `loadMode: "discoverable"`** (hidden behind omp's
   tool search). The spike sends `"essential"` so the model sees them
   directly.
3. **`CLAUDE_CONFIG_DIR` opts Claude's config into omp discovery.** bix-ai's
   container has set it since `f30ddfb`, so inside the container omp would
   discover Claude's MCP servers, **including `bix_mcp`**, and expose those
   tools outside `TOOL_TABLE`. Mitigations (both applied in the spike): strip
   the env var from omp's environment, and put the foreign discovery sources
   in `disabledProviders`.
4. **The bundled Python client (`omp_rpc`) is synchronous.** That conflicts
   with "async throughout". `spike/omp/omp_spike.py`'s `OmpRpc` (~200 lines,
   asyncio subprocess) is the prototype to keep instead.
5. **Governor.** omp has `--max-time` (wall clock) and RPC `abort`. I didn't
   find a turn or token cap equivalent to `LOOP_MAX_TURNS`/`LOOP_MAX_TOKENS`,
   so the host would enforce those by counting `turn_start` and assistant
   `usage` and sending `abort`. The `budget` SSE gauge stays host-generated.
6. **Framing.** The spike stays on protocol v1, where frames are capped at
   1 MiB and oversized events get fields elided. Production must negotiate v2
   and reassemble `rpc_chunk` frames, or large `read_file` results will be
   silently truncated in events.
7. **Defaults to watch.** omp's default approval mode is `yolo`, and it ships
   31 tools (bash, edit, Python/Bun eval kernels, browser, web search,
   subagents). `--no-tools` + `always-ask` + the name allowlist is the minimum.
   **Never** run it with `--yolo`/`--auto-approve`.

## Decision gate

**Go** only if all of the following hold:

- S1, S3, S5, S6, S7, S8, S9 pass.
- S4 passes on at least one local model you'd actually route to.
- Local latency is no worse than today's mode=local by more than about 20%.

**No-go / park** if any of these happens:

- S7 ever writes the canary.
- S4 fails on every local model. Losing the forge-guardrails rescue layer
  would make mode=auto escalate to Claude constantly, which costs money.
- Ollama needs the models.yml workaround *and* tool calls still fail under it.

## Phase 2 — if go: `mode="omp"` backend (sketch, not yet a commitment)

A new `streaming/omp.py`, added next to the existing modes rather than
replacing anything, so the two can be compared side by side:

- **Process model:** one long-lived `omp --mode rpc` per active conversation.
  Each conversation id maps to a `--session-dir` under `DATA_DIR/omp/`,
  re-adopted with `open_session`, and idle processes are reaped. Server-side
  sessions replace both the `history` round-trip and pro mode's `session_id`
  echo.
- **Pre-pass stays:** run `strategy.preprocess` on the user message before
  `prompt`, so blob pointers still apply. omp's own compaction replaces
  `compact.py`.
- **Routing stays:** `routing.decide`, then `set_model` before each prompt,
  and `routing.ndjson` is still written by bix-ai.
- **Tools:** `set_host_tools` generated from `TOOL_TABLE` (a third generated
  wire format next to `FS_TOOLS`/`OLLAMA_TOOLS`), with `_execute_tool` as the
  handler. The role filtering and `stage_write` gating are unchanged.
  Containment stays enforced by `staging.py`, not by omp.
- **Event → SSE mapping** (the UI and golden fixtures stay as they are for
  this mode):

  | omp event | bix-ai SSE |
  |---|---|
  | `turn_start` | `budget` (host-computed) |
  | `message_update` · `text_delta` | `delta` |
  | `tool_execution_start` | `tool_start` + `tool_input` (full args) + `tool_end` |
  | `tool_execution_end` | `tool_result` (4000-char clip) |
  | assistant `message_end` usage | `input_tokens` (first), summed into `metrics` |
  | `retry_fallback_applied` | `model_swap` |
  | `prompt_result` completed | `history` (via `get_messages`, for UI compat), `metrics`, `done` |
  | `prompt_result` error / aborted | `metrics`, `error` (never `done`) |

- **Docker:** install a pinned omp binary in the image, pass the binary
  through the test gate, and run the spike's S1/S7 against `fake_omp.py` as
  regular pytest cases.

## Phase 3 — if Phase 2 holds up for a couple of weeks

Delete `streaming/loop.py`, `providers.py`, `claude.py`, `ollama.py`,
`pro.py`, `local_first.py`'s local leg, `bix_mcp.py`, `compact.py`, and the
forge-guardrails dependency. Re-derive `CLAUDE.md` from the code and update
`PLAN-pi-tools.md`.

## Files

| Path | What |
|---|---|
| `spike/omp/omp_spike.py` | async RPC client + host tools + checks S1–S9 (stdlib only) |
| `spike/omp/spike-config.yml` | omp overlay: `always-ask`, host-tool allowlist, foreign discovery off |
| `spike/omp/fake_omp.py` | scripted stand-in for offline plumbing tests. **Not** evidence about omp |
| `spike/omp/results/` | per-run JSON reports (gitignored) |

`spike/` is in `.dockerignore`, so none of it ships in the image.
