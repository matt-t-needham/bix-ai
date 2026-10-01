# PLAN — back bix-ai's agent loop with oh-my-pi (omp): full implementation

**Audience:** a capable Claude agent running **locally on the bix host** (Linux
Mint 22, the Minisforum HX100G) with shell access, the bix-ai checkout, Docker,
and Ollama. You have **no access to the conversation that produced this plan.**
Everything you need is here or in the files it points to.

**Status when written (2026-10-01):** design only. Nothing here has run against
real omp; the cloud session that wrote it could not install omp. Claims about
omp come from its docs and source at upstream commit `95acb04` (package version
`18.4.9`). **Verify them as you go.** Where reality disagrees, reality wins:
record the difference in §11 "Deviations log" and adapt.

Companion docs:

- `PLAN-omp-spike.md` and `spike/omp/`: the go/no-go spike. **Phase 1 of this
  plan runs it, and nothing after it starts until it passes.**
- `CLAUDE.md`: project rules. They all still apply. Read it first.
- `PLAN-pi-tools.md`: the previous roadmap, which explains why the current loop
  looks the way it does.

---

## 0. Handoff contract: read before doing anything

### 0.1 What you are building

Today bix-ai runs its own agentic loop: `streaming/loop.py` plus
`providers.py`/`claude.py`/`ollama.py`, its own compaction (`compact.py`), and
a forge-guardrails rescue layer for local models. The goal is to make **omp**,
run as a subprocess in RPC mode, the loop for `mode=api`, `mode=local` and
`mode=auto`. bix-ai keeps:

- the FastAPI service, the single-file UI, and the SSE contract the UI consumes;
- **all tools**, served to omp as RPC *host tools* from `tools.TOOL_TABLE`;
- the pre-pass (`strategy.py` + `blobstore.py`);
- the routing decision (`routing.py`) and `routing.ndjson`;
- write containment (`staging.py`, `fs_core.py`). **omp never writes files.**

`mode=pro` (the official `claude` CLI on the Claude subscription) is **out of
scope** and stays as it is (see §0.4).

The rollout is staged: a new `mode="omp"` first, side by side with the old
paths. Then a server flag points api/local/auto at omp. The old loop code is
deleted only after a soak period.

### 0.2 Rules for you, the implementing agent

1. **Branch.** Work on a branch, e.g. `omp-integration`, never `main`. Commit
   per phase with clear messages. Don't push or open PRs unless the human
   asks.
2. **Stop and ask the human** at every point marked **⛔ STOP**. Also stop
   whenever a phase's acceptance criteria can't be met as written. Explain
   what failed and propose options; don't improvise around a gate.
3. **Never** do any of the following:
   - edit `/etc/systemd/system/ollama.service.d/override.conf`;
   - set `HSA_OVERRIDE_GFX_VERSION` or any `*_VISIBLE_DEVICES`;
   - install ROCm (see §1.2);
   - weaken `is_write_denied_path`/`staging.py`;
   - run omp with `--yolo`/`--auto-approve`/`approvalMode: yolo`;
   - regenerate the existing SSE golden fixtures;
   - restart the production container without asking.
4. **The test suite stays offline and green.** `.venv/bin/python -m pytest -q`
   passes at the end of every phase. New tests use the fake omp, never the
   real binary or a real model.
5. **Secrets.** The Anthropic key lives in `/home/matt/apps/bix-infra/.env` as
   `BIX_AI_API_KEY`. Export it into your shell for manual runs. Never write it
   into a file inside the repo, a log, or a commit.
6. **Follow CLAUDE.md's backend standards:**
   - async throughout; httpx only for HTTP;
   - log with the `log.info("k=%s", v)` style;
   - never swallow errors: every failure becomes an `error` SSE event;
   - no clean `done` after a failure.

### 0.3 Decisions already made (don't re-litigate)

| # | Decision | Why |
|---|---|---|
| D1 | Integrate over **RPC (stdio JSONL)**, not the TS SDK | bix-ai is Python; RPC gives process isolation |
| D2 | **Own asyncio client**, not the bundled `omp_rpc` package | `omp_rpc` is synchronous; CLAUDE.md requires async |
| D3 | **Built-in omp tools off** (`--no-tools`); bix-ai tools only, via `set_host_tools` | containment and a single tool registry |
| D4 | `approvalMode: always-ask` + per-tool `allow` list + `--no-ui` | anything not explicitly allowed needs approval nobody can give, so it **fails closed** |
| D5 | **Server-side sessions**: one omp session dir per conversation | tool turns survive across requests without the client `history` round-trip |
| D6 | **`routing.py` still decides the model**; omp is told via `set_model` | routing v2 logic and cost tiers are bix-specific |
| D7 | `strategy.preprocess` runs **before** the prompt reaches omp | omp has no verbatim-excerpt spill for pasted artifacts |
| D8 | omp's compaction replaces `compact.py` for omp-backed modes | no need for two compactors |
| D9 | `mode=pro` untouched | see §0.4 |

### 0.4 Out of scope

- **`mode=pro` / Claude subscription via omp.** omp can log into Anthropic with
  OAuth. Whether a consumer Claude subscription may be used through a
  third-party harness is a terms-of-service question the human must answer.
  Until they do, pro stays on the official `claude` CLI and `streaming/pro.py`
  + `bix_mcp.py` stay.
- omp features beyond the loop (subagents, advisor, collab, browser, LSP,
  Python kernels, web search, TTSR rules). Leave them off.
- Re-enabling self-update/staging deploys (retired 2026-09-25).

---

## 1. Host prerequisites (Linux Mint 22)

### 1.1 Facts about this host

Verify each with the command shown, and record anything that differs in §11.

| Item | Expected | Check |
|---|---|---|
| OS | Linux Mint 22.x (Ubuntu 24.04 "noble" base), x86_64, glibc ≥ 2.39 | `cat /etc/os-release; uname -m; ldd --version \| head -1` |
| CPU | AMD x86_64. omp picks its AVX2 or baseline native build automatically | `grep -o -m1 avx2 /proc/cpuinfo` (nothing printed = baseline; fine) |
| GPU | AMD RX 6600M (gfx1032), used by Ollama through **Vulkan** | `curl -s localhost:11434/api/ps` → `size_vram > 0` after loading a model |
| Ollama | systemd service, `OLLAMA_HOST=0.0.0.0`, `OLLAMA_VULKAN=1`, `OLLAMA_LLM_LIBRARY=vulkan` | `systemctl cat ollama` |
| Docker | Engine + compose plugin; bix-ai runs as compose service `ai-router` (container `apps-ai-router-1`) | `docker compose version; docker ps` |
| Python | 3.12 system python; repo venv at `bix-ai/.venv` | `python3 --version; ls .venv/bin/python` |
| Repo paths | apps tree at `/home/matt/apps/`; bix-ai checkout and `bix-infra/` side by side | `ls /home/matt/apps` |

### 1.2 About "AMD ROCm" on this machine

omp itself needs **no GPU stack at all**. The only parts of omp that do local
inference are its optional "tiny" models (titles and memory, via
ONNX/transformers.js). They run on **CPU by default**, RPC mode already
disables title generation, and we don't enable memory.

Ollama on this host deliberately uses **Vulkan, not ROCm**: Ollama's bundled
rocBLAS has no gfx1032 kernels (details in CLAUDE.md's Gotchas). So:

- don't install ROCm;
- don't set `PI_TINY_DEVICE`/`providers.tinyModelDevice`;
- don't touch the Ollama override.

If anything you install pulls in ROCm packages, **⛔ STOP**.

### 1.3 System packages

```bash
sudo apt update
sudo apt install -y curl ca-certificates git jq python3-venv
# Docker and Ollama are already installed and running; do not reinstall them.
```

### 1.4 Ollama: confirm it serves the OpenAI Responses API

omp's built-in `ollama` provider talks to `/v1/responses` (OpenAI Responses
API), not `/v1/chat/completions`.

```bash
ollama --version
curl -s localhost:11434/v1/responses -H 'content-type: application/json' \
  -d '{"model":"qwen3.5:9b","input":"say ok"}' | jq -r '.output // .error // .' | head -20
```

- **Works** (JSON with `output`) → continue.
- **404 / error** → either:
  - **Option A:** upgrade Ollama. **⛔ STOP and ask the human first.**
    `curl -fsSL https://ollama.com/install.sh | sh` replaces the unit file, but
    `override.conf` in `ollama.service.d/` survives. Afterwards re-verify the
    healthy-GPU checks from CLAUDE.md (`library=vulkan` in
    `journalctl -u ollama -b`, `size_vram > 0`).
  - **Option B:** configure omp to use Ollama's chat-completions endpoint via
    `models.yml` (§3.3, "fallback"). Record which option you used in §11.

### 1.5 Install omp on the host (pinned, checksum-verified)

Use the release binary rather than `bun install -g`. It's a single
self-contained executable (Bun-compiled, Rust natives embedded), identical to
what goes into the Docker image.

```bash
# Pick the version. Use the newest release unless the human pins one.
curl -fsSL https://api.github.com/repos/can1357/oh-my-pi/releases/latest | jq -r .tag_name
export OMP_VERSION=<tag from above>      # e.g. v18.4.9: confirm the exact tag format on the releases page

mkdir -p ~/.local/bin /tmp/omp-dl && cd /tmp/omp-dl
curl -fsSLO "https://github.com/can1357/oh-my-pi/releases/download/${OMP_VERSION}/omp-linux-x64"
curl -fsSLO "https://github.com/can1357/oh-my-pi/releases/download/${OMP_VERSION}/SHA256SUMS.txt" \
  || echo "no SHA256SUMS.txt asset: check the release's asset list for the checksum file name"
grep ' omp-linux-x64$' SHA256SUMS.txt | sha256sum -c -
install -m 0755 omp-linux-x64 ~/.local/bin/omp
command -v omp && omp --version          # Mint's default ~/.profile adds ~/.local/bin to PATH; re-login if not found
```

Record `OMP_VERSION` and the binary's sha256 (`sha256sum ~/.local/bin/omp`).
Phase 6 pins both into the Dockerfile.

On first run omp extracts its native addon under `~/.omp/natives/`. Everything
the spike does lives in an isolated profile (`--profile bixspike`), so your
user-level `~/.omp` config isn't touched.

### 1.6 Shell environment for manual runs

```bash
cd /home/matt/apps/bix-ai                       # adjust if the checkout lives elsewhere
export ANTHROPIC_API_KEY=$(grep BIX_AI_API_KEY /home/matt/apps/bix-infra/.env | cut -d= -f2)
export OLLAMA_HOST=http://localhost:11434
export SUMMARY_LOCAL_MODEL=gemma4:26b           # gemma4:e2b is broken on this host (CLAUDE.md)
unset CLAUDE_CONFIG_DIR                         # see §3.2 finding F3
```

### 1.7 Repo venv and the baseline test run

```bash
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest==8.3.4
.venv/bin/python -m pytest -q                   # record the pass count; it is your baseline
git switch -c omp-integration
```

**Acceptance:** omp runs; Ollama answers `/v1/responses` (or the fallback is
chosen); baseline tests are green. Record the versions in §11.

---

## 2. Phase 1: run the spike (go/no-go)

Follow `PLAN-omp-spike.md` Phase 0 + Phase 1 exactly:

```bash
python3 spike/omp/omp_spike.py --omp "python3 spike/omp/fake_omp.py" --profile ''   # plumbing: 7× PASS
python3 spike/omp/omp_spike.py --claude-model anthropic/claude-sonnet-4-5 --local-model ollama/qwen3.5:9b
python3 spike/omp/omp_spike.py --only S4 --local-model ollama/gemma4:26b
```

The model ids are `provider/model-id`. If `anthropic/claude-sonnet-4-5` isn't
in S1's listed models, pick the closest listed Sonnet id and record it.

While reading the result JSON, **record the real wire shapes**. Phase 4 depends
on them:

- the assistant `usage` object field names (`results.checks.S3.usage`);
- the `tool_execution_end.result` shape;
- any `unknown_frames`;
- the `get_available_models` response shape (S1).

Put them in §11.

**Gate (from the spike plan):** S1, S3, S5, S6, S7, S8, S9 pass, and S4 passes
on at least one local model you'd route to. Local latency is no worse than
today's `mode=local` by more than ~20% (measure today's with the UI or
`routing.ndjson` `elapsed_ms` for a similar prompt).

**⛔ STOP** and report the gate result to the human with the results JSON.
**Do not start Phase 2 without an explicit go.** If S7 ever wrote its canary
file, stop everything: containment failed.

---

## 3. Phase 2: async RPC client (`omp_client.py`)

A new top-level module next to `helpers.py`. Pure library: no FastAPI or
config imports beyond `config.py`.

### 3.1 Requirements

Start from `spike/omp/omp_spike.py`'s `OmpRpc` and harden it:

1. **Protocol v2.** After `ready`, if `2 ∈ supportedProtocolVersions`, send
   `{"id":…,"type":"negotiate_protocol","protocolVersion":2}`. Then reassemble
   `rpc_chunk` frames:
   - validate `chunkId`, `index` (contiguous from 0), `count`, `byteLength`;
   - reject interleaved or interrupted sequences;
   - enforce the advertised `maxReassembledFrameBytes`;
   - concatenate the base64-decoded parts, decode as strict UTF-8, and parse
     one JSON object.

   Treat a protocol violation as fatal for that process: log, fail pending
   futures, kill.
2. **Correlation.** Command responses are matched by `id`, never by order.
   `prompt` completes exactly once:
   - either its ack has `data.agentInvoked: false`,
   - or a later `prompt_result` arrives with the same `id`.
   - A late `response` with `success:false` for an acked prompt id is an async
     dispatch failure. Surface it.
3. **Event delivery.** Don't accumulate. Expose an `async for` stream of
   frames for the in-flight prompt, so the adapter can translate to SSE
   incrementally. Exactly one prompt in flight per client (enforce with a
   lock).
4. **Host tools.** On `host_tool_call`, run the handler as a task, reply with
   `host_tool_result`, and send `isError: true` on exceptions. On
   `host_tool_cancel {targetId}`, cancel that task and reply with an error
   result.
5. **`extension_ui_request`.** Shouldn't occur under `--no-ui`. If it does,
   answer cancelled and log a warning (fail closed).
6. **Lifecycle.**
   - `start()` waits for `ready` but fails fast if the process exits first.
     The spike already does this; keep it.
   - `close()` closes stdin and keeps reading stdout until EOF (the docs
     require that), waits ≤ 15 s, then kills.
   - Expose `alive`, `pid`, and a stderr tail for error messages.
7. **Backpressure.** The reader must never block on a slow consumer. Use a
   bounded `asyncio.Queue`; if it fills, that's a bug: log it and abort the
   prompt.
8. **Env.**
   - The child env is built **from an allowlist**, not by copying
     `os.environ`: `PATH`, `HOME`, `LANG`, `ANTHROPIC_API_KEY`,
     `OLLAMA_BASE_URL`/`OLLAMA_HOST`, `PI_CODING_AGENT_DIR`, `TMPDIR`.
   - `CLAUDE_CONFIG_DIR` must **never** be passed.

### 3.2 Findings from omp's docs that shape the client

These are the reasons behind the design, so keep them in mind:

- **F1. Host tools carry no approval tier**, so they default to `exec`. Under
  `always-ask` each one must be allowed **by name** in
  `tools.approval.<name>: allow`. Anything not listed prompts and fails, which
  is what we want for everything else.
- **F2. Host tools default to `loadMode: "discoverable"`** (hidden behind
  tool search). Send `"essential"`.
- **F3. An explicit `CLAUDE_CONFIG_DIR` opts Claude's user config into omp
  discovery.** It would then import Claude's MCP servers, including
  `bix_mcp`, as extra tools. The container sets that variable (commit
  `f30ddfb`). Defence:
  - strip it from the child env;
  - list foreign discovery sources in `disabledProviders`;
  - run omp with a dedicated, empty `cwd` (project-level discovery reads the
    cwd).
- **F4.** `--max-time` exists (wall clock). There is no turn or token cap, so
  the host enforces `LOOP_MAX_TURNS`/`LOOP_MAX_TOKENS` by counting
  `turn_start` and summing `usage`, then sending `abort`.
- **F5.** A failed provider turn is **not** a failed command: the prompt
  response is `success:true`, and the outcome is in `prompt_result.status`
  (`completed|aborted|error`) and `prompt_result.error`.
- **F6.** `prompt_result` = the agent yielded; `session_settled` = no
  background work left. With built-ins off there's no background work, but
  wait for `sessionSettled: true` (or the `session_settled` frame) before
  recycling a process.

### 3.3 omp config the server ships (`omp/config.yml` in the repo)

```yaml
tools:
  approvalMode: always-ask
  approval: {}            # generated at startup: one `<tool>: allow` per host tool (see 4.2)
disabledProviders: [claude, codex, gemini, github, opencode, cursor, agents-md]
compaction: {}            # leave defaults; verify behaviour in Phase 7 soak
```

Generate the effective config at startup into the omp agent dir, rather than
hand-maintaining the allow list. The allow list must equal exactly the host
tool names of the current `BIX_ROLE`.

**Fallback, only if §1.4 chose Option B.** Write a `models.yml` into the omp
agent dir that overrides the `ollama` provider:

```yaml
providers:
  ollama:
    baseUrl: http://host.docker.internal:11434/v1     # derive from config.OLLAMA_HOST; never hardcode
    api: openai-completions
    discovery: { type: ollama }
```

Verify it by listing models (`get_available_models`) and running spike S4
against it.

### 3.4 Tests (`tests/test_omp_client.py`)

Promote `spike/omp/fake_omp.py` to `tests/fake_omp.py` and extend it with
modes driven by env vars or argv: v2 chunking, a mid-chunk interruption,
host-tool cancel, a crash before ready, a crash mid-prompt, a late
async-failure response, and an unknown event type. Cover:

- ready + negotiation;
- chunk reassembly, including malformed sequences being rejected;
- command correlation with out-of-order responses;
- host-tool success / error / cancel;
- process death failing all waiters with the stderr tail in the message;
- the env allowlist (assert `CLAUDE_CONFIG_DIR` is absent in the child).

The tests spawn `python tests/fake_omp.py` as the "omp" binary; they never
need real omp.

**Acceptance:** the new tests pass; the full suite is green; no new
dependencies in `requirements.txt`.

---

## 4. Phase 3: process pool + tools (`omp_pool.py`, `tools.py`)

### 4.1 Session/process model

- **Conversation id.** The client generates one (UUIDv4) per new chat and
  sends it as `conversation_id` (Phase 5 wires the UI).
  - Validate it server-side: `^[0-9a-f-]{36}$`, otherwise 400.
  - Session dir: `DATA_DIR/omp/sessions/<conversation_id>/`.
- **One omp process per active conversation**, started with
  `--session-dir <dir>` (or `open_session` on a pooled warm process; prefer
  the simpler per-conversation spawn first).
- **Limits** (new knobs in `config.py`, env-overridable, with comments in the
  existing style):
  - `OMP_MAX_PROCS` (default 4); when exceeded, close the least recently used
    idle process. If none is idle, return the SSE `error` "busy".
  - `OMP_IDLE_SECONDS` (default 900); a background reaper closes idle
    processes.
  - `OMP_BIN` (default `omp`), `OMP_AGENT_DIR` (default
    `DATA_DIR/omp/agent`), `OMP_ENABLED` (default `false`), `OMP_BACKEND`
    (default empty; see Phase 7).
- **One prompt at a time per conversation:** a per-conversation
  `asyncio.Lock`. A second `/chat` for a busy conversation gets SSE `error`
  ("conversation busy").
- **Session lost** (dir deleted or evicted) **but the client sent prior
  turns:** start a fresh session and prepend a compact text transcript of the
  client's `messages` (text parts only) to the first prompt. Flag it with a
  `status` event: "Session restored from transcript (tool history not
  available)". Never fail silently.
- **Startup flags (all of them):**
  ```
  omp --mode rpc --no-ui --no-tools --no-lsp --no-pty --no-skills --no-rules
      --no-extensions --approval-mode always-ask
      --config <OMP_AGENT_DIR>/bix-config.yml --session-dir <dir>
      --append-system-prompt <file with identity.identity_system_prompt(...)>
  ```
  - `cwd`: an empty dir, `DATA_DIR/omp/cwd`.
  - env: `PI_CODING_AGENT_DIR=<OMP_AGENT_DIR>`, `HOME=<DATA_DIR>/omp/home`
    (the natives extraction needs a writable HOME in the container).
  - Check in Phase 2 that `--append-system-prompt` keeps omp's default coding
    prompt sensible for chat. If it reads badly, try `--system-prompt` with
    bix-ai's identity prompt alone, and record which you chose.
- **Health.** If a process dies mid-prompt, emit SSE `error` with the stderr
  tail and remove it from the pool. The next request respawns and resumes
  from the session dir.

### 4.2 Host tools from the single registry

In `tools.py` add one generator next to `fs_tools_for_role`/
`ollama_tools_for_role`:

```python
def omp_host_tools_for_role(role: str) -> list[dict]:
    return [{"name": t["name"], "label": t["name"], "description": t["description"],
             "parameters": t["input_schema"], "loadMode": "essential"}
            for t in tool_table_for_role(role)]
```

The handler is always `tools._execute_tool(name, args)`, which already
re-checks role denial at call time (defence in depth stays). Tool results
are text; wrap them as `{"content":[{"type":"text","text": …}]}`. "Error"
results from `_execute_tool` are returned as text today. Keep that, so the
model sees the message the same way as on the old path.

Register on every new process (`set_host_tools`) and assert the returned
`toolNames` equals the role's set. A mismatch is a hard error.

### 4.3 Tests

- `tests/test_omp_pool.py` (fake omp): spawn/reuse, LRU eviction at
  `OMP_MAX_PROCS`, the idle reaper (injected clock), busy-conversation
  rejection, a crash, then respawn + resume, session-lost transcript
  bootstrap, and invalid `conversation_id`.
- Extend `tests/test_roles.py`: in the staging role,
  `omp_host_tools_for_role("staging")` excludes `stage_write`,
  `check_staging` and `ask_staging`, and the generated approval allow list
  matches it exactly.

**Acceptance:** tests green; a manual run against real omp via a scratch
script reuses one process across two prompts and resumes after a kill.

---

## 5. Phase 4: the adapter (`streaming/omp.py`)

`async def _stream_omp(messages, model, max_tokens, *, mode, conversation_id,
route_reason="", on_exhausted="best_effort")`. It yields SSE strings exactly
like `_stream_claude`/`_stream_ollama`.

### 5.1 Flow

1. **Status + pre-pass.**
   - Same `status` events as `claude.py`.
   - Run `strategy.preprocess` on a body made of **only the new user turn**.
     The session already holds the earlier turns, so earlier blocks were
     spilled on earlier requests.
   - On a pre-pass exception: log it and forward the original (CLAUDE.md
     rule).
   - Emit `preprocess` with `compacted: 0`. omp compaction events are
     reported via `status`, not here, so the fixture contract holds.
2. **Model.** `set_model(provider, modelId)`, mapping bix model names to omp
   ids:
   - `claude-*` → `anthropic/<id>`;
   - Ollama names → `ollama/<name>`.

   Keep the mapping in one function with a unit test. A `Model not found`
   failure becomes an `error` event naming the model.
3. **Prompt** with the (pre-processed) newest user message text. Images are
   out of scope; if a turn carries one, reject it with an `error` saying so.
4. **Translate events → SSE** (the contract in CLAUDE.md must not change):

   | omp frame | SSE emitted | Notes |
   |---|---|---|
   | `turn_start` | `budget` | host-side turn counter, elapsed, summed tokens vs `LOOP_MAX_*`. **Breach** → send `abort`, emit `metrics` (partial) + `error` "Loop budget exceeded…", no `done` (same text and semantics as `loop.py`) |
   | `message_update` · `assistantMessageEvent.type == "text_delta"` | `delta {text}` | first one sets TTFT |
   | `message_update` · thinking deltas | (dropped) | |
   | `tool_execution_start` | `tool_start {index,name,id}` → `tool_input {index, partial_json: json.dumps(args)}` → `tool_end {index}` | `index` = running counter; `id` = `toolCallId` |
   | `tool_execution_end` | `tool_result {tool_use_id, content (≤4000 chars), is_error}` | stringify `result.content[].text` |
   | assistant `message_end` with usage | `input_tokens {count}` on the first one only; add to the totals | field names per §11 |
   | `auto_compaction_start` / `_end` | `status {stage:"summarising", message:"Compacting conversation…"}` | |
   | `retry_fallback_applied` | `model_swap` | |
   | `auto_retry_start` | `status` "Retrying…" | |
   | `prompt_result` `completed` | `history` (§5.2), `metrics`, `done` | |
   | `prompt_result` `error` | `metrics`, `error {message}` | 401 → message must say the API key was rejected; quota errors → also `quota_exceeded` |
   | `prompt_result` `aborted` | `metrics`, `error` "aborted" | |
   | client disconnect (`GeneratorExit`/cancel) | send `abort`; release the lock | never leave a run going |

5. **Routing log.** After every outcome, call
   `helpers._write_routing_event(mode, <model that actually ran>,
   reason=route_reason or f"forced:{mode}", input_tokens=…, output_tokens=…,
   ttft_ms=…, elapsed_ms=…)`. `est_cost_usd` keeps working through
   `config.MODEL_COSTS`.
6. **Keep-alive.** The `/chat` route already wraps everything in
   `with_keepalive`. Also wrap local-model runs in `helpers.with_progress`,
   as `ollama.py` does, so slow local prompt evaluation shows heartbeats.

### 5.2 `history` event

The UI adopts `history.messages` as `convHistory`. On the omp path the server
owns history, but the UI still needs a sane `convHistory` for mode switches
and memory saves. After `completed`:

- call `get_messages`;
- convert them to the Anthropic-shaped list the UI already understands
  (`user`/`assistant` with `text`, `tool_use` and `tool_result` blocks);
- emit that.

If conversion meets an unknown block type, include it as text, never drop it.
Unit-test the converter against message shapes recorded from the spike
(sanitised and saved under `tests/fixtures/omp/`).

### 5.3 mode=auto on omp

- `routing.decide` as today.
- **Claude route:** `set_model` to the decision's model (including the Haiku
  downshift) and run.
- **Local route:** `set_model ollama/<OLLAMA_DEFAULT_MODEL>`. On
  `prompt_result.status == "error"` (or a host-detected garbage turn: a
  tool-call failure reported by `tool_execution_end` errors three times in a
  row):
  - `abort` if still running;
  - emit `fallback_triggered` if any delta was sent;
  - `set_model` to the Claude model;
  - send a follow-up prompt: "The previous attempt by a local model failed
    (<reason>). Answer the user's last request fully."
  - Log it as `escalated: <reason>` (same reason text `local_first.py` uses).

  **Verify in Phase 7** that the failed local turn sitting in the session
  doesn't confuse Claude. If it does, use the `branch` RPC command to rewind
  to the entry before the failed turn and re-prompt instead. Record the
  outcome in §11.

### 5.4 Tests

- `tests/test_omp_adapter.py`, driven by the fake omp: text-only turn,
  tool turn, multi-tool turn, provider error, abort/disconnect, budget breach
  (turn cap and token cap), pre-pass spill on the new turn, model-id mapping,
  auto-mode escalation, and the history converter.
- **New golden fixture** `tests/fixtures/sse/omp_tool_turn.json`, recorded
  with the existing harness pattern (`RECORD_SSE=1` for the new test only).
  It must contain the same event names in the same order as the claude
  tool-turn fixture, except for documented differences, which you list in
  the test's docstring. **Do not regenerate the existing fixtures.**

**Acceptance:** suite green, plus a manual run against real omp through the
adapter (scratch script, no HTTP) that prints a plausible SSE stream for a
tool-using prompt on both Claude and Ollama.

---

## 6. Phase 5: route + UI

### 6.1 `main.py`

- Add `conversation_id: str = ""` to `ChatRequest`, plus a comment in the
  existing style.
- Dispatcher: `mode == "omp"` → `_stream_omp(...)`. The model allowlist check
  for omp accepts both `_ALLOWED_CLAUDE_MODELS` and
  `model_admin.is_allowed_local_model`.
- If `OMP_ENABLED` is false, `mode=omp` returns 400 "omp backend disabled".
- An empty or invalid `conversation_id` with `mode=omp` returns 400.
- Optional, nice to have: on shutdown (FastAPI lifespan), close all pool
  processes.

### 6.2 `static/index.html`

Follow CLAUDE.md's frontend design standards: no new colours, system font,
existing classes.

- Generate `conversationId = crypto.randomUUID()` on page load and on every
  "new chat"/clear, and send it as `conversation_id` on every `/chat`. Find
  the existing new-conversation code path (search for where `convHistory` is
  reset) and hook there.
- Add a fourth mode button **only behind the server flag.** `GET /version`
  (or a tiny `GET /omp/status`) reports `omp_enabled`; render a `Pi` button
  in `.mode-group` only when it's true. Use the mauve active colour, like
  `auto`.
- `_resolvedMode()` returns `'omp'` for that button.
- Every SSE event the omp path emits is already handled. If you emitted a
  new `status` stage, check it renders.

### 6.3 Tests

Route tests in the style of `tests/test_staging_routes.py`:

- `mode=omp` disabled → 400;
- bad conversation id → 400;
- happy path streams through the fake omp.

**Acceptance:** with `OMP_ENABLED=true` in a dev run (`uvicorn main:app
--reload --port 8000`, real omp, real Ollama) you can chat in the Pi mode, use
tools, reload the page mid-conversation, and continue (the session resumes).

---

## 7. Phase 6: container

### 7.1 Dockerfile

Install the pinned, checksum-verified binary in the **`base` stage, after
`pip install` and before `COPY . .`**, so it's cached across code changes:

```dockerfile
ARG OMP_VERSION=<pinned tag from §1.5>
ARG OMP_SHA256=<sha256 of omp-linux-x64 from §1.5>
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL -o /usr/local/bin/omp \
      "https://github.com/can1357/oh-my-pi/releases/download/${OMP_VERSION}/omp-linux-x64" \
 && echo "${OMP_SHA256}  /usr/local/bin/omp" | sha256sum -c - \
 && chmod 0755 /usr/local/bin/omp \
 && HOME=/tmp omp --version \
 && apt-get purge -y curl && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*
```

- `python:3.12-slim` is Debian (glibc), so use the glibc `omp-linux-x64` build,
  not the musl one.
- If `omp --version` fails on a missing shared library, install exactly that
  library (likely `libstdc++6`/`libgcc-s1`) and record it.
- Keep the `GIT_SHA`/`BUILT_AT` ARGs at the end of `runtime`. CLAUDE.md
  explains why.
- The test stage needs no real omp (fake only), but `omp --version`
  succeeding at build time is the smoke check.

### 7.2 Compose

The compose file lives in `bix-infra`; find it with
`grep -rl "ai-router" /home/matt/apps/bix-infra`.

- Add env: `OMP_ENABLED=true`, `OMP_BIN=omp`, `OMP_MAX_PROCS=4`.
  `OLLAMA_HOST` is already set; omp reads it too (§3.1 env allowlist).
- `DATA_DIR` (`/app/data`) is already a bind mount. omp's agent dir,
  sessions, HOME and cwd all live under it (`/app/data/omp/...`), so sessions
  survive rebuilds.
- Check the service's `user:`. Whatever uid runs uvicorn must own
  `/app/data/omp`. `CLAUDE_CONFIG_DIR` stays set for `mode=pro`; the client
  strips it for omp (F3).
- **⛔ STOP before editing compose or rebuilding prod.** Show the human the
  diff and the build command:
  `docker compose build ai-router && docker compose up -d ai-router`.

### 7.3 In-container verification

```bash
docker exec apps-ai-router-1 omp --version
docker exec apps-ai-router-1 sh -c 'env | grep -c CLAUDE_CONFIG_DIR'   # set for pro; that's fine
```

Then, from the UI:

- one Pi-mode chat with a tool call;
- `docker exec apps-ai-router-1 ls /app/data/omp/sessions` shows the
  conversation;
- check `docker logs` for `set_host_tools` registering exactly the role's
  tools.

**Containment re-check in the container:** run the spike's S7 inside the
container (copy `spike/omp` in temporarily with `docker cp`, then remove it).
The canary must not appear.

---

## 8. Phase 7: switch-over and soak

1. **Side by side (≥ 3 days of normal use).** Pi mode available; the old modes
   unchanged. Compare in `GET /routing` and `routing.ndjson`:
   - cost per request;
   - `elapsed_ms`/`ttft_ms`;
   - local tool-call success (old path: `guardrail_*` fields; omp:
     `tool_execution_end` errors, logged by the adapter);
   - escalation rate in auto.
2. **Flip** with `OMP_BACKEND=api,local,auto`: the dispatcher sends those
   modes to `_stream_omp` instead of the old adapters. Pi mode becomes
   redundant (keep the button hidden). Rollback = unset the env var and
   restart, no rebuild.
3. **Soak ≥ 7 days.** Watch:
   - omp RSS/CPU (`docker stats`);
   - the session dir's size (add a cleanup: delete session dirs untouched for
     `OMP_SESSION_TTL_DAYS`, default 30, via the reaper; tested);
   - whether compaction behaves on long chats (does the history event stay
     sane?);
   - any `extension_ui_request` warnings (there should be none).

**⛔ STOP** after the soak with a short report (the numbers above) and ask
for the go-ahead to delete the old loop.

---

## 9. Phase 8: retirement

Only with that go-ahead:

- **Delete:** `streaming/loop.py`, `streaming/providers.py`,
  `streaming/claude.py`, `streaming/ollama.py`, `streaming/local_first.py`
  (its routing glue moves into `streaming/omp.py`), `compact.py`, the
  `forge-guardrails` line in `requirements.txt`, and their tests. Drop the
  golden fixtures for the deleted paths; the omp fixture becomes the
  contract.
- **Keep:** `streaming/pro.py`, `bix_mcp.py` (pro mode), `strategy.py`,
  `blobstore.py`, `routing.py`, `tools.py`, `staging*.py`, `fs_core.py`.
- The `/v1/messages` proxy route in `main.py` is independent of the loop;
  keep it.
- Rename `OMP_BACKEND`. omp is now simply the backend; remove the flag and
  Pi button.
- **Re-derive `CLAUDE.md` from the code:** the module map, SSE notes (history
  is now server-derived), test counts, the new Gotchas (omp pin and upgrade
  procedure, F1–F3, session dir cleanup). Add a short "Superseded" header to
  `PLAN-pi-tools.md` pointing here.

**Acceptance:** suite green; image builds; a one-day smoke in prod; `git grep`
finds no imports of the deleted modules.

---

## 10. Upgrading omp later (put this in CLAUDE.md's Gotchas in Phase 8)

1. Read omp's CHANGELOG for RPC protocol, host-tool, approval or config
   changes.
2. Bump `OMP_VERSION`/`OMP_SHA256` in the Dockerfile.
3. Run the spike against the new binary on the host (§2 commands). S7 must
   pass.
4. Rebuild.

Never use an unpinned "latest" in the image.

---

## 11. Deviations log (fill in as you go)

| Phase | Plan said | Reality | Action taken |
|---|---|---|---|
| 1 | Mint 22 / glibc ≥ 2.39 / Vulkan Ollama | | |
| 1 | Ollama serves `/v1/responses` | | |
| 1 | release asset `omp-linux-x64` + `SHA256SUMS.txt`, tag format | | |
| 2 | usage field names | | |
| 2 | `get_available_models` shape; Sonnet model id | | |
| 2 | `--no-tools` still allows host tools | | |
| 3 | `--append-system-prompt` vs `--system-prompt` | | |
| 4 | auto escalation in-session vs `branch` | | |
