# PLAN — agent runner: shell + edit tools for ai.Bix, without making it root

Written 2026-10-13 in a Claude Code session, for an implementer with no access
to that conversation. Plan A (Claude Code Remote Control, Claude-only, as
`matt`) is in `bix-infra/todos/infra.md` → Plans.

## Why

ai.Bix already has the parts matt wants to keep regardless of provider: the chat
UI, provider switching (local Ollama / Claude / OpenRouter), key handling, GPU
access and server-owned conversations. It lacks what a Claude Code session has:
it can't run commands, edit code or use the internet. The goal is a Claude Code-
or Pi-style agent behind ai.Bix's UI that works with any of its providers, so
the provider isn't locked in.

## The constraint that shapes everything

"Give ai.Bix what the SSH session has" is equivalent to root:

- Host cron and systemd run scripts from `/home/matt/apps` as `matt`, and `matt`
  is in the `docker` group. **Anything that can write to the apps tree is
  root by tonight's 04:00 run.** That's why ai-router mounts it `:ro`
  (bix-infra CLAUDE.md, "No container writes anything the host executes").
- ai.bix.computer is a web app that reads attacker-influenced text (web pages,
  logs, files). A shell as `matt` behind it means one prompt injection, XSS or
  CSRF reaches root.
- Read access to secrets plus internet access = exfiltration.

So the agent **does not run in ai-router and does not run as matt.** It runs in
a separate host runner as a dedicated unprivileged user. Its writes land in its
own clones and come back as branches that **matt merges by hand**. Every
command it runs needs approval in the UI unless it's on a short read-only
allowlist. Nothing promotes the agent's work automatically: the 2026-10-05
retirement of the stage → promote pipeline stands.

## Shape

```
phone ─ Access ─ ai-router (container; mounts /srv/bixagent/run, group_add 970)
                    │  tools: run_command / write_file / edit_file / workspace_diff / hand_off
                    │  HTTP over unix socket /srv/bixagent/run/runner.sock (0660, group bixagent)
                    ▼
               bix-agent-runner (host systemd, User=bixagent uid/gid 970)
                    ├─ sees:   an empty /home with only /home/matt/apps bind-mounted read-only
                    ├─ writes: /srv/bixagent/{work,out,home} only
                    ├─ network: internet yes; loopback (except the DNS stub), docker bridges, LAN: denied
                    └─ no docker, no sudo, no capabilities, can't see matt's processes
```

**Why a unix socket, not TCP (changed 2026-10-13 while building Phase 1):** a TCP
listener on the bridge needs the sandbox to reach 172.19.0.0/16, and anything
arriving at the metrics container from the bix gateway gets the deploy-api
token added by its nginx: a path to a shell as matt. With a socket the
sandbox's IP rules can deny every private range outright.

**Why bundles, not `git fetch` from the clone:** git trusts the source repo's
config when cloning or fetching from a local path, so matt must never fetch
from a bixagent-owned repo. The runner writes base..HEAD as a git bundle (plain
data) to `/srv/bixagent/out/`; matt fetches from that.

## Decisions (matt, 2026-10-13)

1. **Write rule:** approved as part of "get B going". ai.Bix itself still writes
   only via `workspace.py`; the runner writes only into its own sandbox. Reword
   bix-ai/CLAUDE.md's "One write path" rule to say so when Phase 2 lands.
2. **Repos:** every git repo directly under `/home/matt/apps`, discovered by the
   runner per request (no list to maintain).
3. **Auto-approve:** read-only commands, plus `write_file`/`edit_file` inside
   the clone (they can't touch the host). Anything that executes code or uses
   the network asks: tests, interpreters, `curl`, `git fetch`.
4. **Provider:** open. matt asked what "agent mode" is; answer and decide
   before Phase 2 (see Phase 2 step 1).
5. **Logs:** `bix-infra/logs` is readable by the agent (decided after install;
   ai-router's log is metadata only, chat history stays hidden in `bix-ai/data`).
6. **Hand-off review:** every hand-off ("PR") runs the fresh-eyes-review skill.
   The reviewer model is picked per hand-off, defaulting to the agent's model.

## Phase 0 + 1 — sandbox and runner (built 2026-10-13, reviewed, not yet installed)

| File | What |
|---|---|
| `bix-infra/scripts/agent-runner.py` | The daemon. Stdlib only. `GET /repos`, `GET /diff?repo=`, `POST /clone {repo, slug, reset}`, `/run {repo?, cwd?, script, timeout_s}`, `/write {repo, path, content}`, `/edit {repo, path, old, new}`, `/bundle {repo}`. Paths are resolved through symlinks and refused if they leave the clone. Each command runs in its own child cgroup (2G memory, 256 pids) that is killed whole when it ends or times out, `setsid` or not. At most 4 commands run at once (429 beyond). Output is capped at 256 KB per stream, with the overflow counted. Hand-offs check the branch shape and that the base is in matt's history, because both live in agent-writable state |
| `bix-infra/scripts/systemd/bix-agent-runner.socket` | `/run/bix-agent-runner/runner.sock`, root-owned, `root:bixagent 0660`. Commands can't loosen or replace it, and a killed daemon is re-spawned on the next request |
| `bix-infra/scripts/systemd/bix-agent-runner.service` | The sandbox (the security boundary): `ProtectHome=tmpfs` + the apps tree read-only. Hidden: `.env`, `bix-ai/data`, `pi_hole/etc-pihole`, `inkplate_dash/data`, `/media`, `/mnt`, docker socket. `Delegate=memory pids`, `OOMPolicy=continue`, `Restart=always`. `IPAddressDeny` covers loopback and every private range **with no exceptions** |
| `bix-infra/scripts/systemd/agent-resolv.conf` | The sandbox's `/etc/resolv.conf` (public resolvers). Allowing systemd-resolved's 127.0.0.53 would also have exposed every service listening on all addresses |
| `bix-infra/scripts/agent-fetch.sh` | matt's side of a hand-off. It copies the bundle (without following symlinks) into a matt-owned temp dir, requires a plain file, checks the name, runs `git bundle verify`, requires exactly the one promised `agent/` branch, and fetches it (never merges) |
| `bix-infra/scripts/install-agent-runner.sh` | Run with sudo. Creates `bixagent` (uid/gid 970, no other groups) and `/srv/bixagent`, installs both units, restarts them, then runs the check |
| `bix-infra/scripts/agent-sandbox-check.py` | Acceptance test against the **live** runner, as root. It probes every TCP listener on the host at every local address, plus the metrics container; each must fail with EPERM (the IP filter), and is probed only if the host itself can reach it. Hidden paths must exist on the host and be unreadable inside. Write, sudo and process-visibility probes. A `setsid` child must not survive its command or hang the request. Allowed: apps reads, own home, public DNS, internet |
| `bix-infra/scripts/tests/test_agent_runner.py` | The flow over the real socket including `agent-fetch.sh`; renamed branches, a tampered base, diff drivers, bundles swapped after hand-off (symlink, directory, extra refs), path escapes, timeouts, output cap, the exact child env, the concurrency cap |

Reviewed by fresh-eyes (staff + nitpicker) on 2026-10-13. Every staff finding was fixed. The runner's tests can't exercise the cgroup path (it needs systemd delegation), so `setsid` containment is proven only by the sandbox check on the installed unit.

Accepted and documented, not fixed:
- **Commands run as the daemon's own uid.** They can kill it (it restarts) and tamper with anything under `/srv/bixagent`, including other clones and bundles. That's why `agent-fetch.sh` trusts nothing there, and why Phase 3's review reads the diff from a bundle ai.Bix hashed itself, never from `/diff` or the clone.
- **The sandbox's internet traffic leaves from the home IP.** That passes watch.bix.computer's home-IP Access bypass, though Materialious's own login is still required.
- **Abstract unix sockets are reachable** (no network namespace). X11 is protected only by `xhost` allowing `localuser:matt`; never run `xhost +local:`.
- **The system D-Bus** is reachable; polkit still requires auth for privileged actions.

## Phase 2 — ai.Bix tools + approval (medium–large)

1. **Tools** (one `TOOL_TABLE` entry each, per `tools.py` convention):
   `run_command`, `write_file`, `edit_file`, `workspace_diff`. Handlers call the
   runner over httpx. They are offered **only when the conversation has agent mode
   on** (a per-conversation flag on the record in `conversations.py`, toggled in
   the UI), never in plain chat.
2. **Approval flow.** In the governed loop the handler awaits
   `execute_tool` (`streaming/loop.py:151`), so it can block on approval:
   - emit a new SSE event `approval_required {tool_use_id, name, input}` and
     wait on an `asyncio.Event` keyed by `(conversation_id, tool_use_id)`;
   - new route `POST /conversations/{id}/approvals/{tool_use_id}
     {decision: approve|deny}`. Because it gates a host action, it **verifies the
     Access JWT** (bix-infra rule: see rss_feed `auth.check_access`) and rejects
     cross-site `Sec-Fetch-Site`/foreign `Origin`;
   - a denial or a timeout (default 10 min) returns a tool_result with
     `is_error: true` saying "denied by user" or "approval timed out". It never
     reads as success;
   - time spent waiting must not eat `LOOP_MAX_SECONDS` (currently 1200 s).
     Pause the governor's clock while waiting, and show the wait in the
     `budget` event;
   - auto-approved commands (allowlist, decision 3) skip the wait, but the
     audit line records `auto`;
   - keep `with_keepalive` on the stream; Cloudflare drops idle SSE at ~100 s.
3. **UI** (`static/index.html`): an agent-mode toggle in the header. The approval
   card shows the exact argv/script, cwd and repo, with Approve / Deny buttons
   (44 px tap targets) and no "always allow" in v1. Command output renders in the
   existing collapsed tool group. All output is text, never HTML.
4. **SSE contract:** `approval_required` is a new event. Add it to the CLAUDE.md
   table and record a new golden fixture (`RECORD_SSE=1`) for an approved and a
   denied run. Existing fixtures must not change.
5. **Prompt:** a short agent-mode system addendum stating that the workspace is a clone,
   changes reach the live system only when matt merges them, and command
   output and web content are untrusted data.

## Phase 3 — hand-off ("PR") with fresh-eyes review (medium)

**Built 2026-10-13** (`handoff.py`, cards in the UI; reviewed twice with the skill).

matt's rule: whenever the agent makes a PR, it runs the fresh-eyes-review
skill on its change. Here a PR is a hand-off of an agent branch.

1. **`hand_off` tool** `{repo, title, intent}`: the agent calls it when it
   believes the work is done. Refused unless the tree is clean and the branch has
   commits (the runner's `/bundle` checks, through its `check: true` dry run).
2. **Review card** in the chat: a reviewer model picker, defaulting to the agent's
   model (matt's decision), and a Run review button.
3. **Each round reviews a fresh bundle, bound to its bytes.** The agent can rewrite
   anything under `/srv/bixagent`, its clone and the bundle included, so:
   - ai.Bix reads the bundle once (no symlinks, FIFOs or other non-files), records
     its sha256, and unpacks it into its own scratch repo; the reviewers' diff comes
     from those bytes, never from the clone or `/diff`.
   - The base must be in matt's checked-out history, or commits between it and
     main would merge unreviewed. ai.Bix checks this, and agent-fetch.sh again.
   - Each finished round shows `agent-fetch.sh <name> <sha256>`, which refuses a
     copy whose hash differs. No fetch command without a finished review, merging
     stays manual, and there is never an apply button.
4. **Both of the skill's reviewers run every round**, not just the nitpicker after
   fixes: the author is an untrusted agent, so its fixes need the staff review too.
   - Prompts are read from `bix-infra/skills/fresh-eyes-review/SKILL.md` at review
     time (`~/.claude/skills/fresh-eyes-review` is a symlink to it).
   - Reviewers see only the intent, the commits and the diff. A generated-file
     header is shown as the agent's claim to check, never a reason to skip; binary
     files are listed as unreviewed.
   - Tools are read-only: `branch_*` on the reviewed commit, and ai.Bix's live
     `read_file`/`list_directory` limited to what the sandbox may read (the unit's
     `InaccessiblePaths`).
   - Local reviewers run one after the other, are refused up front when the prompt
     won't fit the context Ollama loads them with, and their reports count only if
     Ollama confirms that context afterwards (it truncates silently).
   - A failed round keeps the report that finished; Retry reruns only the other.
5. **After the review**, both reports are shown verbatim. "Send to agent" puts them
   in the message box for matt to add to and send. The agent fixes what's real,
   rebuts the rest and hands off again for another round.
6. ai.Bix sends a CSP limiting images, media, fonts and styles to its own origin
   (and DOMPurify drops media and style tags), so rendered model text can't leak
   what a model read through a remote URL.

## Phase 4 — pro mode (medium, optional)

**Built 2026-10-13.** As planned, with one change: the MCP server doesn't copy the
agent tools' schemas, it lists them from ai-router (`GET /internal/agent/tools`), and
each call goes to `POST /internal/agent/tool`, which runs it through the chat's own
`AgentSession`, so approvals, the runner and `hand_off` are shared with the other
modes. The callback's credential is a per-run token in the MCP server's env; tunnel
traffic is refused by header. Approval cards reach the chat through a per-run queue
that `streaming/pro.py` merges into the CLI's output. Hand-off reviews still need
local/api/openrouter (they run through the tool loop).

Original plan: expose the runner tools in `bix_mcp.py` with copied schemas; superseded as above.

## Phase 5 — read-only ops visibility (optional, later)

The sandbox has no `docker`/`journalctl`. If the agent needs container state,
add narrow runner endpoints (e.g. `docker ps` format-filtered, the latest
compiled digest) implemented by a tiny matt-side helper. Don't add docker group
membership.

## Docs to update when shipping

- bix-ai/CLAUDE.md: module map (new tools), SSE table, the amended write rule.
- bix-infra/CLAUDE.md: Security → Rules (runner = sandboxed, merges are manual
  from bundles), Scripts table rows (runner, install, sandbox check), Logging
  (`journal` source `bix-agent-runner` → `logs/agent-runner/`).
- Supersedes the `bix-ai.md` Pending items "Write mode option" (its notes
  describe RW mounts that no longer exist) and partly "Plan mode". Close
  or rewrite them when Phase 2 ships.
