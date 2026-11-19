# Plan: the ai.Bix workbench

Status: **agreed, nothing built** (2026-11-18). §2 records the decisions.

**This is the new way of working.** The workbench replaces the terminal as the primary
interface for development on bix. Matt already runs Claude Code as himself from the
command line; the workbench is the full interface to that same work when he isn't at the
terminal. Sessions run **as matt**, in `~/apps`, on the Pro subscription.

ai.Bix today is a chat page. It can't see or continue the Claude Code work that actually
happens on bix, so it isn't useful for development. This plan turns it into a workbench:
every agent task on bix in one list, a viewport showing what the selected task is doing,
and an **Open** view to continue it. Claude Code on the Pro subscription is the default
engine for every task. Other models are a secondary choice.

## 1. What Matt sees

```
 bix.computer / ai                                            [ + New task ]
 ┌────────────────────────────┬──────────────────────────────────────────────┐
 │ BIX                      4 │ Restructure suite CLAUDE.md          [Open]   │
 │ ! Restructure suite …  2m  │ bix-infra · claude/restructure · Claude Pro   │
 │   Needs you: finish PR or  │                                               │
 │   delete the branch?       │ GOAL   Split CLAUDE.md into on-demand files   │
 │ ● jobs: hook spool    now  │ NOW    Waiting on you                         │
 │ ■ health staging …   14m   │ ! BLOCKED ON YOU  Finish the PR or delete …   │
 │ ○ rss reader tweak    1h   │ PROGRESS  ▓▓▓▓▓▓▓░░  5 of 7 steps             │
 │                            │   ✓ Move logging rules   ✓ Move security …    │
 │ OTHER MODELS             1 │   · Open PR                                   │
 │ ○ (OpenRouter) recipe …    │ CHANGES   [screenshot] [screenshot] [▶ 12 s]  │
 │                            │ CONTEXT   (last few turns, read-only)         │
 └────────────────────────────┴──────────────────────────────────────────────┘
```

- **List (left):** the tiles from jobs.bix.computer, with the same status marks, sorting
  and grouping. They're scrollable, and the needs-you tiles come first. On a phone the
  list is the first screen, and tapping a tile goes to its viewport.
- **Viewport (right):** for the selected task:
  - **Goal:** what it set out to do.
  - **Now:** what it's doing right now.
  - **Progress:** steps done and to do.
  - **Blocked:** what it's blocked on, flagged as *on you* or *on something else*: a
    usage limit, CI, a deploy, another task, or "stalled, unknown".
  - **Changes:** screenshots and short usage videos where the work changes something
    visible.
  - **Context:** the last few turns, read-only.
- **Open:** the full conversation (messages, tool calls collapsed, diffs) with an input
  box. Sending continues that same Claude Code session.
  - Replies **stream** token by token (`--include-partial-messages`).
  - A **model picker**: Opus / Sonnet / Haiku on the Pro subscription by default, other
    models secondary.
  - A **plan usage** summary: how much of the current window and the week is used, when
    it resets, and this session's tokens and cost.
  - Permission prompts appear as approval cards. ai.Bix agent mode already has these.
- **Pick it up anywhere:** every task shows where it's being driven right now (web,
  terminal, Claude app or background), with actions to move it:
  - **Continue here** takes it over in the browser.
  - **Continue in terminal** releases it and shows the `claude --resume <id>` command.
  - **Open in Claude app** makes it a Remote Control session, so it shows up in the
    Claude app on the phone or the Mac.
  
  See §4, "Who drives a session".
- **New task:** pick a project (and by default a fresh git worktree, so tasks don't
  collide), an engine (Claude Code on Pro by default, Opus/Sonnet/Haiku via `--model`;
  "Other models" = ai.Bix's own loop on Ollama/OpenRouter/API) and a first message.

## 2. Decisions

1. **Driving sessions as matt from the web.** **Decided 2026-11-18: sessions run as
   matt.** That's a remote shell as matt behind Access, an explicit exception to "no single
   compromise may reach the host as matt", accepted the way deploy-api was. It gets its own
   SECURITY.md entry when phase 3 is built.
   **Decided: Access alone**, since anything more is too slow for a daily interface. The
   broker checks your Access JWT itself on every action, so it doesn't trust the
   container's word. The accepted risk: a compromised workbench container could replay
   your login to drive sessions until it expires. To limit that:
   - a short Access session duration for this app only;
   - an audit line for every action (who, which session, when, never the message text);
   - no TCP listener on the broker (a unix socket only);
   - the broker refuses anything without a valid JWT for this app's AUD.
2. **Transcript text on the web.** **Agreed:** a fully fledged interface needs it.
   The viewport's context and the Open view put
   transcript text (code, command output, anything a command printed) behind Access. So
   far only names and status lines have left the host. This is fine for bix work, with a
   best-effort secret-redaction pass. The work laptop instance stays local-only either way.
3. **Where it lives.** **Decided 2026-11-18:** **one product
   named ai.Bix, two origins.**
   - The workbench grows out of jobs.bix.computer, which already has the right split: a
     host process reads `~/.claude`, and a minimal container serves the page. It could
     move to e.g. `work.bix.computer`.
   - The existing chat at ai.bix.computer becomes one of its engines, "Other models".
   - Why not inside the ai-router container: ai-router is the suite's most exposed
     container. It renders model output from four backends, mounts the apps tree, writes
     `todos/` and runs agent mode. The thing that drives matt's sessions should sit on
     its own origin with the smallest possible container (the suite rule: one origin per
     trust level).
4. **Sessions running in a terminal.** **Decided: both ways work.** You can pick a
   session up from the terminal, the web or the Claude app, and hand it back. See §4,
   "Who drives a session". The options considered were:
   - **Follow-only** (recommended): watch it live, and continue from the web once it's
     idle and the terminal has closed it.
   - **Fork:** continue a copy (`--fork-session`) from the web while the original stays
     in the terminal.
   - **Take over:** stop it from the web and continue there. Background sessions
     (`claude --bg`) can be taken over safely, since `claude stop` keeps the conversation.
5. **Screenshots and videos.** **Decided:** a stretch goal, but built early, as phase 2
   (§5, §6).

## 3. Status signals

| Viewport field | Source | Confidence |
|---|---|---|
| Goal | job `name`, else the first user message, else a status card (below) | ok; better with cards |
| Now | the newest timeline `detail` (bg jobs), or the last tool call in the transcript tail | good |
| Progress | **status card only.** Claude Code's task tools aren't in the transcripts here (checked: none in any transcript on bix), so there's nothing to derive it from | needs cards |
| Blocked on you | job `blocked` with `needs`, a permission prompt (hook `Notification`, or the broker's own prompt tool), a status card saying so | good |
| Blocked on something else | `needs` that says rate/usage limit (seen: "rate limited — wait and retry"), a status card naming CI, a deploy or another task, `stalled` | ok |
| Changes | the capture service (§6), indexed by session | new |

**Status cards.** Give every session one MCP tool, `report_status(goal, steps[], now,
blocked: {on: "you" | "limit" | "ci" | "deploy" | "task" | "other", why})`. It writes
`~/jobs/cards/<sessionId>.json`, and a line in the user-level CLAUDE.md asks Claude to
call it at the start, at each step and whenever it gets blocked. The collector merges the
card into the task, and cards win over guesses. Sessions started from the workbench get
the tool through `--mcp-config`. Sessions started in a terminal need it in the user-level
MCP config, which is a change outside the apps tree, so it's yours to approve. Without a
card a tile still works; it just shows no step list.

The collector also gains:
- An adapter for `claude agents --json`, the supported command that lists running
  sessions (interactive and background). It replaces parsing `sessions/*.json` where it
  covers the same ground.
- An adapter for ai.Bix conversations, so other-model tasks are tiles too.

## 4. Architecture

```
 browser ──Access──▶ workbench container (minimal; own origin)
                        │  view: snapshot + transcript slices + media (read-only mounts)
                        │  act:  messages / approvals, forwarded with your Access JWT
                        ▼ unix socket
                  host broker (matt) ─── verifies the Access JWT itself
                        │  claude -p --resume <id> --input-format stream-json
                        │         --output-format stream-json --include-partial-messages
                        │         --permission-prompt-tool mcp__workbench__permission
                        │         [--mcp-config workbench+capture] [-w <worktree>]
                        ▼
                  Claude Code sessions (matt, ~/apps), Pro subscription
 jobs-collector (host, no network) ──▶ snapshot + cards + transcript slices
 ai-router (existing) ──▶ "Other models" engine: agent mode in the bixagent sandbox
```

- **Broker:** a new host daemon running as matt. It has to be matt: its children are full
  Claude Code sessions. So the protection sits at its front door, not in a sandbox.
  - It listens on a unix socket only, mounted into the workbench container, and opens no
    TCP port.
  - Its API: `start`, `send`, `approve`, `stop`, `release`, `take-over`, `open-in-app`.
  - It checks the Access JWT forwarded with each request (signature, AUD, expiry) itself
    and writes an audit line per action (decision 1).
  - One driver per session: it hands a session over only through the moves in "Who drives
    a session" below (decision 4).
  - It runs one `claude -p` stream-json process per open session and relays its events.
  - It's the same pattern as ai.Bix Pro mode (`streaming/pro.py` already does
    `--resume` with MCP), moved to the host and pointed at matt's real `~/.claude`.
- **Permission prompts:** `--permission-prompt-tool` sends each prompt to a broker MCP
  tool, which surfaces it as an approval card and waits for your answer. The
  session's own permission mode still applies.
- **Collector:** unchanged in shape. It still reads `~/.claude` with no network. It adds
  cards, the agents adapter and transcript slices (sanitised: text and tool names, tool
  output truncated, best-effort secret redaction). The read-only view never touches the
  broker.
- **Other models:** stay in ai-router, which already runs agent mode in the `bixagent`
  sandbox with hand-off bundles and reviews. Those tasks show as tiles labelled with
  their model, and Open shows the ai.Bix conversation.
- **Not built on:**
  - The undocumented `messagingSocketPath` session socket.
  - `claude attach` (it needs a TTY; a web terminal would be a raw shell as matt).
  - Claude Code's internal file formats beyond what the collector already reads.

### Who drives a session

A session takes input from one place at a time. That place is its **driver**: the web
(the broker's `claude -p` stream-json process), a terminal (an interactive `claude`), the
Claude app (Remote Control) or background (`claude --bg`). The workbench always shows
the driver, and moving a session releases the old driver before starting the new one.
The conversation lives on disk, so it survives the move.

| From → to | How |
|---|---|
| web → terminal | **Continue in terminal**: the broker ends its process once the session is idle, and the page shows `cd <cwd> && claude --resume <id>` to copy |
| terminal → web | **Continue here**: if the terminal session is idle, the broker stops that `claude` process (after a confirm: it closes it in your terminal) and resumes it under the broker. If it's mid-turn, it waits for the turn to end, or offers "interrupt and take over" |
| background → web | `claude stop <id>` (the conversation is kept), then resume under the broker |
| web or terminal → Claude app | **Open in Claude app**: the broker restarts the session with Remote Control enabled (Matt confirmed 2026-11-18 that a host program can do this), so it appears in the Claude app on the phone and the Mac. The `remoteControlAtStartup` setting would instead make every session reachable from the app from the start (offered, not set) |
| Claude app → web or terminal | the same as from a terminal: stop the Remote Control process, then resume |

**Always cleanly resumable.** This is a requirement, not a nicety: the connection from a
phone won't be perfect.
- The broker owns the session process. A browser disconnect never stops the work.
- The page reconnects and replays from the last event it saw. Events carry a
  per-session sequence number, and the broker keeps a ring of recent ones.
- Every send carries a client-made id, so a retried send after a dropped connection is
  delivered once.
- A move between drivers waits for the end of a turn (or is an explicit "interrupt"),
  and only reports success once the next driver has resumed the same session id.
- If a resume fails, the session stays where it was, with an error. It's never left
  with no driver.
- Phase 3 tests this by killing the browser, the broker and the network mid-turn, and
  checking the session continues and no message is lost or doubled.

Where a session was started (web, terminal, app, background) is recorded for the hold
metrics, but nothing depends on it. The driver comes from the broker's own list of
processes, else the session file's `kind`, else "terminal". New sessions are started by
the broker, but can be picked up in a terminal at any time.

## 5. Phases

1. **Read-only workbench:** list plus viewport (goal, now, blocked-on classification, the
   context slice), on the jobs origin. Needs decision 3, plus the hooks you haven't
   approved yet. No new privileges.
2. **Changes and status cards (stretch goal, done early):**
   - The capture service (§6) and the viewport's gallery of screenshots and videos.
   - The `report_status` tool, which brings goal, steps and blocked reasons, and links
     each capture to the step that made it.
   - Both are offered to terminal sessions through the user-level MCP config, which
     needs your OK.
3. **Open and continue as matt:**
   - the broker, with the Access JWT checked at the broker;
   - streaming, the model picker and the plan-usage summary;
   - approval cards;
   - drivers and handover between web, terminal, background and the Claude app.

   Decisions 1 and 4 are settled. This phase crosses the security boundary; it gets its own SECURITY.md entry
   and a probe script like `agent-sandbox-check.py`.
4. **New tasks from the web:** project picker, a worktree per task by default, model
   picker, and other-model tasks through ai-router. After this phase the terminal is
   optional.
5. **Converge:** the chat UI at ai.bix.computer folds into the workbench as the
   other-models engine, or stays as a plain chat. Your call once phase 4 has been used.
   `bix-ai/CLAUDE.md` is rewritten around the workbench.

## 6. Screenshots and usage videos

- **Capture service:** a container with headless Chromium (Playwright). It has no
  secrets, no mounts except an output directory, and network access only to bix
  services' local ports and to a preview port range on the host where sessions run dev
  servers.
- **How sessions use it:** they call an MCP tool `capture(target, steps)`, where `steps`
  is a small fixed vocabulary (goto, click, type, wait, scroll), not a script. So no
  model-written code runs in the browser container. It returns screenshots, or a short
  webm (≤ 15 s), into `~/jobs/media/<sessionId>/`.
- **Before and after:** "before" is the running service; "after" is the session's dev
  server on a preview port. The card links them, and the viewport shows them side by
  side.
- **Behind Access:** services behind Access are captured at their origin port with the
  right Host header, the same way the nightly ping reaches them.

## 7. Hold-up metrics

jobs.bix tracks how often tasks get held up, what held them, and how each hold was
released, across every driver. The design is in `jobs/DESIGN.md` §11. In short:
- the collector records each hold as an episode (start, end, category, release method);
- the workbench and jobs.bix show the trends;
- the broker and the hooks supply the release signals ("replied from the web", "approved
  in the terminal", "limit reset by itself").

It starts with phase 1, because the collector already sees every status change.

## 8. Not yet confirmed

- Whether `claude -p --resume` can take over a background session cleanly after `claude
  stop`. The help says the conversation is kept; untested.
- The exact contract of `--permission-prompt-tool` on 2.1.327 (it's documented for the
  Agent SDK; untested here).
- Whether a terminal session can turn on Remote Control after it has started, and whether
  its claude.ai URL can be found locally (matters only for decision 1d).
- Plan usage: `stats-cache.json` has tokens per model and per day, but no plan-limit
  percentages or reset times, and `policy-limits.json` doesn't exist on bix. Candidates
  for the limits: the data Claude Code gives a status line, the `/usage` command, and
  the reset time in a rate-limited job's `needs` text ("resets 11:30pm").
- How to stop an idle terminal `claude` process from outside so the conversation resumes
  cleanly (`claude stop` documents this for background sessions only). Clean resume is
  required (§4), so phase 3 starts by testing this, and a signal is used only if it
  passes.
- The session file's `entrypoint` values for `-p`, background and Remote Control
  sessions (nice to have; only `cli` has been seen).
