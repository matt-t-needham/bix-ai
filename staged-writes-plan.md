# Plan — staged write capability for bix-ai auto mode

## Goal
Let auto mode (Forge + local model) **propose** file edits repo-wide. Proposals
land in a staging area and touch the live tree only on explicit human approval.
A human can optionally trigger a Claude advisory review (text only). This makes
the "create a draft post" class of task — currently impossible because no write
tool exists — actually work, without handing a weak model unsupervised write
access.

## Shape: propose -> review -> apply
1. Local model calls `stage_write(target_path, content)` -> record in staging
   store, **never** the live tree.
2. Human opens a dedicated review page, sees the diff, optionally clicks "Ask
   Claude to review" (returns text they read).
3. Human **Approve** -> bytes written to the live target (the only privileged
   step); or **Reject** -> discarded.

## Files & changes

### 1. `fs_core.py` — new write denylist (the uncertain part)
Add `is_write_denied_path(p)` covering the *privilege / guardrail-subverting*
class only:
- `*.sh`; anything under a `scripts/` dir
- `Dockerfile*`, `docker-compose*.{yml,yaml}`, `compose*.{yml,yaml}`
- anything under `.github/`
- anything under `bix-ai/` source (self-modification could rewrite these very
  guardrails)

**Allowed** (the point of the feature): ordinary app source + content everywhere
else — `ai_graph_mode`, `beatshare`, `bix-demucs`, blog content/templates, the
Flutter apps, infra docs/todos.

A write must satisfy: within `FS_ROOT` **and** not `is_denied_path` (secrets)
**and** not `is_write_denied_path`.

### 2. `staging.py` — new module, persists under `data/staging/`
One JSON record per change: `id, created_at, proposed_by, target_path, content,
status (pending|approved|rejected), claude_review, reviewed_at, applied_at`.
Functions: `create / list / get / set_review / approve / reject`. Matches
`memory.py`'s async-file persistence style.

### 3. `tools.py` — `stage_write` tool
Validates target (FS_ROOT + both denylists); on denial returns a clear message
the model relays ("can't stage X — protected path"). On success creates a record
and returns "staged for review, id=…, will not apply until you approve." Added
to `FORGE_TOOLS` (auto) and `OLLAMA_TOOLS` (local) — shared dispatch, trivial to
include both. `FORGE_SYSTEM` rewritten: repo-wide staged editing, **no blog
special-casing**, explicit note that writes are reviewed before applying.

### 4. `main.py` — routes (HTML pages mirror the existing memory viewer, `main.py:159`)
- `GET /staging` -> server-rendered list page (Catppuccin, pending first)
- `GET /staging/{id}` -> detail page: content + unified diff vs live file
  (`difflib`), status, any Claude review, Approve/Reject/Ask-Claude buttons
- `POST /staging/{id}/review` -> one-shot non-streaming Claude call (reusing the
  `ANTHROPIC_URL` POST pattern at `main.py:406`), advisory text persisted +
  returned
- `POST /staging/{id}/approve` -> **re-validate both denylists + FS_ROOT**, write
  to live target, mark applied
- `POST /staging/{id}/reject` -> discard

### 5. Claude advisory review prompt
"Review this proposed change to `<path>`; here's the current file (or 'new file')
and the proposed content; flag risks/correctness/anything not safe to approve.
Advisory only — a human decides." The staged content is model-generated and
possibly injection-laundered, so it's wrapped as untrusted data to *review*, not
instructions to follow.

### 6. `static/index.html` — minimal entry point
A small "N pending reviews ->" link that opens `/staging` in a new page. No drawer.

## Safety invariants (must hold)
- `stage_write` never writes live.
- `approve` is the only path that writes live, always human-triggered, and
  re-validates denylists (never trusts the stage-time check alone).
- Claude review is advisory text; it cannot approve or promote.
- Both denylists (secrets + privilege) enforced at **stage and approve**.

## Tests (`tests/`)
- Denied at stage: secret path, `*.sh`, `scripts/`, `Dockerfile`,
  `docker-compose.yml`, `.github/`, `bix-ai/` source.
- Allowed at stage: blog content, `ai_graph_mode` source, infra todo.
- Approve re-validation refuses a record whose target is denied.
- Approve writes the file; reject discards.

## Scope notes
- v1 `stage_write` writes whole-file content (overwrites on edit) — the diff view
  shows exactly what changes, so it's reviewable; `apply_patch` granularity can
  come later.
- No auto-apply, ever. No git commits (per repo rule — you commit). Approved
  writes land as uncommitted working-tree edits and deploy through your normal
  nightly rebuild cycle.

## Open decisions before build
1. **Denylist line** (component 1): deny privilege/guardrail class, allow ordinary
   source + content. This is the point flagged as uncertain.
2. **Scope of `stage_write`**: local mode too, or auto-only?
