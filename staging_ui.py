"""Staged-write review UI — HTML rendering for the /staging routes in main.py.

Same pattern as routing_dash.py: pure rendering + small helpers, stdlib-only
(json / difflib / html / pathlib). No FastAPI imports — main.py owns the routes
and calls render_list / render_detail / build_chat_context.

The detail page is interactive (tabs, selection-to-comment, review/revise
buttons) via inline vanilla JS. Staged content and comments are untrusted
model/user input: everything server-rendered goes through _esc(), and the
record JSON embedded for the client has every "<" encoded as \\u003c so it is
inert inside its <script> blob; the client-side code only inserts it into the
DOM via textContent or after escaping.
"""
import difflib
import html
import json

import staging

_MD_SUFFIXES = (".md", ".markdown", ".txt")


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _fmt_ts(ts: str | None) -> str:
    return (ts or "")[:19].replace("T", " ")


def _fmt_model(m: str | None) -> str:
    # "claude-sonnet-4-6" -> "sonnet-4-6"; drop date-stamped suffixes.
    import re
    return re.sub(r"-\d{8}$", "", (m or "").removeprefix("claude-"))


# ── Diff ──────────────────────────────────────────────────────────────────────

def diff_lines(record: dict) -> list[str]:
    """Unified diff between the current on-disk version and the proposed content.

    Reads staging.current_source_path (the tree approve() actually writes), so a
    self-change diffs against the staging clone, not the untouched prod tree.
    """
    target = staging.current_source_path(record)
    try:
        current = target.read_text(errors="replace").splitlines() if target.exists() else []
    except OSError:
        current = []
    proposed = record["content"].splitlines()
    return list(difflib.unified_diff(
        current, proposed,
        fromfile=f"a/{target}", tofile=f"b/{target}",
        lineterm="",
    ))


def _diff_html(record: dict) -> str:
    lines = []
    for ln in diff_lines(record):
        cls = ""
        if ln.startswith("+") and not ln.startswith("+++"):
            cls = "add"
        elif ln.startswith("-") and not ln.startswith("---"):
            cls = "del"
        elif ln.startswith("@@"):
            cls = "hunk"
        lines.append(f'<span class="dl {cls}">{_esc(ln)}</span>')
    return "\n".join(lines) or '<span class="dl">(no differences)</span>'


# ── Chat / review context ─────────────────────────────────────────────────────

_CTX_DIFF_MAX_LINES = 400


def build_chat_context(record: dict) -> str:
    """Prose context block for seeding a chat (/?staged=<id>) about a record."""
    comments = record.get("comments") or []
    parts = [
        "I want to discuss a staged file change from my bix-ai staging area "
        "(a model-proposed write awaiting my review — nothing is on disk yet).",
        "",
        f"- id: {record['id']}  ·  status: {record.get('status', 'pending')}",
        f"- target file: {record['target_path']}",
        f"- proposed by {record.get('proposed_by', '?')} at {_fmt_ts(record.get('created_at'))}",
    ]
    if record.get("self_change"):
        parts.append(
            f"- this is a change to bix-ai's own source; on approval it is applied to "
            f"the staging clone at {staging.current_source_path(record)}, never prod"
        )
    if record.get("critical"):
        parts.append(
            "- CRITICAL: the target is part of bix-ai's guardrail/privilege surface "
            "(path checks, staging, config, tool registry) — review with extra care"
        )
    if record.get("revisions"):
        parts.append(f"- revised {len(record['revisions'])} time(s) since first proposed")
    parts += ["", "<proposed_content>", record["content"], "</proposed_content>", ""]

    target = staging.current_source_path(record)
    if target.exists():
        dl = diff_lines(record)
        if len(dl) > _CTX_DIFF_MAX_LINES:
            dl = dl[:_CTX_DIFF_MAX_LINES] + [f"… (diff truncated at {_CTX_DIFF_MAX_LINES} lines)"]
        parts += ["<diff_vs_live_file>", "\n".join(dl), "</diff_vs_live_file>", ""]
    else:
        parts += ["(The target file does not exist yet — this is a new file.)", ""]

    if record.get("claude_review"):
        model = _fmt_model(record.get("review_model")) or "claude"
        parts += [f"<latest_review model={model}>", record["claude_review"], "</latest_review>", ""]
    if comments:
        parts.append("<reviewer_comments>")
        parts.append(comments_block_text(record))
        parts += ["</reviewer_comments>", ""]

    parts.append(
        "I'll add my own thoughts next — help me evaluate this change and decide "
        "whether to approve, revise, or reject it."
    )
    return "\n".join(parts)


def comments_block_text(record: dict, quote_max: int = 400) -> str:
    """Plain-text listing of reviewer comments, for prompts/context."""
    lines = []
    for c in record.get("comments") or []:
        status = "RESOLVED" if c.get("resolved") else "OPEN"
        quote = (c.get("quote") or "").strip()
        if len(quote) > quote_max:
            quote = quote[:quote_max] + "…"
        anchor = f' (re: "{quote}")' if quote else ""
        lines.append(f"- [{status}]{anchor} {c.get('text', '')}")
    return "\n".join(lines)


# ── CSS (Catppuccin Mocha — same variables as the chat UI) ────────────────────

CSS = """
:root {
  --bg:#1e1e2e; --surface0:#313244; --surface1:#45475a;
  --text:#cdd6f4; --subtext:#a6adc8;
  --blue:#89b4fa; --mauve:#cba6f7; --green:#a6e3a1; --red:#f38ba8; --yellow:#f9e2af;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:system-ui,-apple-system,sans-serif; background:var(--bg);
  color:var(--text); padding:32px 24px; max-width:980px; margin:0 auto; line-height:1.5;
}
body.detail { max-width:1240px; }
a { color:var(--blue); text-decoration:none; }
a:hover { text-decoration:underline; }
.hdr { border-bottom:1px solid var(--surface1); padding-bottom:14px; margin-bottom:18px; }
.hdr h1 { font-size:1.3rem; font-weight:600; }
.meta { font-size:.78rem; color:var(--subtext); display:flex; gap:14px; flex-wrap:wrap; font-variant-numeric:tabular-nums; }
.row { display:block; padding:12px 14px; margin:8px 0; background:var(--surface0); border-radius:6px; border-left:2px solid var(--surface1); }
.row.pending { border-left-color:var(--yellow); }
.row.approved { border-left-color:var(--green); }
.row.rejected { border-left-color:var(--red); opacity:.7; }
.row .path { color:var(--text); font-size:.9rem; }
.row .sub { color:var(--subtext); font-size:.74rem; font-variant-numeric:tabular-nums; }
.badge { font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; padding:1px 7px; border-radius:3px; background:var(--surface1); color:var(--subtext); }
.badge.pending { color:var(--yellow); } .badge.approved { color:var(--green); } .badge.rejected { color:var(--red); }
.badge.crit { color:var(--red); border:1px solid var(--red); background:none; }
.banner-crit { background:rgba(243,139,168,.1); border-left:2px solid var(--red); color:var(--text);
               padding:10px 14px; margin:14px 0; font-size:.85rem; border-radius:0 4px 4px 0; }
.banner-crit b { color:var(--red); }
.applies-to { color:var(--subtext); font-size:.78rem; font-variant-numeric:tabular-nums; margin:6px 0 0; }
pre.diff { background:#181825; padding:14px; border-radius:6px; overflow-x:auto; font-size:.8rem; line-height:1.45; }
.dl { display:block; white-space:pre; }
.dl.add { color:var(--green); } .dl.del { color:var(--red); } .dl.hunk { color:var(--blue); }
.review { background:var(--surface0); border-left:2px solid var(--mauve); padding:10px 14px; margin:14px 0; font-size:.86rem; border-radius:0 4px 4px 0; }
.review .rbody { white-space:pre-wrap; }
.review .rmeta { font-size:.7rem; color:var(--subtext); margin-bottom:6px; font-variant-numeric:tabular-nums; }
.actions { display:flex; gap:10px; margin:18px 0; flex-wrap:wrap; align-items:center; }
.actions button { font:inherit; font-size:.85rem; padding:7px 16px; border:none; border-radius:5px; cursor:pointer; color:var(--bg); }
.actions button:disabled { opacity:.45; cursor:default; }
.btn-approve { background:var(--green); } .btn-reject { background:var(--red); }
.btn-review, .btn-revise { background:var(--mauve); }
a.btn-chat { display:inline-block; padding:6px 15px; border:1px solid var(--blue); border-radius:5px; color:var(--blue); font-size:.85rem; }
a.btn-chat:hover { background:var(--surface0); text-decoration:none; }
.actions select { font:inherit; font-size:.82rem; padding:6px 10px; background:var(--surface0); color:var(--text); border:1px solid var(--surface1); border-radius:5px; }
.note { color:var(--subtext); font-size:.85rem; font-style:italic; padding:10px 0; }
.err { color:var(--red); font-size:.82rem; padding:6px 0; }
/* Detail layout */
.layout { display:grid; grid-template-columns:minmax(0,1fr) 320px; gap:20px; align-items:start; }
@media (max-width: 900px) { .layout { grid-template-columns:1fr; } }
.tabs { display:flex; gap:2px; border-bottom:1px solid var(--surface1); margin-bottom:12px; }
.tab { font:inherit; font-size:.82rem; background:none; border:none; cursor:pointer;
       color:var(--subtext); padding:7px 14px; border-bottom:2px solid transparent; }
.tab:hover { color:var(--text); }
.tab.active { color:var(--blue); border-bottom-color:var(--blue); }
.pane { display:none; }
.pane.active { display:block; }
pre.raw { background:#181825; padding:14px; border-radius:6px; overflow-x:auto;
          font-size:.8rem; line-height:1.45; white-space:pre-wrap; word-wrap:break-word; }
/* Rendered markdown */
.md { background:var(--surface0); padding:22px 26px; border-radius:6px; font-size:.9rem; overflow-wrap:break-word; }
.md h1,.md h2,.md h3,.md h4 { margin:18px 0 8px; line-height:1.3; }
.md h1:first-child,.md h2:first-child,.md h3:first-child { margin-top:0; }
.md h1 { font-size:1.35rem; } .md h2 { font-size:1.15rem; } .md h3 { font-size:1rem; } .md h4 { font-size:.9rem; }
.md p { margin:8px 0; }
.md ul,.md ol { margin:8px 0; padding-left:24px; }
.md li { margin:3px 0; }
.md pre { background:#181825; padding:12px; border-radius:5px; overflow-x:auto; margin:10px 0; font-size:.8rem; line-height:1.45; }
.md code { background:#181825; padding:1px 5px; border-radius:3px; font-size:.82em; }
.md pre code { background:none; padding:0; font-size:inherit; }
.md blockquote { border-left:2px solid var(--surface1); padding:2px 0 2px 12px; color:var(--subtext); margin:8px 0; }
.md hr { border:none; border-top:1px solid var(--surface1); margin:16px 0; }
/* Comment highlights */
mark.hl { background:rgba(249,226,175,.25); color:inherit; border-bottom:1px dotted var(--yellow);
          border-radius:2px; cursor:pointer; }
mark.hl:hover { background:rgba(249,226,175,.4); }
/* Comments panel */
.cpanel h2 { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:var(--subtext); margin-bottom:4px; }
.comment { background:var(--surface0); border-left:2px solid var(--yellow); border-radius:0 4px 4px 0;
           padding:9px 11px; margin:8px 0; font-size:.8rem; }
.comment.resolved { border-left-color:var(--green); opacity:.6; }
.comment .cq { color:var(--subtext); font-style:italic; font-size:.73rem; border-left:2px solid var(--surface1);
               padding-left:8px; margin-bottom:6px; max-height:60px; overflow:hidden; white-space:pre-wrap; }
.comment .ct { white-space:pre-wrap; }
.comment .cmeta { display:flex; justify-content:space-between; align-items:center; margin-top:7px;
                  font-size:.68rem; color:var(--subtext); font-variant-numeric:tabular-nums; }
.cbtn { font:inherit; font-size:.68rem; background:var(--surface1); color:var(--text);
        border:none; border-radius:3px; padding:2px 9px; cursor:pointer; }
.cbtn:hover { background:var(--bg); }
/* Selection button + composer */
#sel-btn { position:absolute; z-index:10; font:inherit; font-size:.75rem; background:var(--mauve); color:var(--bg);
           border:none; border-radius:4px; padding:4px 11px; cursor:pointer; box-shadow:0 2px 10px rgba(0,0,0,.45); }
#composer { position:fixed; bottom:24px; right:24px; width:min(360px, calc(100vw - 48px)); z-index:20;
            background:var(--surface0); border:1px solid var(--surface1); border-radius:8px; padding:14px; }
#composer .cq { color:var(--subtext); font-style:italic; font-size:.74rem; border-left:2px solid var(--mauve);
                padding-left:8px; margin-bottom:8px; max-height:80px; overflow:auto; white-space:pre-wrap; }
#composer textarea { width:100%; min-height:72px; resize:vertical; font:inherit; font-size:.82rem;
                     background:var(--bg); color:var(--text); border:1px solid var(--surface1); border-radius:4px; padding:8px; }
#composer textarea:focus { outline:none; border-color:var(--mauve); }
#composer .crow { display:flex; gap:8px; justify-content:flex-end; margin-top:8px; }
#composer .crow button { font:inherit; font-size:.78rem; padding:5px 13px; border:none; border-radius:4px; cursor:pointer; }
#composer .c-save { background:var(--mauve); color:var(--bg); }
#composer .c-cancel { background:var(--surface1); color:var(--text); }
"""


# ── List page ─────────────────────────────────────────────────────────────────

def render_list(records: list[dict]) -> str:
    if not records:
        body = '<div class="note">No staged changes.</div>'
    else:
        rows = []
        for r in records:
            st = r.get("status", "pending")
            open_c = sum(1 for c in (r.get("comments") or []) if not c.get("resolved"))
            extra = f' · {open_c} open comment(s)' if open_c else ""
            crit = ' <span class="badge crit">critical</span>' if r.get("critical") else ""
            rows.append(
                f'<a class="row {st}" href="/staging/{_esc(r["id"])}">'
                f'<div class="path">{_esc(r.get("target_path",""))} '
                f'<span class="badge {st}">{_esc(st)}</span>{crit}</div>'
                f'<div class="sub">{_esc(_fmt_ts(r.get("created_at")))} '
                f'· by {_esc(r.get("proposed_by","?"))} · id {_esc(r["id"])}{_esc(extra)}</div></a>'
            )
        body = "\n".join(rows)
    return (
        f'<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        f'<title>Staged writes</title><style>{CSS}</style></head><body>'
        f'<div class="hdr"><h1>Staged writes</h1>'
        f'<div class="meta">{len(records)} record(s) · review before they touch disk</div></div>'
        f'{body}</body></html>'
    )


# ── Detail page ───────────────────────────────────────────────────────────────

def _record_json_blob(record: dict) -> str:
    """Client-side copy of the record. Every "<" is JSON-escaped to \\u003c so
    untrusted content can never terminate or open a tag inside the script blob."""
    slim = {
        "id":            record["id"],
        "status":        record.get("status", "pending"),
        "target_path":   record.get("target_path", ""),
        "content":       record.get("content", ""),
        "comments":      record.get("comments") or [],
        "claude_review": record.get("claude_review"),
        "review_model":  record.get("review_model"),
        "revisions":     [{"at": r.get("at"), "by": r.get("by")} for r in record.get("revisions") or []],
    }
    return json.dumps(slim).replace("<", "\\u003c")


def render_detail(record: dict, *, models: list[str], default_model: str) -> str:
    st  = record.get("status", "pending")
    rid = _esc(record["id"])
    is_md = record.get("target_path", "").lower().endswith(_MD_SUFFIXES)
    default_tab = "rendered" if is_md else "diff"

    # Header meta
    meta = [
        f'<span>{_esc(_fmt_ts(record.get("created_at")))}</span>',
        f'<span>by {_esc(record.get("proposed_by","?"))}</span>',
        f'<span>id {rid}</span>',
    ]
    revs = record.get("revisions") or []
    if revs:
        meta.append(f'<span>revision {len(revs) + 1} · last revised {_esc(_fmt_ts(revs[-1].get("at")))}</span>')
    meta.append('<span><a href="/staging">← all</a></span>')

    # Actions
    chat_btn = f'<a class="btn-chat" href="/?staged={rid}">💬 Chat about this</a>'
    if st == "pending":
        opts = "".join(
            f'<option value="{_esc(m)}"{" selected" if m == default_model else ""}>{_esc(_fmt_model(m))}</option>'
            for m in models
        )
        actions = (
            f'<div class="actions">'
            f'<form method="post" action="/staging/{rid}/approve"><button class="btn-approve">Approve &amp; write</button></form>'
            f'<form method="post" action="/staging/{rid}/reject"><button class="btn-reject">Reject</button></form>'
            f'{chat_btn}'
            f'<select id="model-sel" title="Claude model for review/revise">{opts}</select>'
            f'<button class="btn-review" id="btn-review">Review with Claude</button>'
            f'<button class="btn-revise" id="btn-revise" title="Have Claude rewrite the proposed content to address the open comments">Revise from comments</button>'
            f'</div>'
            f'<div class="err" id="action-msg" hidden></div>'
        )
    else:
        applied = _fmt_ts(record.get("applied_at"))
        extra = f" at {_esc(applied)}" if applied else ""
        deploy_bits = ""
        if st == "approved" and record.get("self_change"):
            if record.get("promoted_at"):
                deploy_bits = (f'<span class="badge approved">promoted '
                               f'{_esc(_fmt_ts(record["promoted_at"]))}</span>')
            else:
                # Queue for the host-side runner: build+run the staging container,
                # or promote this record's content into the prod tree.
                deploy_bits = (
                    f'<form method="post" action="/deploys/enqueue">'
                    f'<input type="hidden" name="action" value="deploy-staging">'
                    f'<input type="hidden" name="note" value="staging record {rid}">'
                    f'<button class="btn-review">Deploy staging</button></form>'
                    f'<form method="post" action="/deploys/enqueue">'
                    f'<input type="hidden" name="action" value="promote">'
                    f'<input type="hidden" name="record_ids" value="{rid}">'
                    f'<button class="btn-approve">Promote to prod</button></form>'
                )
        actions = (
            f'<div class="actions"><span class="note">This change is {_esc(st)}{extra}. '
            f'</span>{deploy_bits}{chat_btn}</div>'
        )

    # Review box
    review_html = ""
    if record.get("claude_review"):
        rmeta = []
        if record.get("review_model"):
            rmeta.append(_esc(_fmt_model(record["review_model"])))
        if record.get("reviewed_at"):
            rmeta.append(_esc(_fmt_ts(record["reviewed_at"])))
        review_html = (
            f'<div class="review"><div class="rmeta">Claude review · {" · ".join(rmeta)}</div>'
            f'<div class="rbody">{_esc(record["claude_review"])}</div></div>'
        )

    crit_badge = ' <span class="badge crit">critical</span>' if record.get("critical") else ""
    crit_banner = ""
    if record.get("critical"):
        crit_banner = (
            '<div class="banner-crit"><b>Critical self-file.</b> This file is part of '
            "bix-ai's guardrail/privilege surface (path checks, staging, config, tool "
            'registry). Review with extra care before approving.</div>'
        )
    applies_html = ""
    if record.get("self_change"):
        applies_html = (
            f'<div class="applies-to">applies to: '
            f'{_esc(str(staging.current_source_path(record)))} '
            f'(staging clone — prod is only written on promote)</div>'
        )

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Staged write — {rid}</title><style>{CSS}</style></head><body class="detail">
<div class="hdr"><h1>{_esc(record.get("target_path",""))} <span class="badge {st}">{_esc(st)}</span>{crit_badge}</h1>
<div class="meta">{"".join(meta)}</div>{applies_html}</div>
{crit_banner}
{actions}
{review_html}
<div class="layout">
  <div class="main">
    <div class="tabs">
      <button class="tab" data-view="rendered">Rendered</button>
      <button class="tab" data-view="raw">Raw</button>
      <button class="tab" data-view="diff">Diff</button>
    </div>
    <div id="view-rendered" class="pane md content-pane"></div>
    <pre id="view-raw" class="pane raw content-pane"></pre>
    <pre id="view-diff" class="pane diff">{_diff_html(record)}</pre>
  </div>
  <div class="cpanel">
    <h2>Comments</h2>
    <div id="comment-list"></div>
    <div class="note" id="comment-hint"></div>
  </div>
</div>
<button id="sel-btn" hidden>💬 Comment</button>
<div id="composer" hidden>
  <div class="cq" id="composer-quote"></div>
  <textarea id="composer-text" placeholder="Comment on the highlighted section…"></textarea>
  <div class="crow">
    <button class="c-cancel" id="composer-cancel">Cancel</button>
    <button class="c-save" id="composer-save">Add comment</button>
  </div>
</div>
<script type="application/json" id="rec-data">{_record_json_blob(record)}</script>
<script>{_DETAIL_JS}
document.addEventListener('DOMContentLoaded', () => initDetail({json.dumps(default_tab)}));
</script>
</body></html>"""


# ── Detail page JS (vanilla, no deps) ─────────────────────────────────────────

_DETAIL_JS = r"""
const REC = JSON.parse(document.getElementById('rec-data').textContent);
const PENDING = REC.status === 'pending';

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// Minimal document-markdown renderer: escape first, then headings, fenced
// code, hr, lists, blockquotes, inline code/bold/em/links, paragraphs.
function renderMd(src) {
  const fences = [];
  let s = src.replace(/```(\w*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    fences.push('<pre><code>' + escHtml(code.replace(/\n$/, '')) + '</code></pre>');
    return '\n\u0000' + (fences.length - 1) + '\u0000\n';
  });
  s = escHtml(s);
  const inline = t => t
    .replace(/`([^`\n]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|\W)\*([^*\n]+)\*(?=\W|$)/g, '$1<em>$2</em>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  const out = [];
  let list = null, para = [], quote = [];
  const flushPara  = () => { if (para.length)  { out.push('<p>' + inline(para.join('<br>')) + '</p>'); para = []; } };
  const flushList  = () => { if (list) { out.push('</' + list + '>'); list = null; } };
  const flushQuote = () => { if (quote.length) { out.push('<blockquote>' + inline(quote.join('<br>')) + '</blockquote>'); quote = []; } };
  for (const rawLine of s.split('\n')) {
    const t = rawLine.trim();
    const ph = t.match(/^\u0000(\d+)\u0000$/);
    if (ph) { flushPara(); flushList(); flushQuote(); out.push(fences[+ph[1]]); continue; }
    const h = t.match(/^(#{1,4})\s+(.*)$/);
    if (h) { flushPara(); flushList(); flushQuote(); out.push(`<h${h[1].length}>${inline(h[2])}</h${h[1].length}>`); continue; }
    if (/^(?:-{3,}|\*{3,}|_{3,})$/.test(t)) { flushPara(); flushList(); flushQuote(); out.push('<hr>'); continue; }
    const bq = t.match(/^&gt;\s?(.*)$/);
    if (bq) { flushPara(); flushList(); quote.push(bq[1]); continue; }
    flushQuote();
    const ul = t.match(/^[-*+]\s+(.*)$/), ol = t.match(/^\d+[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const want = ul ? 'ul' : 'ol';
      if (list !== want) { flushList(); out.push('<' + want + '>'); list = want; }
      out.push('<li>' + inline((ul || ol)[1]) + '</li>');
      continue;
    }
    if (t === '') { flushPara(); flushList(); continue; }
    para.push(t);
  }
  flushPara(); flushList(); flushQuote();
  return out.join('\n');
}

// Wrap the first single-text-node occurrence of each open comment's quote in
// a <mark>. Best effort: quotes spanning element boundaries just don't get a
// highlight — the comment card still shows the quote.
function applyHighlights(container) {
  for (const c of REC.comments) {
    if (c.resolved || !c.quote) continue;
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    let node;
    while ((node = walker.nextNode())) {
      if (node.parentElement && node.parentElement.closest('mark.hl')) continue;
      const idx = node.textContent.indexOf(c.quote);
      if (idx === -1) continue;
      const range = document.createRange();
      range.setStart(node, idx);
      range.setEnd(node, idx + c.quote.length);
      const mark = document.createElement('mark');
      mark.className = 'hl';
      mark.dataset.cid = c.id;
      mark.title = c.text;
      try { range.surroundContents(mark); } catch (e) { /* partial-node overlap — skip */ }
      break;
    }
  }
  container.querySelectorAll('mark.hl').forEach(m => {
    m.addEventListener('click', () => {
      const card = document.getElementById('comment-' + m.dataset.cid);
      if (card) { card.scrollIntoView({ behavior: 'smooth', block: 'center' }); }
    });
  });
}

async function postJson(url, body) {
  const r = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' },
                               body: JSON.stringify(body) });
  let d = {};
  try { d = await r.json(); } catch (e) {}
  if (!r.ok || d.ok === false) throw new Error(d.message || ('HTTP ' + r.status));
  return d;
}

function renderComments() {
  const listEl = document.getElementById('comment-list');
  const hint   = document.getElementById('comment-hint');
  listEl.textContent = '';
  hint.textContent = PENDING ? 'Select text in the content to add a comment.' : '';
  if (!REC.comments.length) {
    if (!PENDING) hint.textContent = 'No comments.';
    return;
  }
  for (const c of REC.comments) {
    const card = document.createElement('div');
    card.className = 'comment' + (c.resolved ? ' resolved' : '');
    card.id = 'comment-' + c.id;
    if (c.quote) {
      const q = document.createElement('div'); q.className = 'cq';
      q.textContent = c.quote.length > 200 ? c.quote.slice(0, 200) + '…' : c.quote;
      card.appendChild(q);
    }
    const t = document.createElement('div'); t.className = 'ct'; t.textContent = c.text;
    const meta = document.createElement('div'); meta.className = 'cmeta';
    const when = document.createElement('span');
    when.textContent = (c.resolved ? 'resolved' : (c.created_at || '').slice(0, 16).replace('T', ' '));
    const btn = document.createElement('button'); btn.className = 'cbtn';
    btn.textContent = c.resolved ? 'Reopen' : 'Resolve';
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      try {
        await postJson(`/staging/${REC.id}/comments/${c.id}/resolve`, { resolved: !c.resolved });
        location.reload();
      } catch (e) { btn.disabled = false; showActionError('Comment update failed: ' + e.message); }
    });
    meta.append(when, btn);
    card.append(t, meta);
    listEl.appendChild(card);
  }
}

function showActionError(msg) {
  const el = document.getElementById('action-msg');
  if (el) { el.textContent = msg; el.hidden = false; }
  else alert(msg);
}

// ── Selection → comment ───────────────────────────────────────────────────────
let pendingQuote = '';

function initSelectionUI() {
  const selBtn   = document.getElementById('sel-btn');
  const composer = document.getElementById('composer');
  const cQuote   = document.getElementById('composer-quote');
  const cText    = document.getElementById('composer-text');

  document.addEventListener('mouseup', (e) => {
    if (composer.contains(e.target) || e.target === selBtn) return;
    setTimeout(() => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed) { selBtn.hidden = true; return; }
      const text = sel.toString().trim();
      const anchor = sel.anchorNode;
      const inPane = anchor && [...document.querySelectorAll('.content-pane, pre.diff')]
        .some(p => p.contains(anchor));
      if (!text || !inPane) { selBtn.hidden = true; return; }
      pendingQuote = text;
      const rect = sel.getRangeAt(0).getBoundingClientRect();
      selBtn.style.top  = (window.scrollY + rect.top - 34) + 'px';
      selBtn.style.left = (window.scrollX + Math.max(0, rect.left + rect.width / 2 - 45)) + 'px';
      selBtn.hidden = false;
    }, 0);
  });

  selBtn.addEventListener('click', () => {
    selBtn.hidden = true;
    cQuote.textContent = pendingQuote.length > 400 ? pendingQuote.slice(0, 400) + '…' : pendingQuote;
    cText.value = '';
    composer.hidden = false;
    cText.focus();
  });

  document.getElementById('composer-cancel').addEventListener('click', () => { composer.hidden = true; });
  document.getElementById('composer-save').addEventListener('click', async () => {
    const text = cText.value.trim();
    if (!text) { cText.focus(); return; }
    const saveBtn = document.getElementById('composer-save');
    saveBtn.disabled = true;
    try {
      await postJson(`/staging/${REC.id}/comments`, { quote: pendingQuote, text });
      location.reload();
    } catch (e) {
      saveBtn.disabled = false;
      showActionError('Comment failed: ' + e.message);
    }
  });
}

// ── Review / revise actions ───────────────────────────────────────────────────
function initReviewActions() {
  const reviewBtn = document.getElementById('btn-review');
  const reviseBtn = document.getElementById('btn-revise');
  if (!reviewBtn) return;
  const openCount = REC.comments.filter(c => !c.resolved).length;
  reviseBtn.disabled = openCount === 0;
  if (openCount === 0) reviseBtn.title = 'Add at least one open comment first';

  async function run(action, btn, busyLabel) {
    const model = document.getElementById('model-sel').value;
    const orig = btn.textContent;
    reviewBtn.disabled = true; reviseBtn.disabled = true;
    btn.textContent = busyLabel;
    document.getElementById('action-msg').hidden = true;
    try {
      await postJson(`/staging/${REC.id}/review`, { model, action });
      location.reload();
    } catch (e) {
      btn.textContent = orig;
      reviewBtn.disabled = false;
      reviseBtn.disabled = REC.comments.filter(c => !c.resolved).length === 0;
      showActionError((action === 'revise' ? 'Revise' : 'Review') + ' failed: ' + e.message);
    }
  }
  reviewBtn.addEventListener('click', () => run('review', reviewBtn, 'Reviewing…'));
  reviseBtn.addEventListener('click', () => run('revise', reviseBtn, 'Revising…'));
}

// ── Tabs + init ───────────────────────────────────────────────────────────────
function initDetail(defaultTab) {
  const rendered = document.getElementById('view-rendered');
  const raw      = document.getElementById('view-raw');
  rendered.innerHTML = renderMd(REC.content);
  raw.textContent    = REC.content;
  applyHighlights(rendered);
  applyHighlights(raw);

  const panes = { rendered, raw, diff: document.getElementById('view-diff') };
  function show(view) {
    document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', b.dataset.view === view));
    Object.entries(panes).forEach(([k, el]) => el.classList.toggle('active', k === view));
  }
  document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => show(b.dataset.view)));
  show(defaultTab);

  renderComments();
  if (PENDING) initSelectionUI();
  initReviewActions();
}
"""
