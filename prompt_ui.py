"""The custom-prompt editor page (GET/POST /prompt in main.py).

Pure rendering + the mode→base-prompt lookup; stdlib-only, no FastAPI imports,
same pattern the blob page follows against main._PAGE_CSS. Everything the page
shows is server-rendered and escaped: the prompt text is Matt's own, but a
preview also contains tool descriptions and doc paths, so it all goes through
html.escape rather than being trusted.

The page is deliberately form-driven — Save, Reset and Restore are plain POSTs
that redirect back — so the only JavaScript is the character counter.
"""
import html

import config
import identity
import prompt_store
import streaming.claude
import streaming.ollama
import streaming.openrouter
import streaming.pro

# The base (immutable) prompt per mode, read through the module at call time so
# a monkeypatched constant is honoured. Each adapter owns its own base because
# the tool lists and doc pointers differ; this is the one place that needs all
# four at once.
_BASES = {
    "local":      (streaming.ollama,     "OLLAMA_SYSTEM"),
    "api":        (streaming.claude,     "_IDENTITY_PROMPT"),
    "openrouter": (streaming.openrouter, "OPENROUTER_SYSTEM"),
    "pro":        (streaming.pro,        "_PRO_IDENTITY_PROMPT"),
}

MODE_LABELS = (
    ("local",      "Local Ollama"),
    ("api",        "Claude API"),
    ("openrouter", "OpenRouter"),
    ("pro",        "Claude CLI"),
)

DEFAULT_PREVIEW_MODE = "local"


def is_preview_mode(mode: str) -> bool:
    return mode in _BASES


def base_for_mode(mode: str) -> str:
    module, attr = _BASES[mode if mode in _BASES else DEFAULT_PREVIEW_MODE]
    return getattr(module, attr)


_DEFAULT_MODELS = {
    "local":      "OLLAMA_DEFAULT_MODEL",
    "api":        "DEFAULT_MODEL",
    "openrouter": "OPENROUTER_DEFAULT_MODEL",
    "pro":        "DEFAULT_MODEL",
}


def example_state(mode: str) -> dict:
    """A representative `This turn` block for the preview. The build facts are
    real; the model is that mode's default, and the tagging rules are shown as
    they appear in a conversation that already has answers in it (a first turn
    carries none, because there is nothing to attribute yet)."""
    st = {
        "mode":    mode,
        "model":   getattr(config, _DEFAULT_MODELS.get(mode, "DEFAULT_MODEL"), ""),
        "agent":   False,
        "tagged":  True,
        "service": identity.service_line(config.GIT_SHA, config.BUILT_AT, "…"),
    }
    if mode == "openrouter":
        st.update(spend_today=0.0, spend_cap=config.OPENROUTER_MAX_USD_PER_DAY)
    return st


def composed_for_mode(mode: str, custom: str) -> str:
    """Exactly what that mode's adapter would send as its system prompt, for a
    non-agent chat (agent mode and hand-off reviews wrap this themselves)."""
    base  = base_for_mode(mode)
    block = identity.runtime_state_block(example_state(mode))
    return identity.with_custom(f"{base}\n\n{block}" if block else base, custom)


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


EXTRA_CSS = """
form.editor { margin:0; }
textarea {
  width:100%; min-height:300px; font:inherit; font-size:.86rem; line-height:1.55;
  background:#181825; color:var(--text); border:1px solid var(--surface1);
  border-radius:6px; padding:12px 14px; resize:vertical;
}
textarea:focus { outline:none; border-color:var(--mauve); }
.counter { font-size:.74rem; color:var(--subtext); font-variant-numeric:tabular-nums; margin-top:6px; }
.counter.warn { color:var(--yellow); } .counter.crit { color:var(--red); }
.bar-wrap { background:var(--surface1); border-radius:4px; height:6px; margin:6px 0 0; }
.bar-fill { background:var(--green); height:100%; border-radius:4px; min-width:2px; transition:width .12s; }
.bar-fill.warn { background:var(--yellow); } .bar-fill.crit { background:var(--red); }
.btn-save { background:var(--mauve); } .btn-restore { background:var(--surface1); color:var(--text); }
.btn-sm { font-size:.72rem !important; padding:3px 10px !important; min-height:0 !important; }
.tabs { display:flex; gap:2px; border-bottom:1px solid var(--surface1); margin:0 0 12px; flex-wrap:wrap; }
.tab { font-size:.8rem; color:var(--subtext); padding:7px 13px; border-bottom:2px solid transparent; }
.tab:hover { color:var(--text); text-decoration:none; }
.tab.active { color:var(--blue); border-bottom-color:var(--blue); }
pre.prompt {
  background:#181825; padding:14px; border-radius:6px; overflow-x:auto;
  font-size:.78rem; line-height:1.5; white-space:pre-wrap; word-wrap:break-word;
  color:var(--subtext);
}
pre.prompt .mine { color:var(--text); }
h2 { font-size:.95rem; font-weight:600; margin:28px 0 10px; }
h2 .hint { font-weight:400; font-size:.78rem; color:var(--subtext); margin-left:8px; }
.rev { display:flex; justify-content:space-between; gap:12px; align-items:center;
       padding:9px 12px; margin:6px 0; background:var(--surface0); border-radius:6px; }
.rev .txt { min-width:0; }
.rev .preview { font-size:.78rem; color:var(--text); overflow-wrap:anywhere; }
.rev .sub { font-size:.72rem; color:var(--subtext); font-variant-numeric:tabular-nums; }
.err { color:var(--red); font-size:.84rem; background:rgba(243,139,168,.1);
       border-left:2px solid var(--red); padding:9px 13px; margin:14px 0; border-radius:0 4px 4px 0; }
.saved { color:var(--green); font-size:.84rem; padding:9px 0; }
.immutable { font-size:.78rem; color:var(--subtext); margin:0 0 10px; }
"""

_COUNTER_JS = """
const ta = document.getElementById('text');
const cap = %(cap)d, out = document.getElementById('counter'), bar = document.getElementById('bar');
function tick() {
  const n = ta.value.length, pct = Math.min(n / cap * 100, 100);
  out.textContent = n.toLocaleString() + ' of ' + cap.toLocaleString() + ' characters';
  bar.style.width = Math.max(pct, 0.5) + '%%';
  const cls = pct >= 95 ? 'crit' : pct >= 80 ? 'warn' : '';
  out.className = 'counter ' + cls; bar.className = 'bar-fill ' + cls;
}
ta.addEventListener('input', tick); tick();
"""


def _preview_html(mode: str, custom: str) -> str:
    """The composed prompt, with Matt's own half brightened so the split
    between immutable and editable is visible at a glance."""
    composed = composed_for_mode(mode, custom)
    section  = identity.custom_section(custom)
    if section and composed.endswith(section):
        base = composed[: -len(section)]
        return f"{_esc(base)}<span class=\"mine\">{_esc(section)}</span>"
    return _esc(composed)


def render(state: dict, revisions: list[dict], preview_mode: str, css: str,
           error: str = "", saved: bool = False) -> str:
    """`css` is main._PAGE_CSS — the shared page base, passed in rather than
    imported so this module keeps no dependency on the app."""
    cap   = state["max_chars"]
    text  = state["text"]
    mode  = preview_mode if preview_mode in _BASES else DEFAULT_PREVIEW_MODE

    tabs = "".join(
        f'<a class="tab{" active" if m == mode else ""}" href="/prompt?preview={m}">{_esc(label)}</a>'
        for m, label in MODE_LABELS
    )

    revs = "".join(
        f'<div class="rev"><div class="txt">'
        f'<div class="preview">{_esc(r["preview"]) or "<em>(blank)</em>"}</div>'
        f'<div class="sub">{_esc(r["saved_at"][:19].replace("T", " "))} · {r["chars"]:,} chars</div>'
        f'</div>'
        f'<form method="post" action="/prompt/restore">'
        f'<input type="hidden" name="revision_id" value="{_esc(r["id"])}">'
        f'<button class="btn-restore btn-sm" type="submit">Restore</button></form></div>'
        for r in revisions
    ) or '<div class="note">No earlier versions yet — saving keeps the last '\
         f'{prompt_store.REVISIONS_KEPT}.</div>'

    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Custom prompt</title><link rel="icon" type="image/svg+xml" href="https://bix.computer/favicon.svg"><style>{css}{EXTRA_CSS}</style></head><body>'

        '<div class="hdr"><h1>Custom prompt</h1><div class="meta">'
        f'<span>{state["chars"]:,} of {cap:,} characters</span>'
        f'<span>{"saved " + _esc(state["updated_at"][:19].replace("T", " ")) if state["updated_at"] else "never saved"}</span>'
        f'<span>{state["revisions"]} earlier version(s)</span>'
        '<span><a href="/">← chat</a></span></div></div>'

        + (f'<div class="err">{_esc(error)}</div>' if error else "")
        + ('<div class="saved">Saved. Every new chat turn uses it from now on.</div>' if saved else "")

        + '<p class="immutable">These instructions are appended to the built-in identity '
          'and ground rules, which stay in the code and can\'t be edited here. They apply '
          'to every backend — local, Claude, OpenRouter and the Claude CLI.</p>'

        + '<form class="editor" method="post" action="/prompt">'
          f'<textarea id="text" name="text" spellcheck="false" '
          f'placeholder="e.g. Prefer metric units. When I ask for code, skip the explanation unless I ask.">{_esc(text)}</textarea>'
          '<div class="bar-wrap"><div class="bar-fill" id="bar"></div></div>'
          '<div class="counter" id="counter"></div>'
          '<div class="actions"><button class="btn-save" type="submit">Save</button></div>'
          '</form>'

        + '<form method="post" action="/prompt/reset" class="actions">'
          '<button class="btn-reject" type="submit">Clear instructions</button></form>'

        + '<h2>What the model receives<span class="hint">per mode — the “This turn” lines are rebuilt on every request</span></h2>'
        + f'<div class="tabs">{tabs}</div>'
        + f'<pre class="prompt">{_preview_html(mode, text)}</pre>'

        + '<h2>Earlier versions</h2>' + revs

        + f'<script>{_COUNTER_JS % {"cap": cap}}</script>'
          '</body></html>'
    )
