"""Routing dashboard — aggregates routing.ndjson and renders GET /routing.

Pure logic (stdlib only, no FastAPI imports): `load_events` reads the ndjson
log (+ its .1 rotation), `aggregate` reduces events to the numbers the page
shows, `render_html` produces the page. main.py owns the route.

Colour semantics follow the UI standard: yellow = local model, mauve = Claude.
Identity is never colour-alone — every coloured mark carries a direct label or
legend, and each breakdown is also a table.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import MODEL_COSTS

_TS_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Catppuccin Mocha — same variables as static/index.html; do not add colours.
_CSS = """
:root {
  --bg:#1e1e2e; --surface0:#313244; --surface1:#45475a;
  --text:#cdd6f4; --subtext:#a6adc8;
  --yellow:#f9e2af; --blue:#89b4fa; --green:#a6e3a1; --red:#f38ba8; --mauve:#cba6f7;
}
*,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
body {
  font-family:system-ui,-apple-system,sans-serif; background:var(--bg);
  color:var(--text); padding:32px 24px; max-width:980px; margin:0 auto; line-height:1.5;
}
a { color:var(--blue); text-decoration:none; }
a:hover { text-decoration:underline; }
.hdr { border-bottom:1px solid var(--surface1); padding-bottom:14px; margin-bottom:18px; }
.hdr h1 { font-size:1.3rem; font-weight:600; }
.meta { font-size:.78rem; color:var(--subtext); display:flex; gap:14px; flex-wrap:wrap; }
h2 { font-size:.95rem; font-weight:600; margin:26px 0 10px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.tile { background:var(--surface0); border-radius:6px; padding:12px 14px; }
.tile .t-label { font-size:.72rem; color:var(--subtext); }
.tile .t-value { font-size:1.35rem; font-weight:600; margin-top:2px; }
.tile .t-sub { font-size:.72rem; color:var(--subtext); margin-top:2px; }
.split { display:flex; height:22px; border-radius:4px; overflow:hidden; gap:2px;
         background:var(--bg); margin:8px 0 6px; }
.split .seg-local  { background:var(--yellow); }
.split .seg-claude { background:var(--mauve); }
.legend { display:flex; gap:16px; font-size:.75rem; color:var(--subtext); }
.legend .sw { display:inline-block; width:10px; height:10px; border-radius:2px;
              margin-right:5px; vertical-align:-1px; }
.chart { display:flex; align-items:flex-end; gap:6px; height:150px;
         border-bottom:1px solid var(--surface1); padding:0 2px; margin-top:14px; }
.col { flex:1; max-width:24px; display:flex; flex-direction:column;
       justify-content:flex-end; height:100%; position:relative; }
.col .fill { background:var(--mauve); border-radius:4px 4px 0 0; min-height:2px; }
.col:hover .fill { background:var(--blue); }
.col .v { position:absolute; top:-18px; left:50%; transform:translateX(-50%);
          font-size:.68rem; color:var(--subtext); white-space:nowrap;
          font-variant-numeric:tabular-nums; }
.xlabels { display:flex; gap:6px; padding:4px 2px 0; }
.xlabels span { flex:1; max-width:24px; font-size:.65rem; color:var(--subtext);
                text-align:center; overflow:visible; white-space:nowrap; }
table { width:100%; border-collapse:collapse; font-size:.82rem; margin-top:6px; }
th { text-align:left; font-size:.7rem; text-transform:uppercase; letter-spacing:.05em;
     color:var(--subtext); font-weight:500; padding:6px 10px; border-bottom:1px solid var(--surface1); }
td { padding:6px 10px; border-bottom:1px solid var(--surface0); }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
.rowbar { display:inline-block; height:8px; border-radius:0 4px 4px 0;
          background:var(--mauve); vertical-align:middle; min-width:2px; }
.rowbar.local { background:var(--yellow); }
.tag { background:var(--surface1); color:var(--subtext); padding:1px 7px;
       border-radius:3px; font-size:.7rem; }
.note { color:var(--subtext); font-size:.85rem; font-style:italic; padding:14px 0; }
"""


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;"))


def load_events(log_path: Path) -> list[dict]:
    """Parse routing.ndjson plus its .ndjson.1 rotation (oldest first).
    Unparseable lines are skipped — the log is best-effort by design."""
    events: list[dict] = []
    for p in (log_path.with_suffix(".ndjson.1"), log_path):
        if not p.exists():
            continue
        try:
            lines = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if isinstance(ev, dict):
                events.append(ev)
    return events


def _parse_ts(ev: dict) -> datetime | None:
    try:
        return datetime.strptime(ev.get("ts", ""), _TS_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _is_claude(ev: dict) -> bool:
    return str(ev.get("model", "")).startswith("claude")


def _cost_of(ev: dict) -> float:
    """Advisory USD cost of one event. Events written before the
    est_cost_usd field existed are backfilled from tokens × MODEL_COSTS —
    the same maths the live pipeline uses."""
    if ev.get("est_cost_usd") is not None:
        return float(ev.get("est_cost_usd") or 0.0)
    rate = MODEL_COSTS.get(str(ev.get("model", "")))
    if not rate:
        return 0.0
    in_t = int(ev.get("input_tokens") or 0)
    out_t = int(ev.get("output_tokens") or 0)
    return (in_t * rate[0] + out_t * rate[1]) / 1e6


def _rule_of(ev: dict) -> str:
    """Leading rule token of the reason field ('code-gen: …' → 'code-gen')."""
    reason = str(ev.get("reason", "")) or "(none)"
    return reason.split(":", 1)[0].strip() or "(none)"


def aggregate(events: list[dict], now: datetime | None = None,
              window_days: int = 30, chart_days: int = 14) -> dict:
    """Reduce routing events to the dashboard numbers. Pure; injectable clock."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=window_days)

    def _bucket():
        return {"requests": 0, "local": 0, "claude": 0, "cost": 0.0,
                "input_tokens": 0, "output_tokens": 0}

    total = _bucket()
    last7 = _bucket()
    today_cost = 0.0
    days: dict[str, dict] = {}
    for i in range(chart_days):
        d = (now - timedelta(days=chart_days - 1 - i)).strftime("%Y-%m-%d")
        days[d] = {"date": d, "cost": 0.0, "requests": 0, "local": 0, "claude": 0}
    by_model: dict[str, dict] = {}
    by_mode: dict[str, dict] = {}
    auto_rules: dict[str, dict] = {}
    cutoff7 = now - timedelta(days=7)
    today = now.strftime("%Y-%m-%d")

    for ev in events:
        ts = _parse_ts(ev)
        if ts is None or ts < cutoff or ts > now:
            continue
        claude = _is_claude(ev)
        cost = _cost_of(ev)
        in_t = int(ev.get("input_tokens") or 0)
        out_t = int(ev.get("output_tokens") or 0)
        model = str(ev.get("model", "")) or "(unknown)"
        mode = str(ev.get("mode", "")) or "(unknown)"

        for bucket in ([total, last7] if ts >= cutoff7 else [total]):
            bucket["requests"] += 1
            bucket["claude" if claude else "local"] += 1
            bucket["cost"] += cost
            bucket["input_tokens"] += in_t
            bucket["output_tokens"] += out_t

        day = ts.strftime("%Y-%m-%d")
        if day == today:
            today_cost += cost
        if day in days:
            days[day]["cost"] += cost
            days[day]["requests"] += 1
            days[day]["claude" if claude else "local"] += 1

        m = by_model.setdefault(model, {"model": model, "requests": 0, "cost": 0.0,
                                        "input_tokens": 0, "output_tokens": 0,
                                        "local": not claude})
        m["requests"] += 1
        m["cost"] += cost
        m["input_tokens"] += in_t
        m["output_tokens"] += out_t

        md = by_mode.setdefault(mode, {"mode": mode, "requests": 0, "cost": 0.0})
        md["requests"] += 1
        md["cost"] += cost

        if mode in ("auto", "auto_fallback"):
            rule = _rule_of(ev)
            r = auto_rules.setdefault(rule, {"rule": rule, "count": 0,
                                             "local": 0, "claude": 0})
            r["count"] += 1
            r["claude" if claude else "local"] += 1

    return {
        "window_days": window_days,
        "total": total,
        "last7": last7,
        "today_cost": round(today_cost, 4),
        "days": list(days.values()),
        "by_model": sorted(by_model.values(), key=lambda m: (-m["cost"], -m["requests"])),
        "by_mode": sorted(by_mode.values(), key=lambda m: -m["requests"]),
        "auto_rules": sorted(auto_rules.values(), key=lambda r: -r["count"]),
    }


def _fmt_usd(x: float) -> str:
    if x >= 1:
        return f"${x:,.2f}"
    if x >= 0.01:
        return f"${x:.3f}"
    if x > 0:
        return f"${x:.4f}"
    return "$0"


def _fmt_int(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 10_000:
        return f"{n / 1e3:.0f}K"
    return f"{n:,}"


def _split_bar(bucket: dict, label: str) -> str:
    total = bucket["local"] + bucket["claude"]
    if total == 0:
        return f'<div class="note">No {_esc(label)} traffic.</div>'
    lp = bucket["local"] / total * 100
    cp = 100 - lp
    return (
        f'<div class="split">'
        f'<div class="seg-local" style="width:{lp:.1f}%" title="local · {bucket["local"]} requests"></div>'
        f'<div class="seg-claude" style="width:{cp:.1f}%" title="Claude · {bucket["claude"]} requests"></div>'
        f'</div>'
        f'<div class="legend">'
        f'<span><span class="sw" style="background:var(--yellow)"></span>'
        f'local {bucket["local"]} ({lp:.0f}%)</span>'
        f'<span><span class="sw" style="background:var(--mauve)"></span>'
        f'Claude {bucket["claude"]} ({cp:.0f}%)</span>'
        f'<span>{_esc(label)}</span>'
        f'</div>'
    )


def _spend_chart(days: list[dict]) -> str:
    max_cost = max((d["cost"] for d in days), default=0.0)
    if max_cost <= 0:
        return '<div class="note">No estimated spend in this window.</div>'
    peak_i = max(range(len(days)), key=lambda i: days[i]["cost"])
    cols, labels = [], []
    for i, d in enumerate(days):
        h = max(d["cost"] / max_cost * 100, 1.5) if d["cost"] > 0 else 0
        # Selective direct labels: the peak and the latest day only.
        show = i == peak_i or i == len(days) - 1
        v = f'<span class="v">{_esc(_fmt_usd(d["cost"]))}</span>' if show and d["cost"] > 0 else ""
        tip = f'{d["date"]} · {_fmt_usd(d["cost"])} · {d["requests"]} req'
        fill = f'<div class="fill" style="height:{h:.1f}%"></div>' if d["cost"] > 0 else ""
        cols.append(f'<div class="col" title="{_esc(tip)}">{v}{fill}</div>')
        labels.append(f'<span>{d["date"][8:]}</span>' if i % 2 == len(days) % 2 else "<span></span>")
    return (f'<div class="chart">{"".join(cols)}</div>'
            f'<div class="xlabels">{"".join(labels)}</div>')


def _model_rows(by_model: list[dict]) -> str:
    if not by_model:
        return '<tr><td colspan="6" class="note">No traffic.</td></tr>'
    max_req = max(m["requests"] for m in by_model)
    rows = []
    for m in by_model:
        w = m["requests"] / max_req * 120
        cls = "rowbar local" if m["local"] else "rowbar"
        rows.append(
            f'<tr><td>{_esc(m["model"])} '
            f'{"<span class=tag>local</span>" if m["local"] else ""}</td>'
            f'<td><span class="{cls}" style="width:{w:.0f}px"></span></td>'
            f'<td class="num">{m["requests"]:,}</td>'
            f'<td class="num">{_esc(_fmt_int(m["input_tokens"]))}</td>'
            f'<td class="num">{_esc(_fmt_int(m["output_tokens"]))}</td>'
            f'<td class="num">{_esc(_fmt_usd(m["cost"]))}</td></tr>'
        )
    return "".join(rows)


def _mode_rows(by_mode: list[dict]) -> str:
    return "".join(
        f'<tr><td>{_esc(m["mode"])}</td>'
        f'<td class="num">{m["requests"]:,}</td>'
        f'<td class="num">{_esc(_fmt_usd(m["cost"]))}</td></tr>'
        for m in by_mode
    ) or '<tr><td colspan="3" class="note">No traffic.</td></tr>'


def _rule_rows(auto_rules: list[dict]) -> str:
    return "".join(
        f'<tr><td>{_esc(r["rule"])}</td>'
        f'<td class="num">{r["count"]:,}</td>'
        f'<td class="num">{r["local"]:,}</td>'
        f'<td class="num">{r["claude"]:,}</td></tr>'
        for r in auto_rules
    ) or '<tr><td colspan="4" class="note">No auto-mode traffic.</td></tr>'


def render_html(agg: dict) -> str:
    total, last7 = agg["total"], agg["last7"]
    local_pct = (total["local"] / total["requests"] * 100) if total["requests"] else 0
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Routing dashboard</title><style>{_CSS}</style></head><body>
<div class="hdr"><h1>Routing dashboard</h1>
<div class="meta"><span>last {agg["window_days"]} days of routing.ndjson</span>
<span>est. costs are advisory (config.MODEL_COSTS)</span>
<span><a href="/">← chat</a></span></div></div>

<div class="tiles">
  <div class="tile"><div class="t-label">Requests ({agg["window_days"]}d)</div>
    <div class="t-value">{total["requests"]:,}</div>
    <div class="t-sub">{last7["requests"]:,} in last 7d</div></div>
  <div class="tile"><div class="t-label">Handled locally</div>
    <div class="t-value">{local_pct:.0f}%</div>
    <div class="t-sub">{total["local"]:,} of {total["requests"]:,} requests</div></div>
  <div class="tile"><div class="t-label">Est. spend ({agg["window_days"]}d)</div>
    <div class="t-value">{_esc(_fmt_usd(total["cost"]))}</div>
    <div class="t-sub">{_esc(_fmt_usd(last7["cost"]))} in last 7d</div></div>
  <div class="tile"><div class="t-label">Est. spend today</div>
    <div class="t-value">{_esc(_fmt_usd(agg["today_cost"]))}</div>
    <div class="t-sub">{_esc(_fmt_int(total["input_tokens"]))} in · {_esc(_fmt_int(total["output_tokens"]))} out tokens</div></div>
</div>

<h2>Local vs Claude</h2>
{_split_bar(total, f"last {agg['window_days']} days")}
{_split_bar(last7, "last 7 days")}

<h2>Estimated daily spend</h2>
{_spend_chart(agg["days"])}

<h2>By model</h2>
<table><thead><tr><th>Model</th><th></th><th class="num">Requests</th>
<th class="num">Tokens in</th><th class="num">Tokens out</th><th class="num">Est. cost</th></tr></thead>
<tbody>{_model_rows(agg["by_model"])}</tbody></table>

<h2>By mode</h2>
<table><thead><tr><th>Mode</th><th class="num">Requests</th><th class="num">Est. cost</th></tr></thead>
<tbody>{_mode_rows(agg["by_mode"])}</tbody></table>

<h2>Auto-mode routing rules</h2>
<table><thead><tr><th>Rule</th><th class="num">Requests</th>
<th class="num">→ local</th><th class="num">→ Claude</th></tr></thead>
<tbody>{_rule_rows(agg["auto_rules"])}</tbody></table>
</body></html>"""
