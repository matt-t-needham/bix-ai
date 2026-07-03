"""Tests for routing_dash — routing.ndjson aggregation + dashboard render."""
import json
from datetime import datetime, timedelta, timezone

import routing_dash

NOW = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)


def ev(days_ago=0, mode="auto", model="claude-sonnet-4-6", reason="",
       cost=0.0, in_t=100, out_t=50):
    ts = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"ts": ts, "mode": mode, "model": model, "reason": reason,
            "est_cost_usd": cost, "input_tokens": in_t, "output_tokens": out_t,
            "ttft_ms": 100, "elapsed_ms": 2000, "summarised": 0, "preprocess_ms": 0}


def test_local_vs_claude_split_by_model_prefix():
    events = [
        ev(model="gemma4:26b", mode="auto", reason="short-chat: short conversational query"),
        ev(model="claude-sonnet-4-6", mode="auto", reason="code-gen: intent", cost=0.01),
        ev(model="claude-haiku-4-5-20251001", mode="auto", reason="long-request: big · downshift", cost=0.001),
    ]
    agg = routing_dash.aggregate(events, now=NOW)
    assert agg["total"]["requests"] == 3
    assert agg["total"]["local"] == 1
    assert agg["total"]["claude"] == 2
    assert round(agg["total"]["cost"], 4) == 0.011


def test_window_excludes_old_events():
    events = [ev(days_ago=0, cost=1.0), ev(days_ago=40, cost=99.0)]
    agg = routing_dash.aggregate(events, now=NOW, window_days=30)
    assert agg["total"]["requests"] == 1
    assert agg["total"]["cost"] == 1.0


def test_last7_and_today_buckets():
    events = [ev(days_ago=0, cost=0.5), ev(days_ago=3, cost=0.25), ev(days_ago=20, cost=0.1)]
    agg = routing_dash.aggregate(events, now=NOW)
    assert agg["total"]["requests"] == 3
    assert agg["last7"]["requests"] == 2
    assert agg["today_cost"] == 0.5


def test_daily_chart_covers_chart_days_and_sums():
    events = [ev(days_ago=1, cost=0.2), ev(days_ago=1, cost=0.3), ev(days_ago=2, cost=0.1)]
    agg = routing_dash.aggregate(events, now=NOW, chart_days=14)
    days = agg["days"]
    assert len(days) == 14
    assert days[-1]["date"] == "2026-07-03"
    by_date = {d["date"]: d for d in days}
    assert round(by_date["2026-07-02"]["cost"], 4) == 0.5
    assert by_date["2026-07-02"]["requests"] == 2
    assert round(by_date["2026-07-01"]["cost"], 4) == 0.1


def test_by_model_sorted_by_cost_and_flags_local():
    events = [
        ev(model="gemma4:26b"),
        ev(model="claude-sonnet-4-6", cost=0.5),
        ev(model="claude-haiku-4-5-20251001", cost=0.01),
    ]
    agg = routing_dash.aggregate(events, now=NOW)
    models = [m["model"] for m in agg["by_model"]]
    assert models[0] == "claude-sonnet-4-6"
    local_flags = {m["model"]: m["local"] for m in agg["by_model"]}
    assert local_flags["gemma4:26b"] is True
    assert local_flags["claude-sonnet-4-6"] is False


def test_auto_rules_counts_only_auto_modes():
    events = [
        ev(mode="auto", model="gemma4:26b", reason="short-chat: short conversational query"),
        ev(mode="auto", model="claude-sonnet-4-6", reason="code-gen: intent"),
        ev(mode="auto", model="claude-sonnet-4-6", reason="code-gen: intent"),
        ev(mode="pro", reason="forced:pro"),
    ]
    agg = routing_dash.aggregate(events, now=NOW)
    rules = {r["rule"]: r for r in agg["auto_rules"]}
    assert set(rules) == {"short-chat", "code-gen"}
    assert rules["code-gen"]["count"] == 2
    assert rules["code-gen"]["claude"] == 2
    assert rules["short-chat"]["local"] == 1


def test_load_events_reads_rotation_then_current(tmp_path):
    log = tmp_path / "routing.ndjson"
    rot = tmp_path / "routing.ndjson.1"
    rot.write_text(json.dumps(ev(days_ago=2)) + "\n")
    log.write_text(json.dumps(ev(days_ago=0)) + "\nnot json\n")
    events = routing_dash.load_events(log)
    assert len(events) == 2  # bad line skipped, rotation included first


def test_load_events_missing_file(tmp_path):
    assert routing_dash.load_events(tmp_path / "routing.ndjson") == []


def test_render_html_smoke():
    events = [
        ev(model="gemma4:26b", mode="auto", reason="short-chat: short conversational query"),
        ev(model="claude-sonnet-4-6", mode="auto", reason="code-gen: intent", cost=0.0123),
    ]
    html = routing_dash.render_html(routing_dash.aggregate(events, now=NOW))
    assert "Routing dashboard" in html
    assert "claude-sonnet-4-6" in html
    assert "gemma4:26b" in html
    assert "code-gen" in html
    # escaping: no raw model/reason injection vector
    evil = [ev(model='claude-x"<script>alert(1)</script>', cost=0.1)]
    html2 = routing_dash.render_html(routing_dash.aggregate(evil, now=NOW))
    assert "<script>alert(1)</script>" not in html2


def test_render_html_empty_log():
    html = routing_dash.render_html(routing_dash.aggregate([], now=NOW))
    assert "Routing dashboard" in html
    assert "No estimated spend" in html


def test_cost_backfilled_for_pre_field_events():
    # Events written before est_cost_usd existed: derive from tokens × MODEL_COSTS.
    legacy = {"ts": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": "pro",
              "model": "claude-sonnet-4-6", "input_tokens": 1_000_000,
              "output_tokens": 0}
    agg = routing_dash.aggregate([legacy], now=NOW)
    assert agg["total"]["cost"] == 3.0  # $3/M input for sonnet tier


def test_cost_not_backfilled_for_local_or_zero_field():
    legacy_local = {"ts": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"), "mode": "auto",
                    "model": "gemma4:26b", "input_tokens": 500, "output_tokens": 500}
    explicit_zero = ev(model="claude-sonnet-4-6", cost=0.0)
    agg = routing_dash.aggregate([legacy_local, explicit_zero], now=NOW)
    assert agg["total"]["cost"] == 0.0
