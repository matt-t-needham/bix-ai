"""OpenRouter support: model catalogue, model-id validation, per-model cost
estimates, spend tracking against the configured caps, and the key-usage
probe behind the UI's budget pill.

Mirrors model_admin.py: async + httpx, config read at call time so tests can
monkeypatch, a short-TTL cache in front of the live catalogue, and a static
fallback when the catalogue fetch fails. Follows bix-graph's OpenRouter
client (ai_graph_mode/lib/ai-providers.ts) where the two should feel the
same — one model id per request, free-text ids behind a shape check, the
catalogue filtered of audio-output models and meta-routers — but with our
own HTTP-Referer / X-OpenRouter-Title so spend is attributed to bix-ai.
"""
import json
import logging
import re
import time
from datetime import datetime, timezone

import httpx

import config

log = logging.getLogger("router")

_CATALOGUE_TTL_S = 300.0
_cache: dict = {"ts": 0.0, "models": None}

# vendor/model[:variant] — the slug shape OpenRouter uses. Anchored, no
# whitespace, at most one '/', an optional ':free'-style suffix.
_MODEL_ID_RE = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}/[a-z0-9][a-z0-9._-]{0,127}(?::[a-z0-9][a-z0-9._-]{0,31})?$"
)

# Used when the live catalogue is unreachable. Prices are USD per 1M tokens
# (prompt, completion) and are advisory — the live listing is the source.
STATIC_MODELS: list[dict] = [
    {"id": "qwen/qwen3-coder",                 "name": "Qwen3 Coder",          "prompt_usd_per_m": 0.30, "completion_usd_per_m": 1.20},
    {"id": "deepseek/deepseek-chat-v3-0324",   "name": "DeepSeek V3",          "prompt_usd_per_m": 0.30, "completion_usd_per_m": 0.88},
    {"id": "meta-llama/llama-4-maverick",      "name": "Llama 4 Maverick",     "prompt_usd_per_m": 0.15, "completion_usd_per_m": 0.60},
    {"id": "google/gemini-2.5-flash",          "name": "Gemini 2.5 Flash",     "prompt_usd_per_m": 0.30, "completion_usd_per_m": 2.50},
    {"id": "mistralai/mistral-small-3.2-24b-instruct", "name": "Mistral Small 3.2", "prompt_usd_per_m": 0.075, "completion_usd_per_m": 0.20},
]


def headers() -> dict:
    return {
        "Authorization":      f"Bearer {config.OPENROUTER_API_KEY}",
        "HTTP-Referer":       config.OPENROUTER_REFERER,
        "X-OpenRouter-Title": config.OPENROUTER_TITLE,
        "Content-Type":       "application/json",
    }


def configured() -> bool:
    return bool(config.OPENROUTER_API_KEY)


def is_valid_model_id(model: str) -> bool:
    """Shape check only — any well-formed slug is sent as typed (it's my own
    key, as in bix-graph's BYOK path). Meta-routers are refused because they
    make the cost estimate meaningless."""
    m = (model or "").strip()
    if not _MODEL_ID_RE.fullmatch(m):
        return False
    return not m.startswith("openrouter/")


def _per_m(raw) -> float:
    """OpenRouter lists prices as USD-per-token strings; convert to per 1M."""
    try:
        return round(float(raw) * 1_000_000, 4)
    except (TypeError, ValueError):
        return 0.0


def _normalise(entry: dict) -> dict | None:
    mid = entry.get("id", "")
    if not is_valid_model_id(mid):
        return None
    arch = entry.get("architecture") or {}
    outputs = arch.get("output_modalities") or []
    if "audio" in outputs:
        return None
    pricing = entry.get("pricing") or {}
    reasoning = entry.get("reasoning") or {}
    return {
        "id":                   mid,
        "name":                 entry.get("name") or mid,
        "vendor":               mid.split("/", 1)[0],
        "context_length":       entry.get("context_length") or 0,
        "prompt_usd_per_m":     _per_m(pricing.get("prompt")),
        "completion_usd_per_m": _per_m(pricing.get("completion")),
        "reasoning_mandatory":  bool(reasoning.get("mandatory")),
        "supports_tools":       "tools" in (entry.get("supported_parameters") or []),
    }


async def _fetch_catalogue() -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{config.OPENROUTER_HOST}/api/v1/models", headers=headers())
        r.raise_for_status()
        data = r.json().get("data") or []
    out = [m for m in (_normalise(e) for e in data if isinstance(e, dict)) if m]
    out.sort(key=lambda m: (m["vendor"], m["id"]))
    return out


def invalidate_cache() -> None:
    _cache["models"] = None
    _cache["ts"] = 0.0


async def list_models(force: bool = False) -> tuple[list[dict], str]:
    """(models, source) — source is "live" or "static". Cached 5 min. Never
    raises: a catalogue outage falls back to STATIC_MODELS and logs."""
    if (not force and _cache["models"] is not None
            and time.monotonic() - _cache["ts"] < _CATALOGUE_TTL_S):
        return _cache["models"], "live"
    try:
        models = await _fetch_catalogue()
    except (httpx.HTTPError, OSError, ValueError) as e:
        log.warning("openrouter catalogue unavailable (%s) — using static list", e)
        return list(STATIC_MODELS), "static"
    _cache["models"] = models
    _cache["ts"] = time.monotonic()
    return models, "live"


async def model_info(model: str) -> dict | None:
    models, _ = await list_models()
    for m in models:
        if m["id"] == model:
            return m
    return None


def estimate_cost_usd(info: dict | None, input_tokens: int, output_tokens: int) -> float:
    if not info:
        return 0.0
    return round((input_tokens * info.get("prompt_usd_per_m", 0.0)
                  + output_tokens * info.get("completion_usd_per_m", 0.0)) / 1e6, 6)


# ── Spend tracking ────────────────────────────────────────────────────────────

def spent_today_usd(log_path=None) -> float:
    """Sum of est_cost_usd for mode=openrouter records dated today (UTC) in
    routing.ndjson. Synchronous file read — callers wrap in to_thread."""
    path = log_path or config.ROUTING_LOG
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = 0.0
    try:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if rec.get("mode") == "openrouter" and str(rec.get("ts", "")).startswith(today):
                    total += float(rec.get("est_cost_usd") or 0.0)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("routing log unreadable for spend total: %s", e)
    return round(total, 6)


def check_caps(projected_request_usd: float, spent_today: float) -> str | None:
    """Return a refusal message when a call would breach a cap, else None."""
    per_req = config.OPENROUTER_MAX_USD_PER_REQUEST
    per_day = config.OPENROUTER_MAX_USD_PER_DAY
    if per_req > 0 and projected_request_usd > per_req:
        return (f"Refused: this request is estimated at ${projected_request_usd:.4f}, over the "
                f"per-request cap of ${per_req:.2f} (OPENROUTER_MAX_USD_PER_REQUEST). "
                "Start a new conversation or pick a cheaper model.")
    if per_day > 0 and spent_today + projected_request_usd > per_day:
        return (f"Refused: ${spent_today:.4f} already spent on OpenRouter today; this request "
                f"(~${projected_request_usd:.4f}) would pass the daily cap of ${per_day:.2f} "
                "(OPENROUTER_MAX_USD_PER_DAY).")
    return None


async def key_usage() -> dict | None:
    """OpenRouter's view of this key: {usage, limit, limit_remaining} in USD.
    None when unreachable or not configured."""
    if not configured():
        return None
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{config.OPENROUTER_HOST}/api/v1/auth/key", headers=headers())
            r.raise_for_status()
            d = r.json().get("data") or {}
    except (httpx.HTTPError, OSError, ValueError) as e:
        log.warning("openrouter key usage unavailable: %s", e)
        return None
    return {
        "usage":           d.get("usage"),
        "limit":           d.get("limit"),
        "limit_remaining": d.get("limit_remaining"),
        "is_free_tier":    d.get("is_free_tier"),
    }


async def chat_text(model: str, messages: list, timeout: float = 120.0) -> str:
    """One non-streaming completion; used for context summaries so the
    summary runs on the provider that's answering, never a background local
    model."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(config.OPENROUTER_URL, headers=headers(), json={
            "model": model, "messages": messages, "stream": False,
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
