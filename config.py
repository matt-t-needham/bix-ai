import os
from pathlib import Path

ANTHROPIC_URL        = "https://api.anthropic.com/v1/messages"
# Bare Ollama host (no path). All Ollama call sites derive their URL from this
# single knob so dev-outside-Docker only needs to override one env var.
OLLAMA_HOST          = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_URL           = f"{OLLAMA_HOST}/v1/chat/completions"
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
# Defence-in-depth for the /v1/messages proxy. Clients must send this value as
# `x-api-key`; the server then substitutes ANTHROPIC_API_KEY before forwarding.
# Empty string disables the check (only safe if Cloudflare Access fully gates the route).
BIX_PROXY_SECRET     = os.environ.get("BIX_PROXY_SECRET", "")
DEFAULT_MODEL        = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
OLLAMA_TOOL_MODEL    = os.environ.get("OLLAMA_TOOL_MODEL", "qwen3.5:9b")
SUMMARY_LOCAL_MODEL  = os.environ.get("SUMMARY_LOCAL_MODEL", "gemma4:e2b")
FS_ROOT              = Path(os.environ.get("FS_ROOT", "/home/matt")).resolve()
DATA_DIR             = Path(os.environ.get("DATA_DIR", "/app/data"))
_MAX_BODY_BYTES      = int(os.environ.get("MAX_BODY_BYTES", "1000000"))
_MAX_TOKENS_CAP      = int(os.environ.get("MAX_TOKENS_CAP", "32000"))
# Governor for the agentic tool loop (streaming/claude.py, streaming/ollama.py).
# Iteration cap (10 turns) is hardcoded at each loop's `range(10)`; these two are
# the remaining budgets — generous defaults so they only bite runaway loops.
LOOP_MAX_TOKENS      = int(os.environ.get("LOOP_MAX_TOKENS", "1000000"))
LOOP_MAX_SECONDS     = float(os.environ.get("LOOP_MAX_SECONDS", "1200"))
LOOP_MAX_TURNS       = int(os.environ.get("LOOP_MAX_TURNS", "50"))
# Local models are slow on this host (long prompt eval, minutes-long tool-JSON
# generation) — when the stream is quiet longer than this, emit a status
# heartbeat so the UI can show "still working" instead of looking hung.
PROGRESS_HEARTBEAT_SECONDS = float(os.environ.get("PROGRESS_HEARTBEAT_SECONDS", "5"))
# Conversation-tail compaction (compact.py, api path). Triggers when the
# narrative token estimate (blob-pointer excerpts excluded) exceeds the
# threshold; the most recent COMPACT_KEEP_TURNS user turns stay verbatim.
COMPACT_THRESHOLD_TOKENS = int(os.environ.get("COMPACT_THRESHOLD_TOKENS", "20000"))
COMPACT_KEEP_TURNS       = int(os.environ.get("COMPACT_KEEP_TURNS", "3"))
_ALLOWED_CLAUDE_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5", "claude-sonnet-4-6",
    "claude-opus-4-5",   "claude-opus-4-7",
}
_ALLOWED_OLLAMA_MODELS = {
    OLLAMA_DEFAULT_MODEL,
    OLLAMA_TOOL_MODEL,
    SUMMARY_LOCAL_MODEL,   # summariser used by strategy.py
    "gemma4:26b",
    "qwen3.5:9b",
}
# Comma-separated allowlist for anthropic-beta header passthrough on /v1/messages.
# Default empty: clients cannot enable beta features without explicit server config.
_ALLOWED_ANTHROPIC_BETAS = frozenset(
    s.strip() for s in os.environ.get("BIX_ALLOWED_ANTHROPIC_BETAS", "").split(",") if s.strip()
)
# Advisory per-token costs, USD per 1M tokens (input, output) — used for the
# est_cost_usd field in routing.ndjson and the UI cost surfaces. Source:
# platform.claude.com pricing as of 2026-07 (Sonnet tier $3/$15, Opus tier
# $5/$25, Haiku 4.5 $1/$5). Local Ollama models cost 0 and aren't listed.
MODEL_COSTS = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-4-5":         (3.00, 15.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-5":           (5.00, 25.00),
    "claude-opus-4-7":           (5.00, 25.00),
}

ENTRIES_PER_FILE  = 200
CLAUDE_CREDS_PATH = Path(os.environ.get("CLAUDE_CONFIG_DIR", "/home/matt/.claude")) / ".credentials.json"
ROUTING_LOG       = Path("logs") / "routing.ndjson"

MEM_DIR     = DATA_DIR / "memories"
CONV_DIR    = DATA_DIR / "convos"

# ── Build identity (GET /version) ─────────────────────────────────────────────
# Stamped by the Dockerfile via --build-arg at image build; "unknown" for dev runs.
GIT_SHA  = os.environ.get("GIT_SHA", "unknown")
BUILT_AT = os.environ.get("BUILT_AT", "unknown")
# Size cap for the content-addressed blob store (blobstore.py). Generous default —
# blobs are spilled artifacts (pasted logs/source/etc.), not primary storage.
# Not derived into a BLOB_DIR constant here — blobstore.py reads config.DATA_DIR
# at call time (mirrors FS_ROOT) so tests can monkeypatch it.
BLOB_STORE_MAX_BYTES = int(os.environ.get("BLOB_STORE_MAX_BYTES", str(500_000_000)))
