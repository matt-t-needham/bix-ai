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
_MAX_TOKENS_CAP      = int(os.environ.get("MAX_TOKENS_CAP", "8192"))
# Governor for the agentic tool loop (streaming/claude.py, streaming/ollama.py).
# Iteration cap (10 turns) is hardcoded at each loop's `range(10)`; these two are
# the remaining budgets — generous defaults so they only bite runaway loops.
LOOP_MAX_TOKENS      = int(os.environ.get("LOOP_MAX_TOKENS", "500000"))
LOOP_MAX_SECONDS     = float(os.environ.get("LOOP_MAX_SECONDS", "300"))
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
ENTRIES_PER_FILE  = 200
CLAUDE_CREDS_PATH = Path("/home/matt/.claude/.credentials.json")
ROUTING_LOG       = Path("logs") / "routing.ndjson"
AUTO_LOG          = Path("logs") / "auto.ndjson"

MEM_DIR     = DATA_DIR / "memories"
CONV_DIR    = DATA_DIR / "convos"
STAGING_DIR = DATA_DIR / "staging"
# Size cap for the content-addressed blob store (blobstore.py). Generous default —
# blobs are spilled artifacts (pasted logs/source/etc.), not primary storage.
# Not derived into a BLOB_DIR constant here — blobstore.py reads config.DATA_DIR
# at call time (mirrors FS_ROOT/STAGING_DIR) so tests can monkeypatch it.
BLOB_STORE_MAX_BYTES = int(os.environ.get("BLOB_STORE_MAX_BYTES", str(500_000_000)))

# Log-review roots (read-only). Internal logs sit under the bix-infra mount;
# Steam's logs dir is mounted read-only into the container separately.
INTERNAL_LOG_ROOT = Path(os.environ.get("INTERNAL_LOG_ROOT", "/home/matt/apps/bix-infra/logs"))
STEAM_LOG_ROOT    = Path(os.environ.get("STEAM_LOG_ROOT", "/home/matt/.steam/debian-installation/logs"))

# Project TODO files (read-only). ALL.md is the compiled all-projects view.
TODOS_DIR = Path(os.environ.get("TODOS_DIR", "/home/matt/apps/bix-infra/todos"))
