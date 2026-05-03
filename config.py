import os
from pathlib import Path

ANTHROPIC_URL        = "https://api.anthropic.com/v1/messages"
OLLAMA_URL           = "http://host.docker.internal:11434/v1/chat/completions"
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
DEFAULT_MODEL        = os.environ.get("DEFAULT_MODEL", "claude-sonnet-4-5")
OLLAMA_DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
FS_ROOT              = Path(os.environ.get("FS_ROOT", "/home/matt")).resolve()
DATA_DIR             = Path(os.environ.get("DATA_DIR", "/app/data"))
_MAX_BODY_BYTES      = int(os.environ.get("MAX_BODY_BYTES", "1000000"))
_MAX_TOKENS_CAP      = int(os.environ.get("MAX_TOKENS_CAP", "8192"))
_ALLOWED_CLAUDE_MODELS = {
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5", "claude-sonnet-4-6",
    "claude-opus-4-5",   "claude-opus-4-7",
}
ENTRIES_PER_FILE  = 200
CLAUDE_CREDS_PATH = Path("/home/matt/.claude/.credentials.json")
ROUTING_LOG       = Path("logs") / "routing.ndjson"

MEM_DIR  = DATA_DIR / "memories"
CONV_DIR = DATA_DIR / "convos"
