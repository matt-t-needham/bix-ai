"""Ollama model management: list installed models, pull new ones (validated
against the registry first so typos fail fast), delete, and back the /chat
local-model allowlist with the live installed set.

Async + httpx like the rest of the service. config is read at call time
(OLLAMA_HOST, _ALLOWED_OLLAMA_MODELS) so tests can monkeypatch. The pull
stream is translated NDJSON→SSE by pull_events(); main.py owns the routes.
"""
import json
import logging
import re
import time

import httpx

import config
from helpers import sse

log = logging.getLogger("router")

REGISTRY_URL = "https://registry.ollama.ai"

# name[:tag] with an optional single repo segment (user/name). Anchored, no
# whitespace, no URL schemes, no path tricks — the parts feed straight into a
# registry URL path. Tags allow uppercase (e.g. q4_K_M); names are lowercase.
_NAME_RE = re.compile(
    r"^[a-z0-9][a-z0-9._-]{0,63}"
    r"(?:/[a-z0-9][a-z0-9._-]{0,63})?"
    r"(?::[A-Za-z0-9][A-Za-z0-9._-]{0,127})?$"
)

_CACHE_TTL_S = 30.0
_cache: dict = {"ts": 0.0, "models": None}


def validate_model_name(name: str) -> tuple[str, str]:
    """Return (repo, tag) for a free-text model name, or raise ValueError.

    'gemma4:26b' -> ('library/gemma4', '26b'); 'user/model' -> ('user/model',
    'latest'). The result is safe to interpolate into a registry URL path.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("model name is empty")
    if not _NAME_RE.fullmatch(name):
        raise ValueError(
            f"invalid model name: {name!r} — expected name[:tag] like 'qwen3.5:9b' "
            "(see ollama.com/library)"
        )
    base, _, tag = name.partition(":")
    repo = base if "/" in base else f"library/{base}"
    return repo, (tag or "latest")


async def registry_manifest_exists(repo: str, tag: str) -> bool:
    """True/False from the Ollama registry manifest endpoint; raises on other
    statuses or network failure (caller distinguishes 'typo' from 'registry down')."""
    url = f"{REGISTRY_URL}/v2/{repo}/manifests/{tag}"
    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        r = await client.get(url, headers={
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"})
    if r.status_code == 200:
        return True
    if r.status_code == 404:
        return False
    raise RuntimeError(f"registry returned HTTP {r.status_code} for {repo}:{tag}")


def invalidate_cache() -> None:
    _cache["models"] = None
    _cache["ts"] = 0.0


async def _fetch_installed() -> list[dict]:
    """Uncached /api/tags + /api/ps merge. Raises when Ollama is unreachable."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        tags = (await client.get(f"{config.OLLAMA_HOST}/api/tags")).json()
        try:
            ps = (await client.get(f"{config.OLLAMA_HOST}/api/ps")).json()
        except (httpx.HTTPError, ValueError):
            ps = {}   # loaded-state is best-effort; the list itself is not
    loaded = {m.get("name"): m for m in ps.get("models") or []}
    out = []
    for m in tags.get("models") or []:
        name = m.get("name", "")
        out.append({
            "name":        name,
            "size":        m.get("size", 0),
            "modified_at": m.get("modified_at", ""),
            "loaded":      name in loaded,
            "size_vram":   loaded.get(name, {}).get("size_vram", 0),
        })
    return out


async def list_installed(force: bool = False) -> list[dict]:
    """Installed models with loaded-state, cached for 30 s."""
    if (not force and _cache["models"] is not None
            and time.monotonic() - _cache["ts"] < _CACHE_TTL_S):
        return _cache["models"]
    models = await _fetch_installed()
    _cache["models"] = models
    _cache["ts"] = time.monotonic()
    return models


async def is_allowed_local_model(model: str) -> bool:
    """mode=local allowlist: any installed model. Falls back to the static
    config list when Ollama is unreachable (never blocks chat on an outage)."""
    try:
        models = await list_installed()
    except (httpx.HTTPError, OSError, ValueError) as e:
        log.warning("ollama unreachable for model allowlist (%s) — using static fallback", e)
        return model in config._ALLOWED_OLLAMA_MODELS
    names = {m["name"] for m in models}
    return model in names or f"{model}:latest" in names


async def pull_events(name: str):
    """Translate Ollama's NDJSON /api/pull stream into SSE events:
    pull_progress {status, digest?, total?, completed?} … pull_done {name}.
    Long quiet phases are the caller's problem — main.py wraps this in
    with_keepalive. A client disconnect is safe: Ollama pulls are resumable."""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{config.OLLAMA_HOST}/api/pull",
                                     json={"model": name, "stream": True}) as r:
                if r.status_code != 200:
                    yield sse("error", {"message": f"Ollama returned HTTP {r.status_code}"})
                    return
                async for line in r.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue
                    if data.get("error"):
                        yield sse("error", {"message": str(data["error"])})
                        return
                    payload = {"status": str(data.get("status", ""))}
                    for k in ("digest", "total", "completed"):
                        if k in data:
                            payload[k] = data[k]
                    yield sse("pull_progress", payload)
                    if payload["status"] == "success":
                        break
    except (httpx.HTTPError, OSError) as e:
        yield sse("error", {"message": f"pull failed: {e.__class__.__name__}: {e}"})
        return
    invalidate_cache()
    yield sse("pull_done", {"name": name})


async def delete_model(name: str) -> tuple[bool, str]:
    """Delete an installed model. Returns (ok, message)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.request("DELETE", f"{config.OLLAMA_HOST}/api/delete",
                                 json={"model": name})
    invalidate_cache()
    if r.status_code == 200:
        return True, f"deleted {name}"
    return False, f"Ollama returned HTTP {r.status_code}: {r.text[:200]}"
