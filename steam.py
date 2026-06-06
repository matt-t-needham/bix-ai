"""Steam installed-game catalog reader.

Pure logic (no imports from main.py / tools.py) so it can be unit-tested in
isolation, mirroring strategy.py.

Source of truth for "all installed games" is Steam's KeyValues (VDF) files:

    <steamapps>/libraryfolders.vdf   — lists every library folder path
    <steamapps>/appmanifest_<appid>.acf — one per installed app (name, size, ...)

These are bind-mounted read-only into the container (see bix-infra/
docker-compose.yml, ai-router service). We only read libraryfolders.vdf and the
top-level appmanifest_*.acf files — we never recurse into common/ or compatdata/,
whose symlinks point outside the mounted tree and would dangle in the container.
"""

import os
import re
from pathlib import Path

# Path to the *primary* library's libraryfolders.vdf. That file enumerates every
# other library, so this single entry point reaches the whole catalog.
LIBRARYFOLDERS_VDF = Path(os.environ.get(
    "STEAM_LIBRARYFOLDERS_VDF",
    "/home/matt/.steam/debian-installation/steamapps/libraryfolders.vdf",
))

# appids that are runtimes/redistributables rather than playable games.
_NON_GAME_APPIDS = {"228980"}  # Steamworks Common Redistributables

# Valve tool apps (Proton, the Steam Linux Runtimes, anticheat shims) install as
# appmanifests too. Match them by name so "all games" means actual games.
_NON_GAME_NAME_RE = re.compile(
    r"^(Proton|Steam Linux Runtime|Steamworks )", re.IGNORECASE
)

_TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|([{}])')


def parse_vdf(text: str) -> dict:
    """Parse Valve KeyValues (VDF) text into nested dicts.

    Handles the subset Steam uses for libraryfolders.vdf / appmanifest_*.acf:
    quoted keys, quoted string values, and nested { } blocks. Good enough for
    these files; not a general VDF implementation (no #include, no conditionals).
    """
    toks: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(text):
        if m.group(2) is not None:
            toks.append(("brace", m.group(2)))
        else:
            toks.append(("str", m.group(1)))

    pos = 0

    def parse_block() -> dict:
        nonlocal pos
        obj: dict = {}
        while pos < len(toks):
            kind, val = toks[pos]
            if kind == "brace" and val == "}":
                pos += 1
                return obj
            # Expect a string key.
            key = val
            pos += 1
            if pos >= len(toks):
                break
            nkind, nval = toks[pos]
            if nkind == "brace" and nval == "{":
                pos += 1
                obj[key] = parse_block()
            else:
                obj[key] = nval
                pos += 1
        return obj

    # Top level is a single "<root>" { ... } pair.
    return parse_block()


def _int(v: str | None) -> int:
    try:
        return int(v or "0")
    except (TypeError, ValueError):
        return 0


def library_steamapps_dirs(libraryfolders_vdf: Path = LIBRARYFOLDERS_VDF) -> list[Path]:
    """Return every existing <library>/steamapps dir listed in libraryfolders.vdf.

    Falls back to the directory containing libraryfolders.vdf if the file is
    missing or unparseable, so a single-library setup still works.
    """
    dirs: list[Path] = []
    try:
        data = parse_vdf(libraryfolders_vdf.read_text(errors="replace"))
        folders = data.get("libraryfolders", {})
        for entry in folders.values():
            if isinstance(entry, dict) and entry.get("path"):
                steamapps = Path(entry["path"]) / "steamapps"
                if steamapps.is_dir():
                    dirs.append(steamapps)
    except FileNotFoundError:
        pass
    except Exception:
        pass

    if not dirs and libraryfolders_vdf.parent.is_dir():
        dirs.append(libraryfolders_vdf.parent)

    # De-dup while preserving order (a library can be listed under its own path).
    seen: set[str] = set()
    unique: list[Path] = []
    for d in dirs:
        key = str(d.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def list_games(
    libraryfolders_vdf: Path = LIBRARYFOLDERS_VDF,
    include_non_games: bool = False,
) -> list[dict]:
    """Enumerate installed Steam apps across every library.

    Returns a list of dicts sorted by name, each with: appid, name, installdir,
    size_on_disk (bytes), last_played (unix ts, 0 if never), library (the
    steamapps dir it was found in). Redistributables/runtimes are excluded
    unless include_non_games is True.
    """
    games: list[dict] = []
    seen_appids: set[str] = set()

    for steamapps in library_steamapps_dirs(libraryfolders_vdf):
        for acf in sorted(steamapps.glob("appmanifest_*.acf")):
            try:
                app = parse_vdf(acf.read_text(errors="replace")).get("AppState", {})
            except Exception:
                continue
            appid = app.get("appid")
            if not appid or appid in seen_appids:
                continue
            name = app.get("name", f"(app {appid})")
            if not include_non_games and (
                appid in _NON_GAME_APPIDS or _NON_GAME_NAME_RE.match(name)
            ):
                continue
            seen_appids.add(appid)
            games.append({
                "appid":        appid,
                "name":         name,
                "installdir":   app.get("installdir", ""),
                "size_on_disk": _int(app.get("SizeOnDisk")),
                "last_played":  _int(app.get("LastPlayed")),
                "library":      str(steamapps),
            })

    games.sort(key=lambda g: g["name"].lower())
    return games


def format_games(games: list[dict]) -> str:
    """Render the game list as a compact text block for a tool result."""
    if not games:
        return "No installed Steam games found."
    lines = [f"{len(games)} installed Steam game(s):", ""]
    for g in games:
        gb = g["size_on_disk"] / 1_000_000_000
        lines.append(f"- {g['name']} (appid {g['appid']}) — {gb:.1f} GB — {g['library']}")
    return "\n".join(lines)
