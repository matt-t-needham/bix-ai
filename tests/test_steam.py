"""Unit tests for steam.py — pure logic, no real Steam install required."""

from pathlib import Path

import steam

_LIBRARYFOLDERS = '''"libraryfolders"
{
\t"0"
\t{
\t\t"path"\t\t"%(lib0)s"
\t\t"label"\t\t""
\t}
\t"1"
\t{
\t\t"path"\t\t"%(lib1)s"
\t\t"label"\t\t""
\t}
}
'''

_ACF = '''"AppState"
{
\t"appid"\t\t"%(appid)s"
\t"name"\t\t"%(name)s"
\t"installdir"\t\t"%(installdir)s"
\t"LastPlayed"\t\t"%(last)s"
\t"SizeOnDisk"\t\t"%(size)s"
}
'''


def _write_acf(steamapps: Path, appid, name, size, installdir="x", last="0"):
    (steamapps / f"appmanifest_{appid}.acf").write_text(
        _ACF % {"appid": appid, "name": name, "installdir": installdir,
                "last": last, "size": size}
    )


def _setup(tmp_path: Path) -> Path:
    lib0 = tmp_path / "ssd"
    lib1 = tmp_path / "hdd"
    (lib0 / "steamapps").mkdir(parents=True)
    (lib1 / "steamapps").mkdir(parents=True)
    vdf = lib0 / "steamapps" / "libraryfolders.vdf"
    vdf.write_text(_LIBRARYFOLDERS % {"lib0": str(lib0), "lib1": str(lib1)})

    _write_acf(lib0 / "steamapps", "72850", "Skyrim", "10000000000")
    _write_acf(lib1 / "steamapps", "976730", "Halo: MCC", "95000000000")
    # Valve tooling that should be filtered out of "games"
    _write_acf(lib0 / "steamapps", "228980", "Steamworks Common Redistributables", "1")
    _write_acf(lib1 / "steamapps", "2805730", "Proton 9.0", "1300000000")
    return vdf


def test_parse_vdf_nested():
    parsed = steam.parse_vdf('"Root"\n{\n\t"k"\t\t"v"\n\t"sub"\n\t{\n\t\t"a"\t\t"1"\n\t}\n}')
    assert parsed["Root"]["k"] == "v"
    assert parsed["Root"]["sub"]["a"] == "1"


def test_library_dirs_span_drives(tmp_path):
    vdf = _setup(tmp_path)
    dirs = steam.library_steamapps_dirs(vdf)
    assert len(dirs) == 2


def test_list_games_excludes_runtimes(tmp_path):
    vdf = _setup(tmp_path)
    games = steam.list_games(libraryfolders_vdf=vdf)
    names = [g["name"] for g in games]
    assert names == ["Halo: MCC", "Skyrim"]          # sorted, runtimes dropped
    assert games[0]["size_on_disk"] == 95000000000


def test_include_runtimes_returns_everything(tmp_path):
    vdf = _setup(tmp_path)
    games = steam.list_games(libraryfolders_vdf=vdf, include_non_games=True)
    assert len(games) == 4


def test_missing_libraryfolders_falls_back(tmp_path):
    games = steam.list_games(libraryfolders_vdf=tmp_path / "nope.vdf")
    assert games == []


def test_format_games_empty():
    assert "No installed Steam games" in steam.format_games([])
