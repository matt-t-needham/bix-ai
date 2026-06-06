"""Unit tests for logtools.py — pure logic, stdlib-only, no real logs needed."""
import pytest

import config
import logtools


@pytest.fixture
def roots(tmp_path, monkeypatch):
    internal = tmp_path / "internal"
    steam = tmp_path / "steam"
    (internal / "ai-router").mkdir(parents=True)
    steam.mkdir(parents=True)
    monkeypatch.setattr(config, "INTERNAL_LOG_ROOT", internal)
    monkeypatch.setattr(config, "STEAM_LOG_ROOT", steam)
    (internal / "ai-router" / "app.log").write_text(
        "\n".join(f"line {i}" for i in range(1, 51))
    )
    (internal / "restart.log").write_text("restarted ok\n")
    (steam / "console-linux.txt").write_text("steam line A\nERROR steam boom\nsteam line B\n")
    return internal, steam


def test_list_sources_shows_both_roots(roots):
    internal, steam = roots
    out = logtools.list_sources()
    assert "Internal app logs" in out and "Steam logs" in out
    assert str(internal / "ai-router" / "app.log") in out
    assert str(steam / "console-linux.txt") in out


def test_read_log_tails(roots):
    internal, _ = roots
    out = logtools.read_log(str(internal / "ai-router" / "app.log"), lines=3)
    assert "line 48" in out and "line 50" in out
    assert "line 47" not in out.split("\n", 1)[1]      # body has only last 3


def test_read_log_contains_filter(roots):
    _, steam = roots
    out = logtools.read_log(str(steam / "console-linux.txt"), contains="error")
    assert "ERROR steam boom" in out
    assert "steam line A" not in out


def test_read_log_contains_no_match(roots):
    _, steam = roots
    out = logtools.read_log(str(steam / "console-linux.txt"), contains="zzz-nope")
    assert "No lines containing" in out


def test_read_log_outside_roots_denied(roots, tmp_path):
    outside = tmp_path / "outside.log"
    outside.write_text("secret")
    out = logtools.read_log(str(outside))
    assert "Access denied" in out


def test_read_log_secret_name_denied(roots):
    internal, _ = roots
    bad = internal / "token.log"            # 'token' keyword -> is_denied_path
    bad.write_text("creds")
    assert "protected" in logtools.read_log(str(bad))


def test_read_log_missing(roots):
    internal, _ = roots
    assert "not found" in logtools.read_log(str(internal / "nope.log")).lower()


def test_read_log_lines_capped(roots):
    internal, _ = roots
    big = internal / "ai-router" / "many.log"
    big.write_text("\n".join(str(i) for i in range(5000)))
    out = logtools.read_log(str(big), lines=99999)      # request over the cap
    body = out.split("\n", 1)[1]
    assert len(body.splitlines()) <= logtools._MAX_LINES


def test_read_log_tails_large_file_via_scan(roots, monkeypatch):
    internal, _ = roots
    monkeypatch.setattr(logtools, "_MAX_SCAN_BYTES", 80)   # force the seek branch
    f = internal / "ai-router" / "big.log"
    f.write_text("\n".join(f"row{i}" for i in range(1, 200)))
    out = logtools.read_log(str(f), lines=5)
    assert "scanned" in out.splitlines()[0]
    assert "row199" in out                                 # newest lines present
