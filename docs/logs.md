# Logs

Two tools, meant to be used in sequence:

1. `list_log_sources()` — enumerates every readable log, both internal
   app/service logs (ai-router, demucs, nginx access/error, nightly deploy
   logs, daily review tickets) and the Steam client logs. Returns each log's
   full path, size, and last-modified time.
2. `read_log(path, lines, contains)` — tails a log from a path
   `list_log_sources` returned. Returns the last N lines (default 200, max
   2000), optionally filtered to lines containing a substring. Large files
   are tailed efficiently, not loaded whole.

Always call `list_log_sources` first rather than guessing a path — the set of
logs and their locations can change, and passing an unlisted path will fail.
