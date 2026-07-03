# Staged writes

`stage_write` (or the MCP `write_file` tool in pro mode) never touches the live
filesystem. Calling it creates a pending record — the target path and full
content — that a human reviews at `/staging` and either approves or rejects.
Approval is the only step that writes to disk; it re-validates every guard
before applying.

## What gets refused

A write is rejected at proposal time (`ValueError`, not a silent no-op) if the
target path is:

- outside the allowed root (`FS_ROOT`)
- a secrets path (anything `is_denied_path` covers — credentials, `.env` files)
- a guardrail path (`is_write_denied_path` — shell scripts, container/CI
  config, bix-ai's own source)

Content is capped at 200 KB.

## How to talk about it

Always tell the user the change was **staged for review**, not written. Don't
say "I created the file" or "I saved it" — say something like "I've staged
`X` for your review at /staging."
