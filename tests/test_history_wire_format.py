"""Tests for Phase 1 wire-format compatibility: block-structured messages
round-tripping through ChatRequest validation and memory rendering.

PLAN-pi-tools.md Phase 1 step 6: ChatRequest.messages already accepts
list[dict[str, Any]] — block-structured content (what a `history` SSE event
produces) passes validation today. This locks that in with a test instead of
a refactor. Also covers the _render_memory_html watch-out: memory saves of
block-structured convHistory (tool_use/tool_result blocks) must still render.
"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

sys.path.insert(0, str(Path(__file__).parent.parent))

import main  # noqa: E402


BLOCK_MESSAGES = [
    {"role": "user", "content": "read a.txt"},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me check that."},
            {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.txt"}},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": "FILE CONTENTS"},
        ],
    },
]


def test_chat_request_accepts_block_structured_history():
    """What a `history` SSE event emits must validate as a fresh ChatRequest body —
    this is exactly what the client resends on the next turn."""
    body = json.dumps({"messages": BLOCK_MESSAGES, "mode": "api", "model": "claude-sonnet-4-6"})
    req = main.ChatRequest.model_validate_json(body)
    assert req.messages == BLOCK_MESSAGES


def test_render_memory_html_renders_tool_blocks():
    entry = {"title": "test", "date": "2026-07-02T00:00:00Z", "model": "claude-sonnet-4-6"}
    convo = {"messages": BLOCK_MESSAGES}
    html = main._render_memory_html(entry, convo)
    assert "tool_use: read_file" in html
    assert "tool_result] FILE CONTENTS" in html
    assert "Let me check that." in html
