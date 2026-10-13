"""Pro mode in an agent chat: approval cards reach the stream while the
claude CLI is silent, waiting on the very tool call they hold."""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from streaming import pro  # noqa: E402


def test_queued_cards_reach_the_stream_while_the_cli_is_silent():
    async def flow():
        stdout, events = asyncio.StreamReader(), asyncio.Queue()
        out = []

        async def drive():
            async for line, event in pro._lines_and_events(stdout, events):
                out.append(event or line.decode().strip())
                if len(out) == 1:            # the CLI's tool_use line: now it waits on the tool
                    await events.put("card")
                elif len(out) == 2:          # the card arrived with stdout still silent
                    stdout.feed_data(b'{"type": "result"}\n')
                    stdout.feed_eof()
        stdout.feed_data(json.dumps({"type": "assistant"}).encode() + b"\n")
        await asyncio.wait_for(drive(), 5)
        return out
    assert asyncio.run(flow()) == ['{"type": "assistant"}', "card", '{"type": "result"}']


def test_only_an_agent_run_gives_the_mcp_server_a_way_back():
    env = json.loads(pro._mcp_config("claude-sonnet-4-6", "tok"))["mcpServers"]["bix"]["env"]
    assert env["BIX_AGENT_TOKEN"] == "tok" and env["BIX_AGENT_URL"] and float(env["BIX_AGENT_TIMEOUT"]) > 1200
    plain = json.loads(pro._mcp_config("claude-sonnet-4-6"))["mcpServers"]["bix"]["env"]
    assert not any(k.startswith("BIX_AGENT_") for k in plain)
