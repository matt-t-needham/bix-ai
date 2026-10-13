"""Agent mode: which commands run without an approval, the loop holding a tool
call until matt decides, the runner client, and the Access gate on the
routes that reach the runner."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import agent  # noqa: E402
import approvals  # noqa: E402
import config  # noqa: E402
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from helpers import parse_sse_data, parse_sse_event  # noqa: E402
from streaming.loop import run_tool_loop  # noqa: E402
from streaming.providers import _to_ollama_wire  # noqa: E402


async def _no_hand_off(conversation, actor, tool_input):
    raise AssertionError("these tests never hand off")


# ── needs_approval ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("script", [
    "ls -la", "cat README.md", "git status", "git log --oneline -5", "git log --format=%h",
    "git diff HEAD~1", "find . -name '*.py'", 'grep -rn "a*b" src', "wc -l main.py",
])
def test_simple_read_only_commands_run_without_approval(script):
    assert not agent.needs_approval("run_command", {"script": script})


@pytest.mark.parametrize("script", [
    "rm -rf build", "curl https://example.com", "python3 x.py", "pytest -q",
    "ls | sh", "cat a > b", "ls; rm x", "ls && rm x", "cat $(echo x)", "cat `id`", "ls *",
    "FOO=1 ls", "git -c core.pager=sh log", "git push", "git fetch", "git branch -D main",
    "git diff --output=/tmp/x", "find . -exec rm {} +", "find . -delete", "ls\nrm x",
    "cat 'unclosed", "", "   ", "cat ~/.gitconfig", "ls src/*.py", "grep -r x [ab]",
])
def test_anything_else_waits_for_approval(script):
    assert agent.needs_approval("run_command", {"script": script})


def test_git_outside_the_clone_root_waits_for_approval():
    assert agent.needs_approval("run_command", {"script": "git status", "cwd": "evil"})
    assert not agent.needs_approval("run_command", {"script": "ls", "cwd": "src"})


def test_only_a_resetting_clone_waits_for_approval():
    assert agent.needs_approval("clone_repo", {"repo": "bix-ai", "reset": True})
    assert not agent.needs_approval("clone_repo", {"repo": "bix-ai"})
    assert not agent.needs_approval("write_file", {"repo": "bix-ai", "path": "x", "content": ""})


# ── the loop's approval gate ──────────────────────────────────────────────────

class _ToolThenAnswer:
    """A provider that asks for one run_command, then answers."""
    name = "fake"
    emits_input_tokens_sse = False

    def __init__(self, script):
        self.script, self.turns = script, 0

    def budget_tokens(self, messages):
        return 0

    async def stream_turn(self, messages):
        self.turns += 1
        if self.turns == 1:
            yield {"kind": "turn_end", "tool_use": True, "truncated": False,
                   "tool_calls": [{"id": "tu1", "name": "run_command", "input": {"script": self.script}}]}
        else:
            yield {"kind": "text_delta", "text": "done"}
            yield {"kind": "turn_end", "tool_use": False, "truncated": False, "tool_calls": []}

    def append_assistant_turn(self, messages):
        messages.append({"role": "assistant", "content": [{"type": "text", "text": "x"}]})

    def append_tool_results(self, messages, results):
        messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": c} for tid, c, _ in results]})

    def history_messages(self, messages):
        return messages

    def final_token_fields(self, elapsed):
        return {"input_tokens": 0, "output_tokens": 0, "tps": 0}

    def breach_token_fields(self, messages):
        return self.final_token_fields(0)

    async def write_routing(self, ttft_ms, elapsed_ms):
        pass

    def stream_status(self):
        return "streaming"


def _run_loop(monkeypatch, script, decide=None, timeout=5.0, max_seconds=60.0, clock=None, result="ok"):
    """Runs the loop with a real AgentSession; `decide(conv, approval_id)` is
    called when the approval card would appear. Returns (events, scripts run)."""
    monkeypatch.setattr(config, "AGENT_APPROVAL_TIMEOUT", timeout)
    ran = []

    async def execute(name, tool_input):
        ran.append(tool_input["script"])
        return result

    async def go():
        events = []
        async for chunk in run_tool_loop(
            _ToolThenAnswer(script), [{"role": "user", "content": "hi"}],
            execute_tool=execute, max_tokens_budget=10**9, max_seconds=max_seconds,
            stats={"summarised": 0, "spilled": 0, "skipped": 0, "failed": 0},
            clock=clock, gate=agent.AgentSession("conv1", "api:test", _no_hand_off),
        ):
            ev, data = parse_sse_event(chunk), parse_sse_data(chunk)
            events.append((ev, data))
            if ev == "approval_required" and decide:
                asyncio.get_running_loop().call_soon(decide, "conv1", data["approval_id"])
        return events

    return asyncio.run(go()), ran


def _results(events):
    return [d for e, d in events if e == "tool_result"]


def test_read_only_command_runs_without_a_card(monkeypatch):
    events, ran = _run_loop(monkeypatch, "ls")
    assert "approval_required" not in [e for e, _ in events] and ran == ["ls"]


def test_approved_command_runs(monkeypatch):
    events, ran = _run_loop(monkeypatch, "rm -rf build", decide=lambda c, a: approvals.decide(c, a, True))
    [card] = [d for e, d in events if e == "approval_required"]
    assert {k: card[k] for k in ("tool_use_id", "name", "input")} == \
        {"tool_use_id": "tu1", "name": "run_command", "input": {"script": "rm -rf build"}}
    assert ran == ["rm -rf build"] and _results(events)[0]["is_error"] is False
    assert events[-1][0] == "done"


def test_denied_command_never_runs_and_the_model_is_told(monkeypatch):
    events, ran = _run_loop(monkeypatch, "rm -rf build", decide=lambda c, a: approvals.decide(c, a, False))
    assert ran == []
    result = _results(events)[0]
    assert result["is_error"] is True and "denied by matt" in result["content"]
    assert events[-1][0] == "done"


def test_unanswered_card_times_out_as_a_refusal(monkeypatch):
    events, ran = _run_loop(monkeypatch, "rm -rf build", timeout=0.05)
    assert ran == [] and "no decision within" in _results(events)[0]["content"]
    assert _results(events)[0]["is_error"] is True


def test_a_decision_only_settles_its_own_pending_approval():
    async def go():
        approval_id = approvals.open_request("conv1")
        wrong_conv = approvals.decide("other-conv", approval_id, True)
        settled = approvals.decide("conv1", approval_id, False)
        refusal = await approvals.wait(approval_id, 1)
        return wrong_conv, settled, refusal, approvals.decide("conv1", approval_id, True)

    assert asyncio.run(go()) == (False, True, "denied by matt", False)


def test_tool_errors_reach_the_ui_as_errors(monkeypatch):
    events, _ = _run_loop(monkeypatch, "ls", result=("Refused by the runner (409): nope", True))
    assert _results(events)[0] == {"tool_use_id": "tu1", "content": "Refused by the runner (409): nope",
                                   "is_error": True}


def test_time_waiting_for_matt_does_not_count_against_the_budget(monkeypatch):
    # The fake clock jumps 100 s while the card is open; the budget is 60 s.
    now = [0.0]

    def jump(conv, approval_id):
        now[0] += 100
        approvals.decide(conv, approval_id, True)

    events, ran = _run_loop(monkeypatch, "rm -rf build", decide=jump, max_seconds=60, clock=lambda: now[0])
    assert ran == ["rm -rf build"] and events[-1][0] == "done"


def test_is_error_reaches_openai_shaped_models_as_text():
    # Anthropic's native is_error is pinned by the claude_agent_denied fixture.
    messages = [{"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "tu1", "content": "Not run: denied.", "is_error": True},
        {"type": "tool_result", "tool_use_id": "tu2", "content": "ok"}]}]
    wire = _to_ollama_wire(messages)
    assert [m["content"] for m in wire] == ["Error: Not run: denied.", "ok"]


# ── runner client ─────────────────────────────────────────────────────────────

def _call(name, tool_input):
    return asyncio.run(agent.call_runner(name, tool_input, conversation="conv1", actor="api:test"))


def test_run_command_result_is_formatted_for_the_model(fake_runner):
    seen, replies = fake_runner
    replies["/run"] = (200, {"exit_code": 1, "timed_out": True, "stdout": "out", "stderr": "err",
                             "stdout_bytes": 999999, "stderr_bytes": 3, "truncated": True})
    text, is_error = _call("run_command", {"script": "make", "repo": "bix-ai"})
    assert seen == [("POST", "/run", {"script": "make", "repo": "bix-ai"}, "conv1")] and is_error is True
    assert "exit code 1 (timed out and killed)" in text and "--- stdout ---\nout" in text
    assert "999999 bytes" in text


def test_runner_refusals_reach_the_model_as_refusals(fake_runner):
    _, replies = fake_runner
    replies["/write"] = (400, {"error": "path escapes the clone: ../x"})
    assert _call("write_file", {"repo": "bix-ai", "path": "../x", "content": ""}) == \
        ("Refused by the runner (400): path escapes the clone: ../x", True)


def test_a_garbled_runner_reply_is_an_error_not_a_crash(fake_runner):
    _, replies = fake_runner
    replies["/repos"] = (200, None)   # the fake sends "null": valid JSON, not an object
    text, is_error = _call("list_repos", {})
    assert is_error is True and text.startswith("Not run — the agent runner failed")


def test_unreachable_runner_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AGENT_RUNNER_SOCKET", str(tmp_path / "missing.sock"))
    text, is_error = _call("list_repos", {})
    assert is_error is True and text.startswith("Not run — the agent runner failed")


def test_session_sends_only_agent_tools_to_the_runner(fake_runner):
    seen, _ = fake_runner
    calls = []

    async def base(name, tool_input):
        calls.append(name)
        return "base"

    run = agent.AgentSession("conv1", "api:test", _no_hand_off).executor(base)
    assert asyncio.run(run("read_file", {"path": "/x"})) == "base"
    asyncio.run(run("list_repos", {}))
    assert calls == ["read_file"] and [s[1] for s in seen] == ["/repos"]


# ── routes ────────────────────────────────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")

    async def allow(model):
        return True
    monkeypatch.setattr(main.model_admin, "is_allowed_local_model", allow)
    return TestClient(main.app)


def _chat(client, headers=None, **body):
    return client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}],
                                      "conversation_id": "new", "mode": "local", "model": "m",
                                      "agent": True, **body}, headers=headers or {})


def test_agent_mode_without_access_is_refused(client, monkeypatch):
    monkeypatch.setattr(config, "ACCESS_AUD", "")
    resp = _chat(client)
    assert resp.status_code == 403 and "Access" in resp.json()["error"]


def test_agent_mode_from_another_site_is_refused(client, monkeypatch):
    monkeypatch.setattr(main.access, "access_verified", lambda request: True)
    assert _chat(client, headers={"Sec-Fetch-Site": "same-site"}).status_code == 403


def test_agent_mode_needs_a_conversation(client, monkeypatch):
    monkeypatch.setattr(main.access, "access_verified", lambda request: True)
    assert _chat(client, conversation_id="").status_code == 400


@pytest.mark.parametrize("mode, model, adapter", [("local", "m", "_stream_ollama"),
                                                   ("pro", "claude-sonnet-4-6", "_stream_pro")])
def test_verified_agent_chat_reaches_the_adapter_with_a_session(client, monkeypatch, mode, model, adapter):
    monkeypatch.setattr(main.access, "access_verified", lambda request: True)
    got = {}

    async def fake(messages, model, *args, agent_session=None, **kw):
        got["agent"] = agent_session
        yield "event: done\ndata: {}\n\n"
    monkeypatch.setattr(main, adapter, fake)
    assert _chat(client, mode=mode, model=model, headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200
    assert isinstance(got["agent"], agent.AgentSession) and got["agent"].actor == f"{mode}:{model}"


def test_plain_chat_gets_no_session(client, monkeypatch):
    got = {}

    async def fake_ollama(messages, model, mode="local", agent_session=None, **kw):
        got["agent"] = agent_session
        yield "event: done\ndata: {}\n\n"
    monkeypatch.setattr(main, "_stream_ollama", fake_ollama)
    assert _chat(client, agent=False).status_code == 200 and got["agent"] is None


def test_approval_route_needs_access_and_a_waiter(client, monkeypatch):
    url = "/conversations/conv1/approvals/abc123"
    monkeypatch.setattr(config, "ACCESS_AUD", "")
    assert client.post(url, json={"approve": True}).status_code == 403
    monkeypatch.setattr(main.access, "access_verified", lambda request: True)
    assert client.post(url, json={"approve": True}, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert client.post(url, json={"approve": True}).status_code == 409


# ── Pro mode: the MCP server's callback ──────────────────────────────────────

def test_a_pro_run_waits_for_approval_like_any_other_mode(fake_runner):
    seen, replies = fake_runner
    replies["/run"] = (200, {"exit_code": 0, "timed_out": False, "stdout": "ok", "stderr": "",
                             "stdout_bytes": 2, "stderr_bytes": 0, "truncated": False})

    async def flow():
        run = agent.open_run(agent.AgentSession("conv1", "pro:m", _no_hand_off))
        call = asyncio.ensure_future(agent.run_tool(run.token, "run_command", {"script": "make test"}))
        card = parse_sse_data(await asyncio.wait_for(run.events.get(), 5))
        assert seen == []                     # nothing runs before the decision
        assert approvals.decide("conv1", card["approval_id"], True)
        result = await asyncio.wait_for(call, 5)
        agent.close_run(run.token)
        return card, result
    card, (text, is_error) = asyncio.run(flow())
    assert card["name"] == "run_command" and card["input"] == {"script": "make test"}
    assert not is_error and "--- stdout ---\nok" in text and seen[0][1] == "/run"


def test_closing_a_pro_run_refuses_its_pending_approvals(fake_runner):
    seen, _ = fake_runner

    async def flow():
        run = agent.open_run(agent.AgentSession("conv1", "pro:m", _no_hand_off))
        call = asyncio.ensure_future(agent.run_tool(run.token, "run_command", {"script": "make test"}))
        await asyncio.wait_for(run.events.get(), 5)
        agent.close_run(run.token)
        return await asyncio.wait_for(call, 5), await agent.run_tool(run.token, "list_repos", {})
    (text, is_error), after = asyncio.run(flow())
    assert is_error and "stream ended" in text and seen == [] and after is None


def test_pro_runs_call_only_agent_tools():
    async def flow():
        run = agent.open_run(agent.AgentSession("conv1", "pro:m", _no_hand_off))
        try:
            return await agent.run_tool(run.token, "add_todo", {"text": "x"})
        finally:
            agent.close_run(run.token)
    assert asyncio.run(flow()) == ("add_todo isn't an agent tool.", True)


@pytest.fixture
def run_token(client, monkeypatch):
    """An open run's token, with TestClient's address counted as the container's loopback."""
    async def open_():
        return agent.open_run(agent.AgentSession("conv1", "pro:m", _no_hand_off)).token
    token = asyncio.run(open_())
    monkeypatch.setattr(main, "_INTERNAL_CLIENTS", {"testclient"})
    yield token
    agent.close_run(token)


def test_internal_routes_serve_an_open_runs_mcp_server(client, run_token):
    resp = client.get("/internal/agent/tools", headers={"X-Bix-Agent-Token": run_token})
    assert resp.json()["tools"] == agent.AGENT_MCP_TOOLS


@pytest.mark.parametrize("headers", [{}, {"X-Bix-Agent-Token": "guess"}, "tunnelled"])
def test_internal_routes_refuse_a_wrong_token_or_tunnel_traffic(client, run_token, headers):
    if headers == "tunnelled":
        headers = {"X-Bix-Agent-Token": run_token, "CF-Connecting-IP": "203.0.113.9"}
    assert client.get("/internal/agent/tools", headers=headers).status_code == 403
    assert client.post("/internal/agent/tool", json={"name": "list_repos"}, headers=headers).status_code == 403


def test_internal_routes_refuse_callers_outside_the_container(client, run_token, monkeypatch):
    monkeypatch.setattr(main, "_INTERNAL_CLIENTS", {"127.0.0.1"})   # the caller is no longer an allowed client
    assert client.get("/internal/agent/tools", headers={"X-Bix-Agent-Token": run_token}).status_code == 403


def test_internal_routes_refuse_a_closed_runs_token(client, run_token):
    agent.close_run(run_token)
    resp = client.post("/internal/agent/tool", json={"name": "list_repos"}, headers={"X-Bix-Agent-Token": run_token})
    assert resp.status_code == 403
