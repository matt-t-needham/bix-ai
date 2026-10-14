"""POST /chat with a server-owned conversation, plus the /conversations routes.

The streaming adapter is replaced with a fake generator so the test pins the
wiring in main.py: the `conversation` + `context` events come first, the new
user turn is stored before dispatch, the `history` tail is folded back into
the record (pointered sent turns + new assistant/tool turns), pro mode
persists its text reply and session id, and an errored turn keeps the
transcript alternating."""
import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

import config  # noqa: E402
import conversations  # noqa: E402
import main  # noqa: E402
from helpers import sse  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONVERSATIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "CONTEXT_BUDGET_LOCAL", 1_000_000)

    async def allow(model):
        return True
    monkeypatch.setattr(main.model_admin, "is_allowed_local_model", allow)
    return tmp_path


@pytest.fixture
def client():
    return TestClient(main.app)


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.split("\n\n"):
        ev, data = None, None
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
        if ev:
            out.append((ev, data))
    return out


def _fake_local(reply: str, tool: bool = False):
    """Stands in for _stream_ollama: echoes the sent messages back on `history`."""
    async def gen(messages, model, mode="local", **kw):
        yield sse("status", {"stage": "streaming", "message": "x"})
        yield sse("delta", {"text": reply})
        hist = list(messages)
        if tool:
            hist.append({"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "/x"}}]})
            hist.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "contents"}]})
        hist.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})
        yield sse("metrics", {"input_tokens": 1, "output_tokens": 1, "elapsed_ms": 1, "ttft_ms": 1,
                              "preprocess_ms": 0, "tps": 1, "summarised": 0, "spilled": 0,
                              "skipped": 0, "failed": 0})
        yield sse("history", {"messages": hist})
        yield sse("done", {})
    return gen


def _post(client, body):
    r = client.post("/chat", json=body)
    assert r.status_code == 200, r.text
    return _events(r.text)


def test_new_conversation_round_trip(env, client, monkeypatch):
    monkeypatch.setattr(main, "_stream_ollama", _fake_local("hello back", tool=True))
    events = _post(client, {"messages": [{"role": "user", "content": "hi there"}],
                            "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    names = [e for e, _ in events]
    assert names[:2] == ["conversation", "context"]
    conv_id = events[0][1]["id"]
    assert events[0][1]["title"] == "hi there"
    ctx = events[1][1]
    assert ctx["total_messages"] == 1 and ctx["sent_messages"] == 1 and ctx["folded_messages"] == 0
    assert names[-1] == "done"

    rec = conversations.get(conv_id)
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]     # tool turns kept
    assert rec["messages"][-1]["content"] == [{"type": "text", "text": "hello back"}]
    assert rec["model"] == "gemma4:26b" and rec["mode"] == "local"

    # Second turn on the same conversation: only the new message is sent,
    # the adapter sees the full stored history.
    seen = {}

    async def spy(messages, model, **kw):
        seen["messages"] = list(messages)
        async for ev in _fake_local("again")(messages, model, **kw):
            yield ev
    monkeypatch.setattr(main, "_stream_ollama", spy)
    events = _post(client, {"messages": [{"role": "user", "content": "second"}],
                            "conversation_id": conv_id, "model": "gemma4:26b", "mode": "local"})
    assert [m["role"] for m in seen["messages"]] == ["user", "assistant", "user", "assistant", "user"]
    assert conversations.get(conv_id)["messages"][-1]["content"] == [{"type": "text", "text": "again"}]
    assert len(conversations.get(conv_id)["messages"]) == 6
    listing = client.get("/conversations").json()["conversations"]
    assert listing[0]["id"] == conv_id and listing[0]["messages"] == 6


def test_unknown_conversation_is_404(env, client):
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "x"}],
                                   "conversation_id": "zzzzzzzzzzzz", "model": "gemma4:26b", "mode": "local"})
    assert r.status_code == 404


def test_errored_turn_keeps_transcript_alternating(env, client, monkeypatch):
    async def failing(messages, model, **kw):
        yield sse("delta", {"text": "partial"})
        yield sse("error", {"message": "Ollama: boom"})
    monkeypatch.setattr(main, "_stream_ollama", failing)
    events = _post(client, {"messages": [{"role": "user", "content": "q"}],
                            "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    conv_id = events[0][1]["id"]
    rec = conversations.get(conv_id)
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant"]
    assert rec["messages"][1]["content"] == "partial"


def test_pro_turn_persists_text_and_session(env, client, monkeypatch):
    async def fake_pro(messages, model, max_tokens, mode="pro", session_id="", agent_session=None, **kw):
        yield sse("pro_session", {"session_id": "abcd-1234"})
        yield sse("delta", {"text": "pro says hi"})
        yield sse("metrics", {"input_tokens": 0, "output_tokens": 3, "elapsed_ms": 1, "ttft_ms": 1,
                              "preprocess_ms": 0, "tps": 1, "summarised": 0, "spilled": 0,
                              "skipped": 0, "failed": 0})
        yield sse("done", {})
    monkeypatch.setattr(main, "_stream_pro", fake_pro)
    events = _post(client, {"messages": [{"role": "user", "content": "q"}],
                            "conversation_id": "new", "model": "claude-sonnet-4-6", "mode": "pro"})
    conv_id = events[0][1]["id"]
    rec = conversations.get(conv_id)
    assert rec["pro_session_id"] == "abcd-1234"
    assert rec["messages"][-1] == {"role": "assistant", "content": "pro says hi"}

    # Resume: only the newest message goes to the CLI, context says so.
    seen = {}

    async def spy(messages, model, max_tokens, mode="pro", session_id="", agent_session=None, **kw):
        seen["messages"], seen["session_id"] = list(messages), session_id
        async for ev in fake_pro(messages, model, max_tokens, mode, session_id):
            yield ev
    monkeypatch.setattr(main, "_stream_pro", spy)
    events = _post(client, {"messages": [{"role": "user", "content": "q2"}],
                            "conversation_id": conv_id, "model": "claude-sonnet-4-6", "mode": "pro"})
    assert seen["session_id"] == "abcd-1234"
    assert [m["content"] for m in seen["messages"]] == ["q2"]
    assert next(d for e, d in events if e == "context")["resumed_session"] == 1


def test_stateless_call_still_works(env, client, monkeypatch):
    monkeypatch.setattr(main, "_stream_ollama", _fake_local("ok"))
    events = _post(client, {"messages": [{"role": "user", "content": "x"}],
                            "model": "gemma4:26b", "mode": "local"})
    assert [e for e, _ in events][0] == "status"            # no conversation/context events
    assert not list(config.CONVERSATIONS_DIR.glob("*.json")) if config.CONVERSATIONS_DIR.exists() else True


def test_conversation_routes(env, client, monkeypatch):
    rec = conversations.create("gemma4:26b", "local")
    rec["messages"] = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    conversations.save(rec)
    assert client.get(f"/conversations/{rec['id']}").json()["messages"][1]["content"] == "hi"
    assert client.get("/conversations/nope").status_code == 404

    saved = {}

    async def fake_save(messages, model_name, in_tokens, out_tokens):
        saved["messages"] = messages
        return {"ok": True, "id": "m1"}
    monkeypatch.setattr(main, "save_memory_entry", fake_save)
    assert client.post(f"/conversations/{rec['id']}/remember").json() == {"ok": True, "id": "m1"}
    assert len(saved["messages"]) == 2

    assert client.delete(f"/conversations/{rec['id']}").json() == {"ok": True}
    assert client.get(f"/conversations/{rec['id']}").status_code == 404


def test_openrouter_mode_requires_config_and_valid_id(env, client, monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "x"}],
                                   "model": "qwen/qwen3-coder", "mode": "openrouter"})
    assert r.status_code == 503
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "sk")
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "x"}],
                                   "model": "openrouter/auto", "mode": "openrouter"})
    assert r.status_code == 400


def test_real_loop_round_trip_keeps_the_reply(env, client, monkeypatch):
    """The fakes above stand in for the whole adapter; this runs the real
    _stream_ollama + run_tool_loop (only Ollama's HTTP is faked), so the
    stored record is what the loop's `history` actually carries. It once
    held only the user turn: the final reply never reached `history`."""
    from unittest.mock import patch

    from streaming import ollama as ollama_mod
    from tests.sse_harness import FakeClient, ollama_turn_lines

    class RecordingClient(FakeClient):
        def __init__(self, turns):
            super().__init__(turns)
            self.sent = []

        def stream(self, method, url, **kwargs):
            self.sent.append(kwargs.get("json", {}).get("messages", []))
            return super().stream(method, url, **kwargs)

    def text(t):
        return ollama_turn_lines([{"choices": [{"delta": {"content": t}}]},
                                  {"choices": [{"delta": {}, "finish_reason": "stop"}]}])
    tool_turn = ollama_turn_lines([
        {"choices": [{"delta": {"tool_calls": [{
            "index": 0, "id": "call_1",
            "function": {"name": "read_file", "arguments": '{"path": "/x"}'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ])
    fake = RecordingClient([tool_turn, text("first answer"), text("second answer")])

    async def fake_tool(name, tool_input):
        return "file contents"

    async def no_routing(*a, **k):
        pass

    with patch.object(ollama_mod.httpx, "AsyncClient", lambda **kw: fake), \
         patch.object(ollama_mod, "_execute_tool", fake_tool), \
         patch.object(ollama_mod, "_write_routing_event", no_routing):
        events = _post(client, {"messages": [{"role": "user", "content": "read /x"}],
                                "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
        assert [e for e, _ in events][-1] == "done"
        conv_id = events[0][1]["id"]
        rec = conversations.get(conv_id)
        assert [m["role"] for m in rec["messages"]] == ["user", "assistant", "user", "assistant"]
        assert rec["messages"][1]["content"][0]["type"] == "tool_use"
        assert rec["messages"][2]["content"][0]["type"] == "tool_result"
        assert rec["messages"][-1]["content"] == [{"type": "text", "text": "first answer"}]

        _post(client, {"messages": [{"role": "user", "content": "and then?"}],
                       "conversation_id": conv_id, "model": "gemma4:26b", "mode": "local"})

    # Turn two went up with turn one's answer in it, after the system prompt,
    # attributed to the model that wrote it.
    sent = [m for m in fake.sent[-1] if m["role"] != "system"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "assistant", "user"]
    assert sent[3]["content"] == "[router-model gemma4:26b]\nfirst answer"
    # …while the stored record stays clean.
    assert "router-model" not in json.dumps(conversations.get(conv_id)["messages"])
    assert sent[-1]["content"] == "and then?"
    rec = conversations.get(conv_id)
    assert [m["role"] for m in rec["messages"]] == ["user", "assistant", "user", "assistant", "user", "assistant"]
    assert rec["messages"][-1]["content"] == [{"type": "text", "text": "second answer"}]


# ── Per-turn model attribution (the `turns` sidecar) ──────────────────────────

def test_each_turn_records_the_model_that_answered(env, client, monkeypatch):
    monkeypatch.setattr(main, "_stream_ollama", _fake_local("local answer"))
    events = _post(client, {"messages": [{"role": "user", "content": "first question"}],
                            "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    conv_id = events[0][1]["id"]
    turns = conversations.get(conv_id)["turns"]
    assert len(turns) == 1
    assert turns[0]["model"] == "gemma4:26b" and turns[0]["mode"] == "local"
    # "from" points at where that turn's assistant output starts.
    rec = conversations.get(conv_id)
    assert rec["messages"][turns[0]["from"]]["role"] == "assistant"

    # A second turn on another model is recorded separately — the record now
    # says which model wrote which part.
    async def fake_claude(messages, model, max_tokens, **kw):
        async for ev in _fake_local("claude answer")(messages, model, **kw):
            yield ev
    monkeypatch.setattr(main, "_stream_claude", fake_claude)
    _post(client, {"messages": [{"role": "user", "content": "second question"}],
                   "conversation_id": conv_id, "model": "claude-opus-4-7", "mode": "api"})
    turns = conversations.get(conv_id)["turns"]
    assert [t["model"] for t in turns] == ["gemma4:26b", "claude-opus-4-7"]
    assert turns[1]["from"] > turns[0]["from"]


def test_the_sidecar_never_reaches_the_provider(env, client, monkeypatch):
    # An unknown key on a message is a 400 at Anthropic, so `turns` must stay
    # beside the messages, never on them.
    seen = {}

    async def spy(messages, model, **kw):
        seen["messages"] = list(messages)
        async for ev in _fake_local("ok")(messages, model, **kw):
            yield ev
    monkeypatch.setattr(main, "_stream_ollama", spy)
    events = _post(client, {"messages": [{"role": "user", "content": "q"}],
                            "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    conv_id = events[0][1]["id"]
    _post(client, {"messages": [{"role": "user", "content": "q2"}],
                   "conversation_id": conv_id, "model": "gemma4:26b", "mode": "local"})
    assert all(set(m) <= {"role", "content"} for m in seen["messages"])
    assert "turns" not in json.dumps(seen["messages"])


def test_an_errored_turn_is_still_attributed(env, client, monkeypatch):
    async def failing(messages, model, **kw):
        yield sse("delta", {"text": "partial"})
        yield sse("error", {"message": "Ollama: boom"})
    monkeypatch.setattr(main, "_stream_ollama", failing)
    events = _post(client, {"messages": [{"role": "user", "content": "q"}],
                            "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    rec = conversations.get(events[0][1]["id"])
    assert [t["model"] for t in rec["turns"]] == ["gemma4:26b"]


def test_an_earlier_turn_goes_up_tagged_with_the_model_that_wrote_it(env, client, monkeypatch):
    monkeypatch.setattr(main, "_stream_ollama", _fake_local("local answer"))
    question = "What is the best way to cache an expensive call in Python?"
    events = _post(client, {"messages": [{"role": "user", "content": question}],
                            "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    conv_id = events[0][1]["id"]

    seen = {}

    async def spy(messages, model, max_tokens, **kw):
        seen["messages"], seen["state"] = list(messages), kw.get("state")
        async for ev in _fake_local("claude answer")(messages, model, **kw):
            yield ev
    monkeypatch.setattr(main, "_stream_claude", spy)
    # The same question again, on a different model.
    _post(client, {"messages": [{"role": "user", "content": question}],
                   "conversation_id": conv_id, "model": "claude-opus-4-7", "mode": "api"})

    assistant = [m for m in seen["messages"] if m["role"] == "assistant"]
    assert assistant, "the earlier turn should have ridden along"
    assert assistant[0]["content"][0]["text"].startswith("[router-model gemma4:26b]")
    # And the prompt is told to expect tags, only because there are some.
    assert seen["state"]["tagged"] is True
    assert seen["state"]["model"] == "claude-opus-4-7" and seen["state"]["mode"] == "api"


def test_tags_are_never_written_back_to_the_record(env, client, monkeypatch):
    # The loop's `history` carries the sent (tagged) messages back; folding them
    # in untouched would persist a tag and then tag it again next turn.
    monkeypatch.setattr(main, "_stream_ollama", _fake_local("first"))
    events = _post(client, {"messages": [{"role": "user", "content": "q1"}],
                            "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    conv_id = events[0][1]["id"]
    for i in range(2):
        monkeypatch.setattr(main, "_stream_ollama", _fake_local(f"reply {i}"))
        _post(client, {"messages": [{"role": "user", "content": f"q{i + 2}"}],
                       "conversation_id": conv_id, "model": "gemma4:26b", "mode": "local"})
    rec = conversations.get(conv_id)
    assert "router-model" not in json.dumps(rec["messages"])
    assert len(rec["turns"]) == 3


def test_the_first_turn_of_a_conversation_is_not_tagged(env, client, monkeypatch):
    # Nothing has been answered yet, so there is nothing to attribute and the
    # prompt must not tell the model to look for tags.
    seen = {}

    async def spy(messages, model, **kw):
        seen["messages"], seen["state"] = list(messages), kw.get("state")
        async for ev in _fake_local("ok")(messages, model, **kw):
            yield ev
    monkeypatch.setattr(main, "_stream_ollama", spy)
    _post(client, {"messages": [{"role": "user", "content": "q"}],
                   "conversation_id": "new", "model": "gemma4:26b", "mode": "local"})
    assert "router-model" not in json.dumps(seen["messages"])
    assert seen["state"]["tagged"] is False


def test_a_stateless_call_still_gets_the_static_state(env, client, monkeypatch):
    seen = {}

    async def spy(messages, model, **kw):
        seen["state"] = kw.get("state")
        async for ev in _fake_local("ok")(messages, model, **kw):
            yield ev
    monkeypatch.setattr(main, "_stream_ollama", spy)
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "q"}],
                                   "model": "gemma4:26b", "mode": "local"})
    assert r.status_code == 200
    assert seen["state"]["model"] == "gemma4:26b"
    assert seen["state"]["tagged"] is False      # no record, nothing to attribute
