"""Routing v2 tests (Phase 6 of PLAN-pi-tools.md).

The load-bearing property: misrouting hard work to the local model is
impossible by construction — Claude signals are checked before any local
rule, and the classifier only routes local on an affirmative EASY; every
failure mode falls open to Claude.
"""
import asyncio

import pytest

import config
import routing


def run(coro):
    return asyncio.run(coro)


async def no_ollama(model, messages):
    raise AssertionError("classifier must not be called for structural decisions")


async def broken_ollama(model, messages):
    raise RuntimeError("ollama down")


def make_classifier(reply):
    async def fake(model, messages):
        return reply
    return fake


def user(text):
    return {"role": "user", "content": text}


# ── Claude structural signals (checked before anything local) ────────────────
@pytest.mark.parametrize("text,rule", [
    ("Can you fix this?\n```python\ndef f():\n    pass\n```", "code-blocks"),
    ("Please write a function that parses nginx logs", "code-gen"),
    ("debug the bug in my script", "code-gen"),
    ("Draft an email to my landlord about the lease", "prose"),
    ("write a blog post about self-hosting", "prose"),
    ("First analyse the logs, then summarise, then draft the fix", "multi-step"),
    ("1. check the config\n2. restart the service\n3. verify", "multi-step"),
])
def test_hard_signals_route_to_claude_structurally(text, rule):
    d = run(routing.decide([user(text)], no_ollama))
    assert d["route"] == "claude"
    assert d["rule"] == rule


def test_long_request_routes_to_claude():
    d = run(routing.decide([user("summarise " + "blah " * 2000)], no_ollama))
    assert d["route"] == "claude"
    assert d["rule"] == "long-request"


def test_large_context_routes_to_claude():
    msgs = []
    for i in range(20):
        msgs.append(user("question %d " % i + "x" * 1000))
        msgs.append({"role": "assistant", "content": "answer %d " % i + "y" * 1000})
    msgs.append(user("ok and now?"))
    d = run(routing.decide(msgs, no_ollama))
    assert d["route"] == "claude"
    assert d["rule"] == "large-context"


# ── Local structural rules ────────────────────────────────────────────────────
def test_short_chat_routes_local():
    d = run(routing.decide([user("what's the capital of France?")], no_ollama))
    assert d == {"route": "local", "rule": "short-chat",
                 "reason": "short conversational query", "claude_model": None}


def test_small_tool_result_digestion_routes_local():
    msgs = [
        user("check the log"),
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "read_log", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "ERROR line\nok\nok"},
        ]},
    ]
    d = run(routing.decide(msgs, no_ollama))
    assert d["route"] == "local"
    assert d["rule"] == "tool-digest"


def test_huge_tool_result_is_not_digested_locally():
    msgs = [
        user("check the log"),
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x" * 40000},
        ]},
    ]
    d = run(routing.decide(msgs, make_classifier("HARD")))
    assert d["route"] == "claude"


# ── Classifier for the ambiguous remainder ────────────────────────────────────
AMBIGUOUS = user(
    "I've been thinking about how we should approach the metrics dashboard and "
    "whether the current nginx log parsing really captures what we care about "
    "when it comes to visitor behaviour across the different bix services and "
    "subdomains, especially now that the tunnel setup changed a bit recently."
)


def test_classifier_easy_routes_local():
    d = run(routing.decide([AMBIGUOUS], make_classifier("EASY")))
    assert d == {"route": "local", "rule": "classifier",
                 "reason": "classified EASY by local model", "claude_model": None}


@pytest.mark.parametrize("reply", ["HARD", "hard.", "I think this is EASY", "", "banana"])
def test_classifier_anything_but_affirmative_easy_fails_open(reply):
    d = run(routing.decide([AMBIGUOUS], make_classifier(reply)))
    assert d["route"] == "claude", f"reply {reply!r} must not route local"


def test_classifier_error_fails_open_to_claude():
    d = run(routing.decide([AMBIGUOUS], broken_ollama))
    assert d["route"] == "claude"
    assert d["rule"] == "classifier-error"
    assert "fail-open" in d["reason"]


def test_hard_signal_never_reaches_classifier():
    # Even a classifier that always says EASY cannot pull code-gen local.
    d = run(routing.decide([user("write a function to sort this list")],
                           make_classifier("EASY")))
    assert d["route"] == "claude"


# ── Cost tiering (claude_model hint) ─────────────────────────────────────────
def test_size_only_route_carries_cheap_model_hint():
    d = run(routing.decide([user("summarise " + "blah " * 2000)], no_ollama))
    assert d["rule"] == "long-request"
    assert d["claude_model"] == config.ROUTING_CHEAP_CLAUDE_MODEL


def test_large_context_without_intent_carries_cheap_model_hint():
    msgs = []
    for i in range(20):
        msgs.append(user("question %d " % i + "x" * 1000))
        msgs.append({"role": "assistant", "content": "answer %d " % i + "y" * 1000})
    msgs.append(user("ok and now?"))
    d = run(routing.decide(msgs, no_ollama))
    assert d["rule"] == "large-context"
    assert d["claude_model"] == config.ROUTING_CHEAP_CLAUDE_MODEL


def test_intent_anywhere_in_history_blocks_downshift():
    # A mid-task "carry on" turn must not demote an ongoing coding conversation.
    msgs = [
        user("write a function that parses nginx logs"),
        {"role": "assistant", "content": "```python\ndef parse():\n    ...\n```"},
        user("ok carry on " + "blah " * 2000),
    ]
    d = run(routing.decide(msgs, no_ollama))
    assert d["route"] == "claude"
    assert d["claude_model"] is None


@pytest.mark.parametrize("text", [
    "Please write a function that parses nginx logs",
    "Draft an email to my landlord about the lease",
    "First analyse the logs, then summarise, then draft the fix",
])
def test_intent_routes_never_downshift(text):
    d = run(routing.decide([user(text)], no_ollama))
    assert d["route"] == "claude"
    assert d["claude_model"] is None


def test_downshift_disabled_by_config(monkeypatch):
    monkeypatch.setattr(config, "ROUTING_CHEAP_CLAUDE_MODEL", "")
    d = run(routing.decide([user("summarise " + "blah " * 2000)], no_ollama))
    assert d["rule"] == "long-request"
    assert d["claude_model"] is None
