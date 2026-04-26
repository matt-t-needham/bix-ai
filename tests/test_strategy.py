import asyncio
import strategy


async def fake_ollama(model, messages):
    return "SHORTSUMMARY"


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_below_threshold_is_untouched():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    new, stats = run(strategy.preprocess(body, fake_ollama))
    assert new == body
    assert stats == {"summarised": 0, "skipped": 0, "failed": 0}


def test_above_threshold_is_summarised():
    big = "x" * (strategy.SUMMARY_THRESHOLD_TOKENS * 4 + 100)
    body = {"messages": [{"role": "user", "content": big}]}
    new, stats = run(strategy.preprocess(body, fake_ollama))
    assert stats["summarised"] == 1
    assert strategy.SUMMARY_MARKER in new["messages"][0]["content"]


def test_already_summarised_is_skipped():
    big = strategy.SUMMARY_MARKER + "\n" + ("x" * 20000)
    body = {"messages": [{"role": "user", "content": big}]}
    new, stats = run(strategy.preprocess(body, fake_ollama))
    assert new == body
    assert stats["summarised"] == 0


def test_tool_result_block_is_summarised():
    big = "y" * (strategy.SUMMARY_THRESHOLD_TOKENS * 4 + 100)
    body = {"messages": [{
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "tu_1", "content": big}],
    }]}
    new, stats = run(strategy.preprocess(body, fake_ollama))
    assert stats["summarised"] == 1
    assert strategy.SUMMARY_MARKER in new["messages"][0]["content"][0]["content"]
