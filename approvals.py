"""Pending Approve/Deny decisions for agent-mode commands.

Each held tool call gets a fresh approval id (never the provider's tool-call
id, which can repeat), so a decision can only ever settle the call its card
showed. The loop parks on wait() while the card is open; the card's POST lands
in decide(). In-process futures, so this relies on ai-router running a single
uvicorn worker (it does: Dockerfile CMD)."""
import asyncio
import uuid

_pending: dict[str, tuple[str, asyncio.Future]] = {}


def open_request(conversation: str) -> str:
    """A new approval id for one held call in `conversation`."""
    approval_id = uuid.uuid4().hex
    _pending[approval_id] = (conversation, asyncio.get_running_loop().create_future())
    return approval_id


async def wait(approval_id: str, timeout: float) -> str | None:
    """None once matt approves; otherwise why the call won't run."""
    _, fut = _pending[approval_id]
    try:
        return await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError:
        return f"no decision within {round(timeout)}s"
    finally:
        _pending.pop(approval_id, None)


def not_run(refusal: str) -> str:
    """What the model sees for a held tool call that won't run (every mode)."""
    return f"Not run: {refusal}."


def cancel(approval_id: str, reason: str) -> None:
    """Settles a still-pending approval as refused, e.g. when its stream ends."""
    entry = _pending.get(approval_id)
    if entry is not None and not entry[1].done():
        entry[1].set_result(reason)


def decide(conversation: str, approval_id: str, approved: bool) -> bool:
    """Settles a pending approval; False when nothing in `conversation` is
    waiting on that id (already decided, timed out, or the stream is gone)."""
    entry = _pending.get(approval_id)
    if entry is None or entry[0] != conversation or entry[1].done():
        return False
    entry[1].set_result(None if approved else "denied by matt")
    return True
