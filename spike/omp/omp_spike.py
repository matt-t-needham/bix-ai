#!/usr/bin/env python3
"""omp RPC spike for bix-ai — see PLAN-omp-spike.md.

Drives `omp --mode rpc` as a subprocess over its JSONL stdio protocol, exposes
bix-ai-style host tools, and runs a fixed set of checks (S1–S9). Stdlib only,
asyncio throughout — this is also the prototype of the async client a real
`streaming/omp.py` backend would use (the bundled `omp_rpc` package is sync).

Nothing here touches the running bix-ai service or the prod tree:
  * omp runs with cwd = a throwaway work dir, an isolated --profile, and every
    foreign config-discovery source disabled (see spike-config.yml);
  * built-in tools are off (--no-tools); the only tools are the host tools
    below, all read-only except `stage_write`, which is a recording stub.

Usage (from the bix-ai repo root, on the PC):
    python3 spike/omp/omp_spike.py --claude-model anthropic/claude-sonnet-4-5 \
        --local-model ollama/qwen3.5:9b
    python3 spike/omp/omp_spike.py --only S1,S3      # subset
    python3 spike/omp/omp_spike.py --omp "python3 spike/omp/fake_omp.py"  # offline self-test

Writes spike/omp/results/<timestamp>.json; paste that back for review.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import shlex
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Discovery sources that would otherwise pull MCP servers / skills / hooks from
# other agents' config (incl. bix_mcp via CLAUDE_CONFIG_DIR). Kept in sync with
# spike-config.yml; also stripped from the child env below.
_STRIP_ENV = ("CLAUDE_CONFIG_DIR",)


# ── Async RPC client ──────────────────────────────────────────────────────────

class RpcError(RuntimeError):
    pass


@dataclass
class PromptRun:
    """Everything observed between sending a prompt and its prompt_result."""
    id: str
    text: str = ""
    tool_starts: list[dict] = field(default_factory=list)
    tool_ends: list[dict] = field(default_factory=list)
    host_calls: list[dict] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    result: dict | None = None
    t0: float = 0.0
    ttft_s: float | None = None
    elapsed_s: float | None = None
    usage: list[dict] = field(default_factory=list)


HostHandler = Callable[[dict], Awaitable[str]]


class OmpRpc:
    """Minimal asyncio client for `omp --mode rpc` (protocol v1, no chunking).

    Stays on protocol v1 deliberately: frames are capped at 1 MiB and oversized
    events get fields elided, which is fine for a spike. A production client
    should negotiate v2 and reassemble `rpc_chunk` frames (see docs/rpc.md).
    """

    def __init__(self, argv: list[str], cwd: Path, env: dict[str, str],
                 host_tools: dict[str, tuple[dict, HostHandler]], log: Callable[[str], None]):
        self.argv, self.cwd, self.env = argv, cwd, env
        self.host_tools = host_tools
        self.log = log
        self.proc: asyncio.subprocess.Process | None = None
        self.ready: dict | None = None
        self._ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future] = {}
        self._run: PromptRun | None = None
        self._run_done: asyncio.Event | None = None
        self._reader: asyncio.Task | None = None
        self._stderr_tail: list[str] = []
        self._ready_evt = asyncio.Event()
        self.unknown_frames: list[str] = []

    async def start(self, ready_timeout: float = 60) -> dict:
        self.proc = await asyncio.create_subprocess_exec(
            *self.argv, cwd=self.cwd, env=self.env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=8 * 1024 * 1024)
        self._reader = asyncio.create_task(self._read_stdout())
        asyncio.create_task(self._read_stderr())
        # Wait for the ready frame, but fail fast if the process dies first.
        ready = asyncio.create_task(self._ready_evt.wait())
        await asyncio.wait({ready, self._reader}, timeout=ready_timeout,
                           return_when=asyncio.FIRST_COMPLETED)
        if not ready.done():
            ready.cancel()
            await asyncio.sleep(0.2)   # let stderr drain into the tail
            why = "exited" if self._reader.done() else f"sent no ready frame within {ready_timeout}s"
            raise RpcError(f"omp {why}; stderr tail: {self.stderr_tail()}")
        return self.ready or {}

    def stderr_tail(self) -> str:
        return " | ".join(self._stderr_tail[-8:])

    async def close(self) -> int | None:
        if not self.proc:
            return None
        if self.proc.stdin and not self.proc.stdin.is_closing():
            self.proc.stdin.close()
        try:
            return await asyncio.wait_for(self.proc.wait(), 15)
        except asyncio.TimeoutError:
            self.proc.kill()
            return await self.proc.wait()

    async def _send(self, frame: dict) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write((json.dumps(frame) + "\n").encode())
        await self.proc.stdin.drain()

    async def call(self, type_: str, timeout: float = 60, **payload: Any) -> Any:
        rid = f"req_{next(self._ids)}"
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        await self._send({"id": rid, "type": type_, **payload})
        resp = await asyncio.wait_for(fut, timeout)
        if not resp.get("success"):
            raise RpcError(f"{type_} failed: {resp.get('error')} (code={resp.get('code')})")
        return resp.get("data")

    async def prompt(self, message: str, timeout: float = 600) -> PromptRun:
        """Send a prompt and wait for its prompt_result (the agent's yield)."""
        if self._run is not None:
            raise RpcError("a prompt is already in flight")
        rid = f"req_{next(self._ids)}"
        run = PromptRun(id=rid, t0=time.monotonic())
        self._run, self._run_done = run, asyncio.Event()
        fut = asyncio.get_running_loop().create_future()
        self._pending[rid] = fut
        try:
            await self._send({"id": rid, "type": "prompt", "message": message})
            ack = await asyncio.wait_for(fut, 60)
            if not ack.get("success"):
                run.result = {"status": "error", "error": {"message": ack.get("error")}}
            elif (ack.get("data") or {}).get("agentInvoked") is False:
                run.result = {"status": "completed", "agentInvoked": False}
            else:
                await asyncio.wait_for(self._run_done.wait(), timeout)
        finally:
            run.elapsed_s = round(time.monotonic() - run.t0, 2)
            self._run, self._run_done = None, None
        return run

    async def _read_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        async for line in self.proc.stderr:
            s = line.decode(errors="replace").rstrip()
            if s:
                self._stderr_tail.append(s)
                del self._stderr_tail[:-50]

    async def _read_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        async for line in self.proc.stdout:
            if not line.strip():
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                self.log(f"non-JSON stdout line: {line[:200]!r}")
                continue
            try:
                await self._dispatch(frame)
            except Exception as e:  # keep reading; surface the failure
                self.log(f"dispatch error on {frame.get('type')}: {e!r}")
        # stdout closed: fail anything still waiting so callers don't hang
        err = RpcError(f"omp exited (code={self.proc.returncode}); stderr tail: {self.stderr_tail()}")
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(err)
        if self._run_done:
            self._run_done.set()

    async def _dispatch(self, f: dict) -> None:
        t = f.get("type")
        run = self._run
        if t == "ready":
            self.ready = f
            self._ready_evt.set()
        elif t == "response":
            fut = self._pending.pop(f.get("id"), None)
            if fut and not fut.done():
                fut.set_result(f)
            elif run and f.get("id") == run.id and not f.get("success"):
                # late async failure of an already-acked prompt
                run.result = {"status": "error", "error": {"message": f.get("error")}}
        elif t == "prompt_result":
            if run and f.get("id") == run.id:
                run.result = f
                self._run_done.set()
        elif t == "host_tool_call":
            asyncio.create_task(self._serve_host_tool(f))
        elif t == "host_tool_cancel":
            self.log(f"host_tool_cancel for {f.get('targetId')}")
        elif t == "extension_ui_request":
            # --no-ui should prevent these; if one arrives, refuse it (fail closed)
            self.log(f"unexpected extension_ui_request: {json.dumps(f)[:300]}")
            await self._send({"type": "extension_ui_response", "id": f.get("id"), "cancelled": True})
        elif run is not None:
            run.event_types.append(t)
            if t == "message_update":
                ev = f.get("assistantMessageEvent") or {}
                if ev.get("type") == "text_delta":
                    if run.ttft_s is None:
                        run.ttft_s = round(time.monotonic() - run.t0, 2)
                    run.text += ev.get("delta", "")
            elif t == "tool_execution_start":
                run.tool_starts.append({"name": f.get("toolName"), "args": f.get("args")})
            elif t == "tool_execution_end":
                run.tool_ends.append({"name": f.get("toolName"), "isError": bool(f.get("isError")),
                                      "result": _clip(json.dumps(f.get("result")), 400)})
            elif t == "message_end":
                msg = f.get("message") or {}
                if msg.get("role") == "assistant" and msg.get("usage"):
                    run.usage.append(msg["usage"])
        elif t not in ("available_commands_update", "session_settled", "notice"):
            self.unknown_frames.append(t)

    async def _serve_host_tool(self, f: dict) -> None:
        name, args = f.get("toolName"), f.get("arguments") or {}
        if self._run:
            self._run.host_calls.append({"name": name, "args": args})
        entry = self.host_tools.get(name)
        try:
            if entry is None:
                raise RpcError(f"unknown host tool {name}")
            text, is_error = await entry[1](args), False
        except Exception as e:
            text, is_error = f"{type(e).__name__}: {e}", True
        frame = {"type": "host_tool_result", "id": f["id"],
                 "result": {"content": [{"type": "text", "text": text}]}}
        if is_error:
            frame["isError"] = True
        await self._send(frame)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"


# ── Host tools ────────────────────────────────────────────────────────────────

STAGED: list[dict] = []   # stage_write calls recorded, never applied


def spike_tools(root: Path) -> dict[str, tuple[dict, HostHandler]]:
    """Read-only fs tools jailed to `root`, plus a recording stage_write stub."""

    def _jail(p: str) -> Path:
        rp = (root / p).resolve() if not os.path.isabs(p) else Path(p).resolve()
        if rp != root and root not in rp.parents:
            raise PermissionError(f"{p} is outside {root}")
        return rp

    async def list_directory(a: dict) -> str:
        p = _jail(a.get("path") or str(root))
        return "\n".join(sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir()))

    async def read_file(a: dict) -> str:
        p = _jail(a["path"])
        return p.read_text(errors="replace")[:200_000]

    async def stage_write(a: dict) -> str:
        STAGED.append({"path": a.get("path"), "bytes": len(a.get("content") or "")})
        return f"Proposed write to {a.get('path')} staged for human review (spike: recorded, not written)."

    def schema(props: dict, req: list[str]) -> dict:
        return {"type": "object", "properties": props, "required": req, "additionalProperties": False}

    path_prop = {"path": {"type": "string", "description": f"Path under {root}"}}
    return {
        "list_directory": ({"description": f"List entries in a directory under {root}.",
                            "parameters": schema(path_prop, ["path"])}, list_directory),
        "read_file": ({"description": f"Read a text file under {root}.",
                       "parameters": schema(path_prop, ["path"])}, read_file),
        "stage_write": ({"description": "Propose writing a file. A human reviews and applies it later; "
                                        "nothing is written immediately.",
                         "parameters": schema({**path_prop, "content": {"type": "string"}},
                                              ["path", "content"])}, stage_write),
    }


def bix_tools(names: list[str]) -> dict[str, tuple[dict, HostHandler]]:
    """The real bix-ai TOOL_TABLE entries (needs the repo venv + config env)."""
    sys.path.insert(0, str(REPO))
    import tools  # noqa: E402  (bix-ai's tools.py)
    table = {t["name"]: t for t in tools.TOOL_TABLE}
    out = {}
    for n in names:
        if n == "stage_write":
            continue   # never wire the real one into a spike
        t = table[n]
        out[n] = ({"description": t["description"], "parameters": t["input_schema"]},
                  (lambda name: (lambda a: tools._execute_tool(name, a)))(n))
    return out


def host_tool_defs(ht: dict[str, tuple[dict, HostHandler]]) -> list[dict]:
    # loadMode "essential": host tools default to "discoverable" (hidden behind
    # tool search) — bix's tools must be directly visible to the model.
    return [{"name": n, "label": n, "loadMode": "essential", **d} for n, (d, _) in ht.items()]


# ── Checks ────────────────────────────────────────────────────────────────────

@dataclass
class Ctx:
    args: argparse.Namespace
    work: Path
    sandbox: Path
    results: dict
    host: dict

    def omp_argv(self, *, model: str | None, session_dir: Path | None,
                 builtin_tools: str | None = None) -> list[str]:
        # omp runs with cwd = the work dir, so pin relative paths in --omp now.
        argv = [str(Path(a).resolve()) if not os.path.isabs(a) and Path(a).exists() else a
                for a in shlex.split(self.args.omp)] + [
            "--mode", "rpc", "--no-ui", "--no-lsp", "--no-pty", "--no-skills", "--no-rules",
            "--no-extensions", "--approval-mode", "always-ask",
            "--config", str(HERE / "spike-config.yml"),
        ]
        if self.args.profile:
            argv += ["--profile", self.args.profile]
        argv += ["--tools", builtin_tools] if builtin_tools else ["--no-tools"]
        argv += ["--session-dir", str(session_dir)] if session_dir else ["--no-session"]
        if model:
            argv += ["--model", model]
        return argv

    def env(self, **extra: str) -> dict[str, str]:
        e = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV}
        e.update(extra)
        return e

    async def client(self, *, model: str | None, session_dir: Path | None = None,
                     builtin_tools: str | None = None, env: dict | None = None,
                     host: dict | None = None) -> OmpRpc:
        c = OmpRpc(self.omp_argv(model=model, session_dir=session_dir, builtin_tools=builtin_tools),
                   self.work, env or self.env(), self.host if host is None else host, log)
        await c.start()
        if c.host_tools:
            data = await c.call("set_host_tools", tools=host_tool_defs(c.host_tools))
            got = set((data or {}).get("toolNames", []))
            if got != set(c.host_tools):
                raise RpcError(f"set_host_tools registered {sorted(got)}, expected {sorted(c.host_tools)}")
        return c


def log(msg: str) -> None:
    print(f"    · {msg}", file=sys.stderr)


def summarize(run: PromptRun) -> dict:
    return {"status": (run.result or {}).get("status"), "error": (run.result or {}).get("error"),
            "elapsed_s": run.elapsed_s, "ttft_s": run.ttft_s, "text": _clip(run.text, 600),
            "tool_starts": run.tool_starts, "tool_ends": run.tool_ends, "host_calls": run.host_calls,
            "usage": run.usage}


def roundtrip_prompt(c: Ctx) -> str:
    return (f"Use the list_directory tool on {c.sandbox} and then read_file on the file whose name "
            "starts with 'note'. Reply with the secret word in that file, and the exact number of "
            "entries the directory listing returned.")


def roundtrip_ok(run: PromptRun) -> tuple[bool, str]:
    names = [h["name"] for h in run.host_calls]
    if (run.result or {}).get("status") != "completed":
        return False, f"prompt status {(run.result or {}).get('status')}"
    if "list_directory" not in names or "read_file" not in names:
        return False, f"expected list_directory + read_file host calls, got {names}"
    if "MARMALADE" not in run.text.upper():
        return False, "answer did not contain the secret word"
    return True, f"host calls {names}"


async def s1_handshake(c: Ctx) -> dict:
    cli = await c.client(model=None)
    try:
        state = await cli.call("get_state")
        models = await cli.call("get_available_models", timeout=120) or []
        ids = [f"{m.get('provider')}/{m.get('id')}" for m in (models.get("models", models)
                                                              if isinstance(models, dict) else models)]
        by_provider: dict[str, int] = {}
        for i in ids:
            by_provider[i.split("/")[0]] = by_provider.get(i.split("/")[0], 0) + 1
        ok = "ollama" in by_provider
        return {"pass": ok, "ready": cli.ready, "providers": by_provider,
                "ollama_models": [i for i in ids if i.startswith("ollama/")],
                "state_keys": sorted((state or {}).keys()),
                "note": "" if ok else "no ollama models discovered — check OLLAMA_HOST/OLLAMA_BASE_URL"}
    finally:
        await cli.close()


async def _roundtrip(c: Ctx, model: str) -> dict:
    cli = await c.client(model=model)
    try:
        run = await cli.prompt(roundtrip_prompt(c), timeout=c.args.turn_timeout)
        ok, why = roundtrip_ok(run)
        builtin = [t["name"] for t in run.tool_starts if t["name"] not in c.host]
        if builtin:
            ok, why = False, f"non-host tools executed: {builtin}"
        return {"pass": ok, "why": why, "model": model, **summarize(run),
                "unknown_frames": sorted(set(cli.unknown_frames))}
    finally:
        await cli.close()


async def s3_claude_roundtrip(c: Ctx) -> dict:
    return await _roundtrip(c, c.args.claude_model)


async def s4_local_roundtrip(c: Ctx) -> dict:
    return await _roundtrip(c, c.args.local_model)


async def s5_s6_history(c: Ctx) -> dict:
    """S5: tool evidence survives into the next prompt (same process).
    S6: …and survives a process restart via --session-dir resume."""
    sdir = c.work / "sessions" / "history"
    q = ("Without calling any tools again: what was the secret word, and what were the "
         "directory entry names you saw earlier?")
    cli = await c.client(model=c.args.claude_model, session_dir=sdir)
    try:
        r1 = await cli.prompt(roundtrip_prompt(c), timeout=c.args.turn_timeout)
        r2 = await cli.prompt(q, timeout=c.args.turn_timeout)
    finally:
        await cli.close()
    s5 = "MARMALADE" in r2.text.upper() and not r2.host_calls
    cli = await c.client(model=c.args.claude_model, session_dir=sdir)
    try:
        opened = await cli.call("open_session", sessionDir=str(sdir))
        r3 = await cli.prompt(q, timeout=c.args.turn_timeout)
    finally:
        await cli.close()
    s6 = bool((opened or {}).get("resumed")) and "MARMALADE" in r3.text.upper() and not r3.host_calls
    return {"pass": s5 and s6, "S5_same_process": s5, "S6_after_restart": s6,
            "open_session": opened, "first": summarize(r1), "followup": summarize(r2),
            "resumed": summarize(r3)}


async def s7_fail_closed(c: Ctx) -> dict:
    """Enable built-in write+bash with always-ask and no allow policy: any attempt
    must fail (no UI to approve) and the canary file must never appear."""
    canary = c.sandbox / "CANARY-should-not-exist.txt"
    cli = await c.client(model=c.args.claude_model, builtin_tools="write,bash", host={})
    try:
        run = await cli.prompt(
            f"Create the file {canary} containing the word hello. Use the write tool; if that "
            f"fails, try bash with `echo hello > {canary}`. Report exactly what happened.",
            timeout=c.args.turn_timeout)
    finally:
        await cli.close()
    attempted = [t["name"] for t in run.tool_starts]
    errored = [t["name"] for t in run.tool_ends if t["isError"]]
    exists = canary.exists()
    if exists:
        verdict, ok = "FAIL — canary written: approval did not fail closed", False
    elif not attempted:
        verdict, ok = "INCONCLUSIVE — model never attempted a tool call", None
    else:
        ok = set(attempted) <= set(errored)
        verdict = "pass — every attempt was refused" if ok else "FAIL — an attempt did not error"
    return {"pass": ok, "verdict": verdict, "canary_exists": exists, **summarize(run)}


async def s8_abort(c: Ctx) -> dict:
    cli = await c.client(model=c.args.claude_model)
    try:
        task = asyncio.create_task(cli.prompt(
            "Write a 2000-word essay on the history of the bicycle.", timeout=120))
        await asyncio.sleep(c.args.abort_after)
        t_abort = time.monotonic()
        await cli.call("abort", timeout=30)
        run = await task
        lag = round(time.monotonic() - t_abort, 2)
    finally:
        await cli.close()
    status = (run.result or {}).get("status")
    return {"pass": status == "aborted" and lag < 10, "status": status, "abort_to_result_s": lag,
            "partial_chars": len(run.text)}


async def s9_provider_error(c: Ctx) -> dict:
    """A bad Anthropic key must surface as status=error — never a clean completion."""
    provider = c.args.claude_model.split("/")[0]
    if provider != "anthropic":
        return {"pass": None, "note": f"skipped: claude model provider is {provider}, not anthropic"}
    cli = await c.client(model=c.args.claude_model,
                         env=c.env(ANTHROPIC_API_KEY="sk-ant-invalid-spike-key"))
    try:
        run = await cli.prompt("Say hi.", timeout=120)
    finally:
        await cli.close()
    status = (run.result or {}).get("status")
    return {"pass": status == "error", "status": status, "error": (run.result or {}).get("error"),
            "note": "if this used OAuth/stored creds instead of the env key, it may complete — "
                    "check auth source in S1 state"}


CHECKS: dict[str, Callable[[Ctx], Awaitable[dict]]] = {
    "S1": s1_handshake,
    "S3": s3_claude_roundtrip,
    "S4": s4_local_roundtrip,
    "S5": s5_s6_history,   # also reports S6
    "S7": s7_fail_closed,
    "S8": s8_abort,
    "S9": s9_provider_error,
}


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--omp", default="omp", help="omp command (default: omp on PATH)")
    ap.add_argument("--profile", default="bixspike", help="isolated omp profile ('' to disable)")
    ap.add_argument("--claude-model", default="anthropic/claude-sonnet-4-5")
    ap.add_argument("--local-model", default="ollama/qwen3.5:9b")
    ap.add_argument("--only", default="", help="comma-separated check ids, e.g. S1,S3")
    ap.add_argument("--turn-timeout", type=float, default=600,
                    help="per-prompt timeout (local models on this host are slow)")
    ap.add_argument("--abort-after", type=float, default=3)
    ap.add_argument("--bix-tools", default="",
                    help="also expose real bix-ai tools, e.g. list_log_sources,read_log "
                         "(needs repo venv + FS_ROOT/DATA_DIR env)")
    ap.add_argument("--keep", action="store_true", help="keep the work dir")
    args = ap.parse_args()

    if not shutil.which(shlex.split(args.omp)[0]):
        print(f"omp command not found: {args.omp}", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="omp-spike-"))
    sandbox = work / "sandbox"
    sandbox.mkdir()
    (sandbox / "note-1.txt").write_text("The secret word is marmalade.\n")
    (sandbox / "alpha.txt").write_text("alpha\n")
    (sandbox / "sub").mkdir()
    host = spike_tools(sandbox)
    if args.bix_tools:
        host.update(bix_tools([n.strip() for n in args.bix_tools.split(",") if n.strip()]))

    ctx = Ctx(args, work, sandbox, {}, host)
    wanted = [s.strip().upper() for s in args.only.split(",") if s.strip()] or list(CHECKS)
    report: dict[str, Any] = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                              "argv_example": ctx.omp_argv(model=args.claude_model, session_dir=None),
                              "checks": {}}
    for cid in wanted:
        fn = CHECKS.get(cid)
        if fn is None:
            print(f"unknown check {cid}", file=sys.stderr)
            continue
        print(f"[{cid}] {fn.__doc__.strip().splitlines()[0] if fn.__doc__ else fn.__name__}", file=sys.stderr)
        t0 = time.monotonic()
        try:
            res = await fn(ctx)
        except Exception as e:
            res = {"pass": False, "exception": f"{type(e).__name__}: {e}"}
        res["check_s"] = round(time.monotonic() - t0, 1)
        report["checks"][cid] = res
        mark = {True: "PASS", False: "FAIL", None: "SKIP/INCONCLUSIVE"}[res.get("pass")]
        print(f"[{cid}] {mark} {res.get('why') or res.get('verdict') or res.get('note') or res.get('exception') or ''}",
              file=sys.stderr)
    report["staged_writes"] = STAGED

    out = HERE / "results" / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport: {out}", file=sys.stderr)
    if args.keep:
        print(f"work dir kept: {work}", file=sys.stderr)
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if all(r.get("pass") is not False for r in report["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
