"""Fixtures shared by the agent-mode and hand-off tests."""
import json
import socketserver
import sys
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: E402


@pytest.fixture
def fake_runner(tmp_path, monkeypatch):
    """A unix-socket HTTP server answering like the runner; records requests."""
    seen, replies = [], {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _answer(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            seen.append((self.command, self.path, body, self.headers.get("X-Bix-Conversation")))
            status, payload = replies.get(self.path.split("?")[0], (200, {}))
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = _answer

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    sock = tmp_path / "runner.sock"
    server = Server(str(sock), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(config, "AGENT_RUNNER_SOCKET", str(sock))
    yield seen, replies
    server.shutdown()
    server.server_close()
