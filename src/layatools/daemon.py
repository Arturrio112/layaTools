"""Warm local HTTP endpoint so the model loads once, not on every call."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .gateway import Gateway, UnknownProfile
from .embed import Embedder, pack
from .profiles import profiles_for

HOST = "127.0.0.1"
PORT = int(os.environ.get("LAYATOOLS_PORT", "8765"))
URL = f"http://{HOST}:{PORT}"
LOG = Path.home() / ".cache" / "layatools" / "daemon.log"


def make_handler(gateway: Gateway) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one forward pass at a time; the model is not thread-safe
    embedder = Embedder()  # has its own lock, so embedding never waits on a Laya judgement

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"ok": True, "embed_model": embedder.model_name})
            elif self.path.startswith("/v1/decisions"):
                cwd = self.path.partition("?cwd=")[2]
                self._send(200, Gateway(gateway.backend, profiles_for(Path(cwd)) if cwd else gateway.profiles).list_decisions())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/v1/embed":
                    vecs = embedder.embed(req["texts"], query=bool(req.get("query")))
                    return self._send(200, {"model": embedder.model_name, **pack(vecs)})
                gw = Gateway(gateway.backend, profiles_for(Path(req["cwd"])) if req.get("cwd") else gateway.profiles)
                with lock:
                    if self.path == "/v1/decide":
                        out = gw.decide(
                            req["profile"], req.get("state", req.get("text")), req.get("min_confidence")
                        )
                    elif self.path == "/v1/rank":
                        out = gw.rank(
                            req["task"], req["items"], req.get("profile", "relevance"), req.get("limit")
                        )
                    else:
                        return self._send(404, {"error": "not found"})
                self._send(200, out)
            except (UnknownProfile, KeyError, ValueError) as exc:
                self._send(400, {"error": f"{type(exc).__name__}: {exc}"})
            except Exception as exc:  # keep the daemon alive on model errors
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args: Any) -> None:
            pass

    return Handler


def serve(gateway: Gateway) -> None:
    ThreadingHTTPServer((HOST, PORT), make_handler(gateway)).serve_forever()


def _healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=1):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ensure_running(timeout: float = 180.0) -> None:
    """Start the daemon detached if it is not answering; wait until it is."""
    if _healthy():
        return
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "layatools.cli", "serve-http"],
            stdout=log, stderr=log, start_new_session=True,
        )
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _healthy():
            return
        time.sleep(0.5)
    raise RuntimeError(f"layatools daemon did not start; see {LOG}")


def call(path: str, payload: dict[str, Any]) -> Any:
    ensure_running()
    req = urllib.request.Request(
        f"{URL}{path}", json.dumps(payload).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        return json.load(exc)
