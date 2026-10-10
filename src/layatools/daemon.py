"""Warm local HTTP endpoint so the model loads once, not on every call."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import decision_log
from .gateway import Gateway, UnknownProfile
from .embed import Embedder, pack
from .profiles import profiles_for

HOST = "127.0.0.1"
PORT = int(os.environ.get("LAYATOOLS_PORT", "8765"))
URL = f"http://{HOST}:{PORT}"
LOG = Path.home() / ".cache" / "layatools" / "daemon.log"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def local_host(host: str | None) -> bool:
    """True if a Host header names this machine. Browsers always send the page's own host, so this
    stops a web page from reaching the daemon through DNS rebinding; non-browser clients may omit it."""
    if not host:
        return True
    name = host.rsplit(":", 1)[0] if not host.endswith("]") else host
    return name.lower() in LOCAL_HOSTS


def make_handler(gateway: Gateway, embedder: Any = None) -> type[BaseHTTPRequestHandler]:
    lock = threading.Lock()  # one forward pass at a time; the model is not thread-safe
    embedder = embedder or Embedder()  # has its own lock, so embedding never waits on a Laya judgement

    def gateway_for(cwd: str | None) -> Gateway:
        return Gateway(gateway.backend, profiles_for(Path(cwd)) if cwd else gateway.profiles)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _forbidden(self) -> bool:
            if local_host(self.headers.get("Host")):
                return False
            self._send(403, {"error": "forbidden host"})
            return True

        def do_GET(self) -> None:
            if self._forbidden():
                return
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/health":
                self._send(200, {"ok": True, "embed_model": embedder.model_name})
            elif url.path == "/v1/decisions":
                cwd = urllib.parse.parse_qs(url.query).get("cwd", [None])[0]
                self._send(200, gateway_for(cwd).list_decisions())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self._forbidden():
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if not isinstance(req, dict):
                    raise ValueError("request body must be a JSON object")
                if self.path == "/v1/embed":
                    vecs = embedder.embed(req["texts"], query=bool(req.get("query")))
                    return self._send(200, {"model": embedder.model_name, **pack(vecs)})
                gw = gateway_for(req.get("cwd"))
                if self.path == "/v1/outcome":
                    path = decision_log.resolve_path(req.get("log_path"))
                    return self._send(200, {"ok": True, "logged": decision_log.log_outcome(req["id"], req["outcome"], path)})
                with lock:
                    if self.path == "/v1/decide":
                        out = decide(gw, req)
                    elif self.path == "/v1/rank":
                        out = rank(gw, req)
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


def _log_opts(req: dict[str, Any]) -> tuple[bool, Path | None, bool]:
    """(log?, per-request log file or None for the global log, redact?). `log_path` is validated here."""
    return bool(req.get("log", True)), decision_log.resolve_path(req.get("log_path")), bool(req.get("redact"))


def _per_item(req: dict[str, Any], key: str, n: int) -> list[Any]:
    values = req.get(key)
    if values is None:
        return [None] * n
    if not isinstance(values, list) or len(values) != n:
        raise ValueError(f"`{key}` must be a list with one entry per item")
    return values


def decide(gw: Gateway, req: dict[str, Any]) -> Any:
    """One item (`text` or `state`) or a batch (`items`: a list of texts/states, answered in order).
    Each answer is logged unless `"log": false`, and carries its log `id` for outcomes. Log options:
    `log_path` (append there instead of the global log), `redact` (store a sha256 of the input, not the input),
    `meta` and `baseline` (the caller's own answer, shaped like `answers`); a batch may instead give
    `metas` / `baselines`, lists with one entry per item (a per-item meta is merged over `meta`)."""
    batch = "items" in req
    items = req["items"] if batch else [req.get("state", req.get("text"))]
    if not isinstance(items, list) or any(not isinstance(i, (str, dict)) for i in items):
        raise ValueError("`items` must be a list of texts or state objects")
    if not batch and items[0] is None:
        raise KeyError("text")
    do_log, path, redact = _log_opts(req)
    metas = _per_item(req, "metas", len(items))
    baselines = _per_item(req, "baselines", len(items))
    if not batch:
        baselines = [req.get("baseline")]
    elif req.get("baseline") is not None:
        raise ValueError("use `baselines` (one per item) with `items`")
    out = []
    for item, item_meta, baseline in zip(items, metas, baselines):
        result = gw.decide(req["profile"], item, req.get("min_confidence"))
        if do_log:
            meta = {**(req.get("meta") or {}), **(item_meta or {})}
            decision_id = decision_log.log_decision(req["profile"], item, result, meta, path, baseline, redact)
            if decision_id:
                result = {"id": decision_id, **result}
        out.append(result)
    return out if batch else out[0]


def rank(gw: Gateway, req: dict[str, Any]) -> Any:
    """Rank `items` ({id: text}) against `task`. With `"log": true` (opt-in) each ranked candidate is logged
    as a decision of the rank profile (answer `relevance`: value = score, conf = p) and its response row gains
    `log_id`, so outcomes can be recorded against it. `log_path`, `redact` and `meta` work as for decide."""
    profile = req.get("profile", "relevance")
    out = gw.rank(req["task"], req["items"], profile, req.get("limit"))
    if req.get("log") is True:
        _, path, redact = _log_opts(req)
        for row in out:
            result = {"answers": {"relevance": {"value": row["score"], "conf": row["p"]}}, "escalate": []}
            state = {"task": req["task"], "content": req["items"][row["id"]]}
            meta = {**(req.get("meta") or {}), "via": "rank", "item": row["id"]}
            log_id = decision_log.log_decision(profile, state, result, meta, path, None, redact)
            if log_id:
                row["log_id"] = log_id
    return out


def serve(gateway: Gateway) -> None:
    decision_log.auto_prune()
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
