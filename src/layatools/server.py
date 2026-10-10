"""MCP server exposing the gateway as two tools."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import decision_log
from .backend import LayaBackend
from .gateway import Gateway, UnknownProfile
from .profiles import profiles_for


def build_gateway(project: Path | None = None) -> Gateway:
    return Gateway(LayaBackend(), profiles_for(project or Path.cwd()))


def build_server(gateway: Gateway) -> MCPServer:
    server = MCPServer("layatools")

    @server.tool()
    def list_decisions() -> str:
        """List the fast decisions available (name, what each answers). Call once, then use `decide`."""
        return json.dumps(gateway.list_decisions(), separators=(",", ":"))

    @server.tool()
    def decide(
        profile: str, text: str, min_confidence: float | None = None, meta: dict[str, Any] | None = None,
        baseline: dict[str, Any] | None = None, log: bool = True, log_path: str | None = None, redact: bool = False,
    ) -> str:
        """Classify/score `text` with a named decision profile in one cheap call (no generation).
        Answers listed under `escalate` were low confidence: decide those yourself.
        The result's `id` can be passed to `record_outcome` once you know the right answer.
        `baseline` is your own answer ({qid: value}) for later agreement reports; `log=false` skips the log,
        `log_path` (absolute) logs to that file, `redact=true` logs a sha256 of `text` instead of the text."""
        try:
            path = decision_log.resolve_path(log_path)
            result: dict[str, Any] = gateway.decide(profile, text, min_confidence)
            decision_id = decision_log.log_decision(
                profile, text, result, {"via": "mcp", **(meta or {})}, path, baseline, redact
            ) if log else None
            if decision_id:
                result = {"id": decision_id, **result}
        except (UnknownProfile, ValueError) as exc:
            result = {"error": str(exc)}
        return json.dumps(result, separators=(",", ":"))

    @server.tool()
    def record_outcome(id: str, outcome: dict[str, Any]) -> str:
        """Record what actually happened for a decision `id`, e.g. {"labels": {"department": "technical"}}
        when the answer was wrong. Used to measure and retrain the profile; costs nothing."""
        try:
            return json.dumps({"logged": decision_log.log_outcome(id, outcome)})
        except ValueError as exc:
            return json.dumps({"error": str(exc)})

    @server.tool()
    def search(question: str, k: int = 5, dir: str = ".") -> str:
        """Find where something is in a codebase before exploring it: returns the best-matching
        files as path, line range and snippet (JSON), best first. Read the top hit first."""
        return search_code(question, k, dir)

    return server


def search_code(question: str, k: int = 5, directory: str = ".", lt: str | None = None) -> str:
    """Runs `lt search --json` (the Rust index); a JSON error when `lt` is missing or fails."""
    exe = lt or shutil.which("lt")
    if not exe:
        return json.dumps({"error": "`lt` is not installed (cargo install --path rust --root ~/.local)"})
    proc = subprocess.run(
        [exe, "-C", directory, "search", question, "-k", str(max(1, min(int(k), 50))), "--json"],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        return json.dumps({"error": (proc.stderr or proc.stdout).strip()[-500:]})
    return proc.stdout.strip()


def main() -> None:
    build_server(build_gateway()).run()
