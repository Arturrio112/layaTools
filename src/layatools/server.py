"""MCP server exposing the gateway as two tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

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
    def decide(profile: str, text: str, min_confidence: float | None = None) -> str:
        """Classify/score `text` with a named decision profile in one cheap call (no generation).
        Answers listed under `escalate` were low confidence: decide those yourself."""
        try:
            result: dict[str, Any] = gateway.decide(profile, text, min_confidence)
        except UnknownProfile as exc:
            result = {"error": str(exc)}
        return json.dumps(result, separators=(",", ":"))

    return server


def main() -> None:
    build_server(build_gateway()).run()
