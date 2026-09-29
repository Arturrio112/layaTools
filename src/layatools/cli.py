"""`layatools` command line. Decisions go through a warm local daemon (auto-started)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import daemon
from .profiles import profiles_for

MAX_CHARS = 6000  # Laya reads a bounded window; keep the head of big files


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(errors="replace")[:MAX_CHARS]


def _gateway():
    from .backend import LayaBackend
    from .gateway import Gateway

    return Gateway(LayaBackend(), profiles_for(Path.cwd()))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="layatools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available decision profiles")

    run = sub.add_parser("decide", help="run a profile on text or files")
    run.add_argument("profile")
    run.add_argument("text", nargs="?", help="text to decide on")
    run.add_argument("--file", "-f", action="append", default=[], help="file to read ('-' = stdin)")
    run.add_argument("--min-confidence", type=float)

    rank = sub.add_parser("rank", help="rank files by relevance to a task")
    rank.add_argument("task")
    rank.add_argument("files", nargs="+")
    rank.add_argument("--limit", type=int, default=10)

    sub.add_parser("serve", help="serve over MCP (stdio)")
    sub.add_parser("serve-http", help="run the warm local endpoint in the foreground")
    args = parser.parse_args(argv)

    cwd = str(Path.cwd())
    if args.cmd == "list":
        print(json.dumps(_gateway().list_decisions(), separators=(",", ":")))
    elif args.cmd == "decide":
        texts = [args.text] if args.text else []
        texts += [_read(f) for f in args.file]
        if not texts:
            parser.error("give text or --file")
        results = []
        for text in texts:
            payload = {"profile": args.profile, "text": text, "min_confidence": args.min_confidence, "cwd": cwd}
            results.append(daemon.call("/v1/decide", payload))
        print(json.dumps(results[0] if len(results) == 1 else results, separators=(",", ":")))
    elif args.cmd == "rank":
        items = {f: _read(f) for f in args.files}
        out = daemon.call("/v1/rank", {"task": args.task, "items": items, "limit": args.limit, "cwd": cwd})
        print(json.dumps(out, separators=(",", ":")))
    elif args.cmd == "serve":
        from .server import build_server

        build_server(_gateway()).run()
    else:
        daemon.serve(_gateway())


if __name__ == "__main__":
    main()
