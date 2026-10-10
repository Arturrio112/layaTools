"""`layatools` command line. Decisions go through a warm local daemon (auto-started)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import calibrate, daemon, decision_log
from .profiles import profiles_for

MAX_CHARS = 6000  # Laya reads a bounded window; keep the head of big files


def _read(path: str) -> str:
    return sys.stdin.read() if path == "-" else Path(path).read_text(errors="replace")[:MAX_CHARS]


def _gateway():
    from .backend import LayaBackend
    from .gateway import Gateway

    return Gateway(LayaBackend(), profiles_for(Path.cwd()))


def labelled_examples(profile: str, path: str | None, log: Path | None = None) -> list[dict]:
    """[{"state": text-or-dict, "labels": {qid: value}}] from a JSONL file, else from the decision log
    (`log`, or the global one). Redacted decisions have no text to re-run, so they are left out."""
    if path:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return [{"state": r.get("state", r.get("text")), "labels": r["labels"]} for r in rows]
    return [
        {"state": r["state"], "labels": r["outcome"]["labels"]}
        for r in decision_log.read(log, profile=profile)
        if "state" in r and isinstance(r["outcome"], dict) and isinstance(r["outcome"].get("labels"), dict)
    ]


def _eval(args, cwd: str) -> dict:
    log = Path(args.log) if args.log else None
    baseline = calibrate.baseline_agreement(decision_log.read(log, profile=args.profile))
    examples = labelled_examples(args.profile, args.labeled, log)
    if not examples:
        if baseline:
            return {"baseline_agreement": baseline}
        return {"error": "no labelled examples (give a JSONL file, or record outcomes with `labels`)"}
    results = daemon.call("/v1/decide", {"profile": args.profile, "items": [e["state"] for e in examples],
                                         "min_confidence": 0.0, "log": False, "cwd": cwd})
    if isinstance(results, dict) and "error" in results:
        return results
    report = calibrate.evaluate(results, [e["labels"] for e in examples], args.target_precision)
    if not args.table:
        for q in report.values():
            q.pop("table")
    if baseline:
        report["baseline_agreement"] = baseline
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="layatools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list available decision profiles")

    run = sub.add_parser("decide", help="run a profile on text or files")
    run.add_argument("profile")
    run.add_argument("text", nargs="?", help="text to decide on")
    run.add_argument("--file", "-f", action="append", default=[], help="file to read ('-' = stdin)")
    run.add_argument("--min-confidence", type=float)
    run.add_argument("--meta", type=json.loads, default=None, help="JSON object stored with the logged decision")
    run.add_argument("--no-log", action="store_true", help="don't record this decision in the decision log")
    run.add_argument("--log-path", help="absolute file to log to instead of the global decision log")
    run.add_argument("--redact", action="store_true", help="log a sha256 of the input, never the input")
    run.add_argument("--baseline", type=json.loads, default=None, help="JSON object: your own answer per question")

    out = sub.add_parser("outcome", help="record what actually happened for a logged decision")
    out.add_argument("id", help="the `id` a logged decision returned")
    out.add_argument("--log", help="the log file the decision was written to (default: the global log)")
    out.add_argument("outcome", type=json.loads, help='JSON object, e.g. \'{"labels": {"tier": "small"}, "passed": true}\'')

    ev = sub.add_parser("eval", help="measure a profile against labelled examples and pick its confidence threshold")
    ev.add_argument("profile")
    ev.add_argument("labeled", nargs="?", help='JSONL of {"text"|"state": ..., "labels": {qid: value}}; default: '
                    "the decision log's decisions whose outcome has `labels`")
    ev.add_argument("--log", help="read decisions from this log file instead of the global one")
    ev.add_argument("--target-precision", type=float, default=0.9)
    ev.add_argument("--table", action="store_true", help="include the full threshold table")

    exp = sub.add_parser("export", help="print logged decisions (joined with outcomes) as JSON lines")
    exp.add_argument("--profile")
    exp.add_argument("--log", help="read decisions from this log file instead of the global one")
    exp.add_argument("--with-outcome", action="store_true", help="only decisions that have an outcome")

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
            payload = {"profile": args.profile, "text": text, "min_confidence": args.min_confidence, "cwd": cwd,
                       "meta": args.meta, "log": not args.no_log, "log_path": args.log_path,
                       "redact": args.redact, "baseline": args.baseline}
            results.append(daemon.call("/v1/decide", payload))
        print(json.dumps(results[0] if len(results) == 1 else results, separators=(",", ":")))
    elif args.cmd == "rank":
        items = {f: _read(f) for f in args.files}
        out = daemon.call("/v1/rank", {"task": args.task, "items": items, "limit": args.limit, "cwd": cwd})
        print(json.dumps(out, separators=(",", ":")))
    elif args.cmd == "outcome":
        if not isinstance(args.outcome, dict):
            parser.error("outcome must be a JSON object")
        print(json.dumps({"logged": decision_log.log_outcome(args.id, args.outcome, Path(args.log) if args.log else None)}))
    elif args.cmd == "eval":
        print(json.dumps(_eval(args, cwd), indent=None if not args.table else 1))
    elif args.cmd == "export":
        for rec in decision_log.read(Path(args.log) if args.log else None, profile=args.profile):
            if rec["outcome"] is not None or not args.with_outcome:
                print(json.dumps(rec, ensure_ascii=False))
    elif args.cmd == "serve":
        from .server import build_server

        build_server(_gateway()).run()
    else:
        daemon.serve(_gateway())


if __name__ == "__main__":
    main()
