"""Append-only log of decisions and their later outcomes, so a profile can be measured and retrained.

Every logged decision gets an id (returned to the caller as `id`). When the caller later learns how
things went (the ticket passed, the file was the right one), it records an outcome against that id.
`read` joins the two; `layatools eval`/`export` build on that.

One JSON object per line:
  {"type": "decision", "id", "ts", "profile", "state", "answers", "escalate", "meta"[, "baseline"]}
  a redacted decision has `"redacted": true` and `"state_sha256"` instead of `state`
  {"type": "outcome",  "id", "ts", "outcome"}
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path.home() / ".local" / "share" / "layatools" / "decisions.jsonl"
MAX_FIELD_CHARS = 6000  # Laya reads a bounded window anyway; keep the log from growing without bound

_lock = threading.Lock()


def log_path() -> Path | None:
    """`LAYATOOLS_DECISION_LOG`: a file path, or `off` to disable logging. Default under ~/.local/share."""
    value = os.environ.get("LAYATOOLS_DECISION_LOG")
    if value and value.lower() in {"off", "0", "none", "false"}:
        return None
    return Path(value).expanduser() if value else DEFAULT_PATH


def log_root() -> Path:
    """Per-request log files must live under this directory: `LAYATOOLS_LOG_ROOT`, default ~/.local/share/layatools/logs."""
    value = os.environ.get("LAYATOOLS_LOG_ROOT")
    return (Path(value).expanduser() if value else DEFAULT_PATH.parent / "logs").resolve()


def resolve_path(value: Any) -> Path | None:
    """A per-request log file: None when not given. Relative paths are relative to the log root; the real path
    (symlinks resolved, `..` collapsed) must be a file inside the root, else ValueError naming the root.
    Subdirectories are created on first write."""
    if value is None:
        return None
    root = log_root()
    if not isinstance(value, str) or not value:
        raise ValueError(f"`log_path` must be a non-empty string naming a file under {root}")
    path = (root / Path(value).expanduser()).resolve()  # an absolute value replaces the root, then is checked below
    if root not in path.parents:
        raise ValueError(f"`log_path` must resolve to a file inside the log root {root} (LAYATOOLS_LOG_ROOT)")
    return path


def state_hash(state: Any) -> str:
    text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clip(state: Any) -> Any:
    if isinstance(state, str):
        return state[:MAX_FIELD_CHARS]
    if isinstance(state, dict):
        return {k: _clip(v) for k, v in state.items()}
    return state


def _append(record: dict[str, Any], path: Path | None) -> bool:
    if path is None:
        return False
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as f:
            # A torn last line (a writer killed mid-record) has no newline: start fresh, or this record is lost too.
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    line = "\n" + line
            f.write(line.encode("utf-8"))
    return True


def log_decision(
    profile: str, state: Any, result: dict[str, Any], meta: dict[str, Any] | None = None, path: Path | None = None,
    baseline: dict[str, Any] | None = None, redact: bool = False,
) -> str | None:
    """Record one decision; returns its id, or None when logging is off. `baseline` is the caller's own
    answer (same shape as `answers`). With `redact`, only a sha256 of the input is stored, never the input."""
    path = path if path is not None else log_path()
    decision_id = uuid.uuid4().hex
    record = {
        "type": "decision", "id": decision_id, "ts": time.time(), "profile": profile,
        "answers": result.get("answers", {}), "escalate": result.get("escalate", []), "meta": meta or {},
    }
    if redact:
        record.update(redacted=True, state_sha256=state_hash(state))
    else:
        record["state"] = _clip(state)
    if baseline is not None:
        if not isinstance(baseline, dict):
            raise ValueError("`baseline` must be a JSON object like `answers`")
        record["baseline"] = baseline
    return decision_id if _append(record, path) else None


def log_outcome(decision_id: str, outcome: dict[str, Any], path: Path | None = None) -> bool:
    """Attach what actually happened to a decision. Several outcomes per id are fine; the last one wins."""
    if not decision_id or not isinstance(outcome, dict):
        raise ValueError("an outcome needs a decision id and a JSON object")
    path = path if path is not None else log_path()
    return _append({"type": "outcome", "id": decision_id, "ts": time.time(), "outcome": outcome}, path)


def read(path: Path | None = None, profile: str | None = None) -> list[dict[str, Any]]:
    """Decisions (optionally of one profile) in log order, each with its latest `outcome` (or None)."""
    path = path if path is not None else log_path()
    if path is None or not path.exists():
        return []
    decisions: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, Any] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line from a crash must not hide the rest
            if rec.get("type") == "decision":
                decisions[rec["id"]] = rec
            elif rec.get("type") == "outcome":
                outcomes[rec["id"]] = rec.get("outcome")
    return [
        {**d, "outcome": outcomes.get(i)}
        for i, d in decisions.items()
        if profile is None or d.get("profile") == profile
    ]


def _count(lines: list[tuple[str, dict[str, Any] | None]]) -> dict[str, int]:
    recs = [r for _, r in lines if r]
    return {"decisions": sum(r.get("type") == "decision" for r in recs),
            "outcomes": sum(r.get("type") == "outcome" for r in recs), "bytes": sum(len(t.encode()) for t, _ in lines)}


def prune(
    path: Path, older_than: float, keep_labelled: bool = True, max_bytes: int | None = None,
    dry_run: bool = False, now: float | None = None,
) -> dict[str, Any]:
    """Drop decisions older than `older_than` seconds (with their outcome rows), except labelled ones when
    `keep_labelled`; then, if `max_bytes` is set, drop the oldest remaining unlabelled decisions until the file
    fits. Torn lines are dropped. The file is rewritten via a temp file + rename, never truncated in place."""
    cutoff = (time.time() if now is None else now) - older_than
    report: dict[str, Any] = {"path": str(path), "dry_run": dry_run}
    with _lock:
        if not path.exists():
            return {**report, "before": {"decisions": 0, "outcomes": 0, "bytes": 0}, "after": {"decisions": 0, "outcomes": 0, "bytes": 0}}
        lines: list[tuple[str, dict[str, Any] | None]] = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                rec = None
            lines.append((raw + "\n", rec if isinstance(rec, dict) else None))
        labelled = {r["id"] for _, r in lines if r and r.get("type") == "outcome"
                    and isinstance((r.get("outcome") or {}).get("labels"), dict)}
        protected = labelled if keep_labelled else set()
        drop = {r["id"] for _, r in lines if r and r.get("type") == "decision"
                and r.get("ts", 0) < cutoff and r["id"] not in protected}

        def kept() -> list[tuple[str, dict[str, Any] | None]]:
            return [(t, r) for t, r in lines if r and r.get("id") not in drop]

        if max_bytes is not None:
            size = _count(kept())["bytes"]
            for t, r in lines:
                if size <= max_bytes:
                    break
                if r and r.get("type") == "decision" and r["id"] not in drop and r["id"] not in protected:
                    drop.add(r["id"])
                    size = _count(kept())["bytes"]
        out = kept()
        report.update(before=_count(lines), after=_count(out))
        if not dry_run:
            tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_text("".join(t for t, _ in out), encoding="utf-8")
            os.replace(tmp, path)
    return report


def auto_prune(now: float | None = None) -> list[dict[str, Any]]:
    """Daemon start-up pruning: `LAYATOOLS_PRUNE_DAYS` (unset = off) applied to every *.jsonl under the log root,
    and to the global log only with `LAYATOOLS_PRUNE_GLOBAL=1`. Labelled decisions are always kept."""
    days = os.environ.get("LAYATOOLS_PRUNE_DAYS")
    if not days:
        return []
    root, global_log = log_root(), log_path()
    paths = [p for p in sorted(root.rglob("*.jsonl")) if p.resolve() != (global_log.resolve() if global_log else None)]
    if os.environ.get("LAYATOOLS_PRUNE_GLOBAL") == "1" and global_log:
        paths.append(global_log)
    return [prune(p, float(days) * 86400, now=now) for p in paths if p.is_file()]
