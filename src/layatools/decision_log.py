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


def resolve_path(value: Any) -> Path | None:
    """A per-request log file: None when not given; otherwise an absolute path whose parent directory exists."""
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("`log_path` must be a non-empty string")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("`log_path` must be an absolute path")
    if not path.parent.is_dir():
        raise ValueError(f"`log_path` parent directory does not exist: {path.parent}")
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
