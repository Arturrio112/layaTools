"""How far can a profile's answers be trusted? Accuracy by confidence, and the threshold to use.

A decision is only worth acting on automatically where it is right often enough. Given the model's
answers (value + confidence) and the true labels, this reports, for each confidence threshold, how
many answers clear it (coverage) and how many of those are right (precision), and picks the lowest
threshold whose precision reaches the target: answers below it go to the LLM (`escalate`).
"""

from __future__ import annotations

from typing import Any

THRESHOLDS = [round(0.05 * i, 2) for i in range(20)]  # 0.0 .. 0.95


def correct(predicted: Any, label: Any) -> bool:
    """Choice: same string. Yes/no: same truth value. Score: same value after rounding."""
    if isinstance(predicted, bool) or isinstance(label, bool):
        return _truthy(predicted) == _truthy(label)
    if isinstance(predicted, (int, float)) and isinstance(label, (int, float)):
        return round(predicted) == round(label)
    return str(predicted).strip().lower() == str(label).strip().lower()


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "y", "1"}
    return bool(v)


def calibrate(pairs: list[tuple[Any, float, Any]], target: float = 0.9) -> dict[str, Any]:
    """`pairs` = (predicted value, confidence, true label) for one question."""
    n = len(pairs)
    hits = [(conf, correct(pred, label)) for pred, conf, label in pairs]
    table = []
    for t in THRESHOLDS:
        kept = [ok for conf, ok in hits if conf >= t]
        table.append({
            "threshold": t,
            "coverage": round(len(kept) / n, 3) if n else 0.0,
            "precision": round(sum(kept) / len(kept), 3) if kept else None,
            "n": len(kept),
        })
    usable = [row for row in table if row["precision"] is not None and row["precision"] >= target]
    return {
        "n": n,
        "accuracy": round(sum(ok for _, ok in hits) / n, 3) if n else None,
        "target_precision": target,
        # Lowest threshold that reaches the target = the most answers Laya can take on its own.
        "recommended_min_confidence": usable[0]["threshold"] if usable else None,
        "coverage_at_recommended": usable[0]["coverage"] if usable else 0.0,
        "table": table,
    }


def evaluate(results: list[dict[str, Any]], labels: list[dict[str, Any]], target: float = 0.9) -> dict[str, Any]:
    """`results[i]` is a decide() result for the item labelled `labels[i]` ({qid: true value}).
    Questions an item has no label for are skipped for that item."""
    per_q: dict[str, list[tuple[Any, float, Any]]] = {}
    for result, truth in zip(results, labels):
        for qid, label in truth.items():
            ans = result.get("answers", {}).get(qid)
            if ans is not None:
                per_q.setdefault(qid, []).append((ans["value"], float(ans["conf"]), label))
    return {qid: calibrate(pairs, target) for qid, pairs in per_q.items()}
