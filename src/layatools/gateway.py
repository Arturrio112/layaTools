"""Gateway: run a named profile against some text and return a compact verdict."""

from __future__ import annotations

from typing import Any

from .backend import Backend
from .profiles import Profile


class UnknownProfile(KeyError):
    pass


class Gateway:
    def __init__(self, backend: Backend, profiles: dict[str, Profile]) -> None:
        self.backend = backend
        self.profiles = profiles

    def list_decisions(self) -> list[dict[str, Any]]:
        return [p.summary() for p in self.profiles.values()]

    def decide(
        self, profile: str, state: str | dict[str, Any], min_confidence: float | None = None
    ) -> dict[str, Any]:
        """Answer every question of `profile` in one forward pass.

        Returns `{"answers": {qid: {"value", "conf"}}, "escalate": [qids]}`. Anything listed in
        `escalate` fell below the confidence threshold: the calling LLM should decide it itself.
        """
        prof = self._profile(profile)
        threshold = prof.min_confidence if min_confidence is None else min_confidence
        if isinstance(state, str):
            state = {prof.state_field or "message": state}
        raw = self.backend.predict(state, prof.questions, prof.model)
        return compact(raw["answers"], threshold)

    def _profile(self, name: str) -> Profile:
        try:
            return self.profiles[name]
        except KeyError:
            raise UnknownProfile(f"unknown profile {name!r}; available: {sorted(self.profiles)}") from None

    def rank(
        self, task: str, items: dict[str, str], profile: str = "relevance", limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Score each item's text against `task`; most relevant first.

        Meant for brownfield exploration: rank candidate files so the LLM reads only the top few.
        """
        prof = self._profile(profile)
        scored = []
        for item_id, text in items.items():
            raw = self.backend.predict({"task": task, "content": text}, prof.questions, prof.model)["answers"]
            score = float(raw["relevance"]["score"])
            p_rel = float(raw["relevant"]["noul"])
            scored.append({"id": item_id, "score": round(score, 2), "p": round(p_rel, 2)})
        scored.sort(key=lambda r: (r["score"], r["p"]), reverse=True)
        return scored[:limit] if limit else scored


def compact(answers: dict[str, Any], threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    escalate: list[str] = []
    for qid, ans in answers.items():
        if "choice" in ans:
            value: Any = ans["choice"]
        elif "score" in ans:
            value = round(float(ans["score"]), 2)
        else:
            value = float(ans["noul"]) >= 0.5
        conf = round(float(ans.get("confidence", 1.0)), 2)
        out[qid] = {"value": value, "conf": conf}
        if conf < threshold:
            escalate.append(qid)
    return {"answers": out, "escalate": escalate}
