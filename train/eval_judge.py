"""Compare judge models on the held-out repos: is the true chunk ranked first among its candidates?

    python train/eval_judge.py <cases.jsonl> [model ...]     (default: base model only)

Groups val cases by question. Each group holds the true chunk plus hard/random negatives. A model is
scored by how often the true chunk gets the highest judge score (top-1) and its mean reciprocal rank.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import laya

from layatools.profiles import SHIPPED_DIR, load_profile_file

BASE = "convaiinnovations/laya"


def score(answers: dict) -> float:
    """Combine the two relevance questions: expected level (0-3, scaled) plus P(relevant)."""
    return answers["relevance"]["score"] / 3 + answers["relevant"]["noul"]


def main() -> None:
    cases = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    groups = defaultdict(list)
    for c in cases:
        if c["split"] == "val":
            groups[(c["repo"], c["task"])].append(c)
    questions = load_profile_file(SHIPPED_DIR / "relevance.yaml").questions

    for model in sys.argv[2:] or [BASE]:
        agent = laya.load(model, device="cuda")
        top1 = rr = 0.0
        for group in groups.values():
            scored = []
            for c in group:
                res = agent.predict({"task": c["task"], "content": c["content"]}, questions)
                scored.append((score(res["answers"]), c["kind"] == "positive"))
            scored.sort(key=lambda s: -s[0])
            rank = next(i for i, (_, pos) in enumerate(scored) if pos) + 1
            top1 += rank == 1
            rr += 1 / rank
        n = len(groups)
        print(f"{model}: top-1 {top1 / n:.3f}, MRR {rr / n:.3f} over {n} held-out questions")


if __name__ == "__main__":
    main()
