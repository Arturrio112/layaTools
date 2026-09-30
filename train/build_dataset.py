"""Build relevance-judge training cases from sampled chunks + Claude-written questions.

    python train/build_dataset.py <oss_dir> <data_dir> <out.jsonl>

For each question: the source chunk is the positive; the top keyword hits from *other files* in the
same repo are hard negatives (they look relevant but the question was written for a different
place); one random chunk from another file is an easy negative. Labels are soft because a hard
negative is occasionally a legitimate answer too.

Each output line: {split, repo, kind, task, content, p_true, levels}.
`split` is "val" for the held-out repos (never trained on) and "train" otherwise.
"""

import json
import random
import subprocess
import sys
from pathlib import Path

VAL_REPOS = {"cobra", "commander.js", "polka", "realworld"}
CONTENT_CHARS = 1500  # must match what `lt search --judge` sends to the judge
HARD_NEGATIVES = 3

# (P(relevant), distribution over the 4 relevance levels: unrelated/tangential/background/directly needed)
LABELS = {
    "positive": (0.95, [0.00, 0.02, 0.08, 0.90]),
    "hard_negative": (0.12, [0.35, 0.40, 0.20, 0.05]),
    "random_negative": (0.02, [0.90, 0.08, 0.02, 0.00]),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    oss, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rng = random.Random(11)
    cases = []
    for qfile in sorted(data.glob("*.questions.jsonl")):
        repo = qfile.name.removesuffix(".questions.jsonl")
        chunks = {c["id"]: c for c in load_jsonl(data / f"{repo}.chunks.jsonl")}
        all_chunks = [json.loads(l) for l in subprocess.run(
            ["lt", "-C", str(oss / repo), "export"], capture_output=True, text=True, check=True).stdout.splitlines()]
        text_of = {(c["file"], c["start"]): c["text"] for c in all_chunks}
        split = "val" if repo in VAL_REPOS else "train"

        for row in load_jsonl(qfile):
            src = chunks.get(row["id"])
            if src is None:
                continue
            for q in (row.get("q_concept"), row.get("q_keyword")):
                if not q or not q.strip():
                    continue
                def add(kind: str, path: str, content: str) -> None:
                    p, levels = LABELS[kind]
                    # The path is part of what the judge sees, exactly as `lt search --judge` sends it:
                    # file names carry much of the answer ("contrast.spec.mjs" for a contrast question).
                    cases.append({"split": split, "repo": repo, "kind": kind, "task": q.strip(),
                                  "content": f"file: {path}\n{content[:CONTENT_CHARS]}", "p_true": p, "levels": levels})

                add("positive", src["file"], src["text"])
                # Optional extra acceptable files (first chunk of each): more than one right answer.
                also = [f for f in row.get("also_files", []) if f != src["file"]][:2]
                for f in also:
                    first = next((c for c in all_chunks if c["file"] == f), None)
                    if first:
                        add("positive", first["file"], first["text"])
                hits = json.loads(subprocess.run(
                    ["lt", "-C", str(oss / repo), "search", "-k", "10", "--lexical", "--json", "--", q],
                    capture_output=True, text=True, check=True).stdout or "[]")
                negs = [h for h in hits if h["path"] != src["file"] and h["path"] not in also][:HARD_NEGATIVES]
                for h in negs:
                    text = text_of.get((h["path"], h["start"]))
                    if text:
                        add("hard_negative", h["path"], text)
                others = [c for c in all_chunks if c["file"] != src["file"]]
                if others:
                    pick = rng.choice(others)
                    add("random_negative", pick["file"], pick["text"])
        print(f"{repo}: {sum(1 for c in cases if c['repo'] == repo)} cases ({split})")

    out.write_text("\n".join(json.dumps(c) for c in cases) + "\n")
    train = sum(c["split"] == "train" for c in cases)
    print(f"wrote {len(cases)} cases: {train} train, {len(cases) - train} val -> {out}")


if __name__ == "__main__":
    main()
