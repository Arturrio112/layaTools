"""Sample diverse chunks from indexed repos for question writing.

    python train/sample_chunks.py <oss_dir> <out_dir>

Writes <out_dir>/<repo>.chunks.jsonl with {id, file, start, end, text}. At most 2 chunks per file so a
few big files cannot dominate, and only chunks with real content (12+ non-blank lines).
"""

import json
import random
import re
import subprocess
import sys
from pathlib import Path

VAL_REPOS = {"cobra", "commander.js"}  # never trained on; used to validate the judge
SKIP = re.compile(r"(^|/)(dist|build|vendor|node_modules|\.github|docs?/_build)/|\.(lock|snap)$|CHANGELOG|LICENSE", re.I)


def main() -> None:
    oss, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(7)
    for repo in sorted(p for p in oss.iterdir() if p.is_dir()):
        lines = subprocess.run(["lt", "-C", str(repo), "export"], capture_output=True, text=True, check=True).stdout.splitlines()
        chunks = [json.loads(l) for l in lines]
        good = [c for c in chunks if not SKIP.search(c["file"]) and sum(1 for t in c["text"].splitlines() if t.strip()) >= 12]
        rng.shuffle(good)
        per_file: dict[str, int] = {}
        picked = []
        target = 60 if repo.name in VAL_REPOS else 100
        for c in good:
            if per_file.get(c["file"], 0) >= 2:
                continue
            per_file[c["file"]] = per_file.get(c["file"], 0) + 1
            picked.append({"id": f"{repo.name}:{c['file']}:{c['start']}", "file": c["file"], "start": c["start"],
                           "end": c["end"], "text": c["text"][:1800]})
            if len(picked) >= target:
                break
        (out / f"{repo.name}.chunks.jsonl").write_text("\n".join(json.dumps(p) for p in picked) + "\n")
        print(f"{repo.name}: {len(picked)} chunks from {len(per_file)} files")


if __name__ == "__main__":
    main()
