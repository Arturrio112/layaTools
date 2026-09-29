"""Score `lt search` against an eval file: top-k accuracy and output size vs a plain grep baseline.

    uv run python eval/run_eval.py eval/mainlandingpage.json <project_dir> [--judge] [-k 5]
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

LT = Path(__file__).parent.parent / "rust" / "target" / "release" / "lt"


def lt_hits(project: Path, question: str, k: int, judge: bool) -> tuple[list[str], int]:
    cmd = [str(LT), "-C", str(project), "search", question, "-k", str(k)]
    if judge:
        cmd.append("--judge")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    files = [m.group(1) for m in re.finditer(r"^(\S+?):\d+-\d+", out, re.M)]
    return files, len(out) // 4  # rough token estimate


def grep_hits(project: Path, question: str, k: int) -> tuple[list[str], int]:
    """What an LLM would do blind: grep for each keyword, read the matching file names."""
    terms = [t for t in re.findall(r"[A-Za-z]{4,}", question)]
    seen: dict[str, int] = {}
    for term in terms:
        out = subprocess.run(
            ["grep", "-rilE", term, ".", "--exclude-dir=node_modules", "--exclude-dir=.git", "--exclude-dir=dist"],
            cwd=project, capture_output=True, text=True,
        ).stdout.split()
        for f in out:
            seen[f.removeprefix("./")] = seen.get(f.removeprefix("./"), 0) + 1
    ranked = sorted(seen, key=lambda f: -seen[f])
    return ranked[:k], sum(len(f) for f in ranked) // 4


def main() -> None:
    spec = json.loads(Path(sys.argv[1]).read_text())
    project = Path(sys.argv[2])
    judge = "--judge" in sys.argv
    k = int(sys.argv[sys.argv.index("-k") + 1]) if "-k" in sys.argv else 5
    lt_ok = grep_ok = 0
    lt_tokens = 0
    for item in spec["questions"]:
        files, tokens = lt_hits(project, item["q"], k, judge)
        gfiles, _ = grep_hits(project, item["q"], k)
        a = any(f in files for f in item["expect"])
        b = any(f in gfiles for f in item["expect"])
        lt_ok += a
        grep_ok += b
        lt_tokens += tokens
        print(f"{'OK ' if a else 'MISS'} lt   {'OK ' if b else 'MISS'} grep  {item['q']}")
    n = len(spec["questions"])
    print(f"\ntop-{k}: lt {lt_ok}/{n}, grep {grep_ok}/{n}; lt output ~{lt_tokens // n} tokens/question")


if __name__ == "__main__":
    main()
