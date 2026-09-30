"""Score grep / lexical / hybrid / hybrid+judge on eval/real.json (built by build_real.py).

    python3 eval/run_real.py --split dev [--modes grep,lexical,hybrid,judge] [-k 5] [--env LT_JUDGE_WEIGHT=0.5]
    python3 eval/run_real.py --split test --per-project

A question is answered if any of its expect+also files appears in the top-k. Reports top-1/3/k,
MRR@k, tokens per answer (lt output / 4), and paired win/loss counts against hybrid.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
LT = HERE.parent / "rust" / "target" / "release" / "lt"
HOME = Path.home()


def lt_hits(project: Path, q: str, k: int, mode: str, env: dict) -> tuple[list[str], int]:
    cmd = [str(LT), "-C", str(project), "search", q, "-k", str(k)]
    if mode == "lexical":
        cmd.append("--lexical")
    if mode == "judge":
        cmd.append("--judge")
    out = subprocess.run(cmd, capture_output=True, text=True, check=True, env={**os.environ, **env}).stdout
    return [m.group(1) for m in re.finditer(r"^(\S+?):\d+-\d+", out, re.M)], len(out) // 4


def grep_hits(project: Path, q: str, k: int) -> tuple[list[str], int]:
    """What an LLM would do blind: grep each keyword, rank files by how many keywords they match."""
    seen: dict[str, int] = {}
    for term in re.findall(r"[A-Za-z]{4,}", q):
        out = subprocess.run(
            ["grep", "-rilE", term, ".", "--exclude-dir=node_modules", "--exclude-dir=.git", "--exclude-dir=dist",
             "--exclude-dir=.astro", "--exclude-dir=vendor", "--exclude=package-lock.json"],
            cwd=project, capture_output=True, text=True,
        ).stdout.split()
        for f in out:
            f = f.removeprefix("./")
            seen[f] = seen.get(f, 0) + 1
    ranked = sorted(seen, key=lambda f: (-seen[f], f))
    return ranked[:k], sum(len(f) for f in ranked[:k]) // 4


def rank_of(files: list[str], ok: set[str]) -> int | None:
    return next((i + 1 for i, f in enumerate(files) if f in ok), None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--modes", default="grep,lexical,hybrid,judge")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to lt (LT_JUDGE_WEIGHT, LT_RRF_K, ...)")
    ap.add_argument("--per-project", action="store_true")
    ap.add_argument("--misses", action="store_true", help="list questions each mode misses")
    ap.add_argument("--json", help="write raw per-question ranks here")
    a = ap.parse_args()
    env = dict(e.split("=", 1) for e in a.env)
    modes = a.modes.split(",")
    qs = [x for x in json.loads((HERE / "real.json").read_text())["questions"] if a.split in ("all", x["split"])]

    def run(item):
        proj = HOME / item["path"]
        ok = set(item["expect"]) | set(item["also"])
        res = {}
        for m in modes:
            files, tok = grep_hits(proj, item["q"], a.k) if m == "grep" else lt_hits(proj, item["q"], a.k, m, env)
            res[m] = (rank_of(files, ok), tok)
        return res

    with ThreadPoolExecutor(4) as ex:
        results = list(ex.map(run, qs))
    if a.json:
        Path(a.json).write_text(json.dumps([{"project": q["project"], "q": q["q"], "ranks": {m: r[m][0] for m in modes}} for q, r in zip(qs, results)]))

    def table(idx: list[int], title: str) -> None:
        n = len(idx)
        print(f"\n{title} (n={n}, k={a.k})")
        print(f"{'mode':9} {'top1':>5} {'top3':>5} {'top'+str(a.k):>5} {'MRR':>6} {'tok/ans':>8}")
        for m in modes:
            rs = [results[i][m][0] for i in idx]
            t1 = sum(r == 1 for r in rs)
            t3 = sum(r is not None and r <= 3 for r in rs)
            tk = sum(r is not None for r in rs)
            mrr = sum(1 / r for r in rs if r) / n
            tok = sum(results[i][m][1] for i in idx) // n
            print(f"{m:9} {t1:5} {t3:5} {tk:5} {mrr:6.3f} {tok:8}")
        if "hybrid" in modes:
            for m in modes:
                if m == "hybrid":
                    continue
                w = l = 0
                for i in idx:
                    rm, rh = results[i][m][0] or 99, results[i]["hybrid"][0] or 99
                    w += rm < rh
                    l += rm > rh
                print(f"  {m} vs hybrid: better rank on {w}, worse on {l}")

    table(list(range(len(qs))), f"split={a.split}")
    if a.per_project:
        by = defaultdict(list)
        for i, q in enumerate(qs):
            by[q["project"]].append(i)
        for p, idx in by.items():
            table(idx, p)
    if a.misses:
        for m in modes:
            print(f"\nmisses [{m}]")
            for q, r in zip(qs, results):
                if r[m][0] is None:
                    print(f"  {q['project']}: {q['q']}  -> {q['expect'][0]}")


if __name__ == "__main__":
    main()
