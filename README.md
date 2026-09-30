# layatools

A System-1 decision gateway for LLMs, built on [Laya](https://github.com/NandhaKishorM/laya).
Instead of spending tokens and tool calls having an LLM reason through routine classification
(routing, urgency, jailbreak checks), it calls a named **decision profile** and gets a compact
verdict from one non-generative Laya forward pass. Low-confidence answers come back under
`escalate`, so the LLM only reasons about the cases Laya is unsure of.

## Usage (uv)

```bash
uv sync
uv run layatools list
uv run layatools decide support_route "Hi, we were billed twice for March."
uv run layatools-mcp          # MCP server over stdio (tools: list_decisions, decide)
uv run pytest
```

Result shape:

```json
{"answers": {"department": {"value": "billing", "conf": 0.94}}, "escalate": []}
```

## Profiles

Laya's built-in presets (`guard`, `moderation`, `triage`, `model_router`, `email`) are always
available. Add your own as YAML in `profiles/` (or set `LAYATOOLS_PROFILES`); see
`profiles/support_route.yaml`. The first real call downloads Laya's model weights.

## `lt`: fast codebase search for LLMs (Rust)

`rust/` builds `lt`, a per-project index (SQLite FTS5, synced by mtime on every call, ~10ms) that
answers "where is X?" with `path:lines` plus a snippet, so an LLM reads hits instead of exploring.

```bash
cargo install --path rust --root ~/.local     # installs `lt`
lt search "where is the router configured" -k 5 [--per-file 2] [--json] [-C <dir>]
lt serve --port 8766                          # POST /v1/search {dir, question, k, ...} on loopback
python3 eval/run_real.py --split test --per-project             # multi-project eval, see "Evaluation"
```

Semantic search: `lt index --embed` embeds chunks (BAAI/bge-small-en-v1.5 in the Python daemon, on GPU);
afterwards `lt search` fuses keyword, name and embedding rankings automatically (new chunks are embedded on
the fly; falls back to keyword-only if the daemon is down; `--lexical` forces that). Top-5 accuracy:
see "Evaluation" below.

`--judge` re-ranks the top candidates with a fine-tuned Laya relevance judge (via the Python daemon,
`layatools serve-http`), fused with the retrieval order. Opt-in. See "Fine-tuning" below.

## Fine-tuning the judge (`train/`)

Question/chunk training data from 10 open-source repos (questions written by Claude subagents, hard
negatives mined with `lt`); `cobra` and `commander.js` are held out entirely.

```bash
python train/sample_chunks.py <oss_dir> <data_dir>            # sample chunks (then write *.questions.jsonl)
python train/build_dataset.py <oss_dir> <data_dir> cases.jsonl
python train/train_relevance.py cases.jsonl train/out/laya-code-relevance-v2 --epochs 3   # ~36 min, RTX 5080
python train/eval_judge.py cases.jsonl <model> ...             # held-out top-1 / MRR
cp -r train/out/laya-code-relevance-v2 ~/.local/share/layatools/models/laya-code-relevance
```

Held-out repos (240 questions, ~5 candidates each): base Laya top-1 0.48 / MRR 0.67 -> tuned 0.86 / 0.93.
On the real-project eval below the judge helps only slightly and not significantly, so it stays opt-in.

## Evaluation (`eval/`)

`eval/real.json` (built by `eval/build_real.py`) has 123 questions over 9 real projects in `~/sites` and
`~/projects` (Astro, React/Vite, Laravel + Next.js, Express, Java). Each question lists acceptable files
(`expect` + `also`), and is split into dev (69) and test (54); `MainLandingPage` is dev-only because the
keyword rules were tuned on it. Weights were swept on dev only, then test was run once. The near-duplicate
`dentist-lv.old` and `MainLandingPage-backup` are excluded. `lt` weights can be overridden for sweeps with
`LT_JUDGE_WEIGHT`, `LT_RRF_K`, `LT_SEM_WEIGHT`, `LT_NAME_WEIGHT`.

```bash
python3 eval/run_real.py --split test --per-project [--misses] [--modes grep,lexical,hybrid,judge] [--env LT_JUDGE_WEIGHT=0.5]
```

Held-out test half, k=5 (n=54; MRR over the top 5):

| mode | top-1 | top-3 | top-5 | MRR | tokens/answer | latency |
|---|---|---|---|---|---|---|
| grep | 6 | 15 | 23 | 0.22 | ~35 | - |
| lexical | 36 | 48 | 48 | 0.765 | ~270 | 9 ms |
| hybrid (default) | 38 | 46 | 47 | 0.776 | ~275 | 24 ms |
| hybrid + judge (v2) | 38 | 46 | 50 | 0.792 | ~290 | 300 ms |
| hybrid + judge (v3) | 39 | 48 | 50 | 0.811 | ~283 | 300 ms |
| hybrid + judge (v4, current) | 40 | 50 | 50 | 0.824 | ~287 | 300 ms |

Dev half (n=69): hybrid 47/60/64 (MRR 0.78) vs lexical 42/55/59 (0.70); judge v2 50/62/63 (0.80); judge v3 49/60/63 (0.79 vs hybrid 0.77).
Honest reading: `lt` finds an acceptable file in the top 5 for ~87-93% of questions (hybrid) vs ~32-43% for
grep. Hybrid beats lexical clearly on dev (better rank on 17 questions, worse on 5) but not on test (7 vs 5 in
lexical's favour), so the embedding gain is real on some projects but not proven overall. The v2 judge
improved the rank on 16 questions and worsened it on 8 across dev+test (sign test p~0.15): not significant.
The v3 judge (retrained with 6.7k extra web-project cases, see below) improves the rank on 16 and worsens it on 3
(p~0.004): a real but modest gain (MRR +0.03), at ~12x the latency, so it is still opt-in. The current fusion weights were already at
the optimum or on a flat plateau on dev, so none were changed.

### v3 and v4 retrains (web-style projects)

Added 9 web-style OSS repos (astro-paper, breeze, commerce, create-t3-app, taxonomy, vitepress,
node-express-realworld-example-app for training; polka and realworld held out), 789 chunks, short locator
questions written by 4 Claude subagents (each with a private working dir), some with `also_files`
(extra acceptable files, used as extra positives by `build_dataset.py`). Trained on the old + new
cases (16.4k train cases, ~62 min on the RTX 5080). Held-out repos (530 questions): top-1 0.879 -> 0.892, MRR
0.934 -> 0.939 (a small change: the OSS benchmark was already near its ceiling); the gain shows up on the real projects.
Questions were written from the first ~260 characters of each chunk, so some labels are approximate.

v4 adds 9.7k cases from 938 fresh chunks (5 new repos: next-auth-example, koa, movies, next-learn, jetstream, plus a
second sample of 6 earlier ones) with questions written from each chunk's *full* text: 26.2k train cases, ~1.9 h.
Held-out OSS repos got marginally worse (top-1 0.892 -> 0.881, MRR 0.939 -> 0.934, noise level), but the real
projects improved a little more: dev MRR 0.793 -> 0.801, test 0.811 -> 0.824. Judge vs hybrid across dev+test with
v4: better rank on 19 questions, worse on 5 (sign test p~0.007). Gains between v3 and v4 are small and within noise
on 123 questions; v4 is active because it was no worse on dev and better on test (test has now had 3 looks).

## End-to-end test: does `lt` save an agent tokens?

12 test-split questions on 7 real projects, each answered by a fresh Sonnet subagent twice: once with plain
Grep/Read/find only, once told to run `lt search --judge -k 5` first (24 runs).

| | without lt | with lt |
|---|---|---|
| subagent tokens (mean) | 29,142 | 28,579 (-1.9%) |
| tool calls (total) | 37 | 24 (-35%) |
| correct file | 11/12 | 10/12 |

Reading: on small projects a plain grep agent already finds the file in 2-4 calls, and ~27k of each run's
tokens is fixed subagent overhead that no search tool can remove, so the token saving is negligible here;
`lt` mainly saves tool calls (about one third fewer), not accuracy. One `lt` miss (`astro.config.mjs` for "site
title and main config") and one strict miss (the view file instead of the route page); the no-lt miss was on the
same ambiguous question. n=12 is too small to separate the accuracy of the two. The benefit should grow with
repo size (these are 8-340 indexed files), which this test does not cover.
