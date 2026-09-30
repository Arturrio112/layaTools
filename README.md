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
| hybrid + judge | 38 | 46 | 50 | 0.792 | ~290 | 300 ms |

Dev half (n=69): hybrid 47/60/64 (MRR 0.78) vs lexical 42/55/59 (0.70); judge 50/62/63 (0.80).
Honest reading: `lt` finds an acceptable file in the top 5 for ~87-93% of questions (hybrid) vs ~32-43% for
grep. Hybrid beats lexical clearly on dev (better rank on 17 questions, worse on 5) but not on test (7 vs 5 in
lexical's favour), so the embedding gain is real on some projects but not proven overall. The judge
improved the rank on 16 questions and worsened it on 8 across dev+test (sign test p~0.15): a small positive
trend that is not statistically significant, at ~12x the latency. The current fusion weights were already at
the optimum or on a flat plateau on dev, so none were changed.
