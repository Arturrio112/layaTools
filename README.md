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
python eval/run_eval.py eval/mireglass.json <project_dir> -k 5   # top-k accuracy vs plain grep
```

Semantic search: `lt index --embed` embeds chunks (BAAI/bge-small-en-v1.5 in the Python daemon, on GPU);
afterwards `lt search` fuses keyword, name and embedding rankings automatically (new chunks are embedded on
the fly; falls back to keyword-only if the daemon is down; `--lexical` forces that). Top-5 accuracy:
MainLandingPage 12 -> 15/16, mireglass 14 -> 14/16 (grep: 4/16, 3/16).

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
On the two real eval projects the judge is roughly neutral (top-1 9/16 both, with or without it), so it
is not on by default.
Held-out eval (mireglass, 16 questions): right file in top 5 for 14/16 vs 3/16 for grep.
