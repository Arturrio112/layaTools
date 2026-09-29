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

`--judge` re-ranks with Laya via the Python daemon (`layatools serve-http`); with the base model it
does not improve results yet (see eval notes), so it is opt-in.
Held-out eval (mireglass, 16 questions): right file in top 5 for 14/16 vs 3/16 for grep.
