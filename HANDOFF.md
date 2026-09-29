# layaTools handoff

Written at the end of session 1 (2026-09-29). Read this fully before doing anything. Last commit: `920e47b`.

## What this project is

A "mini search engine + fast decision engine for LLMs", so an LLM (Claude Code or any harness) finds what
it needs in a codebase, and makes cheap routine decisions, without exploring blindly and burning tokens.
It is a generic engine: it must know nothing about any one project. Project-specific decision profiles live
in the project (`<project>/.layatools/profiles/*.yaml`); user-wide ones in `~/.config/layatools/profiles/`.
The user's main consumer is `~/factory` (a site factory that builds projects into `~/sites/*`); the tools are
meant to run on the projects it builds, not on factory itself.

Two parts:
1. **`lt` (Rust, `rust/`)**: per-project SQLite FTS5 index (cache in `~/.cache/layatools/`, synced by
   mtime/size on every call), hybrid search fusing keyword + file/symbol-name + embedding rankings (RRF),
   optional Laya judge re-rank (`--judge`), `lt serve` (loopback HTTP, `POST /v1/search`), `lt export`.
2. **`layatools` (Python, `src/layatools/`, managed with uv)**: warm daemon on `127.0.0.1:8765`
   (`layatools serve-http`; auto-started by `lt`) hosting Laya (typed decisions), the embedding model
   (BAAI/bge-small-en-v1.5) and the fine-tuned relevance judge; profile gateway, MCP server, CLI.
   Endpoints: `/health`, `/v1/decide`, `/v1/rank`, `/v1/embed`, `/v1/decisions`.

User-level skill: `~/.claude/skills/laya-decide/SKILL.md` (tells Claude Code to run `lt search` first).
Installed binaries: `lt` and `layatools` in `~/.local/bin` (`cargo install --path rust --root ~/.local`;
`uv tool install -e .`). Rust needs `~/.cargo/bin` on PATH. GPU works (RTX 5080; gcc + python3-dev installed).

## Measured results so far

Eval = 16 hand-written questions per project, ground truth = one expected file (or two); metric = expected
file in top-k. Files: `eval/mainlandingpage.json` (real project `~/projects/MainLandingPage`, keyword rules
were tuned on it) and `eval/mireglass.json` (`~/sites/mireglass`, held out: written before running `lt` on it).
Runner: `python3 eval/run_eval.py eval/<x>.json <project_dir> -k 5 [--judge] [--lexical]`.

| setup | MainLandingPage top-1/3/5 | mireglass top-1/3/5 |
|---|---|---|
| plain grep (top-5) | -/-/4 | -/-/3 |
| lt hybrid, no judge | 9 / 12 / 15 | 8 / 12 / 14 |
| lt hybrid + fused v2 judge | 9 / 13 / 14 | 9 / 13 / 14 |

Judge on held-out OSS repos (`cobra`, `commander.js`, never trained on; 240 questions, ~5 candidates each):
base Laya top-1 0.479 / MRR 0.673 -> tuned v2 0.863 / 0.926. So it learned a real, general skill, but on the
two real projects it is roughly neutral. Likely reasons: retrieval already puts the right file in the top 5
~14/16 so there is little to fix; single-expected-file ground truth counts a legitimately relevant doc
(e.g. `CONTEXT.md`) as a miss; 16 questions per project is far too few to see a 1-question change.

## The next task: a bigger, fairer real evaluation

Goal: decide whether the judge (and the retrieval fusion weights) genuinely help, and whether `lt` is good
enough to roll out across `~/factory`'s workflow. Suggested plan:
1. Build ~100+ questions across many real projects (all of `~/sites/*` — `dentist-lv`, `dentist-lv.old`,
   `mireglass`, more if present — plus `~/projects/MainLandingPage`, and any other real repos the user has).
   Question types: locate a feature, find the config, "which test checks X", "why is X done this way",
   "where would I change Y". Do NOT tune on these; keep a dev half and a test half.
2. Allow **multiple acceptable files per question** (graded relevance if practical), and report top-1, top-3,
   top-5, MRR, and tokens per answer. Compare: grep baseline, `--lexical`, hybrid, hybrid + judge.
   Judge fusion weight is `JUDGE_WEIGHT` (=1.0) and RRF constants are in `rust/src/search.rs`; sweep them on
   the dev half only.
3. If the judge still does not help, consider: more/different training data (real-project style questions,
   docs vs code balance, multiple positives per question), training on the user's project types, or
   dropping it and keeping `lt` retrieval-only. Report honestly either way.
4. Also worth measuring: end-to-end token savings for a Claude Code task with vs without `lt`.
Questions can be written by Claude subagents (the user approved that approach), but **give each subagent its
own private working directory**: the parallel agents clobbered each other's scratch files last time.

Known retrieval misses to look at: `App.jsx` for "where is the router configured", `firebase.json` for
production serving, `serve.mjs` for local serving, `behaviors.spec.mjs` for the mobile-menu test.

## Fine-tuning pipeline (`train/`)

`sample_chunks.py` -> (Claude writes `<repo>.questions.jsonl`: `{id,q_concept,q_keyword}` per chunk) ->
`build_dataset.py` (positive + 3 hard negatives from `lt --lexical` + 1 random per question, soft labels;
the judged text is `file: <path>\n<chunk>`) -> `train_relevance.py` (single-GPU port of Laya's RLCD notebook,
bf16, ~36 min for 3 epochs on the 5080) -> `eval_judge.py` (held-out top-1/MRR).
Base model `convaiinnovations/laya` (421M). Everything trained on the 10 OSS repos below; `cobra` and
`commander.js` are validation only.
- Persisted data (survives the scratchpad): `~/.local/share/layatools/train_data/` has every
  `*.chunks.jsonl`, `*.questions.jsonl`, `cases2.jsonl` (10,850 cases) and `train_v2.log`.
- OSS repos were shallow clones of: pallets/flask, psf/requests, expressjs/express, axios/axios,
  honojs/hono, sharkdp/fd, gin-gonic/gin, spf13/cobra, preactjs/preact, pallets/click, sinatra/sinatra,
  tj/commander.js. Re-clone if needed (`git clone --depth 1`); chunk ids depend on the exact files, so for
  new training data re-run `sample_chunks.py` rather than reusing old chunk files with new clones.
- Models: `~/.local/share/layatools/models/laya-code-relevance` (= v2, path-aware, the active one) and
  `laya-code-relevance-v1` (no paths; worse on real projects). Weights are not in git. A profile selects a
  model with `model: <dir name>` (see `src/layatools/builtin_profiles/relevance.yaml`); if the directory is
  missing it falls back to the default Laya router.

## Gotchas learned the hard way

- **Never `pkill -f "<pattern>"` with the pattern also appearing in your own command line**: it kills your own
  shell (exit 144). Kill by exact name (`pkill -x lt`) or in a command that does not contain the pattern.
- `!` in the Claude Code prompt runs a command in-session; typed in a normal shell it negates the exit
  status (`! sudo apt update && ...` silently skipped the install).
- `grep | tee` buffers output: log training runs straight to a file with `python -u` / `PYTHONUNBUFFERED=1`.
- The daemon holds ~2-3 GB of GPU memory; training uses ~15.6 GB of 16 GB. Stop the daemon before training.
- The judge sees file path + first 1500 chars of the chunk; Laya's base checkpoints are near-chance zero-shot,
  and the built-in `relevance` profile is only meaningful with the tuned model.
- Test beds must be treated read-only: `~/sites` and `~/projects` are the user's real work. Copy anything
  you need to modify, and delete the copies afterwards (the user asked for this).
- Tests: `uv run pytest -q` (8 pass), `cd rust && cargo test` (2 pass). Nothing tests the HTTP server or the
  judge fusion; add tests if you change them.

## Not built / open ideas

- `lt serve` has no auth (loopback only) and reloads all vectors per search (fine below ~50k chunks).
- No file watcher (not needed: the index re-syncs on every call).
- Decision profiles (`tool_or_skill`, `watchdog` in `~/factory/.layatools/profiles/`, Laya's `guard`) exist but
  are untested beyond a few smoke calls and are uncalibrated; the user wants them for tool/skill routing,
  guard and watchdog use, but the priority is the search engine.
- The skill at `~/.claude/skills/laya-decide/SKILL.md` says `--judge` gives no benefit; update it with the
  outcome of the bigger eval.
- Nothing is wired into `~/factory` yet beyond the two profile YAMLs in its `.layatools/profiles/`
  (uncommitted in that repo).
