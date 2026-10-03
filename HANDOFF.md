# layaTools handoff

Written at the end of session 1 (2026-09-29), updated in session 2 (2026-09-30, the bigger eval; see below). Read this fully before doing anything.

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

## Session 2: bigger real evaluation (done)

`eval/real.json` (123 questions, 9 projects, dev 69 / test 54, multiple acceptable files) + `eval/run_real.py`;
questions were written by Claude directly (no subagents), before running `lt` on them. Results and the
honest reading are in the README ("Evaluation"). Summary: `lt` top-5 ~87-93% (hybrid) vs grep ~32-43%; hybrid > lexical on
dev (17 vs 5 better) but a wash on test (7 vs 5 for lexical); judge +16/-8 ranks (p~0.15), not significant, ~300 ms
vs ~24 ms, stays opt-in. Fusion weights (JUDGE_WEIGHT, RRF_K, SEM_WEIGHT, NAME_WEIGHT; now env-overridable as
`LT_*`) were swept on dev and were already at the optimum/plateau. Judge weight 0.5 vs 1.0 is within noise.
Caveats: questions are by one author (me), so phrasing is biased toward file names; several projects are
small (3-5 test questions); MainLandingPage is dev-only. Not done: end-to-end token savings for a Claude Code task.

### Session 2d: judge v4 (done)
More data (`train_data_web2/`, `cases4.jsonl`, log `train_v4.log`), questions from full chunk text. Active model = v4; v3 kept as `models/laya-code-relevance-v3`, v2 as `-v2`. Real eval judge vs hybrid: 19 better / 5 worse, test MRR 0.776 -> 0.824. Diminishing returns: v3->v4 is within noise. Test split has had 3 looks; to keep measuring honestly, write a fresh eval set (new questions, new projects) before the next model decision. `sample_chunks.py` gained `--seed/--exclude/--only`.

### Session 2c: end-to-end token test (done)
12 questions x with/without `lt` via subagents: -1.9% tokens, -35% tool calls, 11/12 vs 10/12 correct (details in README). Not worth much on small repos; test on a large repo before claiming savings.

### Session 2b: judge v3 retrain (done)
Training data `~/.local/share/layatools/train_data_web/` (chunks, questions, `cases_web.jsonl`), merged cases
`train_data/cases3.jsonl`, log `train_v3.log`, OSS clones in `~/.local/share/layatools/oss_web/`. Active model
`models/laya-code-relevance` = v3 (v2 kept as `laya-code-relevance-v2`). On the real eval the judge now beats
hybrid on 16 questions and loses on 3 (p~0.004; test MRR 0.776 -> 0.811). Still opt-in (~300 ms). Next: try
`lt search --judge` as default for the skill, measure end-to-end token savings, more real-style training data.
Caveat: test split was also used to compare v2 vs v3 (2 looks); dev shows the same direction.

## The (original) next-task plan, kept for reference

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
- Tests: `uv run pytest -q` (31 pass, incl. the HTTP daemon, decision log and calibration on a random port with fake backend/embedder),
  `cd rust && cargo test` (7 pass, incl. index sync, keyword/semantic search on temp projects via
  `Index::open_at`, `fuse_judge`, and the `lt serve` host check). Keep them green when changing either.

## Session 3 (2026-09-30): filling gaps

- `LAYATOOLS_PROFILES` (README promised it, it did not exist) now adds profile dirs between user and project.
- `lt` honours `LAYATOOLS_PORT` like the Python daemon (was hardcoded 8765) and creates the log dir itself.
- `lt serve` caches chunk vectors per index (invalidated on its own writes and via `PRAGMA data_version`
  for other processes' writes); semantic search fetches the text of only the top chunks by rowid.
- Both HTTP servers reject a non-local `Host` header (DNS-rebinding guard). Still no auth token.
- Daemon: `/v1/decisions?cwd=` is URL-decoded; unknown profile on `/v1/rank` is a 400 like `/v1/decide`.
- Judge fusion is a pure `search::fuse_judge`, unit-tested. Tests added on both sides (see Gotchas).
- The skill now lives in the repo (`skills/laya-decide/SKILL.md`, copy to `~/.claude/skills/`) with the
  v4 judge outcome; the copy in `~/.claude/skills/` must be re-copied by hand.
- Not verified here: real embeddings/Laya models (the cloud sandbox blocks HuggingFace), so the semantic
  path was tested with fixed vectors only. Run `lt index --embed` + a few searches on a real machine.

## Session 3b (2026-09-30): decisions you can measure

- `decision_log.py`: every decide (HTTP, MCP) is appended with an `id`; `outcome` records what happened.
- `calibrate.py` + `layatools eval`: accuracy per question, precision/coverage by confidence, and the
  recommended `min_confidence` for a target precision. `layatools export` joins decisions with outcomes.
- Batch decide (`items`), `meta`, `"log": false`; MCP `record_outcome` and `search` (shells out to `lt`).
- Factory (`landingPageSoftwareFactory`, branch `claude/laya-shadow-routing`): `lib/laya.mjs` asks the
  `step_route` profile before each fresh implement when `factory.config.json` `laya.shadow` is true, and
  records the run's outcome (status, spend, model, fix attempts) at the end. Never acts on the answer.
- Next: turn shadow on, collect a few dozen tickets, label, `eval`. A generic `layatools train <profile>`
  (generalising `train/train_relevance.py` to any profile from logged labels) is the step after that.
- `examples/profiles/status_triage.yaml`, `stall_triage.yaml` (uncalibrated, shadow only). The factory asks
  `stall_triage` in shadow mode when its silence ladder reports a step (it keeps its own copy).
- Decision log: a torn last line (writer killed mid-record) is no longer glued to the next record.
- Factory side of all this: `docs/HANDOFF-2026-09-30-cloud-session.md` on the factory branch.
- Tests: `uv run pytest -q` 31 pass.

## Not built / open ideas

- `lt serve` / the daemon have no auth token (loopback + Host check only).
- A fresh eval set (new questions, new projects) is needed before the next model decision (test split has 3 looks).
- No file watcher (not needed: the index re-syncs on every call).
- Decision profiles (`tool_or_skill`, `watchdog` in `~/factory/.layatools/profiles/`, Laya's `guard`) exist but
  are untested beyond a few smoke calls and are uncalibrated; the user wants them for tool/skill routing,
  guard and watchdog use, but the priority is the search engine.
- Nothing is wired into `~/factory` yet beyond the two profile YAMLs in its `.layatools/profiles/`
  (uncommitted in that repo).
