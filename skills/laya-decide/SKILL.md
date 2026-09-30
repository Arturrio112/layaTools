---
name: laya-decide
description: Find code fast with `lt search` before exploring a project with Grep/Glob/Read, and make routine classifications (routing, urgency, guard checks) with `layatools decide` instead of reasoning them out. Use at the start of any "where is X / how does X work / where would I change X" task in a codebase, and whenever a named decision profile fits.
---

# Search first with `lt`, decide cheaply with `layatools`

## Finding code

Before grepping or reading files blindly, run:

```bash
lt search "<the question in plain words>" -k 5
```

It prints `path:start-end` plus a short snippet per hit, best first. Read the top hit's lines
first; open further hits only if that does not answer the question. The index lives outside the
project (`~/.cache/layatools/`) and re-syncs itself on every call, so it is never stale and needs
no setup. Useful flags:

- `-C <dir>`: search another project.
- `--per-file 2`: more than one chunk per file (long files).
- `--json`: machine-readable hits.
- `--judge`: re-rank with the fine-tuned Laya relevance judge (~300 ms instead of ~25 ms). On
  the real-project eval it improves the rank on about 4x as many questions as it hurts (19 better
  / 5 worse, p~0.007, MRR +0.05). Add it when the question is vague or the first run's top hits
  look off; skip it for questions that name a file or symbol.
- `--lexical`: keyword-only (skip embeddings), if semantic results are noisy.

`lt index --embed` (once per project) enables semantic search; without it `lt` is keyword +
file/symbol-name search, which still finds the right file in the top 5 most of the time. If `lt`
returns nothing useful after one rephrase, fall back to Grep/Glob as usual.

## Routine decisions

For classifications with a named profile, call it instead of reasoning:

```bash
layatools list                                   # available profiles and what each answers
layatools decide <profile> "<text>"              # or --file <path>
```

The result is `{"answers": {id: {"value", "conf"}}, "escalate": [ids]}`. Use the answers as
given; decide the ids under `escalate` (low confidence) yourself. Project profiles live in
`<project>/.layatools/profiles/*.yaml`, user-wide ones in `~/.config/layatools/profiles/`.
