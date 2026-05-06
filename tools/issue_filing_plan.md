---
plan: WS2 — Issue filing friction
status: active
owner: Otter
created: 2026-04-24
last_check: 2026-05-06
forcing_function: tools/check_plan.py + daily brief at 7:30 AM ET
---

# §1 — Lower friction for filing dynamo correctness issues

## Goal

Make filing high-quality dynamo correctness issues a 1-command operation, with rich enough body content that Animesh / dynamo team can triage without bouncing back. Stay under Animesh's 25-issue cap.

## State

- `tools/file_issues.py` has `correctness-report` and `correctness-apply` subcommands (smoke-tested 2026-04-23 against 528-model sweep → 7 family issues, max severity 5.3e9 Phi4).
- 7 family issues identified; **strategy decision blocked** on Peng (open in OPEN-LOOPS.md "Phase 3 issue filing strategy").

## Decision needed (Tier 3)

Animesh budget: file 3 → at 25 (cap), file all 7 → 29 (over). Three options:

1. **Top 3 by severity** (Phi4, AIMv2, Doge) — stays under cap, ships highest-impact signal.
2. **Top 5 by severity** — over cap by 4. Requires Animesh's blessing on cap exception.
3. **All 7** — over cap by 4. Same blessing needed.

## Tasks

| Task | Type | Notes |
|------|------|-------|
| Get Peng's call on filing strategy | needs-input | Surface in daily brief until decided |
| Run `correctness-apply` with chosen plan | blocked | Posts to penguinwu/oss-model-graph-break-corpus/issues |
| Component triage: rerun divergent models with knobs flipped | backlog | Decompositions on/off, fake tensor mode, smaller graphs — narrows which compiler component drifts |

## Done means

- Filing strategy chosen and applied; issues live at penguinwu/oss-model-graph-break-corpus/issues.
- Each filed issue has: family description, affected models list, severity, repro snippet.
- Component triage backlog item moved to `done` or to a follow-on plan with explicit owner.

## Revision log

- 2026-04-24: Plan created from existing OPEN-LOOPS WS2 section.
- 2026-05-06: Renamed top section to §1 (was the whole file); added §2 below for upstream pytorch issue filing.

---

# §2 — Reproducible upstream pytorch issue filing

## Goal

Make filing a pytorch/pytorch issue (or follow-up comment) zero-bounce-back: the assembled body must include everything an upstream maintainer needs to reproduce on their own hardware without asking us for the script, the venv setup, or the env disclosure.

## Why this exists

2026-05-06 issue #182116 round-trip: Peng filed a perf-regression issue with a clear timing repro. Alban replied asking for (a) the cProfile script that produced the "2383 / 772" numbers in the body, (b) clarification on bounds. Both were knowable up front — the cProfile script existed at `/tmp/blt_init_repro.py`, just wasn't included. Cost: one round-trip of Alban's attention. With the right tool and forcing function, that round-trip is zero.

Cited as the cautionary tale at the top of `skills/sweep.md` for sweep launches; same pattern here for issue filing.

## The five must-haves

Every upstream issue / comment must include:

1. **Repro script** — verbatim Python, self-contained, no imports from local trees. Embedded in the body (not linked, not paraphrased).
2. **Setup commands** — exact `pip install` for each venv. For regressions: BOTH the working and broken envs.
3. **Captured output** — actual stdout from running the script in each env (gives the recipient a reference to compare against; also confirms the script runs cleanly on our side).
4. **Env disclosure** — `python -m torch.utils.collect_env` for each venv, OR at minimum torch version + git + OS + GPU + Python version.
5. **For perf issues** — profile output (cProfile / py-spy / autograd profiler) embedded.

## Tool

`tools/file_issues.py pytorch-upstream` (added 2026-05-06; lives alongside the corpus-repo subcommands to share GitHub API plumbing — anti-fragmentation per Peng's 2026-05-06 directive).

```bash
python tools/file_issues.py pytorch-upstream \
  --script /path/to/repro.py \
  --venv pt211:~/envs/torch211/bin/python \
  --venv pt212:~/envs/torch-nightly-cu128/bin/python \
  --pythonpath /home/pengwu/envs/modellibs/transformers-5.5.3 \
  --title "Perf regression: ... in PT 2.12" \
  --summary /tmp/intro.md \
  --output /tmp/issue_body.md
  # Add --post to actually create the issue. Default is dry-run.
  # For follow-up comments to existing issue: --comment 182116 (instead of --title)
```

What the tool does:
- Parses each `--venv NAME:PYTHON_BIN` (repeatable)
- Runs `import torch; print(torch.__version__); print(torch.version.git_version)` in each venv
- Runs `python -m torch.utils.collect_env` in each venv (skip with `--no-collect-env`)
- Runs the repro script in each venv (skip with `--no-run` if outputs are already captured / scripts are slow)
- Assembles the body in the standard template (Marker → Summary → Reproducer → Setup → Run → Captured output → Environment)
- Writes to `--output` (default stdout)
- If `--post`: POSTs to `pytorch/pytorch` via the existing `_proxy_api` helper

The marker `<!-- assembled-by: file_issues.py pytorch-upstream -->` at the top is a sentinel — future tooling can detect tool-assembled bodies vs hand-crafted ones for retroactive analysis.

## Forcing function

Local CLAUDE.md (`~/.myclaw/spaces/AAQANraxXE4/CLAUDE.md`) has a per-project trigger bullet for the corpus repo: "about to file a pytorch upstream issue or post a comment with a repro → use `tools/file_issues.py pytorch-upstream`, walk the five-must-haves checklist." Skipping requires Peng's written approval.

## Tasks

| Task | Type | Notes |
|------|------|-------|
| Auto-derive `pip install` commands from each venv | DONE 2026-05-06 | Tool now probes torch + cuda variant + transformers/diffusers/timm versions per venv and templates `python3 -m venv ~/<NAME>_venv` + `pip install --pre torch==<VER> --index-url .../nightly/<CUDA>` + per-modellib `pip install <pkg>==<VER>` lines. |
| Portable Run section (`~/<NAME>_venv/bin/python repro.py`, never absolute python_bin) | DONE 2026-05-06 | Same commit. |
| Scrub `/home/<user>/...` + `/usr/local/fbcode/...` from captured output before publishing | DONE 2026-05-06 | New `_scrub_paths()` helper applied to both script-run output and `collect_env` output. Internal absolute paths get rewritten to `.../`. Idempotent. |
| Use the tool to assemble the next upstream issue body end-to-end | needs-trigger | The 2026-05-06 Alban-thread follow-up (#182116 comment 4391902719) was assembled by the tool BUT had absolute paths that Peng had to point out — exactly the bug the fixes above address. Next filing should use the post-fix tool from the start. |
| Add `--captured-output NAME:PATH` flag (inject pre-captured output for a venv) | backlog | Useful when scripts are slow and outputs were captured manually. The `--no-run` flag currently leaves a placeholder; the inject flag would let users paste a captured stdout file. |

## Done means

- Tool is live (✓ shipped 2026-05-06).
- Plan documented in this file as §2 (✓).
- Trigger encoded in local CLAUDE.md (✓).
- Used end-to-end on at least one real upstream filing, with self-test (does the assembled body pass the five-must-haves test?) (pending — fires on next filing).
