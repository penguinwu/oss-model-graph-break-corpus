# OSS Model Compiler Quality Corpus

> **Note (2026-05-02):** Skill-discovery work has moved to a separate repo,
> [penguinwu/pt2-skill-discovery](https://github.com/penguinwu/pt2-skill-discovery)
> (`~/projects/pt2-skill-discovery/`). The `discovery/` dir was removed from
> this repo on 2026-05-02 (Phase 6b of the migration); recoverable from git
> history at sha `7af8010`'s parent if needed. All skill-discovery work
> happens in the new repo.

## What This Is

A corpus of 734 open-source models for surfacing `torch.compile` quality issues — graph breaks, numerical divergences, NaN/Inf, shape mismatches, infrastructure failures, and whatever else the compiler trips on. Tracks signals across PyTorch versions with classification, root cause analysis, and issue management. The corpus name still says "graph break" for historical reasons; the scope has grown.

Models come from HuggingFace Transformers, Diffusers, and custom repos. Each model is tested in eval and train modes.

## Project Philosophy

The corpus surfaces compiler errors. It is not a pass/fail certification system.

Every divergence, every graph break, every error is a data point to investigate or file. Never propose a "good enough rate" or an "acceptance threshold." Classification (`numerical_drift` / `divergence` / `nan_inf` / `shape_mismatch`) is for triage order, not for grading what we accept.

If you catch yourself asking "what's the success target?" — you're in the wrong frame. Ask "what gets filed first?" instead.

## Scope Posture

The project is past funding — no fixed scope to defend. We organically expand by tracking the line. Promote a use case to active when it has a real consumer. Demote when speculative. Don't reflexively scope-protect, don't over-commit.

When positioning a use case to consumers, name the niche it serves best. Don't oversell beyond strengths.

Methodology and design rationale live in `design/design-doc.md` (also at Google Doc `1paCL1R8xoN6OajND8c4M5WgA68Uw1iEij-katYFqneM`). This file (CLAUDE.md) is for *how to operate*; the design doc is for *why we built it this way*.

## Checking pytorch/pytorch PR Status — MANDATORY rule

**Never use GitHub's `merged: true/false` field alone as proof of whether a pytorch/pytorch PR has landed.** PyTorch uses ghstack for many PRs. Ghstack lands changes via PyTorch MergeBot (an internal merge flow), NOT via GitHub's Merge button. **GitHub's `merged` field shows `false` even when the commit has actually landed on main and shipped in release branches.**

This has misled us before. On 2026-05-03, we incorrectly reported PR #179611 ("[dynamo] Support copy.deepcopy via polyfill") as "closed without merging" — when in fact it landed via ghstack as commit `61fdec7ddb5d` on 2026-04-11 and IS in `release/2.12`. Communicating this wrong status to compiler developers erodes trust.

**Always use `tools/pr_landing_check.py` to check pytorch PR status.** It handles all four real verdicts:
- `LANDED_GH` — merged via the GitHub button (rare for pytorch nowadays)
- `LANDED_GHSTACK` — merged via ghstack/MergeBot (commit on main; GitHub shows `merged: false`)
- `NOT_LANDED` — closed without any commit reaching main
- `OPEN` — still being reviewed

The script also checks whether the landed commit is in a specific release branch (`--branch release/2.12`).

```bash
# Single PR
python3 tools/pr_landing_check.py 179611

# Check if it's in the upcoming 2.12 stable
python3 tools/pr_landing_check.py 179611 180585 --branch release/2.12
```

**Rule scope:** any time you need to answer "did this pytorch PR land?" or "is this pytorch PR in release X?" — use this script. Do NOT paste GitHub UI quotes / `gh api` PR JSON as authority. Quote `pr_landing_check.py`'s output.

## Closure Discipline

Before marking an item closed in `OPEN-LOOPS.md`, in a status report, or in any "recently fixed" list: verify the artifact exists on disk and is committed. A claim of "shipped" with no commit, or "wrapper built" with no file, corrodes trust faster than the slip itself. If the work isn't on disk, it isn't closed — keep it open with a note about what's blocking.

## Validate Invariants Before Reporting New Experiments

Every new sweep adds a *dimension* — correctness, dynamic shapes, a new compile flag, a new backend. Everything **upstream** of that dimension (model creation, eager execution, anything that doesn't depend on the new thing) must match a comparable prior sweep on the same PyTorch version and model set. Before reporting the headline result, identify what should be invariant and verify it against a baseline. If invariants don't match, report the violation **first** — it likely signals a methodology bug that invalidates the headline.

**How to find what to check:** trace the experiment as a graph. The new dimension is the only thing that should produce new variation. Everything pre-new-dimension should match. For correctness: new = compare outputs; upstream = creation + eager forward; baseline = the regular graph-break sweep on the same PT version. For dynamic shapes: new = trace with dynamic shapes; upstream = creation + static-shape eager; baseline = static-shape sweep.

**Where the baselines live:** see `EXPERIMENTS.md` at the repo root. Add a row when shipping a new experiment type.

*Why:* The Phase 3 correctness sweep produced 169 `create_error` results vs 16 in the pt2.11 graph-break baseline a few days earlier; the 169 were `full_graph` in baseline. The Phase 3 worker had a model-creation bug in the wrapper-variant path that smoke testing didn't cover. We almost reported 12 verification failures that weren't trustworthy.

## Experimentation Discipline (2 tiers)

The corpus runs different kinds of experiments at different cadences and stakes. The discipline scales accordingly.

| Experiment type | Tier | Smoke test | Plan.md gates | Launcher gate | Lifecycle doc |
|---|---|---|---|---|---|
| Sweep / correctness / AutoDev / `tools/run_experiment.py` | **B** — lighter | mandatory (`tools/smoke_test.py`) | n/a (cron-scheduled) | recommended on schema change | "Validate Invariants" section above |
| One-off ad-hoc analysis | **C** — informal | recommended | n/a | n/a | universal floor below |

(Tier A — Discovery experiments — moved to pt2-skill-discovery; see that repo's `CLAUDE.md` and `design/experiment_lifecycle.md`.)

### Universal floor (applies to all tiers)

> Before launching any new run, every loose end from prior runs is CLOSED with verified fix or EXPLICITLY DEFERRED with written reason. No silent drops. No "I'll fix it on the next run."

This rule was hardened on 2026-04-27 after a chain of V8 discovery experiments cascaded multi-hour failures because each "next run" papered over an unresolved loose end from the prior. The pattern: launch → bug surfaces → "I'll fix it next run" → re-launch → same bug surfaces → repeat. The closure rule is the antidote.

### Tier B — Sweep + correctness + AutoDev

**Trigger:** any change to `sweep/worker.py`, `sweep/orchestrator.py`, `sweep/models.py`, `sweep/explain.py`, `sweep/run_sweep.py`, or `tools/run_experiment.py`. ALSO: before proposing or launching any full sweep.

**Procedure:** read and follow `skills/test-sweep-changes/SKILL.md`. It defines the 5 testing gates (unit → smoke → single-trial → reproducibility → mini-sweep), exact commands, pass criteria, and the commit-message gate-evidence template.

**Discipline:** state your gate progression in the conversation as you go. Skipping a gate requires Peng's explicit approval, in writing.

The "Validate Invariants Before Reporting New Experiments" section above is the parallel doctrine for *post-launch* result reporting. Gates 1-5 are *pre-launch*; invariants validation is *pre-headline*.

### Tier C — Ad-hoc analysis

No formal gates. The universal floor still applies: if you change a script, run it before reporting results.

## Discovery Experiments

Migrated to [penguinwu/pt2-skill-discovery](https://github.com/penguinwu/pt2-skill-discovery). See that repo's `CLAUDE.md`, `design/experiment_lifecycle.md`, and `experiments/README.md` for the convention. Scaffold tools (`new_experiment.py`, `new_case_issue.py`, `queue_task.py`) move with the migration; corpus's `tools/queue_task.py` continues to manage the cross-repo board.

The local fallback dir is `corpus/discovery/` — see `discovery/MIGRATED.md` for the dual-run policy.

## Script Map

The codebase has two layers: **sweep/** runs the actual tests, **tools/** analyzes results and manages issues.

### Sweep execution (sweep/)

| Script | Purpose | When to use |
|--------|---------|-------------|
| `sweep/run_sweep.py` | Sweep engine — identify + explain passes | Direct sweep: `python sweep/run_sweep.py sweep` |
| `sweep/worker.py` | Single-model subprocess worker | Debugging: `python sweep/worker.py --model hf/ModelName` |
| `sweep/orchestrator.py` | Parallel dispatch, checkpointing, GPU health | Imported by run_sweep.py (not called directly) |
| `sweep/explain.py` | Graph break analysis (shared) | Imported by worker.py |
| `sweep/models.py` | Model enumeration (HF, Diffusers, TIMM, custom) | Imported by run_sweep.py |
| `sweep/sweep_watchdog.py` | Progress monitor, auto-restart | Long sweeps: `python sweep/sweep_watchdog.py` |

### Unified CLI (tools/)

| Script | Purpose | When to use |
|--------|---------|-------------|
| `tools/run_experiment.py` | Unified front-end — wraps sweep + experiments + nightly | `python tools/run_experiment.py <subcommand>` |
| `tools/file_issues.py` | Post-sweep issue management (sweep-report + sweep-update) | After a sweep completes |
| `tools/analyze_sweep.py` | Status breakdown by source/variant/mode | Quick results overview |
| `tools/analyze_explain.py` | Graph break taxonomy and root cause analysis | Deep-dive on break reasons |
| `tools/analyze_trend.py` | Cross-version trend analysis | Comparing PyTorch releases |
| `tools/compare.py` | Compare two sweep results | Before/after comparison |
| `tools/update_corpus.py` | Merge sweep results into corpus.json | After a sweep, before committing |
| `tools/check_pr_status.py` | Check whether a PyTorch PR landed on main | Before asserting a PR's status |
| `tools/reproduce.py` | Reproduce a single model's graph break | Debugging a specific model |
| `tools/query.py` | Query the corpus (by status, error, etc.) | Exploring corpus data |
| `tools/validate.py` | Corpus integrity checks | After modifying corpus.json |
| `tools/smoke_test.py` | 3-model infrastructure smoke test | Verifying sweep infra works |

### run_experiment.py subcommands

| Subcommand | Purpose |
|------------|---------|
| `sweep` | Run a two-pass sweep (wraps sweep/run_sweep.py) |
| `explain` | Explain-only pass from prior identify results |
| `correctness` | Phase 3 — eager vs compiled forward output comparison on `fullgraph_ok` models (writes `correctness/correctness_results.json`) |
| `nightly` | Full automated pipeline: refresh → staleness check → preflight → canary → sweep → explain → corpus update → summary |
| `corpus` | Build/update corpus from sweep results |
| `selftest` | 3-model smoke test |
| `check-env` | Pre-sweep environment validation |
| `refresh-nightly` | Upgrade nightly venv to latest PyTorch |
| `template` / `validate` / `run` / `merge` | Config-driven experiment system (see [experiments/README.md](experiments/README.md)) |

## Common Workflows

### Run a full sweep

```bash
# Activate your venv (needs PyTorch + transformers + diffusers)
source ~/envs/torch-nightly/bin/activate

# Full sweep: identify + explain, HF + Diffusers + custom models
python sweep/run_sweep.py sweep \
    --source hf diffusers custom \
    --workers 4 \
    --timeout 180

# Or via the unified CLI (equivalent)
python tools/run_experiment.py sweep \
    --source hf diffusers custom
```

### Run the nightly pipeline (automated)

```bash
python tools/run_experiment.py nightly \
    --venv ~/envs/torch-nightly \
    --source hf diffusers custom
```

This runs: venv refresh → staleness check (abort if PyTorch unchanged) → preflight → canary (1 model gate) → full sweep → explain → corpus update → summary.

### Post-sweep: classify and update issues

```bash
# Generate a reviewable plan (read-only, safe to run anytime)
python tools/file_issues.py sweep-report \
    --explain sweep_results/nightly/2026-04-19/explain_results.json \
    --identify sweep_results/nightly/2026-04-19/identify_results.json

# Review the plan file, then apply
python tools/file_issues.py sweep-update \
    --plan sweep_results/nightly/2026-04-19/sweep-report.json
```

sweep-report classifies every graph break into an issue category, computes leverage rankings (which fixes unlock the most fullgraph models), generates issue bodies with affected model tables, and flags close candidates with evidence. sweep-update reads the reviewed plan and PATCHes GitHub issues.

### Run experiments (flag tests, ablations, model subsets)

```bash
# Generate a starter config
python tools/run_experiment.py template > experiments/configs/my-test.json

# Edit the config: set models (list, corpus_filter, sample, all),
# dynamo_flags per config variant, and execution settings

# Validate and preview
python tools/run_experiment.py validate experiments/configs/my-test.json
python tools/run_experiment.py run experiments/configs/my-test.json --dry-run

# Run
python tools/run_experiment.py run experiments/configs/my-test.json

# Merge incremental results into an existing sweep
python tools/run_experiment.py merge --from experiments/results/my-run/ --into sweep_results/pt2.11/
```

Full config schema, output format, and recipes: [experiments/README.md](experiments/README.md)

### Reproduce and debug a graph break

```bash
python tools/reproduce.py ModelName --explain          # Show break reasons
python tools/reproduce.py ModelName --explain --verbose # Full explain output
python tools/reproduce.py ModelName --dynamic mark     # Test with dynamic shapes
TORCH_TRACE=/tmp/trace python tools/reproduce.py ModelName  # Capture trace
```

### Query the corpus

```bash
python tools/query.py                          # Summary
python tools/query.py --status graph_break     # All graph break models
python tools/query.py --error deepcopy         # Search by error text
```

### Compare sweep results

For two-sweep comparison with invariant-checked partition + per-pattern segmentation:

```bash
python tools/sweep_compare.py \
    --baseline sweep_results/nightly/<prior-week> \
    --current  sweep_results/nightly/<this-week> \
    --out experiments/<this-week>-nightly-report.md
```

For per-pattern delta segmented by partition category (the only honest way to ask "did pattern X get worse"):

```bash
python tools/sweep_compare.py --baseline X --current Y \
    --pattern "aten._local_scalar_dense.default"
```

Older `tools/compare.py` is for static-vs-dynamic-mark within ONE sweep (different use case):

```bash
python tools/compare.py --corpus-dynamic
```

### Compose the weekly nightly-sweep brief (for PT2 dynamo team)

When generating a weekly nightly-sweep summary intended for external broadcast (Workplace post + signal boost), use the `weekly-sweep-brief` skill:

- Skill: `skills/weekly-sweep-brief/SKILL.md` (workflow)
- Template: `skills/weekly-sweep-brief/template.md` (8 fixed sections)
- Methodology: `skills/weekly-sweep-brief/methodology.md` (hard rules R1-R7 + soft rules + self-check checklist)

DO NOT iterate the brief structure from scratch each week — the template + methodology are the encoded learning from the 2026-05-04 brief composition session. Skipping the methodology rules has produced wrong numbers / leaked attribution / duplicated issues in the past.

## Conventions

- Batch size must be >= 2 (PyTorch specializes on 0 and 1)
- Backend is always `eager` (tests Dynamo tracing, not inductor codegen)
- Default sources: `hf diffusers custom` (TIMM/dynamic require explicit request)
- Never use 0 or 1 as input dimensions for dynamic shape testing
- Sweep results go in `sweep_results/<label>/` (e.g., `sweep_results/nightly/2026-04-19/`)
- GitHub is canonical for project artifacts (design docs, proposals, charters). Drive is a mirror for share convenience.

## Key Data Files

- `corpus/corpus.json` — main dataset (models + results across versions)
- `sweep_results/nightly/<date>/identify_results.json` — latest sweep raw results
- `sweep_results/nightly/<date>/explain_results.json` — graph break explanations
- `sweep/large_models.json` — models needing extended timeouts
- `sweep/tracked_models.json` — models tracked for specific PR fixes

## Model-Specific Fixes

Fixes for individual models live in `sweep/worker.py`:
- `_fix_config()` — patch config values (e.g., reduce vocab, fix invalid defaults)
- `_create_config()` — composite models needing factory construction
- `_generate_inputs()` — models with non-standard input signatures
- `_reduce_model_size()` — cap layers/hidden dims for GPU fit

After fixing, test with: `python sweep/worker.py --model hf/ModelName --device cuda`

## GitHub Issues

Issues track graph break patterns at https://github.com/penguinwu/oss-model-graph-break-corpus/issues.

- **Dynamo issues** — pattern-level graph break categories (e.g., data-dependent branching, context managers)
- **Model-specific issues** — breaks unique to individual models
- **Corpus-infra issues** — models that fail before compilation (create_error, timeout, eager_error)

The classifier rules live in `tools/file_issues.py` (`GRAPH_BREAK_RULES`). Each rule maps a break reason pattern to an issue number.

Issue bodies include: affected model tables, break reason samples, leverage analysis (models to fullgraph if fixed), and cross-references to related issues. All machine-filed issues contain `<!-- filed-by: otter/file_issues.py -->`.

### Pattern-delta discipline — never aggregate across partition categories

**When computing how a break-reason pattern's count changed between two sweeps, segment by partition category from the start.** Never compute a single corpus-wide `current_total - baseline_total` scalar — that subtraction is mathematically valid but semantically meaningless because it conflates three operationally-distinct populations:

- **cat 3 (compile-success in BOTH sweeps)** — only here can a delta mean "Dynamo regression" or "Dynamo improvement." Apple-to-apple.
- **cat 1 (was error in baseline → success in current)** — patterns that became newly observable because eager / harness fixes unblocked compile testing. This is **EXPOSURE**, not regression.
- **cat 4 (truly new model in current)** — patterns from models that didn't exist in baseline. Also EXPOSURE.

A "+41 corpus-wide regression" finding on 2026-05-04 was wrong precisely because it summed cat 3 (improvement) with cat 1 + cat 4 (exposure) into one scalar.

**Mechanical guard:** use `tools/sweep_compare.py --pattern "<substring>"` (or `pattern_delta()` API) for any per-pattern delta question. The tool returns the segmented breakdown; it intentionally does NOT expose a single corpus-wide scalar.

**Naming discipline:** never write "X regressed by Y" or "X improved by Y" without naming the population (cat 3). Cat 1 / cat 4 contributions are "exposure" or "newly observed."

### Search existing issues BEFORE filing new ones

Before filing a new dynamo / transformers issue: grep open + closed issues for the break-reason substring, the operator name, OR the source location. If a tracked issue already covers the same root cause, **update that issue with current data** instead of filing a new one. Filed-then-closed-as-duplicate creates noise and erodes trust.

```bash
# Quick search via gh API (replace SUBSTR with a distinctive fragment of the break reason)
gh issue list --repo penguinwu/oss-model-graph-break-corpus --state all --search "SUBSTR" --limit 20
```

This rule was learned the hard way on 2026-05-04: I split issue #8 into 5 sub-issues (#112-#116) without searching, and 3 of them (#113 lock+ContextVar, #114 setattr, #115 Tensor.item) were duplicates of existing #11/#23/#24/#56. Closed and redirected, but left fingerprints on the tracker.

### Umbrella-split policy

**One issue = one specific GB type.** Umbrella issues that bundle multiple distinct root causes never close — they hide actual progress. Split them.

- An umbrella issue is one where the breaks under it have ≥2 distinct underlying root causes (different operators, different Dynamo limitations, different source files needing different fixes).
- A "specific GB type" issue tracks ONE root cause: one operator (e.g., `aten.nonzero.default`), one Dynamo limitation (e.g., `setattr` builtin), or one source-file refactor target.
- Each specific issue must have a closure criterion that's a single sweep_compare query (e.g., "ZERO occurrences of pattern X in current sweep").
- Same operator at many sites is NOT an umbrella — it's one specific issue (the fix is the same: support the op or avoid it). Don't split per-site unless the fixes genuinely differ.
- The "trace CALL" / "Encountered graph break when attempting to trace CALL" pattern is bytecode-instruction wrapper, NOT a root cause. Always classify by the underlying explanation/operator beneath the trace-CALL framing.

When you find an umbrella in an existing issue: file per-root-cause sub-issues with the data-derived model lists, comment on the umbrella with the taxonomy + cross-refs, then close the umbrella as `not_planned` (state_reason). Cross-link sub-issues back to the original.

Decided 2026-05-04 after sweep #57 found that issue #8 ("DETR proxy CALL", 234 breaks, 54 models) actually contained 8+ distinct root causes — split into #112-#116 (and re-attributed sub-buckets to existing #18, #55).
