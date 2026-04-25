# OSS Model Compiler Quality Corpus

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

## Closure Discipline

Before marking an item closed in `OPEN-LOOPS.md`, in a status report, or in any "recently fixed" list: verify the artifact exists on disk and is committed. A claim of "shipped" with no commit, or "wrapper built" with no file, corrodes trust faster than the slip itself. If the work isn't on disk, it isn't closed — keep it open with a note about what's blocking.

## Validate Invariants Before Reporting New Experiments

Every new sweep adds a *dimension* — correctness, dynamic shapes, a new compile flag, a new backend. Everything **upstream** of that dimension (model creation, eager execution, anything that doesn't depend on the new thing) must match a comparable prior sweep on the same PyTorch version and model set. Before reporting the headline result, identify what should be invariant and verify it against a baseline. If invariants don't match, report the violation **first** — it likely signals a methodology bug that invalidates the headline.

**How to find what to check:** trace the experiment as a graph. The new dimension is the only thing that should produce new variation. Everything pre-new-dimension should match. For correctness: new = compare outputs; upstream = creation + eager forward; baseline = the regular graph-break sweep on the same PT version. For dynamic shapes: new = trace with dynamic shapes; upstream = creation + static-shape eager; baseline = static-shape sweep.

**Where the baselines live:** see `EXPERIMENTS.md` at the repo root. Add a row when shipping a new experiment type.

*Why:* The Phase 3 correctness sweep produced 169 `create_error` results vs 16 in the pt2.11 graph-break baseline a few days earlier; the 169 were `full_graph` in baseline. The Phase 3 worker had a model-creation bug in the wrapper-variant path that smoke testing didn't cover. We almost reported 12 verification failures that weren't trustworthy.

## Discovery Experiments

Multi-trial discovery experiments (skill-evaluation studies, agent-strategy studies) are tracked under `discovery/experiments/<YYYY-MM-slug>/`. Each experiment has:

- `plan.md` — methodology, matrix, open questions, stop conditions
- `reports/<case>.md` — per-case findings (one per model)
- `synthesis.md` — cross-case writeup (deferred until all cases close)

Convention is documented in `discovery/experiments/README.md` and enforced by `tools/check_experiments.py` (run nightly via the daily brief).

**Use the scaffold tools — do NOT hand-roll directories or issues:**

| Tool | When |
|---|---|
| `tools/new_experiment.py "<slug>" --title "<Title>"` | Starting a new experiment |
| `tools/new_case_issue.py <experiment-slug> <case_id> "<Model name>"` | Adding a case to an existing experiment |
| `tools/queue_task.py "<title>" [--umbrella N]` | Deferring work — creates a Backlog card on project board #1 so the commitment survives session end |

The board is the canonical source of "agreed but not started" work. TodoWrite is in-conversation only; OPEN-LOOPS.md is project-level facts; the board is indefinite-lifetime and visible without local access. When you commit to deferred work, queue it.

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

```bash
python tools/compare.py sweep_results/pt2.11/ sweep_results/nightly/2026-04-19/
```

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
