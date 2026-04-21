# OSS Model Compiler Quality Corpus

## What This Is

A corpus of 734 open-source models for measuring `torch.compile` quality. Tracks `fullgraph=True` success rates across PyTorch versions, with graph break classification, root cause analysis, and issue management.

Models come from HuggingFace Transformers, Diffusers, and custom repos. Each model is tested in eval and train modes.

## Project Philosophy

The corpus surfaces compiler errors. It is not a pass/fail certification system.

Every divergence, every graph break, every error is a data point to investigate or file. Never propose a "good enough rate" or an "acceptance threshold." Classification (`numerical_drift` / `divergence` / `nan_inf` / `shape_mismatch`) is for triage order, not for grading what we accept.

If you catch yourself asking "what's the success target?" — you're in the wrong frame. Ask "what gets filed first?" instead.

Methodology and design rationale live in `design/design-doc.md` (also at Google Doc `1paCL1R8xoN6OajND8c4M5WgA68Uw1iEij-katYFqneM`). This file (CLAUDE.md) is for *how to operate*; the design doc is for *why we built it this way*.

## Closure Discipline

Before marking an item closed in `OPEN-LOOPS.md`, in a status report, or in any "recently fixed" list: verify the artifact exists on disk and is committed. A claim of "shipped" with no commit, or "wrapper built" with no file, corrodes trust faster than the slip itself. If the work isn't on disk, it isn't closed — keep it open with a note about what's blocking.

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
