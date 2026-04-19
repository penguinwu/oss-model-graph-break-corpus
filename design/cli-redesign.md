# run_sweep.py CLI Redesign

**Author:** Otter (builder agent)
**Date:** 2026-04-15
**Status:** Approved by Peng, ready for implementation
**Project:** OSS Model Graph Break Corpus

---

## Problem

`run_sweep.py` has grown organically to 25 flags in a flat namespace. Four different operations (sweep, explain, validate, smoke test) share one `--help` output, making it hard to know which flags apply when. Model selection uses 6+ flags with implicit ordering and non-obvious interactions (e.g., `--models` silently overrides `--source`). Flag names leak implementation jargon (`--dynamic true/mark`, `--subset`).

## Design Principles

1. **Subcommands over mode flags.** Different operations get different `--help`.
2. **Composable model selection.** Filters are lists you combine, not pre-baked combos.
3. **Mutually exclusive flags error loudly.** No silent overrides.
4. **User language over implementation jargon.** `--stability unstable` not `--subset unstable`. `--dynamic-dim batch` not `--dynamic mark`.
5. **Internalize rarely-changed settings.** Config files and hardcoded defaults over CLI flags.

## Subcommands

```
run_sweep.py <command> [options]

Commands:
  sweep       Identify + explain sweep (default if omitted)
  explain     Explain-only from prior identify results
  validate    Two-shape correctness check
```

Each subcommand has its own `--help` showing only relevant flags.

### What happened to test and check-env?

- `--test` → `sweep --selftest` (quick smoke test: 3 models, both passes, exit)
- `--check-env` → `sweep --check-env` (validate versions against corpus, exit)

These are utilities, not distinct workflows. They live as flags on `sweep`.

## Flag Reference

### Global Options (all subcommands)

| Flag | Default | Description |
|------|---------|-------------|
| `--device {cpu,cuda}` | `cuda` | Hardware target |

Note: `--python` was removed. The sweep uses `sys.executable` (activate your venv first). For automation, set `SWEEP_PYTHON` env var. Exact versions are embedded in results metadata at sweep start.

### Run Options (sweep, explain, validate)

| Flag | Default | Description |
|------|---------|-------------|
| `--workers N` | `4` | Parallel worker processes |
| `--timeout N` | `180` | Per-model timeout in seconds |
| `--output-dir PATH` | `sweep_results/` | Output directory |
| `--resume` | off | Resume from JSONL checkpoint |

Workers default changed from 16 → 4. Fewer flaky results from GPU contention. Use `--workers 16` explicitly when speed matters.

### sweep — Model Selection

Applied in this order: source → stability filter → limit.

| Flag | Default | Description |
|------|---------|-------------|
| `--source {timm,hf,diffusers,custom,all}` | `hf diffusers custom` | Model libraries to enumerate. Accepts multiple values. `all` is shorthand for all four (mutually exclusive with individual values). |
| `--models FILE` | — | Explicit JSON model list. **Mutually exclusive with `--source`.** |
| `--stability {stable,unstable}` | — (run all) | Filter by corpus stability. Omit to run all. `stable` = full_graph in all modes, `unstable` = graph_break/error/new. **Only valid with `--source`.** |
| `--limit N` | — | Hard cap on model count (applied last). Works with both `--source` and `--models`. |

Mutual exclusion rules:
- `--source` vs `--models`: error if both specified
- `--stability` with `--models`: error (filter only applies to enumerated models)

### sweep — Execution

| Flag | Default | Description |
|------|---------|-------------|
| `--modes {eval,train}` | `eval train` | Modes to run. Accepts multiple values. |
| `--dynamic-dim {batch,all}` | — (static) | Dynamic shapes: `batch` = batch dim only, `all` = all dims. Omit for static shapes. |
| `--no-auto-retry` | off | Skip automatic retry of timed-out and errored models |
| `--identify-only` | off | Stop after identify pass (skip explain) |

### sweep — Utilities

| Flag | Description |
|------|-------------|
| `--selftest` | Run 3 models through both passes, validate output format, exit |
| `--check-env` | Validate installed PyTorch/transformers/diffusers versions against corpus, exit |

### explain

```
run_sweep.py explain FILE [run-options]
```

`FILE` (positional): Path to identify results JSON from a prior sweep.

Only global + run options apply. No model selection flags — the input file determines what to explain.

### validate

```
run_sweep.py validate [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--from FILE` | — | Identify results JSON; validates `full_graph` models from it. **Mutually exclusive with `--models`.** |
| `--models FILE` | — | Explicit model list to validate. **Mutually exclusive with `--from`.** |
| `--dynamic-dim {batch,all}` | `all` | Default differs from sweep — validation defaults to all-dim dynamic. |

Plus global + run options.

## What Was Removed

| Old Flag | Disposition |
|----------|-------------|
| `--explain-from FILE` | Now the `explain` subcommand: `explain FILE` |
| `--validate-from FILE` | Now `validate --from FILE` |
| `--validate` | Now the `validate` subcommand |
| `--test` | Now `sweep --selftest` |
| `--subset {all,stable,unstable}` | Replaced by `--stability {stable,unstable}` (list-based, like `--source`) |
| `--python PATH` | Dropped. Use venv activation or `SWEEP_PYTHON` env var. |
| `--skip-traces` | Dropped. TORCH_TRACE collection removed from sweep entirely. Use `reproduce.py` for deep debugging individual models. |
| `--timeout-large N` | Internalized. Auto-retry uses 3x base timeout. |
| `--large-models FILE` | Internalized. Always reads `sweep/large_models.json`. |
| `--has-config-only` | Internalized. Folded into diffusers enumeration default. |
| `--skip-models FILE` | Internalized. Auto-loads `sweep/skip_models.json` if present. |
| `--identify-modes` / `--explain-modes` | Merged into single `--modes` flag. Both passes use same modes. |
| `--dynamic {true,mark}` | Renamed to `--dynamic-dim {batch,all}` for clarity. |

## Source Default Change

Old default: `--source all` (includes TIMM).
New default: `--source hf diffusers custom` (excludes TIMM).

Rationale: Our standard sweep scope is HF + Diffusers + Custom. TIMM models require explicit opt-in via `--source timm` or `--source all`.

## Merging Partial Results

When running `sweep --stability unstable`, the sweep produces results for unstable/new models only. These partial results need to merge back into the full corpus.

**Design:** The sweep tool stays dumb — it runs models and writes results. A separate `update_corpus.py` handles the merge:

```bash
# Standard merge (auto-discovers identify + explain + versions)
python tools/update_corpus.py sweep_results/pt2.10/

# Dry run — show changelog without writing
python tools/update_corpus.py sweep_results/pt2.10/ --dry-run

# Skip post-merge validation
python tools/update_corpus.py sweep_results/pt2.10/ --no-validate
```

The tool auto-discovers files in the sweep directory:
- `identify_results.json` (required) — model status data, with versions embedded in `metadata.versions`
- `explain_results.json` (optional) — graph break reasons and counts

Version info is embedded in results metadata by `run_sweep.py` — no separate `versions.json` file. This ensures version info can't be misplaced or separated from the data it describes.

**Version safety (overlay mode):** Before merging, the tool compares PyTorch, transformers, and diffusers versions from results metadata against the corpus. If any differ, the merge is blocked — partial results from different versions produce a mixed corpus. Use `--force` to override (not recommended). For version upgrades, use `--replace` with a full sweep.

Merge rules:
- For each model+mode in the sweep results, replace that entry in the corpus
- Models not in the sweep keep their existing corpus entries unchanged
- New models (in sweep but not in corpus) are added at the end
- No model is silently removed — deletion is an explicit action
- `has_graph_break` is recomputed for all affected models
- Corpus metadata (versions, last_updated) is updated from results metadata

The tool generates a changelog (`{sweep_dir}/changelog.md`) categorizing all changes:
- **Regressions** — full_graph → graph_break
- **Fixes** — graph_break → full_graph
- **New model families** — entirely new architectures
- **New configurations** — task-head variants of existing families

This separation means you can inspect raw sweep results before committing them to the corpus.

## Flag Count

| Metric | Before | After |
|--------|--------|-------|
| Total unique flags | 25 | 16 |
| Max flags in any `--help` | 25 | ~15 (sweep) |
| Flags shown for `explain --help` | 25 | 7 |
| Flags shown for `validate --help` | 25 | 9 |

## Common Invocations

```bash
# Full sweep (most common)
python run_sweep.py sweep

# Incremental: only unstable models
python run_sweep.py sweep --stability unstable

# Source-specific
python run_sweep.py sweep --source timm

# Combined: unstable HF models only, eval mode
python run_sweep.py sweep --source hf --stability unstable --modes eval

# Resume after crash
python run_sweep.py sweep --resume

# Explain from prior results
python run_sweep.py explain sweep_results/identify_results.json

# Validate correctness
python run_sweep.py validate --from sweep_results/identify_results.json

# Quick smoke test
python run_sweep.py sweep --selftest

# Pre-sweep version check
python run_sweep.py sweep --check-env
```
