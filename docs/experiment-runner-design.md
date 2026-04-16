# Experiment Runner Design Sketch

**Status:** Draft for review
**Date:** 2026-04-16

## Problem

We have three workflows that all compile models and collect results, but they use different tooling:

1. **Full sweep** (pt2.10, pt2.11, nightly) — uses `run_sweep.py`, well-tested
2. **Ad-hoc experiments** (flag quality test) — one-off scripts in `/tmp/`, no structure
3. **Incremental updates** (new models added) — manual, no merge tooling

The sweep orchestrator (`run_sweep.py`) has excellent infrastructure — parallel workers, checkpointing, GPU health, resume, timeout escalation — but it's monolithic. Experiments can't reuse it, and sweep results don't capture their full configuration.

## Proposed Architecture

```
sweep/
  orchestrator.py       ← EXTRACTED from run_sweep.py: run_pass(), checkpointing,
                           worker management, GPU health, timeout escalation
  worker.py             ← UNCHANGED: single-model subprocess
  models.py             ← UNCHANGED: model enumeration
  run_sweep.py          ← REFACTORED: thin wrapper that builds a config and
                           calls orchestrator. CLI stays identical.

tools/
  run_experiment.py     ← NEW: config-driven entry point, calls orchestrator

experiments/
  README.md             ← How to run experiments (the contract)
  configs/              ← Experiment recipe files (JSON)
  results/              ← One folder per run (self-describing)
```

### What gets extracted into `orchestrator.py`

These functions move out of `run_sweep.py` unchanged:

- `WorkerHandle` class
- `spawn_worker()` — subprocess launch with process group isolation
- `harvest_worker()` — result collection from completed workers
- `timeout_result()` — timeout result construction
- `escalating_kill()` — SIGTERM → SIGKILL escalation
- `load_checkpoint()` / checkpoint writing
- `run_pass()` — the main poll loop (work queue, spawn, harvest, timeout, GPU health)
- `check_gpu_health()` / `kill_gpu_zombies()`
- `_print_progress()`
- `_log_versions()` — environment capture

`run_sweep.py` keeps:
- `run_sweep()` main function (two-pass workflow, skip-stable, model enumeration)
- `load_corpus_stability()`, `load_large_models()`
- CLI argument parsing
- Single-worker model handling

### The refactor is mechanical

`run_sweep.py` imports from `orchestrator.py` instead of defining those functions locally. No logic changes. The CLI stays identical. This is a pure extraction — zero behavior change for sweeps.

## Experiment Config Schema

```json
{
  "name": "flag-quality-cso",
  "description": "Test capture_scalar_outputs on graph-break models",

  "models": {
    "source": "list",
    "names": ["AriaForConditionalGeneration", "FlaubertModel", "..."]
  },

  "configs": [
    {
      "name": "baseline",
      "dynamo_flags": {}
    },
    {
      "name": "capture_scalar_outputs",
      "dynamo_flags": {"capture_scalar_outputs": true}
    }
  ],

  "settings": {
    "device": "cuda",
    "modes": ["eval"],
    "workers": 4,
    "timeout_s": 180,
    "backend": "counting",
    "pass": 1
  }
}
```

### Model selection options

```json
// Explicit list
"models": {"source": "list", "names": ["Model1", "Model2"]}

// All models from enumeration
"models": {"source": "all"}

// Stratified sample
"models": {"source": "sample", "strategy": "stratified", "size": 50, "seed": 42}

// Only models with specific status in corpus
"models": {"source": "corpus_filter", "status": "graph_break"}

// New models not in existing results (incremental)
"models": {"source": "new_since", "baseline": "results/pt2.10/"}
```

### How people learn the schema

1. **`--template`** — generates a starter config with comments:
   ```bash
   python tools/run_experiment.py --template > experiments/configs/my-test.json
   ```

2. **`--validate`** — validates a config before running, with clear errors:
   ```bash
   python tools/run_experiment.py --validate experiments/configs/my-test.json
   # ERROR: Unknown dynamo flag "capture_scaler_outputs" — did you mean "capture_scalar_outputs"?
   ```

3. **`experiments/README.md`** — field-by-field reference with examples.

## Result Format (Self-Describing)

Every experiment run produces a folder:

```
experiments/results/2026-04-16-flag-quality-cso/
  config.json          # Input config + resolved model list + auto-captured env
  results.jsonl        # One line per (model, config, mode) — granular, mergeable
  summary.md           # Human-readable findings (auto-generated)
```

### `config.json` — resolved config with environment

```json
{
  "experiment": { ... original config ... },
  "resolved": {
    "models": ["actual", "list", "of", "models", "tested"],
    "model_count": 50
  },
  "environment": {
    "torch": "2.12.0.dev20260408+cu128",
    "transformers": "5.5.0",
    "python": "3.12.x",
    "cuda": "12.8",
    "gpu": "NVIDIA PG509-210",
    "corpus_git_commit": "abc123"
  },
  "execution": {
    "started": "2026-04-16T14:00:00",
    "finished": "2026-04-16T15:23:00",
    "duration_s": 4980
  }
}
```

### `results.jsonl` — per-model, per-config results

```json
{"model": "AriaForConditionalGeneration", "config": "baseline", "mode": "eval", "status": "graph_break", "graph_count": 29, "wall_time_s": 12.3}
{"model": "AriaForConditionalGeneration", "config": "capture_scalar_outputs", "mode": "eval", "status": "graph_break", "graph_count": 27, "wall_time_s": 11.8}
```

JSONL is important:
- **Per-line granularity** — merging is appending + deduplicating
- **Streaming** — can process without loading entire file into memory
- **Checkpoint-compatible** — same format as the orchestrator's checkpoint

## Sweep as Experiment

A full sweep is just an experiment with specific settings:

```json
{
  "name": "pt2.10-sweep",
  "description": "Full corpus sweep on PyTorch 2.10 stable",
  "models": {"source": "all"},
  "configs": [{"name": "default", "dynamo_flags": {}}],
  "settings": {
    "device": "cuda",
    "modes": ["eval", "train"],
    "workers": 4,
    "timeout_s": 180,
    "passes": ["identify", "explain"],
    "explain_filter": "graph_break",
    "skip_stable": true
  }
}
```

`run_sweep.py` continues to work as-is (same CLI). But behind the scenes it calls the shared orchestrator — and now every sweep automatically gets the self-describing result format.

## Incremental Merge

When new models are added:

```bash
# 1. Create an incremental experiment config
python tools/run_experiment.py --template --models-source new_since \
  --baseline experiments/results/pt2.10-sweep/ \
  > experiments/configs/pt2.10-incremental.json

# 2. Run it
python tools/run_experiment.py --config experiments/configs/pt2.10-incremental.json

# 3. Merge into existing results
python tools/run_experiment.py --merge \
  --source experiments/results/2026-04-20-pt2.10-incremental/ \
  --target experiments/results/pt2.10-sweep/
```

### Merge rules
- JSONL makes merge trivial: concatenate + deduplicate by (model, config, mode)
- If a model appears in both source and target, **source wins** (newer data)
- Merge updates `config.json` to record the merge event
- Merged result is indistinguishable from a full sweep to downstream tools

## Implementation Plan

### Phase 1: Extract orchestrator (zero behavior change)
- Move generic functions from `run_sweep.py` → `orchestrator.py`
- Update `run_sweep.py` to import from `orchestrator.py`
- Run smoke test to verify nothing breaks

### Phase 2: Build `run_experiment.py`
- Config parsing + validation + `--template`
- Call orchestrator with dynamo flag injection
- Self-describing result output (config.json + results.jsonl)
- Backfill Phase 1 flag quality results as first entry

### Phase 3: Incremental merge
- `--merge` subcommand
- `new_since` model source
- Merge logic with dedup

### Phase 4: Sweep integration (optional)
- Wire `run_sweep.py` to produce experiment-format results alongside existing output
- Preserve backward compatibility with existing result format

## Open Questions

1. **Should `run_sweep.py` fully migrate to experiment configs, or keep its current CLI?**
   My lean: keep both. The sweep CLI is proven; forcing configs on sweep users adds friction for no gain. The shared orchestrator gives us the architectural benefit without forcing a UX change.

2. **Worker modification for dynamo flags?**
   `worker.py` currently doesn't accept arbitrary dynamo flags. We'd need a `--dynamo-flags '{"capture_scalar_outputs": true}'` argument. Small change, contained to worker.py.

3. **Where do sweep results live after this?**
   Current: `results/pt2.10.md` (manually authored summaries)
   Proposed: also produce `experiments/results/pt2.10-sweep/` with full structured data.
   The markdown summaries remain the human-facing artifact; the structured data is the machine-facing one.
