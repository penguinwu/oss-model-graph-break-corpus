# Experiments

Run ad-hoc experiments, flag quality tests, and config ablations using the same
infrastructure as the production sweep.

## Quick Start

```bash
# 1. Generate a starter config
python tools/run_experiment.py --template > experiments/configs/my-test.json

# 2. Edit the config (set models, flags, settings)

# 3. Validate before running
python tools/run_experiment.py --validate experiments/configs/my-test.json

# 4. Run
python tools/run_experiment.py --config experiments/configs/my-test.json
```

## Config Reference

```json
{
  "name": "experiment-name",
  "description": "What you're testing and why",

  "models": { ... },
  "configs": [ ... ],
  "settings": { ... }
}
```

### `models` — which models to test

| source | Required fields | Description |
|--------|----------------|-------------|
| `"list"` | `names: [...]` | Explicit list of model class names |
| `"all"` | — | All models from HF + Diffusers + Custom |
| `"corpus_filter"` | `status: "graph_break"` | Models with a specific status in corpus.json |
| `"sample"` | `size: 50` | Random sample. Optional: `seed`, `strategy` |
| `"new_since"` | `baseline: "path/"` | Models not in an existing results directory |

### `configs` — what configurations to test

Each config is an A/B variant. The first config is treated as baseline in the summary.

```json
"configs": [
  {
    "name": "baseline",
    "dynamo_flags": {}
  },
  {
    "name": "my_flag",
    "dynamo_flags": {
      "capture_scalar_outputs": true
    }
  }
]
```

**Known dynamo flags:** `capture_scalar_outputs`, `capture_dynamic_output_shape_ops`,
`automatic_dynamic_shapes`, `assume_static_by_default`, `specialize_int`,
`suppress_errors`, `verbose`, `cache_size_limit`, `guard_nn_modules`,
`inline_inbuilt_nn_modules`, `optimize_ddp`.

Unknown flags are caught by `--validate` with fuzzy suggestions.

### `settings` — execution parameters

| Field | Default | Description |
|-------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `modes` | `["eval"]` | `["eval"]`, `["train"]`, or `["eval", "train"]` |
| `workers` | `4` | Parallel worker processes |
| `timeout_s` | `180` | Per-model timeout in seconds |
| `pass_num` | `1` | `1` = identify (fullgraph check), `2` = explain (graph break analysis) |

## Output

Each experiment run produces a self-describing folder:

```
experiments/results/2026-04-16-my-test/
  config.json      # Input config + resolved model list + captured environment
  results.jsonl    # One line per (model, config, mode) — granular, mergeable
  checkpoint.jsonl # For resume support
  summary.md       # Auto-generated human-readable report
```

### `config.json`

Contains the original experiment config, resolved model list, environment
(PyTorch version, transformers version, GPU, etc.), and execution timestamps.
This is the "birth certificate" of the result — everything needed to understand
what produced these numbers.

### `results.jsonl`

Per-model, per-config results in JSONL format. Each line:

```json
{"model": "GPT2Model", "config": "baseline", "mode": "eval", "status": "full_graph", "wall_time_s": 12.3}
```

Additional fields appear based on status: `graph_count`, `graph_break_count`,
`error`, `break_reasons`, `dynamo_flags`.

## Merging Results

Merge incremental results into an existing experiment:

```bash
python tools/run_experiment.py --merge SOURCE_DIR TARGET_DIR
```

- Source entries override target on conflict (same model/config/mode)
- New entries are appended
- Merge is idempotent — running twice produces the same result
- `config.json` is updated with merge metadata

## Resuming

If an experiment is interrupted, resume from checkpoint:

```bash
python tools/run_experiment.py --config experiments/configs/my-test.json --resume
```

Completed model/config/mode combinations are skipped.

## Environment

Set `SWEEP_PYTHON` to use a specific Python binary (e.g., a venv):

```bash
SWEEP_PYTHON=/home/pengwu/envs/torch-nightly/bin/python \
  python tools/run_experiment.py --config experiments/configs/my-test.json
```
