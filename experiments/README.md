# Experiments

Run ad-hoc experiments, flag quality tests, and config ablations using the same
infrastructure as the production sweep.

## Quick Start

```bash
# 1. Generate a starter config
python tools/run_experiment.py template > experiments/configs/my-test.json

# 2. Edit the config — set your models, flags, and settings

# 3. Validate before running (catches errors with typo suggestions)
python tools/run_experiment.py validate experiments/configs/my-test.json

# 4. Preview what will run (resolves models, shows work items)
python tools/run_experiment.py run experiments/configs/my-test.json --dry-run

# 5. Run
python tools/run_experiment.py run experiments/configs/my-test.json
```

If you're using a venv (recommended), either activate it first or run the
experiment runner with its python:

```bash
/path/to/venv/bin/python tools/run_experiment.py run experiments/configs/my-test.json
```

## CLI Reference

### `template` — generate a starter config

```bash
python tools/run_experiment.py template > my-config.json
```

Prints a commented JSON config to stdout. Edit it to set your models, dynamo
flags, and execution settings. No arguments.

### `validate <config>` — check a config file

```bash
python tools/run_experiment.py validate my-config.json
```

Validates the config against the schema. Exits 0 on success, 1 on errors.
Catches missing fields, invalid sources, unknown dynamo flags (with typo
suggestions — e.g., `"capture_scaler_outputs"` → `"did you mean
'capture_scalar_outputs'?"`).

Note: `run` also validates automatically before execution. `validate` is useful
for CI/pre-commit checks or catching errors before waiting for model resolution.

### `run <config>` — run an experiment

```bash
# Basic run
python tools/run_experiment.py run my-config.json

# Preview without executing
python tools/run_experiment.py run my-config.json --dry-run

# Override workers or timeout without editing config
python tools/run_experiment.py run my-config.json --workers 1 --timeout 60

# Explicit output directory
python tools/run_experiment.py run my-config.json --output results/my-run/

# Resume a prior interrupted run
python tools/run_experiment.py run my-config.json --resume results/my-run/
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Resolve models, show work item count and list, then exit. No GPU needed. |
| `--output DIR` | Override output directory. Default: `experiments/results/<name>-<YYYYMMDD-HHMMSS>/` |
| `--resume DIR` | Resume from a prior run's output directory. Reads its checkpoint, skips completed items, writes to the same directory. Mutually exclusive with `--output`. |
| `--workers N` | Override worker count from config. Useful for debugging (`--workers 1`). |
| `--timeout N` | Override per-model timeout (seconds) from config. |

### `merge` — merge incremental results

```bash
python tools/run_experiment.py merge --from new-models/ --into full-sweep/
```

Merges source `results.jsonl` into target `results.jsonl`. If a
`(model, config, mode)` entry exists in both, source wins (newer data).
Idempotent — running twice produces the same result. Target's `config.json`
is updated with merge metadata (timestamp, source, counts).

| Flag | Description |
|------|-------------|
| `--from DIR` | Source results directory (must contain `results.jsonl`) |
| `--into DIR` | Target results directory (updated in-place) |

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

| `source` | Required fields | Description |
|----------|----------------|-------------|
| `"list"` | `names: ["GPT2Model", ...]` | Explicit list of model class names |
| `"all"` | — | All models from HF + Diffusers + Custom |
| `"corpus_filter"` | `status: "graph_break"` | Models with a specific status in `corpus/corpus.json` |
| `"sample"` | `size: 50` | Random sample. Optional: `seed` (default 42), `strategy` |
| `"new_since"` | `baseline: "path/to/results/"` | Models not in an existing results directory |

**Model names** are HuggingFace class names (e.g., `GPT2Model`, `LlamaForCausalLM`,
`StableDiffusionPipeline`). Run `--dry-run` to verify your model names resolve
correctly.

### `configs` — what configurations to test

Each config is an A/B variant. The first config is treated as baseline in the
auto-generated summary.

```json
"configs": [
  {
    "name": "baseline",
    "dynamo_flags": {}
  },
  {
    "name": "capture_scalar_outputs",
    "dynamo_flags": {
      "capture_scalar_outputs": true
    }
  }
]
```

`dynamo_flags` are applied via `torch._dynamo.config` before compilation.
An empty `{}` means default dynamo settings.

**Known dynamo flags:** `capture_scalar_outputs`, `capture_dynamic_output_shape_ops`,
`automatic_dynamic_shapes`, `assume_static_by_default`, `specialize_int`,
`suppress_errors`, `verbose`, `cache_size_limit`, `guard_nn_modules`,
`inline_inbuilt_nn_modules`, `optimize_ddp`.

Unknown flags are caught by validation with fuzzy suggestions.

### `settings` — execution parameters

| Field | Default | Description |
|-------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `modes` | `["eval"]` | `["eval"]`, `["train"]`, or both |
| `workers` | `4` | Number of parallel worker processes |
| `timeout_s` | `180` | Per-model timeout in seconds |
| `pass_num` | `1` | `1` = identify (`fullgraph=True` check), `2` = explain (graph break analysis) |

CLI flags `--workers` and `--timeout` override these values without editing the
config file.

## Output Format

Each experiment produces a self-describing output directory:

```
experiments/results/my-test-20260416-193000/
  config.json      — input config + resolved models + environment snapshot
  results.jsonl    — one JSON line per (model, config, mode)
  checkpoint.jsonl — for resume support (same format as results.jsonl)
  summary.md       — auto-generated human-readable report
```

### `config.json` — experiment birth certificate

Contains everything needed to understand and reproduce the results:

```json
{
  "experiment": { ... },         // original input config
  "resolved": {
    "models": ["GPT2Model", "T5Model", ...],
    "model_count": 50
  },
  "environment": {
    "torch": "2.10.0+cu128",
    "transformers": "5.4.0",
    "diffusers": "0.37.1",
    "python": "3.12.13..."
  },
  "execution": {
    "started": "2026-04-16T19:30:00",
    "finished": "2026-04-16T19:45:00",
    "duration_s": 900.3,
    "total_results": 100,
    "python_bin": "/home/user/envs/torch210/bin/python"
  }
}
```

### `results.jsonl` — per-model results

One JSON line per `(model, config, mode)` tuple:

```json
{"model": "GPT2Model", "config": "baseline", "mode": "eval", "status": "full_graph", "wall_time_s": 12.3}
{"model": "GPT2Model", "config": "cso", "mode": "eval", "status": "full_graph", "wall_time_s": 11.8, "dynamo_flags": {"capture_scalar_outputs": true}}
```

Possible `status` values:
- `full_graph` — compiled successfully with no graph breaks
- `graph_break` — compiled but with graph breaks
- `error` — compilation failed
- `timeout` — exceeded time limit

Additional fields by status: `graph_count`, `graph_break_count`, `error`,
`break_reasons`, `dynamo_flags`.

### `summary.md` — auto-generated report

Per-config status breakdown and cross-config comparison. When multiple configs
are present, the first is used as baseline and differences (improvements,
regressions, crashes) are highlighted.

## Common Recipes

### Test a dynamo flag on all graph-breaking models

```json
{
  "name": "cso-quality",
  "description": "Does capture_scalar_outputs cause new crashes?",
  "models": {"source": "corpus_filter", "status": "graph_break"},
  "configs": [
    {"name": "baseline", "dynamo_flags": {}},
    {"name": "cso", "dynamo_flags": {"capture_scalar_outputs": true}}
  ],
  "settings": {"device": "cuda", "modes": ["eval"], "workers": 4, "timeout_s": 180, "pass_num": 1}
}
```

### Sweep new models added to HF Transformers

```json
{
  "name": "new-models-check",
  "description": "Sweep models not in the latest baseline",
  "models": {"source": "new_since", "baseline": "sweep_results/pt2.10/"},
  "configs": [{"name": "default", "dynamo_flags": {}}],
  "settings": {"device": "cuda", "modes": ["eval", "train"], "workers": 4, "timeout_s": 180, "pass_num": 1}
}
```

Then merge into the main results:

```bash
python tools/run_experiment.py merge \
    --from experiments/results/new-models-check-20260416-193000/ \
    --into sweep_results/pt2.10/
```

### Quick random sample for smoke testing

```json
{
  "name": "quick-sample",
  "description": "Random 20-model sample to verify setup",
  "models": {"source": "sample", "size": 20, "seed": 42},
  "configs": [{"name": "default", "dynamo_flags": {}}],
  "settings": {"device": "cuda", "modes": ["eval"], "workers": 4, "timeout_s": 120, "pass_num": 1}
}
```

## How Models Are Tested

Each model is instantiated from its HuggingFace config class (e.g., `GPT2Config()`)
with **random weights** — no pretrained weights are downloaded. The model
architecture (layers, attention heads, FFN blocks) is identical to the real thing.

The model is then compiled with `torch.compile(model, fullgraph=True)` and run
with a small input (batch_size=2, short sequence lengths). Graph break detection,
guard analysis, and Dynamo tracing are all real — only the weights are random,
which does not affect compilation behavior. `torch.compile` traces the
computational graph based on model structure and tensor shapes, not weight values.

Each model runs in an isolated subprocess with its own process group, so crashes
and timeouts in one model don't affect others.

## Environment

Set `SWEEP_PYTHON` to use a specific Python binary without activating a venv:

```bash
SWEEP_PYTHON=/path/to/venv/bin/python \
    python tools/run_experiment.py run experiments/configs/my-test.json
```
