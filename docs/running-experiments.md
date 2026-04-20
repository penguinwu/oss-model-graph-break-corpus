# Running Experiments

Test dynamo flags, config ablations, or model subsets using the same infrastructure as production sweeps. Experiments are config-driven — define what to test in a JSON file, then run it.

## Quick start

```bash
# 1. Generate a starter config
python3 tools/run_experiment.py template > experiments/configs/my-test.json

# 2. Edit the config — set your models, flags, and settings

# 3. Validate (catches errors with typo suggestions)
python3 tools/run_experiment.py validate experiments/configs/my-test.json

# 4. Preview what will run (no GPU needed)
python3 tools/run_experiment.py run experiments/configs/my-test.json --dry-run

# 5. Run
python3 tools/run_experiment.py run experiments/configs/my-test.json
```

If using a venv, either activate it first or run with its Python:

```bash
~/envs/torch211/bin/python tools/run_experiment.py run experiments/configs/my-test.json
```

## CLI reference

### `template` — generate a starter config

```bash
python3 tools/run_experiment.py template > my-config.json
```

Prints a commented JSON config to stdout. No arguments.

### `validate <config>` — check a config file

```bash
python3 tools/run_experiment.py validate my-config.json
```

Validates against the schema. Catches missing fields, invalid sources, unknown dynamo flags, and unknown `compile_kwargs` keys (with typo suggestions — e.g., `"bakend"` suggests `"backend"`).

### `run <config>` — run an experiment

```bash
python3 tools/run_experiment.py run my-config.json
python3 tools/run_experiment.py run my-config.json --dry-run
python3 tools/run_experiment.py run my-config.json --workers 1 --timeout 60
python3 tools/run_experiment.py run my-config.json --output results/my-run/
python3 tools/run_experiment.py run my-config.json --resume results/my-run/
```

| Flag | Description |
|------|-------------|
| `--dry-run` | Resolve models, show work items, then exit. No GPU needed. |
| `--output DIR` | Override output directory |
| `--resume DIR` | Resume from a prior run's checkpoint |
| `--workers N` | Override worker count from config |
| `--timeout N` | Override per-model timeout (seconds) |

### `merge` — merge incremental results

```bash
python3 tools/run_experiment.py merge \
    --from experiments/results/new-models/ \
    --into sweep_results/pt2.11/
```

Merges source results into target. If a `(model, config, mode)` entry exists in both, source wins. Idempotent.

## Config reference

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
| `"sample"` | `size: 50` | Random sample. Optional: `seed` (default 42) |
| `"new_since"` | `baseline: "path/to/results/"` | Models not in an existing results directory |

Model names are HuggingFace class names (e.g., `GPT2Model`, `LlamaForCausalLM`, `StableDiffusionPipeline`). Use `--dry-run` to verify resolution.

### `configs` — what configurations to test

Each config is an A/B variant. The first config is treated as baseline in the summary.

```json
"configs": [
  {
    "name": "baseline",
    "compile_kwargs": {},
    "dynamo_flags": {}
  },
  {
    "name": "aot-eager",
    "compile_kwargs": {"backend": "aot_eager", "fullgraph": false},
    "dynamo_flags": {}
  }
]
```

**`compile_kwargs`** are passed directly to `torch.compile()`. When omitted or empty, defaults to `{"fullgraph": true, "backend": "eager"}` (the standard graph break detection mode).

**Known compile kwargs:** `backend`, `fullgraph`, `dynamic`, `mode`, `options`, `disable`.

Common configurations:

| Use case | `compile_kwargs` |
|----------|-----------------|
| Graph break detection (default) | `{}` or `{"fullgraph": true, "backend": "eager"}` |
| Backend error detection | `{"backend": "aot_eager", "fullgraph": false}` |
| Inductor crash detection | `{"backend": "inductor", "fullgraph": false}` |

**`dynamo_flags`** are applied via `torch._dynamo.config` before compilation. Empty `{}` means default settings.

**Known dynamo flags:** `capture_scalar_outputs`, `capture_dynamic_output_shape_ops`, `automatic_dynamic_shapes`, `assume_static_by_default`, `specialize_int`, `suppress_errors`, `verbose`, `cache_size_limit`, `guard_nn_modules`, `inline_inbuilt_nn_modules`, `optimize_ddp`.

### `settings` — execution parameters

| Field | Default | Description |
|-------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `modes` | `["eval"]` | `["eval"]`, `["train"]`, or both |
| `workers` | `4` | Number of parallel worker processes |
| `timeout_s` | `180` | Per-model timeout in seconds |
| `pass_num` | `1` | `1` = identify, `2` = explain |

CLI flags `--workers` and `--timeout` override these values.

## Output format

```
experiments/results/my-test-20260416-193000/
  config.json      — input config + resolved models + environment snapshot
  results.jsonl    — one JSON line per (model, config, mode)
  checkpoint.jsonl — for resume support
  summary.md       — auto-generated human-readable report
```

### `results.jsonl`

```json
{"model": "GPT2Model", "config": "baseline", "mode": "eval", "status": "full_graph", "wall_time_s": 12.3, "compile_kwargs": {"fullgraph": true, "backend": "eager"}}
{"model": "GPT2Model", "config": "aot-eager", "mode": "eval", "status": "success", "wall_time_s": 11.8, "compile_kwargs": {"fullgraph": false, "backend": "aot_eager"}}
```

Status values depend on the compile configuration:

| Status | When | Meaning |
|--------|------|---------|
| `full_graph` | `fullgraph=True` | Compiled successfully — no graph breaks |
| `graph_break` | `fullgraph=True` | Compiled but with graph breaks |
| `success` | `fullgraph=False` | Compiled and ran without errors |
| `error` | `fullgraph=False` | Compilation or execution raised an exception |
| `timeout` | any | Exceeded per-model time limit |
| `create_error` | any | Model failed to instantiate |
| `eager_error` | any | Model failed during eager execution |

## Recipes

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
  "models": {"source": "new_since", "baseline": "sweep_results/pt2.11/"},
  "configs": [{"name": "default", "dynamo_flags": {}}],
  "settings": {"device": "cuda", "modes": ["eval", "train"], "workers": 4, "timeout_s": 180, "pass_num": 1}
}
```

Then merge into the main results:

```bash
python3 tools/run_experiment.py merge \
    --from experiments/results/new-models-check-*/ \
    --into sweep_results/pt2.11/
```

### Test a backend on a random sample

```json
{
  "name": "aot-eager-sample",
  "description": "Does aot_eager surface errors beyond graph breaks?",
  "models": {"source": "sample", "size": 50, "seed": 42},
  "configs": [
    {"name": "default", "compile_kwargs": {}, "dynamo_flags": {}},
    {"name": "aot-eager", "compile_kwargs": {"backend": "aot_eager", "fullgraph": false}, "dynamo_flags": {}}
  ],
  "settings": {"device": "cuda", "modes": ["eval"], "workers": 4, "timeout_s": 180}
}
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

## How models are tested

Each model is instantiated from its HuggingFace config class (e.g., `GPT2Config()`) with **random weights** — no pretrained weights are downloaded. The model architecture is identical to the real thing.

By default, the model is compiled with `torch.compile(model, fullgraph=True, backend="eager")` for graph break detection. With custom `compile_kwargs`, any `torch.compile()` configuration can be tested (e.g., `backend="aot_eager"` or `backend="inductor"`). Models are run with a small input (batch_size=2, short sequence lengths). Graph break detection, guard analysis, and Dynamo tracing are all real — only the weights are random, which does not affect compilation behavior.

Each model runs in an isolated subprocess with its own process group, so crashes and timeouts don't affect other models.

## Environment variable

Set `SWEEP_PYTHON` to use a specific Python binary without activating a venv:

```bash
SWEEP_PYTHON=~/envs/torch211/bin/python \
    python3 tools/run_experiment.py run my-config.json
```

## Next steps

- [Running Sweeps](running-sweeps.md) — full corpus sweeps for version-level measurement
- [Understanding Results](understanding-results.md) — interpret experiment output
- [Issue Management](issue-management.md) — classify and track graph breaks
