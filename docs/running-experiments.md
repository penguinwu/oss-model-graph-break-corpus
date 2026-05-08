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
    --into sweep_results/baseline/pt2.11/
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

| `source` | Required fields | Optional fields | Description |
|----------|----------------|-----------------|-------------|
| `"list"` | `names: ["GPT2Model", ...]` | — | Explicit list of model class names |
| `"all"` | — | — | All models from HF + Diffusers + Custom |
| `"corpus_filter"` | `status: "graph_break"` | `from: "<path>"`, `source_sha256: "<hex>"` | Models with a specific status. Default reads `corpus/corpus.json`; pass `from` to read any sweep results file (identify/explain). With `source_sha256` validation REFUSES launch on file drift. |
| `"sample"` | `size: 50` | `seed` (default 42), `from: <models block>` | Random sample. `from` lets you sub-sample from another resolved source (e.g., `corpus_filter` results). |
| `"new_since"` | `baseline: "path/to/results/"` | — | Models not in an existing results directory |

Model names are HuggingFace class names (e.g., `GPT2Model`, `LlamaForCausalLM`, `StableDiffusionPipeline`). Use `--dry-run` to verify resolution.

**Anchoring on a prior result file (`corpus_filter` + `from` + `source_sha256`):**

```json
"models": {
  "source": "corpus_filter",
  "from": "sweep_results/experiments/nested-gb-2026-05-05-2026-05-05/explain_results.json",
  "status": "ok",
  "source_sha256": "a96d3e1ed5c30b26e223031a99b5968479109b714754cfbe61d8784e33dfbf2b"
}
```

This is how NGB-verify-style experiments anchor their cohort on a specific prior result set. The `source_sha256` pin catches the failure mode where the prior file is regenerated between launch and analysis (silent drift). Used by `tools/derive_sweep_commands.py` for reproducible multi-stage launches.

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
| Inductor numeric quality | `{"backend": "inductor", "fullgraph": false}` (read `numeric_status` field) |

**Numeric correctness check runs for every config.** Each result includes `numeric_status` (`match` / `divergence` / `nan_inf_introduced` / `shape_mismatch` / `dtype_mismatch` / `skipped`), `numeric_max_diff`, `numeric_severity_ratio`, etc. — see [Understanding Results §Numeric correctness fields](understanding-results.md#numeric-correctness-fields-identify-pass). For non-default backends like inductor, expect some `divergence` reports as legitimate compiler-quality data (TF32, fused ops). The less_flaky retry separates noise-floor divergence from real bugs.

**`dynamo_flags`** are applied via `torch._dynamo.config` before compilation. Empty `{}` means default settings.

**Known dynamo flags:** `capture_scalar_outputs`, `capture_dynamic_output_shape_ops`, `automatic_dynamic_shapes`, `assume_static_by_default`, `specialize_int`, `suppress_errors`, `verbose`, `cache_size_limit`, `accumulated_cache_size_limit`, `guard_nn_modules`, `inline_inbuilt_nn_modules`, `optimize_ddp`, `nested_graph_breaks`.

### `settings` — execution parameters

| Field | Default | Description |
|-------|---------|-------------|
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |
| `modes` | `["eval"]` | `["eval"]`, `["train"]`, or both |
| `workers` | `4` | Number of parallel worker processes |
| `timeout_s` | `180` | Per-model timeout in seconds |
| `pass_num` | `1` | `1` = identify, `2` = explain |
| `python_bin` | `null` | Absolute path to interpreter (e.g., `~/envs/torch-nightly-cu126/bin/python`). Optional for `run` (uses `SWEEP_PYTHON` env var as fallback), but **required by `tools/derive_sweep_commands.py`** for stack pinning. |
| `modellib_pins` | `null` | Dict mapping `transformers`/`diffusers`/`timm` to version strings. Resolved into PYTHONPATH against `~/envs/modellibs/<lib>-<ver>/`. **Required by `tools/derive_sweep_commands.py`.** Example: `{"transformers": "5.6.2", "diffusers": "0.38.0"}`. |

CLI flags `--workers` and `--timeout` override these values.

## Output format

```
experiments/results/my-test-20260416-193000/
  config.json              — input config + resolved models + environment snapshot
  results.jsonl            — first line: provenance metadata; subsequent lines: one JSON per (model, config, mode)
  identify_streaming.jsonl — symlink to results.jsonl (sweep_watchdog.py compatibility)
  sweep_state.json         — phase + progress + spec_path + spec_sha256 (sweep_watchdog.py compatibility)
  summary.md               — auto-generated human-readable report
```

### `results.jsonl`

**First line — provenance metadata header** (added 2026-05-07):

```json
{"_record_type": "metadata", "pass": "identify", "spec_path": "/home/.../experiments/configs/ngb-verify-2026-05-07.json", "spec_sha256": "a96d3e1ed5...", "spec_name": "ngb-verify-2026-05-07", "started": "2026-05-08T01:16:32Z", "total_models": 190, "total_work_items": 380, "modes": ["eval", "train"], "configs": ["ngb"], "_writer": "tools/run_experiment.py run subcommand"}
```

The header lets `tools/check_cohort_invariants.py --post-sweep` verify SP1 (provenance — spec sha256 match). Consumers must skip records where `_record_type == "metadata"` (or filter on the absence of `model`/`status`).

**Subsequent lines — per-result rows:**

```json
{"model": "GPT2Model", "config": "baseline", "mode": "eval", "status": "full_graph", "wall_time_s": 12.3, "compile_kwargs": {"fullgraph": true, "backend": "eager"}, "numeric_status": "match", "numeric_max_diff": 0.0, "numeric_bitwise_equal": true}
{"model": "GPT2Model", "config": "aot-eager", "mode": "eval", "status": "success", "wall_time_s": 11.8, "compile_kwargs": {"fullgraph": false, "backend": "aot_eager"}, "numeric_status": "match", "numeric_max_diff": 1.19e-06, "numeric_bitwise_equal": false}
```

The `numeric_*` fields appear on every identify-pass result (not just baseline). See [Understanding Results §Numeric correctness fields](understanding-results.md#numeric-correctness-fields-identify-pass) for the full schema.

## Multi-stage launches

For experiments warranting the gate → sample → full sequence, use `tools/derive_sweep_commands.py`:

```bash
tools/derive_sweep_commands.py experiments/configs/my-test.json --stage gate --run    # 5 models, ~5 min
tools/derive_sweep_commands.py experiments/configs/my-test.json --stage sample --run  # 20 models, ~15 min
tools/derive_sweep_commands.py experiments/configs/my-test.json --stage full --run    # whole cohort
```

All three stages mechanically use the same config flags — only the sub-sample size + output dir vary. The tool requires `settings.python_bin` and `settings.modellib_pins` (see settings table above). See [Running Sweeps §Multi-stage launches](running-sweeps.md#multi-stage-launches-via-derive_sweep_commands).

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
  "models": {"source": "new_since", "baseline": "sweep_results/baseline/pt2.11/"},
  "configs": [{"name": "default", "dynamo_flags": {}}],
  "settings": {"device": "cuda", "modes": ["eval", "train"], "workers": 4, "timeout_s": 180, "pass_num": 1}
}
```

Then merge into the main results:

```bash
python3 tools/run_experiment.py merge \
    --from experiments/results/new-models-check-*/ \
    --into sweep_results/baseline/pt2.11/
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
