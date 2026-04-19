# Running Sweeps

A sweep compiles every model in the corpus with `torch.compile(fullgraph=True)` and records the result. Use sweeps to measure compile quality for a specific PyTorch version or validate a compiler change.

## Quick start

```bash
# Set up a venv with torch + model libraries
python3 -m venv ~/envs/torch-test
source ~/envs/torch-test/bin/activate
pip install -r requirements.txt

# Run a sweep
python3 sweep/run_sweep.py sweep \
    --source hf diffusers custom \
    --workers 4 \
    --timeout 180
```

Or use the unified CLI (equivalent):

```bash
python3 tools/run_experiment.py sweep \
    --source hf diffusers custom
```

## Two-pass architecture

Sweeps run in two passes:

1. **Identify** — compile each model with `fullgraph=True`, record pass/fail/error
2. **Explain** — for graph-breaking models, re-run with Dynamo logging to extract break reasons, locations, and counts

By default, `sweep` runs both passes. To run identify-only (faster):

```bash
python3 sweep/run_sweep.py sweep --identify-only \
    --source hf diffusers custom
```

## Using a specific Python binary

Point `--python` at your venv's Python to avoid activating it:

```bash
python3 sweep/run_sweep.py sweep \
    --python ~/envs/torch-test/bin/python \
    --source hf diffusers custom \
    --workers 4
```

Or set `SWEEP_PYTHON`:

```bash
SWEEP_PYTHON=~/envs/torch-test/bin/python \
    python3 sweep/run_sweep.py sweep --source hf diffusers custom
```

## Model sources

| Source | What it includes |
|--------|-----------------|
| `hf` | HuggingFace Transformers models (base + ForCausalLM + ForConditionalGeneration) |
| `diffusers` | HuggingFace Diffusers pipelines |
| `custom` | Custom model definitions in `sweep/custom_models/` |
| `timm` | PyTorch Image Models (requires explicit `--source timm`) |

Default sources: `hf diffusers custom`. TIMM requires explicit request.

## Dynamic shape testing

```bash
# Mark batch + sequence length dims as dynamic
python3 sweep/run_sweep.py sweep --dynamic mark --source hf diffusers custom

# All dims symbolic
python3 sweep/run_sweep.py sweep --dynamic true --source hf diffusers custom
```

## Crash recovery

Sweeps checkpoint after every model. To resume after a crash:

```bash
python3 sweep/run_sweep.py sweep --resume --source hf diffusers custom
```

The `--resume` flag reads the checkpoint file and skips already-completed models.

## Output

Results go to `sweep_results/<label>/`:

```
sweep_results/pt2.11/
  identify_results.json   — pass/fail/error for each model
  explain_results.json    — graph break details for breaking models
  checkpoint.jsonl        — incremental checkpoint for crash recovery
```

## Monitoring long sweeps

For full sweeps (734 models, ~2-4 hours on A100):

```bash
python3 sweep/sweep_watchdog.py
```

The watchdog monitors progress and auto-restarts on failures.

## Update the corpus after a sweep

```bash
python3 tools/update_corpus.py \
    --identify sweep_results/pt2.11/identify_results.json \
    --explain sweep_results/pt2.11/explain_results.json
```

Validate after updating:

```bash
python3 tools/validate.py
```

## Conventions

- Batch size must be >= 2 (PyTorch specializes on 0 and 1)
- Backend is always `eager` (tests Dynamo tracing, not Inductor codegen)
- Never use 0 or 1 as input dimensions for dynamic shape testing
- Default sources: `hf diffusers custom` (TIMM/dynamic require explicit request)

## Next steps

- [Running Experiments](running-experiments.md) — test specific flag combinations or model subsets
- [Issue Management](issue-management.md) — classify graph breaks and update GitHub issues after a sweep
- [Understanding Results](understanding-results.md) — interpret sweep output
