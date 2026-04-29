# Contributing

How to add models, fix compiler quality issues (graph breaks, numerical divergences, infra failures), run sweeps, and extend the corpus.

## Fix a model

Model-specific fixes live in `sweep/worker.py`:

| Function | Purpose |
|----------|---------|
| `_fix_config()` | Patch invalid config values (e.g., reduce vocab, fix invalid defaults) |
| `_create_config()` | Composite models needing factory construction |
| `_generate_inputs()` | Models with non-standard input signatures |
| `_reduce_model_size()` | Cap layers/hidden dims for GPU fit |

After fixing, verify:

```bash
python3 sweep/worker.py --model hf/ModelName --device cuda
```

## Run a sweep

```bash
# Set up a venv
python3 -m venv ~/envs/torch-test
~/envs/torch-test/bin/pip install -r requirements.txt

# Run a sweep
python3 tools/run_experiment.py sweep \
    --device cuda \
    --python ~/envs/torch-test/bin/python \
    --source hf diffusers custom \
    --workers 4 \
    --timeout 180 \
    --output-dir sweep_results/$(date +%Y%m%d)
```

Add `--dynamic mark` or `--dynamic true` for dynamic shape testing. Add `--resume` to recover from crashes.

See [Running Sweeps](running-sweeps.md) for full details.

## Update the corpus

After a sweep:

```bash
# Merge results into corpus.json
python3 tools/update_corpus.py \
    --identify sweep_results/baseline/pt2.11/identify_results.json \
    --explain sweep_results/baseline/pt2.11/explain_results.json

# Validate integrity
python3 tools/validate.py
```

## Architecture

| Script | Role |
|--------|------|
| `sweep/models.py` | Model enumeration from HF/Diffusers/TIMM (base + ForCausalLM + ForConditionalGeneration) |
| `sweep/worker.py` | Single-model subprocess (create, eager, compile) |
| `sweep/run_sweep.py` | Sweep engine (parallel workers, timeouts, checkpointing) |
| `sweep/orchestrator.py` | Shared worker management (spawn, harvest, timeout, kill, checkpoint) |
| `sweep/explain.py` | Graph break analysis via Dynamo logging |
| `sweep/sweep_watchdog.py` | Progress monitor + auto-restart on failure |
| `sweep/test_explain.py` | Unit tests for the graph-break analysis module — `python3 sweep/test_explain.py` (no GPU) |
| `sweep/large_models.json` | Models needing extended timeouts |
| `sweep/tracked_models.json` | Models tracked for specific PR fixes |
| `tools/run_experiment.py` | Unified CLI — wraps sweep + experiments + nightly pipeline |
| `tools/file_issues.py` | Post-sweep issue management (sweep-report + sweep-update) |
| `tools/query.py` | Query corpus by status, error, dynamic comparison |
| `tools/reproduce.py` | Reproduce a single model's graph break |
| `tools/analyze_explain.py` | Graph break taxonomy and root cause analysis |
| `tools/analyze_trend.py` | Version trend analysis across PyTorch releases |
| `tools/validate.py` | Corpus integrity checks (golden set, schema) |
| `tools/compare.py` | Compare two sweep results side-by-side |
| `tools/update_corpus.py` | Update corpus.json from sweep results |
| `tools/generate_index.py` | Generate corpus dashboard (docs/index.html) |
| `tools/generate_traces.py` | Pre-generate tlparse reports for top N models |

The sweep uses process-group isolation, non-blocking polling, GPU pressure backoff, and JSONL checkpointing for crash recovery.

## Conventions

- Batch size must be >= 2 (PyTorch specializes on 0 and 1)
- Default backend is `eager` (tests Dynamo tracing, not Inductor codegen) — override via `--compile-kwargs '{"backend": "..."}'`
- Default sources: `hf diffusers custom` (TIMM/dynamic require explicit request)
- Never use 0 or 1 as input dimensions for dynamic shape testing
- Sweep results go in `sweep_results/<label>/` (e.g., `sweep_results/nightly/2026-04-19/`)

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.5.3 |
| Diffusers | 0.37.1 |
| Python | 3.12.13 |
| GPU | NVIDIA A100 80GB |

## Next steps

- [Getting Started](getting-started.md) — quick start guide
- [Running Experiments](running-experiments.md) — config-driven flag testing
- [Issue Management](issue-management.md) — post-sweep issue workflow
