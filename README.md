# OSS Model Graph Break Corpus

A corpus of **468 open-source models** tested for `torch.compile(fullgraph=True)` compatibility on PyTorch 2.10.0.

## Results at a Glance

| Status | Eval | Train |
|--------|------|-------|
| **Clean** | 352 (75%) | 337 (72%) |
| **Graph Break** | 93 (20%) | 107 (23%) |
| Eager Error | 13 (3%) | 14 (3%) |
| Create Error | 6 (1%) | 6 (1%) |
| Timeout | 4 (1%) | 4 (1%) |

**110 models** have graph breaks in at least one mode. 3 targeted PRs (deepcopy→clone, un-skip audio callables, Logger support) would fix **49 models (45%)**.

With `mark_dynamic` on batch + seq_len dims, 23 additional models lose clean status — primarily from constraint violations where models specialize on dimensions marked as dynamic.

## Quick Start (Corpus Consumers)

### Browse the corpus

```bash
# Summary
python tools/query.py

# All graph break models
python tools/query.py --status graph_break

# Search by error (e.g., deepcopy, Logger, unbacked)
python tools/query.py --error deepcopy

# Compare static vs dynamic shapes
python tools/query.py --compare-dynamic
```

### Reproduce a graph break

```bash
pip install torch==2.10.0 transformers==5.4.0 diffusers==0.37.1

python tools/reproduce.py BartModel
python tools/reproduce.py BartModel --mode train
```

### Compare sweep results

```bash
python tools/compare.py results_a.json results_b.json --labels "2.9" "2.10"
python tools/compare.py --corpus-dynamic
```

### Corpus format

`corpus/corpus.json` — each model has eval/train results with status, compile time, error details, and optional `dynamic_mark` results. See `corpus/summary.md` for a human-readable overview.

## Contributing (Corpus Builders)

### Run a sweep

```bash
python -m venv env && source env/bin/activate
pip install torch transformers diffusers

python sweep/run_sweep.py \
    --device cuda \
    --source hf+diffusers \
    --pass1-only \
    --pass1-modes eval train \
    --workers 4 \
    --timeout 180 \
    --timeout-large 600 \
    --output-dir sweep_results/$(date +%Y%m%d)
```

Add `--dynamic mark` or `--dynamic true` for dynamic shape testing. Add `--resume` to recover from crashes.

### Fix a model

Model-specific fixes live in `sweep/worker.py`:
- `_fix_config()` — patch invalid config values
- `_create_config()` — composite models needing factory methods
- `_generate_inputs()` — model-specific input overrides
- `_reduce_model_size()` — cap layers/hidden dims for GPU fit

Verify your fix:
```bash
python sweep/worker.py --model hf/ModelName --device cuda
```

### Architecture

| Script | Role |
|--------|------|
| `sweep/models.py` | Model enumeration from HF/Diffusers/TIMM |
| `sweep/worker.py` | Single-model subprocess (create → eager → compile) |
| `sweep/run_sweep.py` | Orchestrator (parallel workers, timeouts, checkpointing) |
| `sweep/sweep_watchdog.py` | Progress monitor + auto-restart on failure |

The sweep uses process-group isolation, non-blocking polling, GPU pressure backoff, and JSONL checkpointing for crash recovery.

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |
| Python | 3.12.13 |
| GPU | NVIDIA A100 80GB |

## Design Doc

Full methodology, taxonomy, and analysis: [docs/design-doc.md](docs/design-doc.md)

Graph break pattern analysis for PT2 team: [analysis/graph-break-analysis.md](analysis/graph-break-analysis.md)
