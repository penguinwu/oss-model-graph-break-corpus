# OSS Model Graph Break Corpus

## What This Is

A corpus of 468 HuggingFace models tested for `torch.compile(fullgraph=True)` compatibility on PyTorch 2.10.0. Results include static and dynamic shape (mark_dynamic) compilation status.

## Key Files

- `corpus/corpus.json` — main dataset (468 models, eval + train modes, static + dynamic results)
- `tools/query.py` — query the corpus (by status, error text, dynamic comparison)
- `tools/reproduce.py` — reproduce a single model's graph break
- `tools/compare.py` — compare two sweep results
- `sweep/run_sweep.py` — run a full sweep (orchestrator)
- `sweep/worker.py` — single-model test worker (model creation, input generation, compile test)
- `sweep/sweep_watchdog.py` — monitor sweep progress, auto-restart on failure

## Common Tasks

### Query the corpus
```bash
python tools/query.py                          # summary
python tools/query.py --status graph_break     # all graph break models
python tools/query.py --error deepcopy         # search by error text
python tools/query.py --compare-dynamic        # static vs dynamic=mark
```

### Reproduce a graph break
```bash
python tools/reproduce.py BartModel            # eval mode
python tools/reproduce.py BartModel --mode train
```

### Run a sweep
```bash
python sweep/run_sweep.py \
    --device cuda \
    --python /path/to/python \
    --source hf+diffusers \
    --pass1-only \
    --pass1-modes eval train \
    --workers 4 \
    --timeout 180 \
    --timeout-large 600 \
    --output-dir sweep_results/$(date +%Y%m%d)
```

Add `--dynamic mark` or `--dynamic true` for dynamic shape testing.
Add `--resume` to resume from a crash.

### Adding a model fix
Model-specific fixes live in `sweep/worker.py`:
- `_fix_config()` — patch config values
- `_create_config()` — composite models needing factory methods
- `_generate_inputs()` — model-specific input overrides
- `_reduce_model_size()` — cap layers/hidden dims for GPU fit

After fixing, re-run the single model to verify:
```bash
python sweep/worker.py --model hf/ModelName --device cuda
```

## Conventions

- Batch size must be >= 2 (PyTorch specializes on 0 and 1)
- Backend is always `eager` (tests Dynamo tracing, not inductor codegen)
- Never use 0 or 1 as input dimensions for dynamic shape testing
- Run HF models first in sweeps (highest graph break density)
