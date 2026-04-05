# OSS Model Compiler Quality Corpus

## What This Is

A reusable corpus of 468 open-source models for measuring and improving `torch.compile` quality. The first application tracks `fullgraph=True` success rates across PyTorch versions; the infrastructure extends to dynamic shape behavior, recompilation patterns, and other compiler diagnostics.

## Key Files

- `corpus/corpus.json` — main dataset (468 models, eval + train modes, static + dynamic results)
- `tools/query.py` — query the corpus (by status, error text, dynamic comparison, top errors)
- `tools/reproduce.py` — reproduce a single model's graph break
- `tools/analyze_explain.py` — graph break taxonomy and root cause analysis
- `tools/analyze_trend.py` — version trend analysis across PyTorch releases
- `tools/validate.py` — corpus integrity checks (golden set, schema)
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
    --identify-only \
    --identify-modes eval train \
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

## Agent Recipes

### Fix a data inconsistency
1. Identify the issue in `corpus/corpus.json` (wrong status, missing field, stale count)
2. Fix the data directly in corpus.json
3. Run `python tools/validate.py --fix` to regenerate summary block and fix has_graph_break flags
4. Run `python tools/validate.py` to confirm all checks pass
5. Check golden set: if you changed a golden set model, flag for Peng (Tier 3 — never update golden_set.json yourself)

### Update corpus from sweep results
1. Run the sweep: `python sweep/run_sweep.py --device cuda ...`
2. Run `python tools/update_corpus.py <sweep_results_dir>` to merge results into corpus.json
3. Run `python tools/validate.py` to confirm integrity
4. Run `python tools/compare.py <old_results> <new_results>` to generate a changelog
5. Commit and flag Peng for review and push

### Add a CLI flag to query.py
1. Add the argument in the argparse block (search for `parser.add_argument`)
2. Add filtering logic in the main query loop
3. Ensure `--json` output includes the new field if applicable
4. Test: `python tools/query.py --your-new-flag` and `python tools/query.py --your-new-flag --json`
5. Run `python tools/validate.py` to confirm tool output checks still pass

### Add a model-specific fix in worker.py
1. Identify the failure mode (create_error, eager_error, graph_break with specific error)
2. Choose the right fix point:
   - `_fix_config()` — patch config values (e.g., reduce vocab, fix invalid defaults)
   - `_create_config()` — composite models needing factory construction
   - `_generate_inputs()` — models with non-standard input signatures
   - `_reduce_model_size()` — models that OOM even at 2 layers
3. Add the fix with a comment explaining why
4. Test: `python sweep/worker.py --model hf/ModelName --device cuda`
5. Re-run `python tools/validate.py` after updating corpus

### Check environment before a sweep
```bash
python tools/version_check.py          # Compare installed vs corpus versions
python sweep/run_sweep.py --check-env  # Pre-sweep validation (exits pass/fail)
```

### Reproduce and debug a graph break
```bash
python tools/reproduce.py ModelName --explain          # Show break reasons
python tools/reproduce.py ModelName --explain --verbose # Full explain output
python tools/reproduce.py ModelName --dynamic mark     # Test with dynamic shapes
TORCH_TRACE=/tmp/trace python tools/reproduce.py ModelName  # Capture trace
pip install tlparse && tlparse /tmp/trace -o /tmp/report     # Visualize trace
```
