# OSS Model Graph Break Corpus

A corpus of open-source models tested for `torch.compile(fullgraph=True)` compatibility.

## What is this?

When `torch.compile(model, fullgraph=True)` fails, it means the model contains operations that force **graph breaks** — points where the compiler must split the computation graph, preventing full-model optimization.

This corpus systematically tests **468 models** from HuggingFace Transformers and Diffusers against `torch.compile(fullgraph=True, backend="eager")` in both eval and train modes, producing a structured dataset of which models compile cleanly and which have graph breaks.

## Corpus Summary

| Status | Eval | Train |
|--------|------|-------|
| Clean (fullgraph=True works) | 220 | 231 |
| Graph Break | 56 | 66 |
| Timeout | 84 | 74 |
| Eager Error | 72 | 64 |
| Create Error | 36 | 33 |

**70 models** have graph breaks in at least one mode (eval or train).

## Quick Start

### Browse the corpus

```bash
# All graph-break models
python -c "
import json
corpus = json.load(open('corpus/corpus.json'))
for m in corpus['models']:
    if m['has_graph_break']:
        print(f\"{m['source']}/{m['name']:30s}  eval={m['eval']['status']:12s}  train={m['train']['status']}\")
"
```

### Reproduce a graph break

```bash
pip install torch transformers diffusers

# Quick check for a single model
python reproduce.py BartModel

# With train mode
python reproduce.py BartModel --mode train
```

### Run the full sweep

```bash
# Set up environment
python -m venv env && source env/bin/activate
pip install torch transformers diffusers timm

# Run sweep (eval + train, all HF + Diffusers models)
python sweep/run_sweep.py \
    --device cuda \
    --source hf+diffusers \
    --pass1-only \
    --workers 4 \
    --timeout 180
```

The sweep writes results incrementally to `sweep_results/pass1_checkpoint.jsonl` and can resume from crashes with `--resume`.

## Corpus Format

`corpus/corpus.json` contains:

```json
{
  "metadata": { "pytorch_version": "...", "methodology": "..." },
  "summary": { "total_models": 468, "graph_break_models": 70, ... },
  "models": [
    {
      "name": "BartModel",
      "source": "hf",
      "has_graph_break": true,
      "eval": { "status": "graph_break", "fullgraph_ok": false, "error": "..." },
      "train": { "status": "graph_break", "fullgraph_ok": false, "error": "..." }
    }
  ]
}
```

Each model entry has:
- `name` — model class name
- `source` — `hf` (HuggingFace Transformers) or `diffusers` (HuggingFace Diffusers)
- `has_graph_break` — `true` if graph break in either eval or train mode
- `eval` / `train` — per-mode results with status, timing, memory usage, and error details

Status values:
- `clean` — `fullgraph=True` compiles successfully
- `graph_break` — compilation fails due to graph break
- `timeout` — model did not complete within timeout
- `eager_error` — model fails even in eager mode (input compatibility issue)
- `create_error` — model cannot be instantiated with default config

## Sweep Scripts

| Script | Purpose |
|--------|---------|
| `sweep/models.py` | Model enumeration — discovers models from HF Transformers and Diffusers |
| `sweep/worker.py` | Single-model worker subprocess — creates model, runs eager + compile |
| `sweep/run_sweep.py` | Orchestrator — manages parallel workers, timeouts, checkpointing |
| `sweep/sweep_watchdog.py` | Watchdog — monitors sweep progress, auto-restarts on failure |

### Sweep Architecture

The sweep uses a **poll-loop orchestrator** with process-group isolation:

- Each model runs in its own subprocess (process group via `setsid`)
- The orchestrator polls workers every 5 seconds (non-blocking)
- Timed-out workers get escalating kill signals: SIGTERM → SIGKILL → abandon
- GPU memory pressure reduces parallelism instead of aborting the sweep
- Results are checkpointed to JSONL after each model completes
- `--resume` skips already-completed models for crash recovery

## Methodology

1. For each model, instantiate with default config (no pretrained weights)
2. Generate synthetic inputs matching the model's expected input format
3. Run eager forward pass to verify the model works
4. Run `torch.compile(model, fullgraph=True, backend="eager")` and execute
5. If compilation raises `torch._dynamo.exc.Unsupported`, the model has a graph break
6. Test both `model.eval()` and `model.train()` modes

The `eager` backend isolates graph break detection from inductor-specific issues.

## Regenerating the Corpus

To regenerate with a newer PyTorch version:

```bash
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
