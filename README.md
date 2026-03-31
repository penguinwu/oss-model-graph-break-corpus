# OSS Model Graph Break Corpus

A corpus of open-source models tested for `torch.compile(fullgraph=True)` compatibility.

## What is this?

When `torch.compile(model, fullgraph=True)` fails, it means the model contains operations that force **graph breaks** — points where the compiler must split the computation graph, preventing full-model optimization.

This corpus systematically tests **468 models** from HuggingFace Transformers and Diffusers against `torch.compile(fullgraph=True, backend="eager")` in both eval and train modes, producing a structured dataset of which models compile cleanly and which have graph breaks.

### Environment

This corpus was generated with:

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |
| Python | 3.12.13 |
| CUDA | 12.8 |
| GPU | NVIDIA A100 80GB |

To reproduce, install matching versions:
```bash
pip install torch==2.10.0 transformers==5.4.0 diffusers==0.37.1
```

## Corpus Summary

| Status | Eval | Train |
|--------|------|-------|
| Clean (fullgraph=True works) | 295 | 288 |
| Graph Break | 69 | 77 |
| Eager Error | 96 | 95 |
| Create Error | 6 | 6 |
| Timeout | 2 | 2 |

**80 models** have graph breaks in at least one mode (eval or train).

Results reflect 6 rounds of refinement (R1-R6): initial sweep, retry passes, model size reduction, config fixes for composite models, MoE topk guards, mrope_section fixes, and head_dim/rope_theta patches.

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
pip install torch==2.10.0 transformers==5.4.0 diffusers==0.37.1

# Quick check for a single model
python reproduce.py BartModel

# With train mode
python reproduce.py BartModel --mode train

# List all graph-break models
python reproduce.py --list
```

### Run the full sweep

```bash
# Set up environment
python -m venv env && source env/bin/activate
pip install torch transformers diffusers

# Run sweep (eval + train, all HF + Diffusers models)
python sweep/run_sweep.py \
    --device cuda \
    --source hf+diffusers \
    --pass1-only \
    --workers 4 \
    --timeout 180
```

The sweep writes results incrementally to `sweep_results/pass1_checkpoint.jsonl` and can resume from crashes with `--resume`.

## Graph Break Taxonomy

The **80 graph-break models** fall into **10 root cause categories**. The top 3 account for 56% of all breaks and are all fixable in PyTorch core or HuggingFace model code — not inherent model limitations.

| Root Cause | Count | % | Fix Location |
|------------|-------|---|-------------|
| **copy.deepcopy()** | 19 | 25% | HF model code: replace deepcopy with clone() |
| **Skipped/forbidden callable** | 15 | 20% | PyTorch core: support these callables in Dynamo |
| **as_proxy() missing** | 8 | 11% | PyTorch core: implement as_proxy() for failing types |
| **Data-dependent branching** | 7 | 9% | Model code: requires torch.cond() or restructuring |
| **Untraceable builtin (`callable`)** | 6 | 8% | PyTorch core: teach Dynamo to trace `callable()` |
| **logging.Logger** | 5 | 7% | PyTorch core: skip/inline logger calls in Dynamo |
| **Unbacked symbols** | 2 | 3% | Hard: model generates shapes dynamically |
| **Observed exception (try/except)** | 2 | 3% | Dynamo exception handler support |
| **requires_grad_()** | 1 | 1% | Dynamo mutation support |
| **Non-Tensor return** | 1 | 1% | Dynamo op return type support |

**Key insight:** `copy.deepcopy()` alone causes 25% of all graph breaks — all 19 are encoder-decoder models (BART, T5, Pegasus, Whisper, etc.) that clone decoder layers from encoder layers during init. A single HF PR replacing deepcopy with explicit clone() would fix all 19 models.

## Methodology

### Two-Pass Sweep

The sweep uses a two-pass approach to efficiently identify and analyze graph breaks:

**Pass 1 (identification)** — runs `torch.compile(model, fullgraph=True, backend="eager")` on every model. `fullgraph=True` fails immediately on the first graph break, so broken models fail in <0.1s while clean models take 3-10s. This makes Pass 1 fast even on the full corpus. The `--pass1-only` flag runs only this pass.

**Pass 2 (diagnostics)** — runs `torch._dynamo.explain()` + `TORCH_TRACE` only on models that broke in Pass 1. `explain()` reports *all* graph breaks (not just the first), with source locations, graph counts, and subgraph sizes. `TORCH_TRACE` collects full tracebacks and FX output graphs. This avoids running expensive diagnostics on the ~90% of models that compile cleanly.

### Why `backend="eager"`?

The `eager` backend tests Dynamo's tracing ability without adding inductor codegen overhead. Benchmarks show it's 2-5x faster than `aot_eager` and 5.5x faster than `inductor`, with identical graph break detection. The corpus captures Dynamo-level graph breaks — backend-specific issues (inductor bugs, codegen failures) are a separate concern.

| Backend | resnet50 eval | Relative |
|---------|--------------|----------|
| eager | 3.3s | 1.0x |
| aot_eager | 8.1s | 2.4x |
| inductor | 38.2s | 11.6x |

### Why batch_size=2?

PyTorch specializes on dimension values 0 and 1, treating them as constants rather than symbols. Using `batch=1` bypasses dynamic shape machinery, giving false confidence that compilation works. Any batch size >= 2 avoids this specialization.

We use `bs=2` instead of `bs=3` because they produce identical graph break detection results, but `bs=2` uses 33% less GPU memory — allowing more parallel workers and faster sweeps.

### Why HF Transformers + Diffusers (not TIMM)?

TIMM (1,284 vision models) is excluded from the default sweep because it's nearly solved:

| | TIMM | HF Transformers + Diffusers |
|---|---|---|
| Models | 1,284 (73% of corpus) | 468 (27% of corpus) |
| Graph breaks | 3 (0.2%) | 70 (15%) |
| Root causes | 1 (RNN wrapping) | 10 categories |
| Sweep time | ~5 hours | ~1 hour |

TIMM contributes 73% of sweep time but only 4% of graph break signal (3 out of 73 breaks, all the same RNN root cause). The full corpus including TIMM is available via `--source all`.

### Dynamic Shapes Don't Add New Graph Breaks

A `dynamic=True` sweep on statically-clean models showed that dynamic shapes do not introduce new graph break *categories*. The same root causes (deepcopy, skipped callables, data-dependent branching, etc.) appear whether shapes are static or dynamic. Dynamic shapes primarily add compile timeouts from symbolic constraint explosion.

The default sweep runs static shapes only. Dynamic sweeps are available via `--dynamic true` for identifying compile performance regressions.

## Corpus Format

`corpus/corpus.json` contains:

```json
{
  "metadata": {
    "pytorch_version": "2.10.0+cu128",
    "python_version": "3.12.13",
    "methodology": "..."
  },
  "summary": { "total_models": 468, "graph_break_models": 70 },
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
