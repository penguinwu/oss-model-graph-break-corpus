# OSS Model Compiler Quality Corpus

A reusable corpus of **468 open-source models** for measuring and improving `torch.compile` quality. Structured, reproducible, extensible.

The first application tracks `fullgraph=True` success rates across PyTorch versions. But the corpus is designed to extend to other compiler quality studies: dynamic shape behavior, recompilation patterns, graph break taxonomy, and fix validation.

## Who Is This For?

- **Compiler engineers** — measure graph break rates, track regressions across versions, identify high-impact fix targets
- **Skill builders** — validate graph break fix tools against a known corpus of breaks (e.g., testing an automated fix skill on all `deepcopy`-related breaks)
- **Researchers** — study dynamic shape behavior, recompilation patterns, or compiler diagnostics at scale

## Results at a Glance (PyTorch 2.10)

|  | Static (full\_graph / break) | `mark_dynamic` (full\_graph / break) | `dynamic=True` (full\_graph / break) |
|---|---|---|---|
| **eval** | 337 / 90 | 335 / 97 | 339 / 90 |
| **train** | 323 / 105 | 318 / 113 | 321 / 107 |

- **116 models** (25%) have graph breaks in at least one of 6 configurations (static/mark\_dynamic/dynamic=true × eval/train)
- **86 models** (74% of those with breaks) break in all 6 configurations
- All graph breaks come from HF Transformers; Diffusers models (5/5) all compile clean
- Train mode has more graph breaks than eval (+15 models in static)

### Version Trend (PyTorch 2.8 → 2.9 → 2.10)

Same 468 models, identical sweep code. Only variable: PyTorch version.

- full\_graph eval: 298 (64%) → 324 (69%) → 337 (72%)
- full\_graph train: 288 (62%) → 314 (67%) → 323 (69%)
- **12 graph breaks fixed in v2.10**, zero full\_graph→graph\_break regressions across two major releases

Reproducible: `python3 tools/analyze_trend.py`

## Quick Start

### Install dependencies

```bash
bash scripts/setup_env.sh        # creates venv and installs deps
source env/bin/activate

# Or manually:
pip install -r requirements.txt  # torch 2.10.0, transformers 5.4.0, diffusers 0.37.1
```

### Browse the corpus

```bash
# Summary
python3 tools/query.py

# All graph break models
python3 tools/query.py --status graph_break

# Search by error pattern (e.g., deepcopy, Logger, unbacked)
python3 tools/query.py --error deepcopy

# Top error categories with counts
python3 tools/query.py --top-errors

# Compare static vs dynamic shapes
python3 tools/query.py --compare-dynamic

# Machine-readable output
python3 tools/query.py --status graph_break --json
```

### Reproduce a graph break

```bash
# No GPU needed — runs on CPU by default
python3 tools/reproduce.py BartModel
python3 tools/reproduce.py BartModel --mode train

# Show ALL graph breaks (not just the first)
python3 tools/reproduce.py BartModel --explain
python3 tools/reproduce.py BartModel --explain --verbose  # include stack traces

# Test with dynamic shapes
python3 tools/reproduce.py BartModel --dynamic mark  # batch + seq_len dims
python3 tools/reproduce.py BartModel --dynamic true  # all dims symbolic

# Or explicitly on GPU
python3 tools/reproduce.py BartModel --device cuda
```

### Analyze graph breaks

```bash
# Graph break taxonomy — distribution of root causes
python3 tools/analyze_explain.py

# Actionability breakdown — fixable in user code vs needs library PR vs needs compiler change
python3 tools/analyze_explain.py --actionability

# Per-model deep dive
python3 tools/analyze_explain.py --model BartModel

# Version trend analysis
python3 tools/analyze_trend.py

# Export to CSV for further analysis
python3 tools/analyze_explain.py --csv
```

### Check your environment

```bash
# Verify installed versions match the corpus
python3 tools/version_check.py

# Pre-sweep environment validation
python3 sweep/run_sweep.py --check-env
```

### Compare sweep results

```bash
python3 tools/compare.py results_a.json results_b.json --labels "2.9" "2.10"
python3 tools/compare.py --corpus-dynamic
```

### Validate corpus integrity

```bash
# Run all validation checks (golden set, schema, consistency)
python3 tools/validate.py
```

## Corpus Format

`corpus/corpus.json` — 468 models with eval + train results across static and dynamic configurations. Each model includes status, compile time, error details. See `corpus/summary.md` for a human-readable overview.

`corpus/golden_set.json` — 2,133 expected-status checks for regression detection. Run `python3 tools/validate.py` to verify the corpus matches the golden set.

Sweep results by version: `sweep_results/{v2.8,v2.9,v2.10}/` — JSONL checkpoints for identify and explain passes.

## Debugging Graph Breaks

### TORCH_TRACE + tlparse

For deep investigation of graph breaks, capture the full compilation trace:

```bash
# Capture trace artifacts to a directory
TORCH_TRACE=/tmp/trace python3 tools/reproduce.py BartModel

# Parse the trace into a browsable report
pip install tlparse  # trace visualization tool for torch.compile artifacts
tlparse parse /tmp/trace -o /tmp/report

# Opens an HTML report with:
#   - Full graph IR for each subgraph
#   - Guard expressions that caused breaks
#   - Dynamo decision log
#   - Performance counters
```

### Batch-generate trace reports

Pre-generated traces exist for all 214 model-mode combinations (v2.10). To generate browsable HTML reports for all of them:

```bash
python3 tools/generate_trace_reports.py                    # all 214 traces → trace_reports/
python3 tools/generate_trace_reports.py --skip-existing    # resume interrupted run
python3 tools/generate_trace_reports.py --version v2.9     # different version
```

This generates an `index.html` with a filterable table linking to each model's tlparse report.

**Note:** Reports are generated locally (not hosted). To share, consider uploading to a manifold bucket or GitHub Pages.

### Typical debugging workflow

1. **Identify** — find the model in the corpus: `python3 tools/query.py --error deepcopy`
2. **Reproduce** — confirm the break: `python3 tools/reproduce.py ModelName`
3. **Explain** — see all break reasons: `python3 tools/reproduce.py ModelName --explain --verbose`
4. **Trace** — capture full artifacts: `TORCH_TRACE=/tmp/trace python3 tools/reproduce.py ModelName`
5. **Parse** — browse the trace: `tlparse /tmp/trace`
6. **Fix** — patch the model or upstream (see "Fix a model" below)
7. **Verify** — re-run: `python3 tools/reproduce.py ModelName`

### Export models for batch investigation

```bash
# Export all deepcopy-broken models for a targeted sweep
python3 tools/query.py --error deepcopy --output deepcopy_models.json

# Run a sweep on just those models
python3 sweep/run_sweep.py --models deepcopy_models.json --device cuda
```

## Contributing (Corpus Builders)

### Run a sweep

```bash
python3 -m venv env && source env/bin/activate
pip install torch transformers diffusers

python3 sweep/run_sweep.py \
    --device cuda \
    --source hf+diffusers \
    --identify-only \
    --identify-modes eval train \
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
python3 sweep/worker.py --model hf/ModelName --device cuda
```

### Architecture

| Script | Role |
|--------|------|
| `sweep/models.py` | Model enumeration from HF/Diffusers/TIMM |
| `sweep/worker.py` | Single-model subprocess (create → eager → compile) |
| `sweep/run_sweep.py` | Orchestrator (parallel workers, timeouts, checkpointing) |
| `sweep/sweep_watchdog.py` | Progress monitor + auto-restart on failure |
| `sweep/large_models.json` | 17 models needing extended timeouts during sweeps |
| `tools/query.py` | Query corpus by status, error, dynamic comparison |
| `tools/reproduce.py` | Reproduce a single model's graph break |
| `tools/analyze_explain.py` | Graph break taxonomy and root cause analysis |
| `tools/analyze_trend.py` | Version trend analysis across PyTorch releases |
| `tools/validate.py` | Corpus integrity checks (golden set, schema) |
| `tools/compare.py` | Compare two sweep results side-by-side |
| `tools/version_check.py` | Verify environment matches corpus versions |
| `tools/update_corpus.py` | Update corpus.json from sweep results |
| `tools/generate_trace_reports.py` | Batch-generate tlparse HTML reports with index |
| `scripts/setup_env.sh` | One-command virtual environment setup |

The sweep uses process-group isolation, non-blocking polling, GPU pressure backoff, and JSONL checkpointing for crash recovery.

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |
| Python | 3.12.13 |
| GPU | NVIDIA A100 80GB |

## Feedback

Questions, bug reports, or feature requests: join the [Corpus Feedback space](https://chat.google.com/room/AAQABmB_3Is).

## Design Doc

Full methodology, taxonomy, and analysis: [docs/design-doc.md](docs/design-doc.md)

Graph break pattern analysis for PT2 team: [analysis/graph-break-analysis.md](analysis/graph-break-analysis.md) — root cause taxonomy (12 categories), per-category fix paths, fix impact analysis, and cross-configuration comparison.
