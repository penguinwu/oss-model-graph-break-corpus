# OSS Model Compiler Quality Corpus

A reusable corpus of **716 open-source models** (424 families) for measuring and improving `torch.compile` quality. Structured, reproducible, extensible.

The first application tracks `fullgraph=True` success rates across PyTorch versions. But the corpus is designed to extend to other compiler quality studies: dynamic shape behavior, recompilation patterns, graph break taxonomy, and fix validation.

## Who Is This For?

Compiler developers working on `torch.compile`. Three workflows:

1. **Find & fix graph breaks** — reproduce any break with one command, see root causes and fix hints, prioritize by impact across 716 models
2. **Prioritize work** — see which break categories affect the most models, track version-over-version progress, identify high-ROI fixes
3. **Validate tools** — test graph break fix skills, compiler changes, or diagnostics against a known corpus of real-world breaks

## Results at a Glance (PyTorch 2.10, expanded corpus)

|  | eval | train |
|---|---|---|
| **full\_graph** | 521 (73%) | 478 (67%) |
| **graph\_break** | 169 (24%) | 217 (30%) |
| **error** | 26 (4%) | 21 (3%) |

- **716 models** across model families — base models, ForCausalLM, and ForConditionalGeneration
- **242 models** have graph breaks in at least one mode (including dynamic shapes)
- All graph breaks come from HF Transformers; Diffusers models compile clean
- ForConditionalGeneration has the highest break rate (~43%) — vision-text merge paths

### Version Trend (PyTorch 2.8 → 2.11, original 468 base models)

| Version | eval full\_graph | train full\_graph | eval break | train break | Models | Fixes | Regressions |
|---------|-----------------|-------------------|------------|-------------|--------|-------|-------------|
| 2.8  | 298 (64%) | 288 (62%) | 96  | 106 | 468 | —  | — |
| 2.9  | 324 (69%) | 314 (67%) | 101 | 110 | 468 | 0  | 0 |
| 2.10 | 337 (72%) | 323 (69%) | 90  | 105 | 468 | 12 | 0 |
| 2.11 | 350 (74%) | 333 (70%) | 87  | 104 | 473 | 2  | 0 |

Steady improvement across four releases. **Zero full\_graph→graph\_break regressions** in any release.

The 2.10 expanded corpus (716 models) adds ForCausalLM and ForConditionalGeneration variants, testing the full model stack users actually compile.

Nightly tracking and per-version details: [`results/`](results/)

Reproducible: `python3 tools/analyze_trend.py`

## Quick Start

### Install dependencies

```bash
# Option 1: Auto-setup (creates venv at ./env by default, or supply a custom path)
bash scripts/setup_env.sh                    # creates ./env
bash scripts/setup_env.sh ~/envs/torch210    # creates at a custom location

# Option 2: Manual
python3 -m venv ~/envs/torch210   # or wherever you prefer
source ~/envs/torch210/bin/activate
pip install -r requirements.txt   # torch, transformers, diffusers, timm
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

# Show ALL graph breaks with doc links (not just the first)
python3 tools/reproduce.py BartModel --explain
python3 tools/reproduce.py BartModel --explain --verbose  # also capture TORCH_TRACE

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

## Browsing the Corpus

### Corpus dashboard

A browsable HTML dashboard of all 716 models is available at `docs/index.html`:

```bash
python3 tools/generate_index.py    # generates docs/index.html
open docs/index.html               # browse locally
```

The dashboard shows each model's status, break count, root cause category, fixability rating (Easy/Medium/Hard), and links to tlparse trace reports when available. Filterable by status, mode, and fixability.

### Trace reports

Pre-generated trace directories exist in `sweep_results/pt2.10/traces/` (214 model-mode combinations). To generate browsable HTML reports for the top models:

```bash
# Generate tlparse reports for the top 30 models by break count
python3 tools/generate_traces.py

# Preview which models will be selected
python3 tools/generate_traces.py --list

# Customize
python3 tools/generate_traces.py --top 50             # more models
python3 tools/generate_traces.py --skip-existing       # resume interrupted run

# Regenerate dashboard to pick up trace links
python3 tools/generate_index.py
```

Reports are written to `docs/traces/` (gitignored — ~20 MB per model). After generating, refresh the dashboard to see "view" links in the Trace column.

To generate a report for a single model manually:

```bash
TORCH_TRACE=/tmp/trace python3 tools/reproduce.py BartModel
tlparse parse /tmp/trace -o /tmp/report
open /tmp/report/index.html
```

## Model Variants

The sweep tests three tiers of HuggingFace model classes, reflecting how users actually apply `torch.compile`:

| Tier | Example | What it tests | Why it matters |
|------|---------|---------------|----------------|
| **Base Model** | `Gemma4Model` | Transformer backbone only | Core attention/MLP compile quality |
| **ForCausalLM** | `Gemma4ForCausalLM` | Backbone + LM head + loss | The standard text generation compile target |
| **ForConditionalGeneration** | `Gemma4ForConditionalGeneration` | Backbone + task head + multimodal merge | Real-world VLM compile — where vision-text integration breaks surface |

Users don't compile base models — they compile `ForCausalLM` (text generation) or `ForConditionalGeneration` (multimodal generation). The `ForConditionalGeneration` wrapper adds vision-text merge logic that can introduce graph breaks invisible at the base model level. For example, `Gemma4Model` compiles clean, but `Gemma4ForConditionalGeneration` has 5 graph breaks from dynamic shape ops in the vision merge path.

Each variant is a separate entry in the corpus, so you can compare compile quality across tiers for the same architecture.

## Corpus Format

`corpus/corpus.json` — models with eval + train results across static and dynamic configurations. Each model includes status, compile time, error details. See `corpus/summary.md` for a human-readable overview.

`corpus/golden_set.json` — 1,432 expected-status checks for regression detection. Run `python3 tools/validate.py` to verify the corpus matches the golden set.

Sweep results by version: `sweep_results/{pt2.8,pt2.9,pt2.10,pt2.11}/` — JSONL checkpoints for identify and explain passes. See [`results/`](results/) for human-readable summaries and nightly tracking.

## Debugging Graph Breaks

### Typical debugging workflow

1. **Browse** — find the model in the dashboard: `open docs/index.html`
2. **Query** — or search by error: `python3 tools/query.py --error deepcopy`
3. **Reproduce** — confirm the break: `python3 tools/reproduce.py ModelName`
4. **Explain** — see all break reasons with doc links: `python3 tools/reproduce.py ModelName --explain`
5. **Trace** — capture full artifacts: `TORCH_TRACE=/tmp/trace python3 tools/reproduce.py ModelName`
6. **Parse** — browse the trace: `tlparse parse /tmp/trace -o /tmp/report`
7. **Fix** — patch the model or upstream (see "Fix a model" below)
8. **Verify** — re-run: `python3 tools/reproduce.py ModelName`

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
# Set up a venv with torch + model libraries (wherever you prefer)
python3 -m venv ~/envs/torch210
~/envs/torch210/bin/pip install -r requirements.txt

# Run a sweep — point --python to your venv
python3 sweep/run_sweep.py \
    --device cuda \
    --python ~/envs/torch210/bin/python \
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
| `sweep/models.py` | Model enumeration from HF/Diffusers/TIMM (base + ForCausalLM + ForConditionalGeneration) |
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
| `tools/generate_index.py` | Generate corpus dashboard (docs/index.html) |
| `tools/generate_traces.py` | Pre-generate tlparse reports for top N models |
| `tools/feedback_monitor.py` | Monitor GChat feedback space for user reports |
| `tools/generate_nightly_summary.py` | Generate nightly comparison markdown (auto-run by `run_nightly.sh`) |
| `tools/github_issue_monitor.py` | Monitor GitHub issues for new activity |
| `scripts/setup_env.sh` | One-command virtual environment setup |

The sweep uses process-group isolation, non-blocking polling, GPU pressure backoff, and JSONL checkpointing for crash recovery.

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.5.3 |
| Diffusers | 0.37.1 |
| Python | 3.12.13 |
| GPU | NVIDIA A100 80GB |

## Feedback

Questions, bug reports, or feature requests: [open a GitHub issue](https://github.com/penguinwu/oss-model-graph-break-corpus/issues).

Use `for:*` labels to route issues to the right team:
- `for:dynamo-team` — PyTorch Dynamo compiler issues
- `for:hf-transformers` — HuggingFace Transformers model/library fixes
- `for:corpus-tooling` — corpus pipeline and tooling improvements

Also available: [Corpus Feedback GChat space](https://chat.google.com/room/AAQABmB_3Is).

## Design Doc

Full methodology, taxonomy, and analysis: [docs/design-doc.md](docs/design-doc.md)

Graph break pattern analysis for PT2 team: [analysis/graph-break-analysis.md](analysis/graph-break-analysis.md) — root cause taxonomy (12 categories), per-category fix paths, fix impact analysis, and cross-configuration comparison.
