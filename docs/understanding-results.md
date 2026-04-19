# Understanding Results

How to interpret sweep output, model variants, the corpus format, and the browsable dashboard.

## Status values

Every model-mode combination gets one of these statuses:

| Status | Meaning |
|--------|---------|
| `full_graph` | Compiled successfully with `fullgraph=True` — no graph breaks |
| `graph_break` | Compiled but with graph breaks (Dynamo falls back to eager for some subgraphs) |
| `error` | Compilation failed (Dynamo internal error, unsupported op, etc.) |
| `timeout` | Exceeded the per-model time limit (default 180s) |
| `create_error` | Model failed to instantiate before compilation |
| `eager_error` | Model failed during eager (non-compiled) execution |
| `cuda_error` | CUDA error during eager execution |

The first three are compiler quality signals. The last three are infrastructure issues (the model itself is broken, not the compiler).

## Model variants

The corpus tests three tiers of HuggingFace model classes:

| Tier | Example | What it tests |
|------|---------|---------------|
| **Base Model** | `Gemma4Model` | Transformer backbone only |
| **ForCausalLM** | `Gemma4ForCausalLM` | Backbone + LM head + loss computation |
| **ForConditionalGeneration** | `Gemma4ForConditionalGeneration` | Backbone + task head + multimodal merge |

Users don't compile base models — they compile `ForCausalLM` (text generation) or `ForConditionalGeneration` (multimodal generation). The `ForConditionalGeneration` wrapper adds vision-text merge logic that can introduce graph breaks invisible at the base model level.

Each variant is a separate entry in the corpus, so you can compare compile quality across tiers for the same architecture.

## Corpus format

### corpus.json

`corpus/corpus.json` contains all 734 models with eval + train results across static and dynamic configurations. Each model entry includes:

- Model name and source (hf, diffusers, custom)
- Status per mode (eval, train)
- Compile time
- Error details and break reasons (for graph_break models)
- Results across PyTorch versions

### golden_set.json

`corpus/golden_set.json` contains 1,432 expected-status checks for regression detection:

```bash
python3 tools/validate.py
```

This verifies the corpus matches the golden set and catches unintended status changes.

### Sweep results by version

Raw sweep data lives in `sweep_results/{pt2.8,pt2.9,pt2.10,pt2.11}/` as JSONL checkpoints. Human-readable summaries are in [`results/`](../results/).

## Browsable dashboard

A browsable HTML dashboard of all 734 models is available at `docs/index.html`:

```bash
python3 tools/generate_index.py    # generates docs/index.html
open docs/index.html               # browse locally
```

The dashboard shows each model's status, break count, root cause category, fixability rating (Easy/Medium/Hard), and links to trace reports. Filterable by status, mode, and fixability.

## Trace reports

Pre-generated trace directories exist in `sweep_results/pt2.10/traces/`. To generate browsable HTML reports:

```bash
# Generate reports for the top 30 models by break count
python3 tools/generate_traces.py

# Preview which models will be selected
python3 tools/generate_traces.py --list

# Customize
python3 tools/generate_traces.py --top 50
python3 tools/generate_traces.py --skip-existing

# Regenerate dashboard to pick up trace links
python3 tools/generate_index.py
```

Reports are written to `docs/traces/` (~20 MB per model). After generating, refresh the dashboard to see "view" links.

To generate a report for a single model:

```bash
TORCH_TRACE=/tmp/trace python3 tools/reproduce.py BartModel
tlparse parse /tmp/trace -o /tmp/report
open /tmp/report/index.html
```

## Graph break analysis

### Taxonomy

```bash
# Root cause distribution
python3 tools/analyze_explain.py

# Actionability breakdown
python3 tools/analyze_explain.py --actionability

# Per-model details
python3 tools/analyze_explain.py --model BartModel
```

### Version trends

```bash
python3 tools/analyze_trend.py
```

This shows fullgraph counts, graph break counts, fixes, and regressions across PyTorch versions.

## Export for further analysis

```bash
# Export models matching a filter
python3 tools/query.py --error deepcopy --output deepcopy_models.json

# Run a targeted sweep on exported models
python3 sweep/run_sweep.py --models deepcopy_models.json --device cuda

# Export graph break analysis as CSV
python3 tools/analyze_explain.py --csv
```

## Next steps

- [Getting Started](getting-started.md) — reproduce your first graph break
- [Running Sweeps](running-sweeps.md) — test your own PyTorch version
- [Issue Management](issue-management.md) — how graph breaks map to tracked issues
