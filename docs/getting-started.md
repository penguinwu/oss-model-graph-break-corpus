# Getting Started

Install the corpus tools, browse model data, and reproduce your first graph break in under 5 minutes.

## Install

```bash
# Option 1: Auto-setup (creates venv at ./env, or supply a custom path)
bash scripts/setup_env.sh
bash scripts/setup_env.sh ~/envs/torch211

# Option 2: Manual
python3 -m venv ~/envs/torch211
source ~/envs/torch211/bin/activate
pip install -r requirements.txt   # torch, transformers, diffusers, timm
```

Verify your environment matches the corpus:

```bash
python3 tools/version_check.py
```

## Browse the corpus

The corpus tracks 734 open-source models. Query it without a GPU:

```bash
# Summary stats
python3 tools/query.py

# All models with graph breaks
python3 tools/query.py --status graph_break

# Search by error pattern
python3 tools/query.py --error deepcopy

# Top error categories with counts
python3 tools/query.py --top-errors

# Compare static vs dynamic shapes
python3 tools/query.py --compare-dynamic

# Machine-readable output
python3 tools/query.py --status graph_break --json
```

## Reproduce a graph break

No GPU needed — runs on CPU by default:

```bash
# Basic reproduction
python3 tools/reproduce.py BartModel
python3 tools/reproduce.py BartModel --mode train

# Show ALL graph breaks with doc links (not just the first)
python3 tools/reproduce.py BartModel --explain
python3 tools/reproduce.py BartModel --explain --verbose

# Test with dynamic shapes
python3 tools/reproduce.py BartModel --dynamic mark   # batch + seq_len dims
python3 tools/reproduce.py BartModel --dynamic true   # all dims symbolic

# Run on GPU
python3 tools/reproduce.py BartModel --device cuda
```

## Analyze graph breaks

```bash
# Graph break taxonomy — distribution of root causes
python3 tools/analyze_explain.py

# Actionability — fixable in user code vs needs library PR vs needs compiler change
python3 tools/analyze_explain.py --actionability

# Per-model deep dive
python3 tools/analyze_explain.py --model BartModel

# Export to CSV
python3 tools/analyze_explain.py --csv
```

## Debug a specific graph break

A typical workflow for investigating and fixing a graph break:

1. **Find** the model — browse the [dashboard](index.html) or search: `python3 tools/query.py --error deepcopy`
2. **Reproduce** — confirm the break: `python3 tools/reproduce.py ModelName`
3. **Explain** — see all break reasons with doc links: `python3 tools/reproduce.py ModelName --explain`
4. **Trace** — capture full Dynamo artifacts: `TORCH_TRACE=/tmp/trace python3 tools/reproduce.py ModelName`
5. **Parse** — browse the trace report: `tlparse parse /tmp/trace -o /tmp/report && open /tmp/report/index.html`
6. **Fix** — patch the model or file a PR upstream (see [Contributing](contributing.md))
7. **Verify** — re-run: `python3 tools/reproduce.py ModelName`

## Compare sweep results

```bash
python3 tools/compare.py results_a.json results_b.json --labels "2.10" "2.11"
python3 tools/compare.py --corpus-dynamic
```

## Version trend analysis

```bash
python3 tools/analyze_trend.py
```

## Next steps

- [Running Sweeps](running-sweeps.md) — test your own PyTorch version against the full corpus
- [Running Experiments](running-experiments.md) — test dynamo flags, config ablations, model subsets
- [Understanding Results](understanding-results.md) — how to interpret statuses, model variants, and the corpus format
- [Contributing](contributing.md) — add models, fix graph breaks, extend the corpus
