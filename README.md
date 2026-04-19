# OSS Model Compiler Quality Corpus

A reusable corpus of **734 open-source models** for measuring and improving `torch.compile` quality. Structured, reproducible, extensible.

## Who Is This For?

Compiler developers working on `torch.compile`. Three workflows:

1. **Find & fix graph breaks** — reproduce any break with one command, see root causes and fix hints, prioritize by impact across 734 models
2. **Prioritize work** — see which break categories affect the most models, track version-over-version progress, identify high-ROI fixes
3. **Validate changes** — test compiler changes, dynamo flags, or diagnostics against a known corpus of real-world breaks

## Results at a Glance (PyTorch 2.11)

|  | eval | train |
|---|---|---|
| **full\_graph** | 531 (67%) | 489 (62%) |
| **graph\_break** | 177 (22%) | 219 (28%) |
| **error** | 82 (10%) | 82 (10%) |

**240 models** have graph breaks in at least one mode. Zero full\_graph→graph\_break regressions across four releases (2.8→2.11).

Nightly tracking and per-version details: [`results/`](results/)

## Quick Start

```bash
# Install
bash scripts/setup_env.sh

# Browse the corpus
python3 tools/query.py
python3 tools/query.py --status graph_break
python3 tools/query.py --error deepcopy

# Reproduce a graph break (no GPU needed)
python3 tools/reproduce.py BartModel --explain
```

## Guides

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting-started.md) | Install, browse, reproduce your first graph break |
| [Running Sweeps](docs/running-sweeps.md) | Test your own PyTorch version against the full corpus |
| [Running Experiments](docs/running-experiments.md) | Config-driven flag testing, ablations, model subsets |
| [Understanding Results](docs/understanding-results.md) | Interpret statuses, model variants, corpus format, dashboard |
| [Issue Management](docs/issue-management.md) | Post-sweep graph break classification and GitHub issue tracking |
| [Contributing](docs/contributing.md) | Add models, fix graph breaks, architecture reference |

## Design Doc

Full methodology, taxonomy, and analysis: [design/design-doc.md](design/design-doc.md)

Graph break pattern analysis for PT2 team: [analysis/graph-break-analysis.md](analysis/graph-break-analysis.md)

## Feedback

Questions, bug reports, or feature requests: [open a GitHub issue](https://github.com/penguinwu/oss-model-graph-break-corpus/issues).

Use `for:*` labels to route issues:
- `for:dynamo-team` — PyTorch Dynamo compiler issues
- `for:hf-transformers` — HuggingFace Transformers model/library fixes
- `for:corpus-tooling` — corpus pipeline and tooling improvements
