# Issue Management

After a sweep, graph breaks are classified into issue categories and tracked as GitHub issues. This page covers the classification workflow, issue structure, and how to update issues from sweep results.

## Overview

Issues live at [github.com/penguinwu/oss-model-graph-break-corpus/issues](https://github.com/penguinwu/oss-model-graph-break-corpus/issues) and fall into three categories:

- **Dynamo issues** — pattern-level graph break categories (e.g., data-dependent branching, context managers). These are the actionable compiler issues.
- **Model-specific issues** — breaks unique to individual models that don't fit a pattern.
- **Corpus-infra issues** — models that fail before compilation (create_error, timeout, eager_error). Not compiler quality issues.

## Two-command workflow

Issue management uses a plan-then-apply pattern:

### 1. Generate a report (read-only)

```bash
python3 tools/file_issues.py sweep-report \
    --explain sweep_results/nightly/2026-04-19/explain_results.json \
    --identify sweep_results/nightly/2026-04-19/identify_results.json
```

This produces a JSON plan file containing:
- **Issue updates** — new affected model tables, break reason samples, and updated counts for every open issue
- **Leverage rankings** — which fixes would unlock the most fullgraph models (single-fix impact analysis)
- **Close candidates** — issues where all previously-listed models have been reclassified, moved to fullgraph, or removed from the corpus, with model-by-model disposition evidence
- **Unclassified patterns** — break reasons that don't match any existing rule

### 2. Review and apply

Review the JSON plan, then apply:

```bash
python3 tools/file_issues.py sweep-update \
    --plan sweep_results/nightly/2026-04-19/sweep-report.json
```

This PATCHes GitHub issues: updates bodies with current model tables, adjusts titles with counts, and closes issues that qualify.

## Classification rules

The classifier maps graph break reasons to issue numbers using pattern-matching rules defined in `tools/file_issues.py` (`GRAPH_BREAK_RULES`). Rules are matched top-to-bottom — more specific rules must come before catch-all rules.

Each rule specifies:
- A regex pattern matching the break reason text
- The GitHub issue number to classify under
- A human-readable key for the rule

## Leverage analysis

The report computes **leverage** — how many models would become `full_graph` if a single issue were fixed. A model counts toward leverage only if the issue is its **only** break reason.

This is the key metric for prioritization: fixing issue #54 (data-dependent branching) unlocks 77 models to fullgraph, far more than any other single fix.

## Issue body structure

Machine-filed issues contain:

- **Affected models table** — every model with this break pattern, with mode and status
- **Break reason samples** — representative error messages
- **Leverage count** — models to fullgraph if fixed
- **Cross-references** — related issues

All machine-filed issues are tagged with `<!-- filed-by: otter/file_issues.py -->`.

## Close candidate evidence

When the report identifies a close candidate, it includes **model disposition** — for every model previously listed in the issue:

- `fullgraph on current sweep` — the model now compiles clean
- `reclassified -> #X (rule_key)` — the break is now classified under a different, more specific issue
- `removed from corpus` — the model was removed
- `still breaking, pattern unclassified` — the model still breaks but the pattern doesn't match any rule (blocks closing)

An issue qualifies for closing only when every model is accounted for.

## Labels

Use `for:*` labels to route issues:
- `for:dynamo-team` — PyTorch Dynamo compiler issues
- `for:hf-transformers` — HuggingFace Transformers model/library fixes
- `for:corpus-tooling` — corpus pipeline and tooling improvements

## Next steps

- [Running Sweeps](running-sweeps.md) — generate the sweep data that feeds issue management
- [Understanding Results](understanding-results.md) — interpret sweep output
- [Contributing](contributing.md) — fix graph breaks and contribute back
