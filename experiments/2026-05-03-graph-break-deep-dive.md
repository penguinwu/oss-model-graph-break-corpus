---
plan: PT 2.12 graph-break technical report
status: active
owner: Otter (drafting); Peng (reviewing)
created: 2026-05-03
last_check: 2026-05-03T13:30Z
forcing_function: Today's iteration — Peng comments on gdoc draft
revision: 2
---

# PT 2.12 Graph-Break Technical Report — Project Tracker

> **Rev 2 (2026-05-03):** Reframed from open-ended "deep-dive" to a single-deliverable project: a comprehensive technical report on PT 2.12 graph-break sweep state, targeted at Dynamo compiler developers. Long-form gdoc → corpus repo .md → Workplace post (extracted later).

## Goal

Produce a technical report that gives Dynamo compiler developers a clear picture of:
- Where graph capture stands on HF transformers in PT 2.12
- What changed from PT 2.11 (regressions + improvements)
- Which remaining breaks are Dynamo-fixable vs require model-side rewrites
- Train vs eval asymmetry, with explanation

Audience: PT2 / Dynamo team. Scope: stats + trends. Out of scope: corpus infra improvements.

## Deliverables

| # | Item | Status | Where |
|---|---|---|---|
| 1 | Long-form draft (gdoc, for Peng comments) | in progress | https://docs.google.com/document/d/1ecTEnmc6J83oPOs5R5YY54ruH1qoqxbSwawZQYDo0BU/edit |
| 2 | `.md` in corpus repo (publication form, derived from finalized gdoc) | pending | `experiments/pt2.12-sweep-analysis.md` |
| 3 | Workplace post (short form, extracted from .md) | pending | (drafted after .md is final) |

## Sections (status)

| # | Section | Status | Notes |
|---|---|---|---|
| 1 | Executive summary / TL;DR | ✅ done | First-pass written; refine after Peng comments |
| 2 | Top-line stats (eval / train separated) | ✅ done | Includes break-count distribution + top break reasons + top-5 most-broken |
| 3 | New-models-since-2.11 subsection | ✅ done | 29 added; per-status lineup + new-vs-existing comparison; only 34.5% fullgraph |
| 4 | Q3: graph_count vs graph_break_count accounting | ✅ done | 0 violations across 446 explain rows |
| 5 | Q2: deep-stack origin → cascading downstream breaks | ✅ done | 53% concentrated; deepcopy at 164 breaks is biggest amplifier |
| 6 | Train vs eval asymmetry investigation | 🟡 partial | Headline + top-10 + IBert insight; deeper categorization of train-only extras still TODO |
| 7 | Deep-dive: #102 + #103 (compiler gives up wrapper pattern) | ✅ done | 193 + 141 models; inner reason categorization; 58% data-dep / 42% Dynamo-fixable for #102; #103 dominated by CALL_FUNCTION_EX |
| 8 | Regression vs 2.11 (reuse from reference doc) | ✅ done | 7 status flips; Blt eager regression flagged |
| 9 | Remaining breaks: Dynamo-fixable vs requires-model-rewrite | 🟡 partial | Initial categorization done; cross-reference with filed issues still TODO |
| 10 | Q1: nested graph break feature impact | LAST | Deferred per Peng directive — requires understanding of nested-GB feature first |

## Data sources

| What | Where | State |
|---|---|---|
| 2.11 baseline (canonical) | `sweep_results/baseline/pt2.11-fresh-2026-04-29/` | complete, 2026-04-29 |
| 2.12 baseline (pre-release) | `sweep_results/baseline/pt2.12-2026-04-30/` | partial (INTERRUPTED file present — to investigate) |
| 2.13-nightly (in-flight today) | `sweep_results/nightly/2026-05-03/` | running, currently 143/1734 |
| Comparison tool | `tools/compare.py` | existing |
| Filed issues for current breaks | GitHub `penguinwu/oss-model-graph-break-corpus` | live |

**On 2.12 wheel publication:** PT 2.12 is not yet stable-released. Today's draft uses the 2026-04-30 pre-release sweep. When the stable 2.12 wheel publishes, refresh data and re-run analysis (Peng's plan).

## Reference materials

- **Prior regression analysis (reference doc):** [PT 2.11 vs PT 2.12 — Sweep Comparison & Root-Cause Analysis (2026-04-30)](https://docs.google.com/document/d/1vG0tob5_D6PBPEs84-pe8F0BqV35VqoaKf2htgsAOG8/edit) — author: Otter; comments: Peng. Local: `sweep_results/comparisons/pt2.11-vs-pt2.12-2026-04-30/REPORT.md`
- **Issue tracker:** [penguinwu/oss-model-graph-break-corpus/issues](https://github.com/penguinwu/oss-model-graph-break-corpus/issues) — #102, #103, #104 are central to §7

## Per-section details

### §1 Executive summary
Headline numbers + 3-5 bullets on key findings. Written LAST so it can summarize what's actually in the report.

### §2 Top-line stats (eval / train separated)
From `identify_results.json`:
- # models tested (eval count, train count, intersection)
- # full_graph (% of total) eval, train
- # graph_break, eager_error, create_error, timeout
- Break-count distribution: histogram of breaks per model
- Top break reasons (categorized) — eval vs train differences

### §3 New-models-since-2.11
- Identify "new" via model-list diff between 2.11 + 2.12 baselines
- For each new model: graph break count, status, category
- Compare new-model fullgraph rate vs older-model fullgraph rate
- Implication: are newer model architectures more or less amenable to torch.compile?

### §4 Q3 — graph_count vs graph_break_count accounting
- Read explain_results.json
- For each model: extract `graph_count` (total subgraphs) + `graph_break_count` (breaks reported)
- Expected invariant: `graph_count = graph_break_count + 1`
- Find violations: which models, by how much, why
- If invariant holds: increases trust in subsequent stats. If violated: report the gap.

### §5 Q2 — deep-stack cascading
- Hypothesis: a single break deep in the call stack causes N downstream breaks
- Method: for each break, look at frame context (depth in stack); look at follow-on breaks within the same forward
- Report: distribution of cascade sizes, top "amplifier" patterns

### §6 Train vs eval asymmetry
- Per-model: train_breaks - eval_breaks delta
- Categorize the extra train breaks by reason
- Hypotheses to test:
  - Backward graph adds breaks (autograd-specific patterns)
  - Train-mode-only ops (dropout, batch_norm running stats)
  - Loss computation differs

### §7 Deep-dive: #102 + #103 wrapper pattern
- #102: `Failed to handle graph break gracefully` — 166 models
- #103: `Cannot resume from graph break` — 107 models
- These are "compiler gives up" wrappers — the real reason hides inside
- Method: extract inner reason from explain_results.json frames
- Categorize inner reasons; cross-reference #104 (better error messages)
- Quantify: how many resolve with better error surfacing vs require model rewrite

### §8 Regression vs 2.11
- Reuse from reference doc: 7 work items flipped status
  - 2 compile_error → graph_break (Gemma4 — strict capability win)
  - 1 graph_break → full_graph (MraModel — Dynamo improvement)
  - 4 full_graph → timeout (Blt × 2 modes — eager-side nn.Module construction regression, NOT torch.compile)
- Add: any newly-found regressions from re-analysis

### §9 Remaining breaks: Dynamo-fixable vs model-rewrite
- For each unique break reason in PT 2.12: classify
  - **Dynamo-fixable** — Dynamo can support this op/pattern with engineering work
  - **Model-rewrite** — model code needs change (e.g., data-dependent control flow fundamental to algorithm)
  - **Ambiguous** — could go either way
- Quantify by count; rank by impact (# models affected)

### §10 Q1 — nested graph break feature (LAST)
- Requires understanding the nested-GB feature in upstream PyTorch first
- Hypothesis: nested-GB support may reduce reported counts by treating "deep stack returns to capture frame" cases differently
- Defer to end so other sections aren't blocked

## Open questions / pending decisions

(captured as work proceeds — Peng's comments on the gdoc accumulate here)

## Pick-up state if not finished today

When ending a session, update each row in the §Sections table with one of:
- ✅ done — section text in gdoc; data validated
- 🟡 partial — outline + some text; gaps marked with TODO
- pending — not started

The gdoc has the canonical text; this plan doc tracks what's done. Tomorrow's session reads this table + picks up the highest-priority `pending` or `🟡 partial`.
