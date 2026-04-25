---
plan: WS1 — Skill Discovery Phase 3
status: active
owner: Otter
created: 2026-04-24
last_check: 2026-04-25
forcing_function: tools/check_plan.py + daily brief
---

# WS1 Phase 3 — Discovery agent across diverse graph break cases

**Methodology lives in [`experiments/2026-04-cross-case-skill-discovery/plan.md`](experiments/2026-04-cross-case-skill-discovery/plan.md).** This file is the workstream-level pointer; do not duplicate methodology here.

## Goal

Run the discovery harness against 4 cases that span the widest graph-break shape diversity available in the corpus. Goal is **strategy discovery**, not fix-rate maximization.

Selection criterion: distinct break shapes, NOT user-fixability. Per Peng 2026-04-24: "Graph break skills should not differentiate between dynamo fixable or not."

## Case queue

| # | Case | gb count | Why selected | Per-case issue | Status |
|---|------|----------|--------------|----------------|--------|
| 3a | Mistral3 | 16 | Widest single-model diversity probe (multimodal multi-shape) | #59 | Running (24-trial matrix launched 2026-04-25 04:18 UTC, ~12hr wall) |
| 3b | VitsModel (train) | 22 | Train mode + layerdrop + dropout-active break categories | #61 | Ready (case authored, smoke green) |
| 3c | Aria | 28 | Multimodal vision+text + MoE | #62 | Ready (case authored, smoke green) |
| 3d | PaddleOCRVL | 19 | Operator-generalization across data-dep ops | #63 | Ready (case authored, smoke green) |

## Done means

- All 4 cases have merged report PRs under `discovery/experiments/2026-04-cross-case-skill-discovery/reports/`
- Cross-case synthesis written into `synthesis.md` (same dir)
- Mental-model doc landed in PARA + Workplace post sent

## Revision log

- 2026-04-24: Plan created. Order set diversity-first.
- 2026-04-24 21:45 ET: Case 3a (Mistral3) authored, smoke green.
- 2026-04-25 00:30 ET: Master experiment plan extracted to `experiments/2026-04-cross-case-skill-discovery/plan.md`; this file slimmed to a workstream-level pointer. Cases 3b/3c/3d authored in parallel (commits authored under #61/#62/#63). Mistral3 24-trial matrix launched.
