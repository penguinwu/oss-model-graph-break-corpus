---
plan: WS1 — Skill Discovery Phase 3
status: active
owner: Otter
created: 2026-04-24
last_check: 2026-04-24
forcing_function: tools/check_plan.py + daily brief at 7:30 AM ET
---

# WS1 Phase 3 — Discovery agent across diverse graph break cases

## Goal

Run the discovery harness against 4 cases that span the widest graph-break shape diversity available in the corpus. Goal is **strategy discovery**, not fix-rate maximization. Per Peng 2026-04-24: "Graph break skills should not differentiate between dynamo fixable or not." Selection criterion: distinct break explanations, not user-fixability.

## Frame

Each case = one graph-break shape. Per case: M=2 prompt variants (with-skill, no-skill) × N=6 trials → strategy fingerprints → cross-case attractor analysis.

Reference: `discovery/design.md` v0.4 §6 (case file schema), §8 (Phase 1 closed Qs + Phase 3 open Qs).

## Queue

| # | Case | gb count | Distinct types | Why selected | Status |
|---|------|----------|----------------|--------------|--------|
| 3a | **Mistral3** | 16 | 8 | Widest single-model diversity probe | not started |
| 3b | **VitsModel (train)** | 29 | 7 | Train mode + `as_proxy` + `find_spec` — novel categories | not started |
| 3c | **Aria** | 27/28 | 7 | Second probe of Mistral3-shaped break space | not started |
| 3d | **PaddleOCRVL** | 19 | 6 | Operator-generalization probe across data-dep ops | not started |

Order is by diversity-first; do NOT reorder without writing why in the revision log.

## Per-case execution shape

For each case in order:

1. Author `discovery/cases/<case>_<break-id>.{py,baseline.json}` matching the schema in `design.md` §6.
2. Pre-flight: validate model loads, repro the break, baseline correctness recorded.
3. Run harness sequentially (no parallel — Pilot 3 race-condition lesson). Wall budget: ~30 min/case.
4. Tier-2 enrichment: run `enrich_tier2.py` on the variant outputs.
5. Write per-case findings to `discovery/reports/<case>_<date>.md`. Distill: distinct strategies, fingerprint axes, surprises.
6. Update this plan: status → done, link the report.

## Cross-case synthesis (after 3a–3d done)

- Cross-case attractor analysis: do the same strategy clusters appear, or does each case have its own attractor set?
- Fingerprint axis stability: do the 5 axes locked in Phase 1 still partition strategies cleanly across all 4 cases?
- Mental-model deliverable: doc + Workplace post (the WS1 forcing function from 2026-04-23).

## Done means

For this plan to be marked complete:
- All 4 cases have a finished report under `discovery/reports/`.
- `agent_diff.patch`, `prompt.txt`, `result.json` per trial are in git (per gitignore convention; raw `stream.jsonl` archived to Drive).
- Cross-case synthesis section above is written into `discovery/design.md` §8 (Phase 3 closed Qs).
- Mental-model doc landed in PARA + Workplace post sent.

## Revision log

- 2026-04-24: Plan created. Order set by Peng 2026-04-24 evening (diversity-first, T5 deferred, FLOP reasoning de-emphasized).
