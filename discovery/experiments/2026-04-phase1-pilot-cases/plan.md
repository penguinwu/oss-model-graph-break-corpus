---
plan: 2026-04-phase1-pilot-cases
status: done
owner: Otter
created: 2026-04-25
last_check: 2026-04-25
---

# Phase 1 Pilot Cases — Skill Discovery (Jamba + Dbrx)

> **Status: closed.** This is a **retroactive plan** for the two pilot cases that ran before the Phase 2 discovery harness existed. The reports under `reports/` capture what was learned. Methodology is **not** the same as the current `2026-04-cross-case-skill-discovery` experiment — see the report-level "Note" header in each findings doc for caveats.

## What this experiment was

Pre-harness exploration of the question: "When the `debug-graph-breaks` skill is loaded into the agent, does the agent's reasoning, fix-space, or fix-shape change vs. bare Claude?" Two pilot models, two break shapes, hand-rolled `run_trial.sh` harness.

| Case | Model | Break shape | Trials | Report |
|---|---|---|---|---|
| `jamba_mask_branch` | `JambaForCausalLM` | data-dep `if` with provably-safe deletion fix | 6 (3 with_skill + 3 no_skill) | [findings.md](reports/jamba_mask_branch/findings.md) |
| `dbrx_moe_data_dep` | `DbrxForCausalLM` | data-dep MoE expert-dispatch loop, no obvious deletion fix | 6 (V0×3 + V1×3) | [findings.md](reports/dbrx_moe_data_dep/findings.md) |

## What it produced (carry-forward to Phase 2)

- The pilot harness pattern (run_trial.sh + per-trial dirs) — **superseded by `discovery/run_case.py`** in Phase 2.
- Variant axis V0/V1 — **expanded to V0/V2/V4/V6** in Phase 2 (`2026-04-cross-case-skill-discovery/plan.md`).
- The framing "skill changes reasoning, not just success rate" — **carried forward** as the central question in Phase 2.
- The fingerprint-axis vocabulary (fix-shape, fix-locus, escape-hatches, op-order) — **promoted to a pinned schema** in Phase 2 (`per-case-analysis/fingerprint_schema.md`).

## Why this dir exists

To keep the Phase 1 pilot reports inside the experiments tree (consistent with the convention in `discovery/experiments/README.md`) so cross-case synthesis pulls from a single root. Without this dir the Dbrx report sat in `discovery/reports/` outside any experiment, and the Jamba report sat only on Drive.
