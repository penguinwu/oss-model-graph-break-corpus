---
plan: WS2 — Issue filing friction
status: active
owner: Otter
created: 2026-04-24
last_check: 2026-04-24
forcing_function: tools/check_plan.py + daily brief at 7:30 AM ET
---

# WS2 — Lower friction for filing dynamo correctness issues

## Goal

Make filing high-quality dynamo correctness issues a 1-command operation, with rich enough body content that Animesh / dynamo team can triage without bouncing back. Stay under Animesh's 25-issue cap.

## State

- `tools/file_issues.py` has `correctness-report` and `correctness-apply` subcommands (smoke-tested 2026-04-23 against 528-model sweep → 7 family issues, max severity 5.3e9 Phi4).
- 7 family issues identified; **strategy decision blocked** on Peng (open in OPEN-LOOPS.md "Phase 3 issue filing strategy").

## Decision needed (Tier 3)

Animesh budget: file 3 → at 25 (cap), file all 7 → 29 (over). Three options:

1. **Top 3 by severity** (Phi4, AIMv2, Doge) — stays under cap, ships highest-impact signal.
2. **Top 5 by severity** — over cap by 4. Requires Animesh's blessing on cap exception.
3. **All 7** — over cap by 4. Same blessing needed.

## Tasks

| Task | Type | Notes |
|------|------|-------|
| Get Peng's call on filing strategy | needs-input | Surface in daily brief until decided |
| Run `correctness-apply` with chosen plan | blocked | Posts to penguinwu/oss-model-graph-break-corpus/issues |
| Component triage: rerun divergent models with knobs flipped | backlog | Decompositions on/off, fake tensor mode, smaller graphs — narrows which compiler component drifts |

## Done means

- Filing strategy chosen and applied; issues live at penguinwu/oss-model-graph-break-corpus/issues.
- Each filed issue has: family description, affected models list, severity, repro snippet.
- Component triage backlog item moved to `done` or to a follow-on plan with explicit owner.

## Revision log

- 2026-04-24: Plan created from existing OPEN-LOOPS WS2 section.
