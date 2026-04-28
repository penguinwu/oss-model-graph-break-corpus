# discovery/experiments/

Per-experiment plans, per-case reports, and cross-case synthesis docs.

## Convention

Each experiment is a directory:

```
<YYYY-MM-descriptive-slug>/
├── plan.md              # methodology, matrix, questions, stop conditions
├── reports/
│   └── <case>.md        # per-case findings, one file per model
└── synthesis.md         # cross-case writeup, written when all cases close
```

Date prefix gives chronological order. Slug describes the path-finding goal — no opaque labels.

## Active experiments
| `2026-04-2026-04-vits-corrected-validator-rerun` | active | [plan.md](2026-04-2026-04-vits-corrected-validator-rerun/plan.md) | — |

| Slug | Status | Plan | Umbrella issue |
|---|---|---|---|
| `2026-04-cross-case-skill-discovery` | active | [plan.md](2026-04-cross-case-skill-discovery/plan.md) | #60 |
| `2026-04-parallel-runner-vits-validation` | active | [plan.md](2026-04-parallel-runner-vits-validation/plan.md) | — |

## Closed experiments

| Slug | Status | Plan | Notes |
|---|---|---|---|
| `2026-04-phase1-pilot-cases` | done (retroactive) | [plan.md](2026-04-phase1-pilot-cases/plan.md) | Pre-harness Jamba + Dbrx pilots; methodology superseded by Phase 2. |

## Why this directory exists separately from `discovery/cases/` and `discovery/runs/`

- `discovery/cases/` holds per-case **configuration** (the model, the inputs, the prompt) — reusable across experiments.
- `discovery/runs/` holds per-trial **raw artifacts** (`agent_diff.patch`, `result.json`, gitignored `stream.jsonl`) — produced by the harness, indexed by case + run-id.
- `discovery/experiments/` holds per-experiment **interpretation** — what did we set out to learn, what did we find, what does it mean. Multiple experiments can use the same case.
