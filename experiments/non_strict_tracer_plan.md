---
plan: WS3 — Non-strict tracer evaluation
status: active
owner: Otter
created: 2026-04-24
last_check: 2026-04-24
forcing_function: tools/check_plan.py + daily brief at 7:30 AM ET
---

# WS3 — Non-strict tracer evaluation (Edu)

## Goal

Compare dynamo-traced vs non-strict-traced model outputs against eager. Goal: surface where the new non-strict tracer in nightly PT diverges in correctness from dynamo, so Edu's team has empirical signal beyond synthetic tests.

## Open work

| Task | Type | Notes |
|------|------|-------|
| Find non-strict tracing entry point in nightly PT | backlog | First scoping step. Likely under `torch.export` or new `torch._tracing` namespace — needs verification. |
| Design dual-trace correctness comparison | backlog | Reuse Phase 3 correctness machinery (validate.py, max_abs_diff fields). Add a third backend column. |
| Wire dual-trace into nightly sweep | backlog | Nightly only (non-strict not in stable). Don't pollute baseline/stable sweeps. |

## Done means

- A nightly-only sweep mode that captures `eager`, `compile`, `non_strict_trace` outputs side-by-side.
- A report showing models where non-strict diverges from eager AND/OR diverges from compile.
- One-liner status surfaced in daily brief: "non-strict diverges on N models this week."

## Revision log

- 2026-04-24: Plan created from existing OPEN-LOOPS WS3 section.
