---
plan: 2026-04-parallel-runner
status: active
owner: Otter
created: 2026-04-27
last_check: 2026-04-27
forcing_function: tools/check_plan.py + lifecycle gate
---

# Experiment Plan: Parallel Discovery Runner — Stage 1 (subprocess per config)

**Slug:** `2026-04-parallel-runner`
**Type:** Infrastructure validation (not a discovery-on-a-case experiment)
**Owner:** Otter
**Workstream:** WS1 (skill eval) — infra readiness
**Umbrella issue:** [#69](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/69)
**Design doc:** `discovery/parallel-runner-design.md` (signed off by Peng 2026-04-27)
**Status:** active

---

## Gate 0 — Frame

### Question

Can we run M discovery configs as M independent subprocesses, each with its own per-trial transformers source clone (via PYTHONPATH), and produce results identical to the current sequential mechanism — at ~5-6x wall-time speedup at K=12?

### Done criteria

1. Single VITS config end-to-end through `run_config.py`: result.json schema-equivalent to `runner.py:run_trial` output (same fields, same value classes).
2. 3 parallel VITS configs (V8 × {dgb, noskill} × 1 each, plus 1 V0 noskill): each produces independent result.json with no cross-config contamination, no shared-file races, no schema regressions.
3. Wall-time speedup ≥ 2x at 3-way parallelism vs sequential (real measurement, not projection).
4. `discovery/smoke_test.py:test_parallel_runs_isolated` passes.

### Loose ends from prior runs (closure rule)

- *State-contamination root cause* — DEFERRED. The parallel design sidesteps in-process state contamination by giving each config its own subprocess; the in-process bug remains undiagnosed but unblocked. Filed in OPEN-LOOPS.
- *Phase 2 corpus-wide determinism fix* — DEFERRED. Tier-A discovery work (this experiment) does not gate on Phase 2 (which is mechanical propagation across cases). Phase 2 will pick up the corpus-canonical seeding pattern after this lands.
- *V8 deep-dive on clean data* — DEFERRED. The V8 v3 batch was killed in the spirit of methodology; will re-run via `launch_parallel.py` once it ships, getting clean data + parallel speedup in one pass.

No loose ends BLOCK this experiment.

### Infra changes since last similar run

This experiment IS the infra change. New components:
- `discovery/run_config.py` — single-config end-to-end (refactored from `runner.py:run_trial`)
- `discovery/launch_parallel.py` — parallel launcher with lifecycle gate
- `discovery/merge_results.py` — post-hoc cross-config aggregator
- Per-case `get_case_spec_sandboxed(sandbox)` method (start with VITS only)
- `discovery/smoke_test.py:test_parallel_runs_isolated` (synthetic regression test)

Gates 1-3 will catch any regression these introduce.

### Estimated wall time

- Gate 1 smoke: <1 min
- Gate 2 single-config validation: ~3-5 min (one VITS config end-to-end)
- Gate 3 small-batch validation: ~5-8 min (3 parallel VITS configs)
- Gate 4 full scale (deferred to next experiment): not part of this validation

---

## Pre-launch gates

- [ ] Gate 0: question + done criteria + loose ends listed (above)
- [ ] Gate 1: smoke_test exit 0; new behaviors covered; watched files restored
- [ ] Gate 2: single-config result.json fully inspected; no anomalies; schema matches sequential output
- [ ] Gate 3: 3-config small batch shows reproducibility; no cross-config drift; no race conditions
- [ ] All loose ends from prior runs CLOSED or DEFERRED with reason (see above)
- [ ] Estimated wall time documented (above)
- [ ] Gate 4 launch authorized (this experiment doesn't run a Gate 4 — that's the next experiment using this infra)

---

## Implementation order

1. Per-case `get_case_spec_sandboxed()` for VITS only — validate the sandbox path works
2. `discovery/run_config.py` — single-config end-to-end
3. Manual Gate 2 walk: invoke `run_config.py --case vits_model_train --variant V8 --skill none --trial-label test1 --out-dir /tmp/runs/parallel-test/cfg1/`
4. Inspect result.json: schema correct? Fields populated? Compare side-by-side with a recent sequential V8 trial.
5. `discovery/launch_parallel.py` (lifecycle gate + spawn loop + wait)
6. `discovery/merge_results.py` (post-hoc aggregator)
7. Manual Gate 3 walk: launch 3 parallel VITS configs, inspect for cross-config issues
8. Smoke test: `test_parallel_runs_isolated` against synthetic case (3-way)
9. Document: update design doc with measured speedup, any surprises discovered
10. Propagate `get_case_spec_sandboxed()` to other 4 cases (mistral3, dbrx, jamba, aria, paddle_ocr_vl) once VITS pattern is solid

---

## Risk register (flagged in the design doc)

- *PYTHONPATH precedence subtlety:* `run_config.py` must NOT import transformers in its own process; only its subprocesses do.
- *Cleanup discipline:* failed trials leave sandbox dirs (~50 MB each). Need `discovery/clean_sandboxes.py` housekeeping before this graduates from prototype to production.
- *Per-process startup overhead* (~5-8s) — could limit speedup at high parallelism. Measure during Gate 3.
- *Inductor + Triton cache contention* — unmeasured. Will surface during Gate 3 if it matters.

## Revision log

- *2026-04-27* — Plan created; Gate 0 framed. Awaiting Peng review then walk through gates.
