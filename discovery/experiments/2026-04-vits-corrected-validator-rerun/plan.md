---
plan: 2026-04-vits-corrected-validator-rerun
status: done
owner: Otter
created: 2026-04-28
last_check: 2026-04-28  # closed 17:05 ET — see Resolution below
forcing_function: tools/check_plan.py + lifecycle gate
---

> **Resolution (2026-04-28 17:05 ET).** All 15 corrected-validator trials complete.
> 3 of 15 (V4_1, V6_1, SKILL_V9_1 waveB) flagged filesystem-contaminated by
> post-hoc audit; re-launched in PARALLEL under chmod-RO + Layer A/B/C
> protections; all 3 came back clean. Findings folded at commit `739af75`.
> Headline finding strengthens: skill-trap reproducible at TWO constraint
> levels (V6+V9 noskill 3/3 general vs SKILL 0/3 general). The 3-layer
> contamination prevention (detection + chmod-RO + intent-check) shipped at
> commits `5ba1b80`, `dc8e1c1`, `bb6c854`, `ec92224`. Plan closed.

# Experiment Plan: VITS Skill Discovery Re-run with Corrected Validator + V9

**Slug:** `2026-04-vits-corrected-validator-rerun`
**Type:** Discovery (re-run with corrected infra to validate prior findings)
**Owner:** Otter
**Workstream:** WS1 — skill discovery via corpus
**Umbrella issue:** none (local re-run; results fold into the cross-case skill discovery report)
**Status:** active

---

## Gate 0 — Frame

### Question

Do the prior cross-case skill discovery findings on VitsModel (24-trial Apr 25 + 6-trial Apr 26 V8 batch) hold up under the corrected validator (HF `set_seed`, perf-shape sanity, canonical-input check)? AND: when we close the last setup-layer escape door (V9), does the agent reach a true model-layer fix, or does it expose an irreducible PyTorch limitation?

### Done criteria

1. *Cross-validator deltas surfaced.* For each variant V0/V2/V4/V6, compare prior `fix_status` (Apr 25-26) vs current `fix_status` (Apr 28). Identify any cells where the verdict flips (especially `general` → something else, indicating prior validator was over-permissive).
2. *V9 outcomes documented.* For 4 V9 trials (2 arms × 2 seeds), classify each into one of:
   - `general` — agent finds a true model-layer fix that survives canonical evaluation
   - `none` (with documentation) — agent identifies and documents the PyTorch limitation honestly
   - `none` (without documentation) — agent fails silently; this is signal that the door-closing prompt didn't land
3. *Report-ready dataset.* Combined dataset = 3 V8 trials (parallel-runner, Apr 27-28, corrected validator) + 12 new trials (this experiment) = 15 trials of corrected-validator data, plus the prior 30 trials as the baseline-comparison reference.

### Loose ends from prior runs (closure rule)

- *Prior 30-trial findings.md retraction* — explicit at top of `findings.md`. The L/M/S/R taxonomy and V8 "convergent failure" framing relied on a wrong noise-floor model. This re-run produces the corrected dataset for the rewrite.
- *Validator perf-shape sanity bug* — fixed (commit `f135b19`). Stays fixed.
- *_eager_self_check shape-mismatch crash* — fixed (commit `4538580`). Stays fixed.
- *_measure_case.py perf-infra bug* — fixed (commit `4538580`). Stays fixed.
- *HF set_seed in validator* — fixed (commit `0cd779c`). Stays fixed.

No loose ends BLOCK this experiment.

### Infra changes since last similar run

1. Validator switched to HF `set_seed` (commit `0cd779c`)
2. perf-shape sanity check + `runtime_failure` + `fix_survives_perf` (commit `f135b19`)
3. `_eager_self_check` crash fix on stochastic models (commit `4538580`)
4. Subprocess isolation via parallel runner (`launch_parallel`, commit `96a61b2`)
5. R5 — case body now tells the agent to self-verify against canonical via `validate.py` (this experiment)
6. V9 — new variant closing the declared-`_dynamo.config` door V8 left open (this experiment)

### Estimated wall time

- *Smoke (Phase A):* 2 trials @ 2-way parallel = ~35 min
- *Full batch (Phase B):* 10 remaining trials @ 4-way parallel = ~105 min (3 waves × 35 min)
- *Inspection + writeup (Phase D):* ~2h
- *Total:* ~4h wall, ~3h Otter active

---

## Pre-launch gates

- [x] Gate 0: question + done criteria + loose ends listed (above)
- [x] Gate 1: smoke_test exit 0 (`python -m discovery.smoke_test --skip-cases` — 9/9 passed 2026-04-28 06:28 ET)
- [x] Gate 2: single-trial validation — V8 step2/step3 trials from parallel-runner-vits-validation experiment (Apr 27 evening) ARE the single-trial validation for the corrected validator. Schemas correct, sandbox isolation verified.
- [x] Gate 3: 2-trial parallel — same experiment's step3 ran 2 V8 trials in parallel cleanly. Sandboxes isolated.
- [x] All loose ends from prior runs CLOSED or DEFERRED with reason (see above)
- [x] Estimated wall time documented (above)
- [x] Gate 4 launch authorized for Phase A (smoke); Phase B authorized only after smoke green and inspected

---

## Implementation order

### Phase A — Smoke validate (R5 + V9 changes)

```
python -m discovery.launch_parallel \
    --case vits_model_train --variants V0,V9 --skills none --n 1 \
    --experiment-dir /tmp/runs/vits-r5v9-smoke --max-parallel 2 \
    --plan discovery/experiments/2026-04-vits-corrected-validator-rerun/plan.md
```

Inspect: V0 noskill should produce a setup-required result (matches prior); V9 noskill should produce either a documented `none` (agent identifies the Inductor limitation) or — surprise — a true model-layer fix. Either way is signal.

### Phase B — Full batch (10 remaining trials)

After Phase A green-lights, launch in waves.

Wave 1 (4-way parallel): V0 SKILL, V2 SKILL, V2 noskill, V4 SKILL — ~35 min
Wave 2 (4-way parallel): V4 noskill, V6 SKILL, V6 noskill, V9 SKILL — ~35 min
Wave 3 (2 trials, 2-way): V9 noskill seed=2, V9 SKILL seed=2 — ~35 min

### Phase C — In-flight signal checks

- Cron-watchdog at +50 min from launch: verify wave-1 health (≥3 of 4 trials complete, all `import_ok`, no GPU hangs)
- Cron-watchdog at +1h35m: verify wave-2 health
- Cron-watchdog at +2h10m: verify wave-3 complete

### Phase D — Findings rewrite

Reuse skeleton at `discovery/experiments/2026-04-cross-case-skill-discovery/reports/vits_model_train/findings.md`. Drop retraction banner. Rewrite Phases B-F against the unified dataset (3 V8 from parallel-runner + 12 from this experiment + reference to prior 30 as the contrast). Keep Phase A (model-layer fix is convergent) — that finding holds across both batches.

---

## What this experiment is NOT

- NOT a from-scratch re-discovery — we're testing whether the prior findings hold under corrected infra
- NOT a benchmark — we want categorical truth (does V9 surface true model-layer fixes?), not perf characterization
- NOT a replacement for the original 30-trial dataset — that data stands; this experiment adds the corrected layer

---

## Risk register

- Inductor doesn't lower unbacked-symbolic conv1d. Known PyTorch limitation. V9 may produce 4/4 documented `none` if no model-layer fix exists. Acceptable — that IS the finding.
- Skill arm consistency. SKILL agents may behave differently from noskill on V9. Watch for divergence.
- Wall-time overrun. If GPU contention pushes 4-way parallel past 1.5x single-trial wall, drop to 2-way.

---

## Revision log

- 2026-04-28: created
