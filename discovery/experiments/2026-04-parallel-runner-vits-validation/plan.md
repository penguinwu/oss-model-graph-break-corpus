---
plan: 2026-04-parallel-runner-vits-validation
status: active
owner: Otter
created: 2026-04-27
last_check: 2026-04-27
forcing_function: tools/check_plan.py + lifecycle gate
---

# Experiment Plan: Parallel Runner Validation on VITS V8

**Slug:** `2026-04-parallel-runner-vits-validation`
**Title:** Parallel Runner Validation on VITS V8
**Type:** Validation (testing the new infra, not learning about VITS)
**Owner:** Otter
**Workstream:** WS1 (skill eval) — infra validation
**Umbrella issue:** none (local validation; pass --with-umbrella-issue if findings warrant team-visible tracking)
**Status:** active

---

## Gate 0 — Frame

### Question

Does `discovery/launch_parallel.py` (Stage 1 parallel runner, commits 96a61b2 + 40dec90) produce VITS V8 trial results equivalent in shape to what the existing sequential `run_case.py` produces — when running with a real claude agent against the sandboxed transformers source?

### Done criteria

1. *Single VITS V8 trial via launch_parallel.py with a real agent run* (`--n 1 --variants V8 --skills none --max-parallel 1`):
   - Agent invokes successfully and edits files in the per-trial sandbox (NOT site-packages — verifiable via `git status` showing site-packages untouched, AND agent_diff.patch shows sandbox paths)
   - result.json populated with all schema fields (validation, perf, perf_tier2, fix_survives_perf, flags)
   - No mutation flags (`flags == []` or only expected `file-mutated:` entries)
   - Validator + perf both ran successfully against the sandboxed transformers (verifiable via gb_call_sites pointing at sandbox paths)
2. *2-trial parallel batch* (`--n 1 --variants V8 --skills none,/path/to/dgb/SKILL.md --max-parallel 2`):
   - Both trials complete independently (each with own sandbox)
   - Per-trial result.json schemas correct
   - merge_results.py produces summary.md + summary.json with both trials represented
   - No cross-trial contamination (sandbox isolation works under concurrency)
3. *Wall time*: 2-trial parallel ≤ 1.3x the wall time of one trial (i.e., parallelism saves >30%)

### Loose ends from prior runs (closure rule)

- *State-contamination root cause* — DEFERRED. Subprocess design sidesteps the in-process bug; that's the foundational fix for the parallel runner.
- *V8 deep-dive on clean data* — INCLUDED in done criteria #1 (a single-trial V8 result IS the start of the V8 deep-dive on clean infra)
- *Phase 2 corpus-wide determinism fix* — DEFERRED. VITS case file already uses HF set_seed (per validate_runner.py + smoke_test guard). Other cases unaffected.

No loose ends BLOCK this experiment.

### Infra changes since last similar run

The parallel runner IS the change. Already smoke-tested via:
- Synthetic 3-way parallel (commits 96a61b2 + 40dec90)
- VITS probe with `--skip-agent` (sandbox setup, validator inheritance, schema correctness — all verified)

This experiment is the FIRST end-to-end run with a real agent against VITS using the new runner.

### Estimated wall time

- Step 1: single-trial validation — agent timeout 1800s + setup + validate + perf ≈ 30-35 min
- Step 2: 2-trial parallel — same per trial; parallel = ~30-35 min wall (not 60-70)
- Inspection + diff vs sequential V8 trial = ~15 min

Total: ~80 min wall, ~30 min Otter active time (rest is agent runs in background).

---

## Pre-launch gates

- [x] Gate 0: question + done criteria + loose ends listed (above)
- [ ] Gate 1: smoke_test exit 0 (will re-run immediately before launch)
- [ ] Gate 2: single-trial validation — inspect result.json end-to-end; assert sandbox paths in agent_diff.patch + gb_call_sites
- [ ] Gate 3: 2-trial parallel — sandboxes isolated, no cross-trial contamination, schemas consistent
- [ ] All loose ends from prior runs CLOSED or DEFERRED with reason (see above)
- [ ] Estimated wall time documented (above)
- [ ] Gate 4 launch authorized (this experiment IS the Gate 4 — full-scale = 2 trials at 2-way parallel; sufficient for validation purposes)

---

## Implementation order

1. *Gate 1*: re-run `python -m discovery.smoke_test --skip-cases` immediately before launch
2. *Gate 2*: launch single trial:
   ```
   python -m discovery.launch_parallel \
       --case vits_model_train --variants V8 --skills none --n 1 \
       --experiment-dir /tmp/runs/parallel-vits-validation/step2 \
       --max-parallel 1 \
       --plan discovery/experiments/2026-04-parallel-runner-vits-validation/plan.md
   ```
   Inspect: result.json, agent_diff.patch (sandbox paths), validator stderr
3. *Gate 3*: launch 2-trial parallel (V8 noskill + dgb):
   ```
   python -m discovery.launch_parallel \
       --case vits_model_train --variants V8 \
       --skills none,/home/pengwu/projects/oss-model-graph-break-corpus/discovery/skills/debug-graph-breaks/SKILL.md \
       --n 1 \
       --experiment-dir /tmp/runs/parallel-vits-validation/step3 \
       --max-parallel 2 \
       --plan discovery/experiments/2026-04-parallel-runner-vits-validation/plan.md
   ```
   Inspect: 2 result.json files, sandboxes isolated, summary.md sane
4. *Findings*: if all gates pass, commit a brief writeup to this experiment dir noting:
   - Wall time observed (vs estimate)
   - Sandbox setup overhead (target ~0.5s)
   - Any surprises (Inductor cache contention? per-process startup?)
   - Direction for next step (full V8 6-trial batch via parallel runner; or address surprises first)

---

## What this experiment is NOT

- NOT a V8 deep-dive (that's its own follow-up — once the runner is validated, V8 batch becomes the natural first investigation)
- NOT a benchmark (we're confirming correctness + rough speedup, not characterizing across many models)
- NOT a sweep (single case, small N)

---

## Risk register (carried from design doc)

- *PYTHONPATH precedence subtlety* — verified clean for VITS in the probe; should hold for the agent run since the agent is a fresh subprocess inheriting our env
- *Per-process Python startup* — ~5-8s; visible at higher parallelism; not a concern at 1-2 trial scale
- *Inductor + Triton cache contention under concurrent compile* — UNMEASURED; will surface during Gate 3 if it matters

## Revision log

- *2026-04-27* — Plan created. Gate 0 ticked. Gates 1-4 unticked. Awaiting Peng greenlight to launch Gate 1 (smoke test).
