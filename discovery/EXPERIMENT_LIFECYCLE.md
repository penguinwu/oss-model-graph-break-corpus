# Discovery Experiment Lifecycle — 5 Gates

**Status:** mandatory for all discovery-harness experiments (multi-trial runs, harness changes, schema changes, case additions).
**Owner:** Otter
**Last revised:** 2026-04-27

## Why this exists

Without explicit gates, the natural failure mode is: launch a multi-hour run → bug surfaces → fix → re-launch → bug surfaces again. Each "I'll fix it on the next run" burns ~30 minutes of human attention and erodes trust in the data we ship. This doc forces the *bug-vs-cost asymmetry* to favor catching bugs early at low compute cost rather than late at high attention cost.

Concrete bug list this would have caught (2026-04-27 V8 work):
- `_eager_self_check` shape-mismatch crash → Gate 1 smoke (broken-path test)
- State contamination in repeated `_run_canonical_check` → Gate 3 small-batch reproducibility check
- Incomplete `set_seed` reseed (torch-only, missed numpy / python.random) → Gate 2 single-trial validation (eager_self_diff != 0 with np.random in forward)
- `_dynamo.explain` deprecated in corpus → Gate 0 forced "what's the corpus-canonical alternative?" check

## The 5 gates (non-skippable, in order)

### Gate 0 — Frame (no compute)

Write to the experiment's `plan.md` BEFORE touching code or compute:

- *Question:* one sentence — what does this experiment answer?
- *Done criteria:* specific observables that constitute a clean answer (e.g., "fix_status distribution across N trials, with at least one trial in each `general` / `setup-required` / `none` category")
- *Loose ends from prior runs:* either *closed* (with the verifying commit / observation) or *explicitly deferred* (with written reason). NO "I'll come back to this."
- *Infra changes since last similar run:* enumerate file changes; identify which gates are most at risk (e.g., new schema field → Gate 2 schema check; new validator behavior → Gate 1 smoke test addition)
- *Estimated total wall time:* sum of per-trial × trial count; if > 30 min, must complete Gates 1-3 first

### Gate 1 — Infra readiness (cheap, < 1 min)

- Run `python -m discovery.smoke_test`. Must exit 0.
- If a NEW component or behavior was added since last smoke run that isn't covered by an existing test, ADD A TEST for it FIRST. Both happy-path AND broken-path. Until that test exists, Gate 1 is failed.
- Verify all watched files are restored to `.original` (`ls discovery/cases/*.original 2>/dev/null` matches expected; or programmatic check).
- Verify `git status` clean for any harness file you've touched (don't leave uncommitted methodology changes during an experimental run).

### Gate 2 — Single-trial validation (~5 min)

- Launch ONE trial with the smallest valid configuration.
- Inspect EVERY field of `result.json`:
  - Schema: every expected field present?
  - Values: any unexpected `null` / `NaN` / placeholder / default-only values?
  - Cross-field consistency: e.g., `fix_status="general"` should imply `gb_in_agent_run=0`; `perf_shape_sanity="ok"` should imply `eager_ms` is a real number
- If ANY anomaly: STOP. Diagnose. Fix. Re-run single trial. Do not advance.
- Only proceed if single-trial result is end-to-end clean.

### Gate 3 — Small-scale validation (3 trials, ~15 min)

- Launch 3 same-shape trials.
- Confirm reproducibility: 3 trials with the same setup should produce similar (not necessarily bit-identical, but pattern-similar) results across schema fields.
- Confirm cross-trial state hygiene: trial 3 should not look qualitatively different from trials 1 + 2 in ways that have no causal explanation. If trial 3 differs from trials 1+2 *and* you can't name a cause, you have state contamination — STOP.
- Inspect aggregate signals (distributions, summaries): do they match what per-trial inspection would predict?
- Only proceed if reproducibility is clean.

### Gate 4 — Full-scale run

- Only after Gates 0-3 are all green AND closed.
- Mid-run: spot-check partial results every ~5 trials. If a pattern emerges that wasn't visible at small scale, halt and diagnose — don't let the full run finish on bad data.
- Post-run: aggregate findings; revisit your Gate 0 done criteria — were they actually answered, or did you drift?

## The closure rule (the one we keep violating)

> Before launching any new run, every loose end from the prior run is either CLOSED with verified fix or EXPLICITLY DEFERRED with a written reason in the experiment's `plan.md`. No silent drops. No "I'll come back to this." No "next run will tell us."

If a prior run surfaced 3 issues and you've fixed 2, the third must be either (a) closed before launch, (b) deferred with explicit "deferred because X, will revisit by Y", or (c) the run charter must be revised to acknowledge it operates on the un-fixed state.

## How to use this doc

For every discovery experiment:

1. **First touch** — read this doc top-to-bottom. Update `plan.md`'s top section with Gate 0 framing.
2. **Before any compute** — run through Gate 0 checklist; verify all loose ends closed/deferred.
3. **Before any launch** — run Gate 1 (smoke test) AND verify the gate it gates is appropriate.
4. **For multi-trial runs** — Gates 2 → 3 → 4 in strict order; cannot skip Gate 2 even if Gate 1 passed; cannot skip Gate 3 even if Gate 2 passed.

## Pre-launch checklist (copy into plan.md)

```
## Pre-launch gates — experiment <NAME>

- [ ] Gate 0: question + done criteria + loose ends listed
- [ ] Gate 1: smoke_test exit 0; new behaviors covered; watched files restored
- [ ] Gate 2: single-trial result.json fully inspected; no anomalies
- [ ] Gate 3: 3-trial small batch shows reproducibility; no cross-trial drift
- [ ] All loose ends from prior runs CLOSED or DEFERRED with reason
- [ ] Estimated wall time: __ min × __ trials = __ min total
- [ ] Gate 4 launch authorized

Gate 0 framing:
- Question: <one sentence>
- Done criteria: <observables>
- Loose ends from prior runs:
  - <item>: closed by <commit> / deferred because <reason>
- Infra changes since last similar run:
  - <change>: gates affected: G_, mitigation: <test added / behavior verified>
```

## When to revise this doc

- A bug surfaces in production that the current gates wouldn't have caught → ADD a check to the relevant gate
- A gate becomes consistently noisy / false-positive → tune it; don't drop it
- Discovery harness gets a new structural component → Gate 1's smoke test gains a corresponding test

Doc is in git so revisions are auditable. Version tag in revision log when changes are non-trivial.

## Revision log

- *2026-04-27 (v1.0)* — Initial draft after Peng called out the rush-to-launch anti-pattern + the V8 multi-bug-cascade morning. Gates 0-4 + closure rule + checklist template.
