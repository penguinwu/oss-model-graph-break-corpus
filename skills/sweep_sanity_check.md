# Sweep Sanity Check

Living checklist of what "looks right" for a sweep. Walk it at three points: pre-launch sample, mid-sweep peek, post-completion. Every time a new failure mode slips past, an invariant is added.

**STRICT** failures HALT. **FLAG** failures get noted, may proceed.

## When to apply

**Pre-launch sample (~10-15 min on 20 random models):** apply to the sample's output before launching the full sweep. Any STRICT failure → halt the full launch and fix the cohort generator / harness. Do not exclude individual models and proceed; that lets the bug ship.

**Mid-sweep peek:** apply the status invariants (C-family) on whatever results are in. Skip count/cross-run checks. STRICT failures stop the sweep immediately.

**Post-completion (mandatory before any analysis or issue-filing):** apply everything. Classify each FAIL as `accepted` (with reason in the run's plan.md) or `blocking`. Untriaged errors (G1) are always blocking.

## Invariants

Default is STRICT for curated cohorts (verify, correctness, explain follow-ups, hand-built model lists). For full-corpus identify sweeps, A1-A3 and C1-C2 are LENIENT (some error at corpus scale is normal). For cohort-expansion runs, see §Cohort expansion.

### Cohort
- **A1.** No `module X has no attribute Y` create_errors (cohort drift / version contamination)
- **A2.** If `--models <flat.json>`: cohort file has a `_metadata` block declaring `source_run` + `filter` + `target_versions`
- **A3.** Every cohort model exists in the declared source per the declared filter — no extras (cautionary: NGB verify 2026-05-06 had MiniCPMV4_6 + PPFormulaNet that weren't in the explain pass it was supposedly filtered from)
- **A4.** No cohort model is in `skip_models.json`

### Status
- **C1.** No `create_error` (cohort / loader / network bug)
- **C2.** No `eager_error` unless env-induced (env = CUDA OOM, cudnn, contention only; input-shape mismatch is harness, not env)
- **C3.** No input-generation error (harness bug — STRICT in all contexts including expansion)
- **C4.** No generic `error` row that's actually a Python exception in harness code

### Cross-run
- **E1.** No status regression on previously-clean models — `(name, mode)` clean in baseline → must be clean in this run

### Verification (verify / correctness sweeps only)
- **D1.** Catastrophic divergences (`max_diff > 1e-3`): every one HALT-and-investigate. No threshold.
- **D2.** Persistent divergences (`less_flaky=divergence`, `max_diff < 1e-3`): FLAG every model — usually low-magnitude real signal worth filing
- **D3.** Noise-floor divergences (`less_flaky=match`): expected; only FLAG if rate > 20%

### Hygiene
- **G1.** Every non-success row is triaged — entry in `known_errors.json` (matching `error_pattern` + `applies_to_versions` covering active torch) OR in `skip_models.json`. Untriaged errors → HALT. (The harness's `--strict-known-errors` mode mechanically enforces this; sanity check sets the discipline of when to use it.)

### Execution
- **B1.** Result count = `len(cohort) × len(modes)` — no silent drops
- **B2.** eval and train counts within 1
- **B3.** worker_error + zombie + timeout combined < 0.5%

## Cohort expansion runs

Triggered by: bumping a library version, installing a new library, importing new corpus sources.

The expansion run is a discovery pass — it exists to surface what doesn't yet work. Two rules:

1. **Newly-added models** (not present in the prior corpus): errors are EXPECTED. The deliverable is the triage — every error gets fixed, added to `known_errors.json` with a reason, or `skip_models.json` with a reason. The run is "done" when a fresh 20-random sample shows zero untriaged errors, not when the sweep process exits.

2. **Pre-existing models** (present in the prior corpus before the expansion): all invariants apply STRICTLY. These models worked before; if they regress under the new library version, that is a real bug. Fix or file an upstream issue. Do NOT triage them into `known_errors.json` — that hides regressions.

## Revision log

| Date | Change |
|---|---|
| 2026-05-07 | Initial v3 — simplified from v2.1 (~80 lines vs 286). Cut: per-invariant 5-field blocks, four named apply contexts, sweep-type matrix, lifecycle hygiene family (operational, belongs in sweep.md), D1/D2 sub-flavors of expansion, G2/G3 (belong in known_errors.json `_doc` and a weekly hygiene routine respectively), triage-authority and wiring sections. Kept: the invariants themselves, the cohort-expansion bump-vs-new rule, the triage discipline. |
