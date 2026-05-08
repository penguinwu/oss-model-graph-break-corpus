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
- **A1.** No `module X has no attribute Y` create_errors (cohort drift / version contamination). Mechanically checked by `tools/check_cohort_invariants.py --post-sweep <results.json>`.
- **A2.** If `--models <cohort.json>`: cohort file has a `_metadata` block declaring `derived_from` + `filter` + `model_count` + `source_versions` (NOT "target_versions" — name was historically wrong; corrected 2026-05-07 per adversary-review case_id 2026-05-07-124100). Mechanically enforced at load time by `sweep/cohort_validator.py` (REJECTED unless `--allow-bare-cohort`).
- **A3.** Every cohort model exists in the declared source per the declared filter — no extras (cautionary: NGB verify 2026-05-06 had MiniCPMV4_6 + PPFormulaNet that weren't in the explain pass it was supposedly filtered from). Mechanically checked by `tools/check_cohort_invariants.py <cohort.json>` (pre-launch).
- **A4.** No cohort model is in `skip_models.json`. Mechanically checked by `tools/check_cohort_invariants.py <cohort.json>`.

### Status
- **C1.** No `create_error` (cohort / loader / network bug). Mechanically checked by `tools/check_cohort_invariants.py --post-sweep`.
- **C2.** No `eager_error` unless env-induced (env = CUDA OOM, cudnn, contention only; input-shape mismatch is harness, not env). Mechanically checked by `tools/check_cohort_invariants.py --post-sweep`.
- **C3.** No input-generation error (harness bug — STRICT in all contexts including expansion).
- **C4.** No generic `error` row that's actually a Python exception in harness code.

### Cross-run
- **E1.** No status regression on previously-clean models — `(name, mode)` clean in baseline → must be clean in this run.

### Verification (verify / correctness sweeps only)
- **D1.** Catastrophic divergences (`max_diff > 1e-3`): every one HALT-and-investigate. No threshold. Mechanically checked by `tools/check_cohort_invariants.py --post-sweep` (STRICT_FAIL severity).
- **D2.** Persistent divergences (`less_flaky=divergence`, `max_diff < 1e-3`): FLAG every model — usually low-magnitude real signal worth filing. Mechanically checked by `tools/check_cohort_invariants.py --post-sweep` (FLAG severity, threshold `1e-7 < max_diff <= 1e-3`).
- **D3.** Noise-floor divergences (`less_flaky=match`): expected; only FLAG if rate > 20%.

### Hygiene
- **G1.** Every non-success row is triaged — entry in `known_errors.json` (matching `error_pattern` + `applies_to_versions` covering active torch) OR in `skip_models.json`. Untriaged errors → HALT. (The harness's `--strict-known-errors` mode mechanically enforces this; sanity check sets the discipline of when to use it.) Also mechanically checked by `tools/check_cohort_invariants.py --post-sweep`.

### Provenance
- **SP1.** Results file's metadata header records the spec it ran from (`spec_path` + `spec_sha256` + run context, written as the FIRST line of `results.jsonl` by `tools/run_experiment.py run`); the spec file's current sha256 must match what was recorded at launch time. STRICT for any results file produced by the `run` subcommand on or after 2026-05-07; FLAG (older format) for older files. Mechanically checked by `tools/check_cohort_invariants.py --post-sweep`. Catches: results file mis-attributed to wrong spec, silent spec edits between launch and analysis, results file moved between dirs and provenance lost.

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
| 2026-05-07 | v3.1 (adversary-review case_id 2026-05-07-124100-cohort-regen-fix): A2 wording corrected (`target_versions` → `source_versions`; was a live drift between this skill text and the code's `_metadata.source_versions`). A1, A2, A3, A4 each cross-referenced to mechanical executors (`sweep/cohort_validator.py`, `tools/check_cohort_invariants.py`) so the markdown checklist is no longer the only enforcement. |
| 2026-05-07 | v3.2 (adversary-review case_id 2026-05-07-213400-doc-vs-impl): cross-referenced C1, C2, D1, D2, G1 to `tools/check_cohort_invariants.py --post-sweep` (the executor was checking these, but skill didn't say so). NEW: SP1 (Provenance) — checks the results file's metadata header against the spec it claims to be from (sha256 drift + missing-spec detection). SP1 was added to the executor 2026-05-07 21:24 ET per Peng directive ("In the result sanity check check-off list, we should add one more check against the experiment spec"); skill now records it as the canonical invariant definition. |
