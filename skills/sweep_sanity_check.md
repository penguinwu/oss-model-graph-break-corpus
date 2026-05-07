# Sweep Sanity Check

Living checklist of invariants we expect of every sweep, and the apply contexts that say which invariants are STRICT/LENIENT/EXEMPT for which sweep types.

This skill is the basic guardrail on sweep correctness. Every time a sweep surfaces a new failure mode this skill missed, an invariant is added. The list is intended to grow monotonically.

**Read this before:** launching a full sweep, inspecting an in-flight sweep's results, or declaring a completed sweep "done."

---

## Apply contexts

A sanity check is applied in one of four contexts. Each context narrows which invariants matter and what severity their failures carry.

### APPLY-A — Pre-launch sample sweep
Random-sample 20 models from the planned full cohort (uniform random by default; stratified by source if cohort is heterogeneous). Launch a sample sweep with **identical flags** to the planned full launch — same venv, modellibs, workers, timeouts, compile-kwargs, dynamo-config. Apply the sweep-type's STRICT subset of invariants. Any STRICT failure → HALT full launch and investigate. Skipping requires Peng's written approval.

The sample sweep is also a good place to detect cohort-formation bugs that pre-launch reasoning doesn't catch, because it exercises the actual cohort with the actual flags end-to-end.

### APPLY-B — Mid-sweep peek
On-demand or watchdog-triggered peek of a partial sweep. Apply only invariants that work on partial data (mostly C-family status checks; skip count-based and cross-run). HALT failures STOP the sweep immediately; FLAG failures get noted and the sweep continues.

### APPLY-C — Post-completion review
Mandatory before declaring sweep results valid. Apply the full STRICT subset for the sweep type. Every FAIL must be classified as `accepted` (with reason recorded in the run's plan.md) or `blocking` (must fix before next sweep). No publishing, no issue-filing, and no analysis until blocking fails are resolved.

### APPLY-D — Cohort expansion run
Triggered by adding new models, bumping a model library version, installing a new library for the first time, or importing a new corpus source. The expansion run is a discovery pass — its purpose is to surface what doesn't yet work. **NEW** errors are expected on **NEW** models; pre-existing models that regress are real bugs (see two sub-flavors below).

Two flavors:
- **D1 — Brand-new library or source.** All models are new; errors expected anywhere. INV-G1 is EXEMPT during the run; STRICT on completion (all errors must be triaged before the run is considered done).
- **D2 — Library version bump.** Only the delta (newly added models, by name) is in discovery mode. Pre-existing models that worked under the old version MUST still work under the new version. Any regression on a pre-existing model is a real bug and must be fixed or filed, NOT triaged into known_errors.

In both flavors, the deliverable of a cohort-expansion run is:
1. updated `sweep/known_errors.json` and/or `sweep/skip_models.json` with full triage entries for every newly-discovered acceptable error, AND
2. filed issues / fixes for every regression on pre-existing models, AND
3. a re-run sample (fresh 20-random) showing 0 untriaged errors

The run isn't done when the sweep finishes. It's done when the triage closes G1.

---

## Invariant families

### A — Cohort integrity

```
INV-A1: Every cohort model is reachable in target stack
What:    No model in the cohort triggers `module X has no attribute Y` at create.
Applies: Curated cohorts on a pinned version stack.
Failure: HALT — likely contamination (model removed in target version) or
         unhandled rename. Investigate cohort generator.

INV-A2: Cohort declares its provenance
What:    Cohort file has a _metadata block recording: source_run path, filter
         expression, target_versions (torch/transformers/diffusers it was
         filtered FOR), generated_at, generated_by.
Applies: Any sweep using --models <flat.json>.
Failure: HALT — flat-list cohort with no provenance is opaque; can't detect
         version mismatch or contamination at launch.

INV-A3: Cohort matches its declared source
What:    Every model in the cohort exists in the source_run's results per the
         declared filter. No "extras" added.
Applies: Curated cohorts derived from prior sweeps.
Failure: HALT — investigate cohort generator. Cautionary tale: NGB verify
         2026-05-06 had MiniCPMV4_6 + PPFormulaNet in the cohort even though
         these weren't in the explain results the cohort was supposedly
         filtered from.

INV-A4: Cohort excludes skip-listed models
What:    No model in the cohort is in skip_models.json.
Applies: Curated cohorts.
Failure: HALT — cohort generator is broken; re-run cohort generation with
         skip-list applied.
```

### B — Harness execution

```
INV-B1: Result count matches expected work-item count
What:    len(results) == len(cohort) * len(modes), no silent drops.
Applies: All sweeps post-completion.
Failure: HALT — investigate worker reaper, checkpoint resume, or output writer.

INV-B2: Mode-balanced
What:    eval count ≈ train count (within 1) when both modes ran.
Applies: Both-modes sweeps.
Failure: FLAG — usually indicates one mode silently dropped some models.

INV-B3: No worker_error / zombie above 0.5%
What:    Worker process crashes or zombie reapers under threshold.
Applies: All sweeps.
Failure: FLAG — may be transient cudnn/CUDA contention; if reproducible,
         HALT and investigate worker pool.

INV-B4: sweep_state finished cleanly
What:    sweep_state.json status is "done", phase is "report", no orphan PIDs
         match the run-name pattern.
Applies: Post-completion.
Failure: HALT — sweep didn't finish or watchdog left something running.
```

### C — Status integrity

```
INV-C1: No create_error
What:    Sum of status=create_error across all results is 0.
Applies: STRICT for curated cohorts (verify, correctness, explain follow-ups).
         LENIENT for full corpus identify (some create_error is normal at
         corpus scale due to broken model configs).
         EXEMPT during cohort-expansion runs; STRICT on expansion completion
         (every create_error must be triaged into known_errors.json).
Failure: HALT — investigate root cause: cohort contamination, version
         mismatch, network/Hub access, or loader regression.

INV-C2: No eager_error unless env-induced
What:    Sum of status=eager_error across all results is 0.
         Env-induced is narrowly defined: CUDA OOM, cudnn handle, GPU
         contention. Anything else (input shape mismatch, missing config
         field, model-internal bug) is a harness bug, not env.
Applies: STRICT for curated cohorts. LENIENT for full corpus identify.
         EXEMPT during expansion runs; STRICT on expansion completion.
Failure: HALT — investigate input-gen, model spec, or harness loader.

INV-C3: No input-generation error
What:    No result with error_type matching input shape mismatch, type
         mismatch, missing key in inputs dict, or similar input-gen failure.
         (Sub-class of C2 but called out separately because input-gen
         failures are always harness bugs, never model or env.)
Applies: STRICT in all contexts including expansion.
Failure: HALT — input-gen harness bug; fix in models.py or worker.py.

INV-C4: No `error` status that's actually a harness exception
What:    Generic `error` status rows are real dynamo / compile errors
         (real signal), not Python exceptions in harness code.
Applies: All sweeps.
Failure: HALT — harness exception masquerading as model error.
```

### D — Verification (numeric)

```
INV-D1: Tier-1 noise-floor divergence rate ≤ 20%
What:    success rows with numeric_status=divergence AND noise_floor_dominant
         AND less_flaky_status=match. Informational ceiling.
Applies: Verify / correctness sweeps.
Failure: FLAG — high counts may indicate baseline tolerance drift.

INV-D2: Tier-2 persistent divergence rate ≤ 5%
What:    success rows with numeric_status=divergence AND less_flaky_status=
         divergence AND max_diff < 1e-3. Real signal at low magnitude.
Applies: Verify / correctness sweeps.
Failure: FLAG every model in this band — these are candidates for issue
         filing or further investigation, but rarely showstoppers.

INV-D3: Catastrophic divergences (max_diff > 1e-3) — flag every one
What:    success rows with numeric_status=divergence AND max_diff > 1e-3.
         No threshold; even one is news.
Applies: Verify / correctness sweeps.
Failure: HALT-and-investigate every single one. These are the headlines.
```

### E — Cross-run consistency

```
INV-E1: No status regressions on previously-clean models
What:    For every (name, mode) where source baseline had status in
         {ok, success, graph_break}, current run must also have status in
         the same set. Any clean → error transition is a regression.
Applies: Verify / explain follow-up sweeps with a known baseline.
Failure: HALT — investigate regression. The model worked before; something
         in this run broke it. Likely root causes: harness change since
         baseline, library version drift, environment change.

INV-E2: GB characteristics preserved on common models
What:    For (name, mode) ok-in-both, graph_break_count should not shift
         by more than 1 between runs (when both runs measure it).
Applies: Explain-or-verify on the same cohort + stack as a prior run.
Failure: FLAG — differences worth examining; could be real (NGB toggle
         changing capture) or noise.
```

### F — Lifecycle hygiene

```
INV-F1: Watchdog cron removed after sweep done
What:    crontab has no entry referencing this run's output dir after
         status=done.
Applies: Post-completion.
Failure: FLAG — clutter, will fire on stale state until removed.

INV-F2: No stale INTERRUPTED marker
What:    Output dir has no INTERRUPTED marker file after a successful
         status=done.
Applies: Post-completion.
Failure: FLAG — confusing for future readers; remove or convert to a
         note about prior interruption + recovery.

INV-F3: Auto-retry checkpoints empty or processed
What:    auto_retry_*_checkpoint.jsonl files are either empty or every
         entry has been re-run.
Applies: Post-completion.
Failure: FLAG — pending retries left undone.
```

### G — Error triage hygiene

```
INV-G1: Every non-success row is triaged
What:    For every result with status in {create_error, eager_error, error,
         worker_error, timeout}, the (name, mode) is recorded in EITHER
         known_errors.json (with a matching error_pattern + applies_to_versions
         covering the active torch) OR skip_models.json with a reason.
         The harness already validates this when run with --strict-known-errors.
Applies: STRICT for all sweep types EXCEPT cohort-expansion runs (where
         untriaged errors are the discovery output, not a failure).
Failure: HALT post-completion review. Untriaged errors mean either:
         (a) a new failure mode exists and a cohort-expansion sub-task is
             needed to process it, or
         (b) a fix regressed and silently re-introduced an error.
         Either way: investigate before publishing, filing issues, or
         running the next sweep.

INV-G2: Triage entries are complete
What:    Every entry in known_errors.json has at minimum: status, model,
         modes, error_pattern, applies_to_versions, added (date), reason.
         (skip_models.json entries should grow a reason field too — see
         OPEN-LOOPS for the schema-tightening task.)
Applies: known_errors.json maintenance — checked at the time of any add.
Failure: Reject the entry; cannot add until complete.

INV-G3: Triage entries are periodically revisited
What:    Once per N sweeps (or weekly), every known_errors entry is
         re-evaluated. If the model now passes in a recent sweep, remove
         the entry. If it still fails, check the issue link for movement.
Applies: Background hygiene — not blocking any single sweep.
Failure: FLAG — informational; prompts maintenance sweep.
```

---

## Sweep-type matrix

| Invariant | Verify / correctness | Full corpus identify | Cohort expansion (D1: brand-new) | Cohort expansion (D2: lib bump) |
|---|---|---|---|---|
| A1-A4 cohort integrity | STRICT | LENIENT | LENIENT (cohort being defined) | STRICT for pre-existing subset |
| B1-B4 harness execution | STRICT | STRICT | STRICT | STRICT |
| C1 No create_error | STRICT | LENIENT | EXEMPT during run | STRICT on pre-existing; EXEMPT on new |
| C2 No eager_error | STRICT | LENIENT | EXEMPT during run | STRICT on pre-existing; EXEMPT on new |
| C3 No input-gen error | STRICT | STRICT | STRICT | STRICT |
| C4 No harness-exception masked as error | STRICT | STRICT | STRICT | STRICT |
| D1-D3 verification | STRICT | N/A | N/A | N/A |
| E1 No status regression | STRICT | N/A | N/A | **STRICT — pre-existing models that regress are real bugs, not triage-able** |
| E2 GB characteristics preserved | FLAG | N/A | N/A | FLAG |
| F1-F3 lifecycle hygiene | FLAG | FLAG | FLAG | FLAG |
| G1 All errors triaged | STRICT | STRICT | EXEMPT during; STRICT on completion | EXEMPT during for new models; STRICT for pre-existing always |
| G2 Triage entries complete | STRICT | STRICT | STRICT | STRICT |
| G3 Triage entries revisited | FLAG | FLAG | FLAG | FLAG |

---

## Triage authority

When a non-success row needs triaging:

- **Obvious cases** — model was removed in the target library version (`module transformers has no attribute X`), or model is a known-flaky on this hardware. Agent updates `known_errors.json` directly with `added_by: <agent_name>, run <run_name>`; reports to Peng in the next reply.
- **Ambiguous cases** — worker crash, custom-model loader regression, novel error pattern. Agent proposes the triage classification (FIX / KNOWN-ERROR / SKIP) with reasoning; Peng approves the registry add.
- **Library-bump regression on pre-existing model** — never triaged. Always FIX (file a corpus issue or harness PR). The whole point of the bump-protection rule is that pre-existing models cannot quietly degrade across versions.

---

## Wiring

This skill is invoked by `skills/sweep.md`:
- **Pre-flight gate** (mandatory): APPLY-A sample sweep
- **Mid-sweep watchdog**: APPLY-B if anything unexpected appears
- **Post-completion** (Step 0 of §8 Due Diligence): APPLY-C, before any analysis
- **Whenever bumping a library or adding a corpus source**: APPLY-D

---

## Revision Log

| Date | Change | Source |
|------|--------|--------|
| 2026-05-07 | Initial v2.1 — four apply contexts, seven invariant families A-G, sweep-type matrix, library-bump distinction (D1 vs D2). | NGB verify run 2026-05-06 surfaced 22 create_error + 4 eager_error that prior `--strict-known-errors` validation alone didn't catch as a process gap; Peng's directive to encode the discipline as a skill, not a script. |
