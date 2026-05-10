---
case_id: adv-2026-05-10-150000-audit-new-models-design
subagent: adversary-review
date_utc: 2026-05-10T19:00:00Z
trigger: design-review
files: [sweep/AUDIT_NEW_MODELS_DESIGN.md, sweep/WEEKLY_SWEEP_WORKFLOW.md, sweep/SWEEP_COMPARE_WIRING_DESIGN.md, sweep/large_models.json, sweep/skip_models.json, tools/sweep_compare.py, tools/audit_new_errors.py]
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 9614991ea1821f235a7dfea44e23c3acb0e4d05886d1e6b6aba820e8e60c60ca
---

## Reviewer raw output

```
VERDICT: approve-with-gaps
CONFIDENCE: high

GAPS_FOUND:

1. [HIGH] REMOVED-model classification path is unreachable: tools/sweep_compare.py:186-202 builds cat4/cat5 with `if k[0] not in skip_models`. Skip-listed removed models go into `skip_listed`, not cat5. The "intentional-skip" branch on cat 5 will NEVER fire. The 25 removed all classified as skip_listed, not cat5. SUGGESTED_FIX: change input contract to also read `skip_listed` from compare-vs-baseline.json.

2. [HIGH] Third REMOVED path missing: model present in baseline as `skipped` due to known_errors.json, with no current row → lands in cat5 yet has neither intentional-skip nor refactor semantics. SUGGESTED_FIX: add `known-error-evolution` branch (precedence: known_errors check BEFORE unexpected-removal).

3. [HIGH] cat 4 dedupe-by-name missing: cat 4 is per-(name, mode); a model with eval+train shows as 2 entries. Without dedupe, audit proposes 2 large_models.json entries for the same name. SUGGESTED_FIX: per-model rollup, MAX(wall_time_s) across modes; cohort_delta.new = unique-name count + new_pairs separately.

4. [HIGH] GPU mem thresholds (5GB/15GB) have no precedent — every entry in current large_models.json uses wall_time_s only. The 5GB knee is folklore being locked under an adversary-review gate. SUGGESTED_FIX: drop GPU mem from v1 (pure wall-clock); add later with evidence.

5. [MEDIUM] Missing wall_time_s "skip + warn" is dead code — error rows go to Check 3 (fixture-fix), not Check 2. SUGGESTED_FIX: replace warn with assert wall_time_s is not None for non-error rows; fail loud.

6. [MEDIUM] Threshold-change discipline has no enforcement beyond prose. SUGGESTED_FIX: pin constants in test (test_thresholds_pinned).

7. [MEDIUM] skip_models.json schema is flat string array — can't represent TEMPORARY metadata. Proposing TEMPORARY entries is meaningless without schema change. SUGGESTED_FIX: out-of-scope skip_models.json edits in v1; surface as "human must decide, schema doesn't support TEMPORARY"; bundle schema upgrade as separate task.

8. [MEDIUM] "Link to audit_new_errors recommendation" unspecified: read sister sidecar or duplicate logic? SUGGESTED_FIX: declare audit-new-errors.json as Input #6, exit code 3 if absent (graceful degradation).

9. [LOW] Cross-tool exit-code gating not declared: should audit_new_models block on audit_new_errors failure? SUGGESTED_FIX: independent; both allow_fail=True in run_experiment.py.

SUGGESTED_TESTS: test_cat5_skip_listed_path, test_cat5_known_error_baseline_only_model, test_cat4_dedupe_by_name_picks_max_wall, test_threshold_constants_pinned, test_boundary_wall_exactly_60s, test_branch_precedence_error_with_high_wall, test_audit_runs_when_audit_new_errors_sidecar_absent, test_compare_absent_exit_code, test_thresholds_match_workflow_doc_string_match.

NOTES:
- Re-verify cat 4/5 semantics from sweep_compare.py:186-202 BEFORE implementation; skip_listed is the missing piece.
- Sister tool exit-code semantics: audit_new_errors returns 3 when compare absent — match exactly.
- Strongly prefer importing _FIXTURE_BUG_SUBSTRINGS from sister module rather than copy-paste.
```

## My disposition

| # | Severity | Disposition | Action |
|---|---|---|---|
| 1 | high | **Addressed** in rev 2 | Read `skip_listed` from compare-vs-baseline.json for intentional-skip classification; cat5 = unexpected-removal only |
| 2 | high | **Addressed** in rev 2 | Add `known-error-evolution` REMOVED-class; check known_errors.json BEFORE unexpected-removal |
| 3 | high | **Addressed** in rev 2 | Per-name rollup: dedupe cat4 by model name; MAX(wall_time_s) across modes; report cohort_delta.new (unique names) + new_pairs separately |
| 4 | high | **Addressed** in rev 2 | DROP GPU mem from tier classification in v1 — wall-clock only (matches all 37 existing large_models.json entries). Add later if evidence warrants. |
| 5 | medium | **Addressed** in rev 2 | Replace "warn" with `assert wall_time_s is not None` on non-error rows; fail-loud. |
| 6 | medium | **Addressed** in rev 2 | Pin via `test_threshold_constants_pinned` (literal value asserts). |
| 7 | medium | **Deferred** (new PLAN.md task) | skip_models.json schema upgrade is its own work. v1 of audit emits TEMPORARY proposals as surface-text-only ("human must decide; schema doesn't support TEMPORARY today"); does NOT propose JSON edits to skip_models.json. NEW PLAN.md task: schema upgrade + check_skip_models.py. |
| 8 | medium | **Addressed** in rev 2 | Declare audit-new-errors.json as Input #6; degraded-mode warning when absent; do NOT crash. |
| 9 | low | **Addressed** in rev 2 | Both audits independent; allow_fail=True in run_experiment.py per sister tool. |

All 9 SUGGESTED_TESTS implemented in `tools/test_audit_new_models.py`.
