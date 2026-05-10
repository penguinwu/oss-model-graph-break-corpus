# tools/audit_new_models.py — design (rev 2)

**Author:** Otter
**Date:** 2026-05-10 (rev 2: 15:00 ET — adversary-driven changes per case adv-2026-05-10-150000)
**Status:** Implementation in flight
**WS1 task:** "Build `tools/audit_new_models.py`"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 2b
**Adversary case file:** `subagents/adversary-review/invocations/adv-2026-05-10-150000-audit-new-models-design.md` (9 gaps; 8 addressed in rev 2, 1 deferred to skip_models schema upgrade)

---

## What it does

For NEW models in current cohort (not in baseline), propose tier classification + per-row triage. For REMOVED models, classify by source (intentional-skip / known-error-evolution / unexpected-removal).

Surfaces only — never auto-writes config files. Reviewer (Peng) approves each action.

## Inputs (rev 2)

1. `<sweep_dir>/identify_results.json` — effective rows (load via `sweep/results_loader.py`)
2. `<sweep_dir>/compare-vs-baseline.json` — needs cat 4 (NEW) + cat 5 (unexpected REMOVED) + `skip_listed` (intentional-skip REMOVED).
3. `<sweep_dir>/audit-new-errors.json` — sister output; for fixture-fix linkage on NEW-model error rows. Optional; degraded if absent.
4. `sweep/large_models.json` — current tier registry
5. `sweep/skip_models.json` — out-of-scope set
6. `sweep/known_errors.json` — for known-error-evolution REMOVED classification

## Tier classification (rev 2 — wall-clock only)

```python
WALL_LARGE_S = 60
WALL_VERY_LARGE_S = 300
```

Rule (assert wall_time_s is not None on non-error rows):
- `wall_time_s <= 60`  → regular (no entry needed)
- `60 < wall_time_s <= 300` → large
- `wall_time_s > 300` → very_large

GPU memory dropped from v1 per adversary gap #4 — every existing large_models.json entry uses wall-clock only. Threshold constants pinned via `test_threshold_constants_pinned`; future change requires test update + adversary-review.

## Per-NEW-model checks (rev 2)

cat4 entries are per-(name, mode). Per-model rollup:
1. Group cat4 by model name.
2. For each model name, take MAX(wall_time_s) across modes for tier classification.
3. Report cohort_delta with `new` (unique names) AND `new_pairs` (cat4 entry count).

Per model:
- All modes `full_graph` or `graph_break` → tier-classify only
- Any mode is an error row → fixture-fix needed; cross-reference `audit-new-errors.json` for triage_class
- Timeout in any mode → propose tier upgrade (very_large if `phase_at_timeout==create` else large)

If `audit-new-errors.json` absent: error-row entries get action "fixture-fix link unavailable — run audit_new_errors first" (graceful degradation).

## Per-REMOVED-model checks (rev 2)

Three sources of "removed":
1. `skip_listed` in compare json (only_in: `baseline`) — `intentional-skip` (in skip_models.json)
2. `cat5` AND name ∈ known_errors.json for active torch — `known-error-evolution`
3. `cat5` otherwise — `unexpected-removal` (transformers refactor)

Precedence: intentional-skip > known-error-evolution > unexpected-removal.

## skip_models.json proposals (rev 2 — surface only)

skip_models.json is a flat string array today (no metadata). The audit emits proposals as TEXT in the report ("propose adding ModelX to skip_models.json with reason: ... and follow-up task: ..."), but does NOT emit copy-pasteable JSON. Reviewer must hand-decide given the schema limitation. Schema upgrade + check_skip_models.py is a separate PLAN.md task.

## Output

### Markdown (`<sweep_dir>/audit-new-models.md`)
- Summary header (cohort delta, source breakdown)
- NEW models grouped by status bucket: compile-clean / compile-with-breaks / errored / timeout
- Proposed tier upgrades (per-name, with evidence)
- REMOVED models by classification
- Action checklist for Peng

### JSON (`<sweep_dir>/audit-new-models.json`)
```json
{
  "sweep_date": "...",
  "baseline_date": "...|null",
  "cohort_delta": {"new": N_unique_names, "new_pairs": N_cat4_pairs, "removed": N},
  "new_models": [{name, modes, max_wall_s, max_gpu_mem_mb, proposed_tier, has_error_row, fixture_fix_link}],
  "removed_models": [{name, classification, evidence}]
}
```

## CLI

```
python3 tools/audit_new_models.py <sweep_dir>
```

Exit codes (match audit_new_errors):
- 0 — report written
- 1 — input parse error
- 3 — `compare-vs-baseline.json` absent (degraded mode, partial report)

## Test plan (rev 2)

9 tests addressing all 9 adversary gaps:
1. test_cat5_skip_listed_path
2. test_cat5_known_error_baseline_only_model
3. test_cat4_dedupe_by_name_picks_max_wall
4. test_threshold_constants_pinned
5. test_boundary_wall_exactly_60s (and 300)
6. test_branch_precedence_error_with_high_wall
7. test_audit_runs_when_audit_new_errors_sidecar_absent
8. test_compare_absent_exit_code
9. test_thresholds_match_workflow_doc_string_match (deferred — workflow doc updated separately)

## Implementation scope

- `tools/audit_new_models.py`: ~200 lines
- `tools/test_audit_new_models.py`: ~250 lines
- One PLAN.md task: skip_models.json schema upgrade + check_skip_models.py (deferred from gap #7)

Adversary-review case `adv-2026-05-10-150000` covers this design + impl.
