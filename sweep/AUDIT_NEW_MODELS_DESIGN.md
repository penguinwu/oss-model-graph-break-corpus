# tools/audit_new_models.py — design

**Author:** Otter
**Date:** 2026-05-10
**Status:** DRAFT — awaiting Peng review
**WS1 task:** "Build `tools/audit_new_models.py`"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 2b
**Depends on:** `compare-vs-baseline.json` (sweep_compare wiring design)

---

## What it does

For each model in current cohort that was NOT in baseline cohort (cat 4 NEW), perform tier classification + per-row error triage. For each model REMOVED (cat 5), confirm intentional removal.

Emits a markdown report listing per-model decisions + proposed config-file edits (`large_models.json`, `skip_models.json`, suggested fixture-fix targets).

## Per Peng directive (encoded in workflow doc)

- create_errors on new models are entirely infra's fault and MUST be fixed before model is "ready" for the cohort. Tool surfaces these as fixture-fix candidates, not skip-list candidates.
- Per-REMOVED-model: confirm in `skip_models.json` (intentional). If not, investigate as transformers refactor.

## Inputs

1. `<sweep_dir>/compare-vs-baseline.json` — cat 4 (NEW) + cat 5 (REMOVED) lists
2. `<sweep_dir>/identify_streaming.jsonl` — for wall-clock + (when populated) GPU-mem fields per row
3. `sweep/large_models.json` — current tier registry
4. `sweep/skip_models.json` — out-of-scope list (validates REMOVED models)
5. Baseline `<baseline_dir>/identify_results.json` (only to confirm a "removed" model was indeed in baseline)

No model re-execution. Pure analysis over the sweep evidence already collected.

## Per-NEW-model checks (the tool does these in order)

### Check 1 — Status in identify

Bucket by `current_status`:
- `full_graph` → "compiles cleanly; tier-classify only"
- `graph_break` → "compiles with breaks; tier-classify only"
- `eager_error` / `create_error` → "infra fault, fixture fix needed; tier-classify deferred"
- `worker_error` / `timeout` → "needs auto-retry result; if still fails, tier-classify or skip"

### Check 2 — Tier classification (only for non-error rows)

Read the row's `wall_time_s` (already populated by `sweep/run_sweep.py`) and (when available) GPU mem from sweep evidence. Propose tier:

| Wall-clock | GPU mem | Proposed tier | Action |
|---|---|---|---|
| ≤ 60s | ≤ 5GB | `regular` (default) | No `large_models.json` entry needed. |
| > 60s OR > 5GB | — | `large` | Propose `large_models.json` entry with `timeout_tier: "large"`. |
| > 300s OR > 15GB | — | `very_large` | Propose `large_models.json` entry with `timeout_tier: "very_large"`. |

Thresholds match the workflow doc; the tool refuses to silently change them — any threshold change is a separate adversary-review-gated commit to the design doc + tool.

### Check 3 — Fixture-fix triage (only for error rows)

For `eager_error` / `create_error` on a NEW model, propose action:
- If error matches `audit_new_errors.py`'s `fixture-bug` heuristic: link to that audit's recommendation.
- Otherwise: surface as `unknown — manual triage required`.

For `worker_error` / `timeout` on a NEW model: cross-reference with `auto_retry_*_checkpoint.jsonl`. If retry succeeded, classify by retry-success status. If retry also failed, propose either (a) tier upgrade (timeout case) or (b) `skip_models.json` add with TEMPORARY label + follow-up task ref.

**Skip-list adds are TEMPORARY — must come with a follow-up fix task.** The tool emits each skip-list proposal with a templated PLAN.md task entry the human reviewer can copy in.

## Per-REMOVED-model checks

For each `(name, *)` in cat 5:
- If `name` ∈ `skip_models.json`: classification = `intentional-skip`. No action.
- If `name` ∉ `skip_models.json`: classification = `unexpected-removal`. Surface to human reviewer — likely a transformers refactor (model renamed, deprecated, or moved to a different module). Proposed action: investigate; either update the corpus model registry to track the new name OR document the deprecation.

## Output format

### Markdown report (`<sweep_dir>/audit-new-models.md`)

```
# New / removed models triage — sweep YYYY-MM-DD vs baseline YYYY-MM-DD

cohort delta: +N new (cat 4), -N removed (cat 5)

## NEW models (cat 4)

### Compile-clean (count) — tier-classify only
| Model | Mode | Wall (s) | GPU (MB) | Proposed tier | large_models.json edit |
|...|

### Compile-with-breaks (count) — tier-classify only
[same structure]

### Errored (count) — fixture-fix needed
| Model | Mode | Status | Error | Proposed action | Follow-up task |
|...|

## REMOVED models (cat 5)

### Intentional-skip (count) — verified in skip_models.json
- ModelA (reason: <skip_models.json reason field if present>)
...

### Unexpected-removal (count) — needs investigation
| Model | Last seen status (baseline) | Suggested investigation |
|...|

## Proposed config-file edits

### sweep/large_models.json (N additions)
[copy-pasteable JSON entries]

### sweep/skip_models.json (N additions, TEMPORARY)
[copy-pasteable JSON entries with REASON + follow-up task templates]

## Action checklist for Peng review
- [ ] Approve large_models.json edits (each entry: model, tier, evidence)
- [ ] Approve skip_models.json TEMPORARY entries + register follow-up tasks in PLAN.md
- [ ] Triage unexpected-removal cases (transformers refactor investigation)
```

### JSON sidecar (`<sweep_dir>/audit-new-models.json`)

```json
{
  "sweep_date": "...",
  "baseline_date": "...",
  "cohort_delta": {"new": N, "removed": N},
  "new_models": [
    {
      "name": "...", "mode": "...", "status": "...",
      "wall_time_s": N, "gpu_mb": N,
      "proposed_tier": "regular|large|very_large",
      "proposed_action": "...",
      "config_edit_proposal": {...}
    }
  ],
  "removed_models": [
    {"name": "...", "classification": "intentional-skip|unexpected-removal", "evidence": "..."}
  ]
}
```

## Manual gates

| Gate | What requires human approval |
|---|---|
| `large_models.json` edits | Reviewer approves each entry; tool emits proposal text but does NOT auto-write |
| `skip_models.json` TEMPORARY entries | Reviewer approves AND registers a follow-up task in PLAN.md |
| Unexpected-removal investigation | Reviewer assigns ownership |
| Fixture-fix for new-model errors | Standard `worker.py` commit path (with adversary-review per local CLAUDE.md trigger) |

## CLI

```
python3 tools/audit_new_models.py <sweep_dir> [--baseline <baseline_dir>] [--out-dir <dir>]
```

Defaults match `audit_new_errors.py`. Same exit-code semantics.

## Test plan

- Unit: pin tier classification — wall × gpu_mb cross-product fixtures with asserted tier output. Threshold values pinned (changing them requires changing the test).
- Boundary: wall-clock missing → tier classification skipped + warning row emitted (tool doesn't silently default to `regular`).
- Boundary: GPU mem field missing → wall-clock alone determines tier (acceptable degradation).
- Integration: run against `sweep_results/nightly/2026-05-09` (live data) — verify the 25 removed models all classify as `intentional-skip` (matches manual check in PLAN.md WS2).
- Cohort-delta sanity: `len(new_models) + len(removed_models) == cat4_count + cat5_count` from `compare-vs-baseline.json`.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Tier thresholds drift (set once, never re-tuned) | Adversary-review required to change thresholds; design doc references the source-of-truth values |
| `skip_models.json` becomes long-term dumping ground | TEMPORARY label + follow-up task template baked into proposal text; quarterly known-errors-rot audit (workflow doc § "Known-errors rot audit") catches it |
| Unexpected-removal classification flags real refactors as "lost" | Reviewer judges; tool surfaces, doesn't decide |
| GPU mem field absent for old sweep rows | Wall-clock alone is enough for tier in 95% of cases (sweep evidence shows wall_time_s is always populated); fall through gracefully |

## Implementation scope

- `tools/audit_new_models.py`: ~200-300 lines
- `tools/test_audit_new_models.py`: ~100-150 lines
- One-line addition to `tools/run_experiment.py nightly` Step 5e (chain after audit_new_errors)
- Adversary-review on initial commit (tier-threshold + classification logic = scoring)

Estimated effort: 3-5 focused hours.
