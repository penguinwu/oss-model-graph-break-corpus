# tools/audit_new_errors.py — design

**Author:** Otter
**Date:** 2026-05-10
**Status:** DRAFT — awaiting Peng review
**WS1 task:** "Build `tools/audit_new_errors.py`"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 2a
**Depends on:** `compare-vs-baseline.json` (sweep_compare wiring design)

---

## What it does

Walks the current sweep's HF rows looking for **errors that are NEW this week** (not in baseline at the same status, not in `known_errors.json`, not in `skip_models.json`), classifies each via heuristic, and emits a markdown report + JSON sidecar with proposed triage decisions.

**Operationally** this replaces the manual one-off Python audit Otter has re-written each week.

## Per Peng directive (encoded in workflow doc)

- Bias INFRA-FIX over add-to-list. The known_errors / skip_models lists are escape hatches; default action is "fix root cause."
- create_errors are entirely infra's fault and MUST be fixed at root.
- The output is a TRIAGE PROPOSAL — Otter does not auto-act. Human reviewer approves each unknown / upstream-bug item before action.

## Inputs

1. `<sweep_dir>/compare-vs-baseline.json` (single source of truth, R1 — produced by sweep_compare wiring)
2. `<sweep_dir>/identify_results.json` + `identify_streaming.jsonl` (effective + raw rows; for error_message text)
3. `sweep/known_errors.json` (per-version applicability)
4. `sweep/skip_models.json` (out-of-scope models)
5. `<sweep_dir>/sweep_state.json` → `versions.torch` (for known_errors version filter)

No GitHub API calls. No re-running models. Pure analysis over local files.

## Candidate selection

A row `(name, mode)` is a "new error candidate" if ALL hold:

1. Status in current sweep ∈ `{eager_error, create_error}` (status taxonomy from `sweep_compare.py:55-57`)
2. NOT in `skip_models.json`
3. NOT in `known_errors.json` for the active torch version (per the `applies_to_versions` filter)
4. NOT present in baseline at the same status (i.e., `(name, mode)` is in cat 2 OR cat 4 OR cat 6-with-status-flip; NOT cat 6-stable)

The cat-based filter (#4) is the load-bearing one — stable failures (cat 6) are reported by sweep_compare itself, not re-surfaced here. We focus on what flipped.

## Triage classes (heuristic-driven)

The classifier reads `row.error_message` (truncated first line) + `row.status` + optional `row.fullgraph_error`. Each class has a substring-match rule + escalation action:

| Class | Match heuristic | Action proposed |
|---|---|---|
| `fixture-bug` | error contains `Audio must be mono` / `Image features and image tokens do not match` / `expected sequence of length` / known fixture-shape patterns | Fix `sweep/worker.py` input synthesis. PR-blocker — must land before brief. |
| `gpu-contention` | error contains `CUDA out of memory` AND model is NOT in `large_models.json` | Wait for auto-retry serial pass; if still OOMs, propose `large` or `very_large` tier add. |
| `cuda-context-pollution` | error contains `device-side assert` / `CUDA error` (non-OOM) | Wait for auto-retry; if still fails serially, propose upstream filing. |
| `subprocess-crash` | row.status = `worker_error` OR error contains `subprocess died` / `worker exited` / SIGSEGV | Wait for auto-retry. If RETRY_ELIGIBLE whitelist already covers this, no action. |
| `upstream-bug` | error matches a known upstream pattern OR no other class matched AND error mentions `transformers` / `torch.` API | Propose new GitHub issue via file-issue subagent (Step 2c). |
| `unknown` | nothing matched | Surface to human reviewer; default action is manual triage. |

**The heuristic table is hardcoded in the script with each entry tagged by the case_id that introduced it** (so when a future class needs a new rule, we add it with attribution). New rules require an adversary-review (heuristic logic = scoring logic).

## Re-run scope rule (load-bearing — workflow doc §"Re-run scope rule")

If any `fixture-bug` candidate gets a `worker.py` fix committed during Step 2a, the affected models' identify_results.json rows are STALE. The tool emits a warning at the bottom of the report:

```
WARNING: 3 fixture-bug candidates flagged. If any worker.py fix lands during this cycle,
re-run JUST the affected models BEFORE Step 2c:
  python3 tools/run_experiment.py sweep --models 'HiggsAudio Mistral3 ...' \
      --output-dir <sweep_dir> --resume
```

The warning is text-only; the actual re-run is the human reviewer's decision.

## Output format

### Markdown report (`<sweep_dir>/audit-new-errors.md`)

```
# New errors triage — sweep YYYY-MM-DD vs baseline YYYY-MM-DD

torch: <version>  transformers: <version>  diffusers: <version>

## Summary
- N total HF error rows in current sweep
- N covered by known_errors.json (skipped from this audit)
- N stable failures (cat 6 in baseline+current; reported by sweep_compare, not here)
- N candidates surfaced below

## Candidates (in priority order)

### fixture-bug (count) — PR-blocker, fix worker.py
| Model | Mode | Status | Error (first line) | Suggested fix |
|...|

### gpu-contention (count) — wait for auto-retry, then tier
| Model | Mode | First-seen-this-week? | Notes |
|...|

### upstream-bug (count) — file via file-issue subagent
| Model | Mode | Error | Search-existing-issues query proposed |
|...|

### unknown (count) — manual triage
| Model | Mode | Error | Notes |
|...|

## Re-run scope warning (if applicable)
[the warning block above]

## Action checklist for Peng review
- [ ] Approve fixture-bug fixes (link to suggested worker.py changes if Otter drafted any)
- [ ] Approve `upstream-bug` filings → invoke file-issue subagent for each
- [ ] Triage `unknown` candidates manually
```

### JSON sidecar (`<sweep_dir>/audit-new-errors.json`)

```json
{
  "sweep_date": "...",
  "baseline_date": "...",
  "torch_version": "...",
  "summary": {"total_errors": N, "in_known_errors": N, "stable_cat6": N, "candidates": N},
  "candidates": [
    {
      "name": "...", "mode": "...", "status": "...",
      "error_first_line": "...",
      "triage_class": "fixture-bug|gpu-contention|...",
      "suggested_action": "...",
      "first_seen_this_sweep": true,
      "baseline_status": "..."
    }
  ]
}
```

## Manual gates

| Gate | What requires human approval |
|---|---|
| `unknown` triage | Human reviewer reads each one |
| `upstream-bug` filing | Per file-issue Step 5 authority gate |
| Adding to `known_errors.json` | Discouraged (escape hatch); requires explicit reviewer approval — the audit tool refuses to write known_errors.json itself |
| Adding to `skip_models.json` | Same — tool emits proposal, doesn't auto-write |

## CLI

```
python3 tools/audit_new_errors.py <sweep_dir> [--baseline <baseline_dir>] [--out-dir <dir>]
```

Defaults:
- `--baseline`: auto-discover from `<sweep_dir>/compare-vs-baseline.json` (the JSON includes `baseline_dir` in metadata; sweep_compare wiring writes it)
- `--out-dir`: `<sweep_dir>` (writes `audit-new-errors.{md,json}` next to other sweep outputs)

Exit codes:
- 0: report written, candidates surfaced for review
- 1: no `compare-vs-baseline.json` found (run sweep_compare first)
- 2: input parse error

## Test plan

- Unit: pin the classification table — for each `triage_class`, a fixture row that matches the heuristic + an asserted classification result. Tests live in `tools/test_audit_new_errors.py`.
- Boundary: row that matches MULTIPLE heuristics → priority order (fixture-bug > gpu-contention > cuda-context > subprocess-crash > upstream-bug > unknown).
- Integration: run against `sweep_results/nightly/2026-05-09` (live data) — verify the 4 candidates I manually surfaced this morning (HiggsAudio, Mistral3, Blip2, Sam3) get classified correctly.
- Regression: any new heuristic rule must come with a fixture row in the test file.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Heuristic over-classifies as `fixture-bug`, triggers unnecessary re-run | Conservative substring rules; human reviewer approves before re-run |
| Heuristic mis-classifies an upstream-bug as `unknown`, suppressing surface | `unknown` is a surface, not a suppress — human reviewer SEES it |
| Tool keeps growing as new error patterns appear → drift | Heuristic table requires adversary-review for changes; entries tagged with case_id for attribution |
| `compare-vs-baseline.json` schema changes break consumer | Single-source-of-truth = single test pin in `tools/test_sweep_compare.py` (existing) catches schema drift |

## Implementation scope

- `tools/audit_new_errors.py`: ~250-350 lines (heuristic classifier + report + JSON)
- `tools/test_audit_new_errors.py`: ~150-200 lines (per-class fixtures)
- One-line addition to `tools/run_experiment.py nightly` Step 5d (chain after sweep_compare wiring): `audit-new-errors --sweep-dir <dir>` (allow_fail=True; surface only)
- Adversary-review on initial commit (heuristic table = scoring logic)

Estimated effort: 4-6 focused hours including test fixtures.
