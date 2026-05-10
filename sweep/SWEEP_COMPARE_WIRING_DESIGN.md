# sweep_compare wiring into nightly pipeline — design

**Author:** Otter
**Date:** 2026-05-10
**Status:** DRAFT — awaiting Peng review
**WS1 task:** "Wire `tools/sweep_compare.py` into nightly pipeline"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 1 → Step 2 bridge

---

## What changes

After explain pass + `sweep-report` + `close-stale` complete in `tools/run_experiment.py nightly`, automatically invoke `tools/sweep_compare.py` against the prior week's nightly results. Output lands at `<sweep_dir>/compare-vs-baseline.{md,json}` and is the canonical input for Steps 2a / 2b / 2d.

Today: `sweep_compare.py` is invoked manually after each sweep; Steps 2a/2b/2d each have to either re-invoke it or fall back to ad-hoc comparison. That ad-hoc fallback is what the methodology.md R1 rule was written to prevent.

## Why now

R1 ("single source of truth: `tools/sweep_compare.py`") is encoded in methodology.md but enforced only by discipline. The reliable way to make R1 hold is to pre-compute the comparison once per sweep so downstream tools (audit_new_errors, audit_new_models, compose_brief) have one obvious input file and no excuse to roll their own.

## Mechanism

### Step in `run_experiment.py nightly`

After the existing `close-stale` dry-run step (around `tools/run_experiment.py:1711`), add Step 5c:

```python
# Step 5c: sweep-compare vs prior week's baseline
baseline_dir = _find_prior_baseline(output_dir)  # see _find_prior_baseline below
if baseline_dir:
    cmd = [python, str(tools_dir / "sweep_compare.py"),
           "--baseline", str(baseline_dir),
           "--current",  str(output_dir),
           "--out",      str(output_dir / "compare-vs-baseline.md"),
           "--json",     str(output_dir / "compare-vs-baseline.json"),
           "--verbose"]
    _run(cmd, "sweep_compare vs prior baseline", allow_fail=True)
```

Use `allow_fail=True` because invariant failures (exit 1) and explain-coverage gaps (exit 2) should not abort the whole nightly — they're the very signal we want surfaced to the human reviewer (and the watchdog completion message can include "compare failed" in that case).

### `_find_prior_baseline(current_dir)`

Walk `<current_dir>/..` for the dated directory immediately preceding `current_dir.name` (lexicographic sort on `YYYY-MM-DD`). Skip:
- `current_dir` itself
- Directories without `identify_results.json` (incomplete sweeps)
- Directories whose `sweep_state.json` shows `status != "done"`

If no qualifying baseline found, return `None`. The pipeline step is a no-op (logged as "no baseline; skipped"); not an error.

Edge case: if the prior baseline is >2 weeks old (e.g. weekly sweeps were skipped), still use it but the watchdog completion message flags "baseline is N days stale."

### Output file convention

- `compare-vs-baseline.md` — human-readable report (the existing markdown sweep_compare emits, no template change)
- `compare-vs-baseline.json` — machine-readable partition (`cats_to_json` output, already implemented at `tools/sweep_compare.py:674`)

Downstream tools key off `<sweep_dir>/compare-vs-baseline.json` as their single input.

### Watchdog completion message (small extension)

Currently the watchdog cycle script's completion-detected path (`sweep/sweep_watchdog_cycle.sh:48-69`, just shipped) reports identify rows + explain size + paths. Extend to include sweep_compare result if `compare-vs-baseline.json` exists at completion time:
- exit_0: "compare: cat1=N cat2=N cat3=N cat4=N cat5=N cat6=N (baseline=<date>)"
- exit_1 / exit_2: "compare: INVARIANT FAILURE — review compare-vs-baseline.md"
- absent: silently omit

This makes the completion notification the one-look summary of what changed.

## What this UNBLOCKS

Steps 2a / 2b / 2d can now assume `<sweep_dir>/compare-vs-baseline.json` exists and is the source of truth. Their tool implementations (audit_new_errors / audit_new_models / compose_brief) consume that JSON's pre-computed cat 1-6 partition rather than re-implementing the comparison.

## Test plan

- Unit: `tools/test_run_experiment_corpus_filter.py` already pins the nightly entry point structure; add a small pin that verifies the `sweep_compare` step runs after `close-stale` AND that absent-baseline returns no-op (not error).
- Integration (one-shot, against current sweep):
  ```
  python3 tools/run_experiment.py nightly --resume --output-dir sweep_results/nightly/2026-05-09
  ```
  Verify `sweep_results/nightly/2026-05-09/compare-vs-baseline.{md,json}` exist and the JSON's cat counts match `python3 tools/sweep_compare.py --baseline <prior> --current 2026-05-09 --check-only` output.
- Failure mode: rename baseline's `identify_results.json` → verify the step emits "no baseline; skipped" and doesn't fail the nightly.

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| Baseline disappears between sweeps (manual cleanup) | `_find_prior_baseline` walks until first qualifying dir, not just N-1 |
| `sweep_compare.py` exit 1/2 (invariant failure) aborts nightly | `allow_fail=True` + watchdog flags it in completion message |
| Stale baseline (> 2 weeks) gives misleading "regressions" | Watchdog flags age in completion message; brief composition tool will refuse to compose without an in-window baseline (compose_brief.py design covers this) |

## Rollback

Single point of insertion in `run_experiment.py`. Revert the diff that adds Step 5c. Downstream tools that consume `compare-vs-baseline.json` would then need a manual step before each weekly cycle (matches today's state).

## Implementation scope

- `tools/run_experiment.py`: ~30 lines (Step 5c + `_find_prior_baseline` helper)
- `tools/test_run_experiment_corpus_filter.py`: ~15 lines (the new pin)
- `sweep/sweep_watchdog_cycle.sh`: ~10 lines (completion-message extension)

Estimated effort: 1-2 focused hours.
