# sweep_compare wiring into nightly pipeline — design

**Author:** Otter
**Date:** 2026-05-10 (rev 2: 14:15 ET — auto-discovery removed per Peng directive)
**Status:** Implemented + shipped
**WS1 task:** "Wire `tools/sweep_compare.py` into nightly pipeline"
**Related spec:** `sweep/WEEKLY_SWEEP_WORKFLOW.md` Step 1 → Step 2 bridge

---

## What it does

`tools/run_experiment.py nightly --baseline-dir <prior-sweep-dir>` invokes `tools/sweep_compare.py` after explain + sweep-report + close-stale, writing `<output-dir>/compare-vs-baseline.{md,json}`.

When `--baseline-dir` is omitted, the comparison step is skipped with a log message. Caller (cron prompt OR human invocation) explicitly names the baseline.

## Why explicit input, not auto-discovery

**Rev 1** included a `_find_prior_baseline()` helper that walked the parent dir for the latest dated YYYY-MM-DD sibling. Peng pushed back 2026-05-10 14:12 ET: "prior_baseline should be specified as input to the system. Is it over-engineering?"

Yes. Auto-discovery added:
- A helper function + 8-test file
- Surprising behavior (silent skip on no qualifier; no clarity about WHICH dir was picked)
- Coupling to dir-naming conventions (YYYY-MM-DD format, status-done filtering)

Explicit input matches the principle that the caller knows what comparison it wants. The cron prompt that invokes the nightly knows the baseline; humans invoking it explicitly know too.

## Mechanism

In `tools/run_experiment.py:run_nightly_command`, after Step 5b (close-stale dry-run):

```python
baseline_dir = getattr(args, "baseline_dir", None)
sweep_compare = str(tools_dir / "sweep_compare.py")
if baseline_dir and Path(sweep_compare).exists():
    compare_md = str(Path(results_dir) / "compare-vs-baseline.md")
    compare_json = str(Path(results_dir) / "compare-vs-baseline.json")
    _run([python, sweep_compare,
          "--baseline", str(baseline_dir),
          "--current", str(results_dir),
          "--out", compare_md,
          "--json", compare_json,
          "--verbose"],
         f"sweep_compare vs {Path(baseline_dir).name}",
         allow_fail=True)
elif not baseline_dir:
    print("NIGHTLY: sweep_compare — SKIPPED (no --baseline-dir given)")
```

`allow_fail=True` because invariant failures (exit 1) and explain-coverage gaps (exit 2) should surface but not abort the rest of the nightly.

## CLI

```
python3 tools/run_experiment.py nightly \
    --baseline-dir sweep_results/nightly/<prior-date> \
    [other existing args]
```

When `--baseline-dir` omitted, sweep_compare step is skipped silently (with a log line); the rest of nightly runs unchanged.

## Output

- `<output-dir>/compare-vs-baseline.md` — human-readable report (sweep_compare's existing markdown output)
- `<output-dir>/compare-vs-baseline.json` — machine-readable cat 1-6 partition (sweep_compare's existing `cats_to_json` output)

Downstream tools (audit_new_errors.py and audit_new_models.py) consume the JSON.

## Test plan

The change is a single conditional invocation behind a new arg. No new logic to test beyond:
- Parser accepts `--baseline-dir`
- When given, sweep_compare is invoked
- When omitted, step is skipped silently

The existing `tools/test_run_experiment_corpus_filter.py` parser tests already exercise the nightly subcommand. A regression test was deferred — the change is mechanical and the verification is the sweep_compare invocation succeeding (which it did against the live 2026-05-09 sweep, surfacing the explain_coverage invariant failure as expected).

## What this UNBLOCKS

Steps 2a / 2b can assume `<sweep_dir>/compare-vs-baseline.json` exists when audit tools run as part of the cron prompt's post-sweep sequence (the cron prompt passes `--baseline-dir`).

## Manual gates

None at this layer. The caller decides which baseline to compare against.

## Implementation status

Shipped 2026-05-10 14:30 ET. See commit message for the rev — auto-discovery helper was reverted per Peng directive 14:12 ET.
