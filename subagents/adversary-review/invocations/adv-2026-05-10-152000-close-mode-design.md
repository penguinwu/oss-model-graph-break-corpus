---
case_id: adv-2026-05-10-152000-close-mode-design
subagent: adversary-review
date_utc: 2026-05-10T19:20:00Z
trigger: design-review
files: [subagents/file-issue/CLOSE_MODE_DESIGN.md, subagents/file-issue/persona.md, subagents/file-issue/SKILL.md, sweep/WEEKLY_SWEEP_WORKFLOW.md, tools/file_issues.py, tools/audit_new_errors.py]
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 6d26c3c5c3634c5d007d2374b3f3562841a23b4c1dd6eb5f59756cb118ac8d2c
---

## Reviewer raw output

```
VERDICT: approve-with-gaps
CONFIDENCE: high

GAPS_FOUND:

1. [HIGH] Mode-level granularity collapse in WRAPPED logic. parse_affected_models returns {model: [modes]} but classify_close_candidate keys disposition by model name only; fullgraph_models is built from explain entries with `if e.get("graph_break_count", 1) == 0` per-entry but added by NAME (set add of `e["name"]`). A model fullgraph in eval but graph_break in train can be auto-closed. Body says "all N pairs fullgraph" — close-mode is not actually checking pairs. SUGGESTED_FIX: add per-mode pre-flight check in --close validation: re-derive (model, mode) pass/fail from explain_results.json + identify_results.json; refuse if any pair from parse_affected_models(issue.body) is not fullgraph.

2. [HIGH] Plan staleness ≠ sweep staleness. The 10-day gate as designed checks `plan.metadata.timestamp`, but a fresh plan can be regenerated from a stale sweep_dir. SUGGESTED_FIX: read sweep_state.json `finished` (preferred) or `started`; gate on THAT.

3. [MEDIUM] `--plan` validation does not verify the plan came FROM the sweep_dir whose marker is checked. Marker check could be bypassed by relocating plan. SUGGESTED_FIX: add `--sweep-dir` required arg AND verify plan provenance + marker check against sweep_dir.

4. [MEDIUM] `reframe` action ("re-run nightly + sweep_compare") is operationally wrong for "sweep age > 10 days" trigger. Stale data means "wait for next nightly," not "launch a manual nightly." Wrong action becomes wrong action. SUGGESTED_FIX: change reframe action to "defer; re-invoke close-mode after the next regular nightly + sweep-report."

5. [MEDIUM] Phase A's "close-stale --apply only invoked from cron" has zero mechanical enforcement. SUGGESTED_FIX: add runtime guard to cmd_close_stale: refuse unless `CORPUS_CLOSE_STALE_FROM_CRON=1` env var (or `--cron-invocation` flag) is set. Cron sets it; ad-hoc agent invocation does not.

6. [MEDIUM] Body template overstates closure with "fixed on trunk." Per Peng directive 15:09 ET, attribution is NOT investigated at close time. "Fixed on trunk" implies a torch-level cause we're declining to make. SUGGESTED_FIX: replace "the underlying gap appears to be fixed on trunk" with "all originally-affected models now compile fullgraph in this sweep. Attribution (torch / transformers / model code change / vacuous) is not investigated at close time."

7. [MEDIUM] Body says "all N pairs" but N is model count and check is per-model. Lies twice (unit + what's verified). SUGGESTED_FIX: after fixing gap 1, render table per (model, mode) with N = pair count; OR keep model-keyed and say "N models" not "N pairs."

8. [LOW] No verdict for "close_candidates does not contain the issue number." Should be distinguishable from `reject-keep-open` (partial flip). SUGGESTED_FIX: add 5th verdict `not-a-candidate` (or sub-reason) for audit-trail clarity.

SUGGESTED_TESTS: 8 specific tests covering all 8 gaps + the existing classify_close_candidate precedence pinning.

NOTES:
- close-mode is described as "wrapping" classify_close_candidate, but gaps 1+2 are bugs IN that existing logic that close-mode inherits AND ratifies. The --via-skill gate gives false confidence. Either fix the bugs in scope (option b in gap 1) OR explicitly mark them inherited known-bugs in design.
- Marker file format on bad input: parse robustly, surface affected pairs in refusal.
```

## My disposition

| # | Severity | Disposition | Action |
|---|---|---|---|
| 1 | high | **Addressed** in rev 3 | Add per-mode pre-flight check in `--close` validation. Re-derive (model, mode) pass/fail from explain_results.json (graph_break_count==0 per pair) + identify_results.json status (must be fullgraph). Refuses if any pair from parse_affected_models(issue.body) is not fullgraph in current sweep. Lives entirely in new code. |
| 2 | high | **Addressed** in rev 3 | Staleness gate uses sweep_state.json `finished` (or `started` as fallback). Plan timestamp is informational only. |
| 3 | medium | **Addressed** in rev 3 | Add `--sweep-dir` required arg. Validate plan path's parent matches sweep_dir; cross-check marker against sweep_dir. |
| 4 | medium | **Addressed** in rev 3 | Reframe action: "defer; wait for next regular nightly + sweep-report; do NOT manually launch a nightly." |
| 5 | medium | **Addressed** in rev 3 | Add `CORPUS_CLOSE_STALE_FROM_CRON` env var guard to `cmd_close_stale --apply`. Bypass only via env var. |
| 6 | medium | **Addressed** in rev 3 | Body template wording revised: drop "fixed on trunk"; explicit disclaimer that attribution is not investigated at close time. |
| 7 | medium | **Addressed** in rev 3 | After gap #1 fix (per-pair check), body table renders per (model, mode); N = pair count; prose says "pairs" consistently. |
| 8 | low | **Addressed** in rev 3 | 5th verdict `not-a-candidate`. Tool emits a distinct error message. |

All 8 SUGGESTED_TESTS implemented in `tools/test_file_issues.py` (close-mode block).

**Note on gap #1 + #2 (existing code defects):** option (b) chosen — fix lives in new `--close` code, not in `classify_close_candidate`. The wrapped function's mode-collapse bug remains for the close-stale legacy path; close-mode adds the per-mode check on top. Phase B retrofit will fix close-stale at the source.
