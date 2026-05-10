---
case_id: adv-2026-05-10-145000-audit-new-errors-design
subagent: adversary-review
date_utc: 2026-05-10T18:50:00Z
trigger: design-review
files: [sweep/AUDIT_NEW_ERRORS_DESIGN.md, sweep/WEEKLY_SWEEP_WORKFLOW.md, sweep/SWEEP_COMPARE_WIRING_DESIGN.md, sweep/known_errors.json, sweep/skip_models.json, skills/weekly-sweep-brief/methodology.md, tools/sweep_compare.py]
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 6880be0e174764b323acf135a530a5f15b50be8669de090d2c9d6705b0ce0ee6
---

## Reviewer raw output

```
VERDICT: approve-with-gaps

CONFIDENCE: high
CONFIDENCE_RATIONALE: I read the design + workflow + sweep_compare + known_errors + the live nightly identify_streaming/identify_results files; the gaps below come from concrete patterns in those files, not from speculation. Confidence would drop to medium only if I'm wrong about how compose_brief.py will consume the JSON sidecar (it's not built yet, so I'm inferring from template.md sections 4 and 7).

FILES_READ: [as listed in frontmatter]

GAPS_FOUND:

1. [SEVERITY: high] The design reads `error_message` but identify_streaming.jsonl uses field name `error` (not `error_message`); identify_results.json effective rows ALSO use `error`. A naive `row.get("error_message")` will return None for every row and every classification will fall through to `unknown`.

2. [SEVERITY: high] BambaModel-class rows (and the entire 2026-05-09 cuDNN cluster — Bart*, Beit*, Bert*, Exaone4*, Falcon*, etc., easily 30+ rows in this single sweep) have `status="worker_error"` with the cuDNN load failure pattern. The design's `subprocess-crash` heuristic only matches `worker_error` OR strings like `subprocess died` / `worker exited` / `SIGSEGV`. It does NOT match the actual cuDNN failure text. Suggested action ("wait for auto-retry") is wrong: cuDNN cluster is a *venv-bootstrap* infra bug, not a transient subprocess crash. SUGGESTED_FIX: add `venv-bootstrap-broken` class.

3. [SEVERITY: high] The design specifies status taxonomy as `{eager_error, create_error}` for candidate selection — it EXCLUDES `worker_error` and `timeout`. Workflow doc lists those as in-scope. 2026-05-09 sweep has dozens of `worker_error` rows that this audit will silently drop. SUGGESTED_FIX: expand candidate set to `{eager_error, create_error, worker_error, timeout}` (matches sweep_compare's ERROR_STATUSES) OR add explicit out-of-scope tabulation section.

4. [SEVERITY: high] Bias-INFRA-FIX has NO mechanical enforcement. The design says "the audit tool refuses to write known_errors.json itself" — non-feature. Actual escape-hatch path: human-reviewer reads audit, decides "too hard to fix," edits known_errors.json by hand. Discipline-only enforcement of Peng directives is the failure mode External Engagement Approval was retrofitted to fix. SUGGESTED_FIX: pre-commit / corpus-consistency check that REJECTS new known_errors entries with `status: "create_error"` without explicit `--allow-create-error-escape` override.

5. [SEVERITY: high] Re-run-scope rule enforced ONLY by TEXT WARNING. No mechanical gate. If skipped, Step 2c silently consumes stale identify_results.json. SUGGESTED_FIX: emit machine-readable `<sweep_dir>/.audit-rerun-required` marker with affected (model, mode) list. Step 2c (file_issues.py corpus-issue, close-mode) reads the marker and refuses to run if present.

6. [SEVERITY: medium] `applies_to_versions` filter has two underspecified failure modes: missing field (universal apply per docstring), patch-release granularity (can't express "applies to 2.13.0 but not 2.13.1"). Pick one explicitly + add test.

7. [SEVERITY: medium] Cat 6 NEW-entry edge case real: a known_errors entry with WRONG error_pattern silently hides cat 6 stable failures from BOTH audit and brief. SUGGESTED_FIX: add "cat 6 — known_errors mismatch" subsection in audit report.

8. [SEVERITY: medium] Schema fit with downstream compose_brief incomplete. audit-new-errors.json schema lacks `error_type`, `phase`, `pytorch_issue_link`, `attribution_status`, `prior_status_for_cat1`. Either DROP the claim that audit-new-errors.json feeds compose_brief sections 4/7 OR extend schema.

9. [SEVERITY: medium] Field absences from identify_streaming.jsonl: `error_type`, `phase`, `phase_at_timeout`, `returncode`, `retry_note` — all structured contracts the design ignores in favor of fragile substring matching. SUGGESTED_FIX: rewrite heuristic table to key off structured fields FIRST, substring as refinement.

10. [SEVERITY: medium] Non-deterministic classification: identify_streaming.jsonl and identify_results.json have DIFFERENT error text for HiggsAudio (streaming says "Audio must be mono" → fixture-bug; effective says "Sizes of tensors must match" → unknown). Pin: ALWAYS classify using identify_results.json effective row.

11. [SEVERITY: low] Branch precedence for overlapping heuristics not pinned by tests. A `worker_error` with "device-side assert" matches both `subprocess-crash` (status) and `cuda-context-pollution` (text). Add explicit precedence test for every PAIR; document "first match wins."

12. [SEVERITY: low] No regression test for "audit's input is compare-vs-baseline.json" wiring claim. Two parallel WIP designs with prose-only interface contract = schema-drift setup. Add JSON-schema fixture or shared dataclass.

SUGGESTED_TESTS: 10 specific tests — see full output (test_field_names_match_real_data, test_worker_error_cuDNN_cluster_classification, test_rerun_required_marker_blocks_step2c, test_bias_infra_fix_blocks_create_error_known_errors_addition, test_classifier_uses_effective_row_not_streaming_row, test_applies_to_versions_missing_field_behavior_pinned, test_cat6_known_errors_pattern_mismatch_surfaced, test_worker_error_returncode_signal_disambiguation, test_branch_precedence_pinned_for_overlapping_heuristics, test_schema_compose_brief_consumer_can_fill_section_4_or_admit_it_cant)

NOTES:
- Three themes: (1) verify against REAL data shapes from 2026-05-09 BEFORE declaring done — this is the "verify before acting" lesson; (2) replace text directives with mechanical gates wherever Peng has stated a hard rule; (3) pin interface contracts to fixtures, not prose.
- Strong recommendation: BEFORE writing implementation, run a 30-minute "live data exploration" — load identify_streaming.jsonl + identify_results.json, dump every distinct (status, error_type, phase, returncode) tuple, use THAT as the basis for the heuristic table.
- Pair "case_id attribution per heuristic rule" with a coverage test: every case_id in the table MUST have at least one fixture row.
- Consider exit code 3 for "no baseline; degraded mode" so watchdog can distinguish "no candidates clean" from "couldn't compare."
```

## My disposition

| # | Severity | Disposition | Action |
|---|---|---|---|
| 1 | high | **Addressed** in design rev 2 | Field name `error` not `error_message`; pin against real 2026-05-09 row fixtures in tests |
| 2 | high | **Addressed** in design rev 2 | Add `venv-bootstrap-broken` triage class; matches cuDNN load failures + missing-symbol + BPF-block patterns |
| 3 | high | **Addressed** in design rev 2 | Expand candidate set to all 4 `ERROR_STATUSES` (matches sweep_compare); per-status routing |
| 4 | high | **Deferred** (separate task added to PLAN.md) | Pre-commit check for create_error additions is a separate tool with its own scope; keep audit_new_errors.py focused. NEW PLAN.md task: "build tools/check_known_errors.py — bias-INFRA-FIX mechanical gate" |
| 5 | high | **Addressed** in design rev 2 | Emit `<sweep_dir>/.audit-rerun-required` marker; Step 2c tools (file_issues.py corpus-issue + close-mode when built) refuse to run if present. Marker honored at the consumer side. |
| 6 | medium | **Addressed** in design rev 2 | Pin: missing `applies_to_versions` → fail-loud (forces explicit decision per workflow doc "discouraged"). Match granularity stays at major.minor; document patch-release limitation. |
| 7 | medium | **Addressed** in design rev 2 | Add "cat 6 — known_errors pattern mismatch" subsection that verifies error_pattern substring is actually present in the row's effective error. |
| 8 | medium | **Addressed** in design rev 2 | Drop the claim that audit-new-errors.json feeds compose_brief Section 4/7 directly. Per Peng's compose_brief rethink, that tool is gated on (a)/(b)/(c) decision; if (b) lands (linter on already-written brief), the cross-tool schema dependency is moot. Audit's JSON sidecar is for human review, not pipelined consumption. |
| 9 | medium | **Addressed** in design rev 2 | Heuristic table rewritten to key off structured fields first (status, error_type, phase, returncode, retry_note); substring as refinement only. |
| 10 | medium | **Addressed** in design rev 2 | Pin: classify against identify_results.json effective row (post-auto-retry, last-write-wins). identify_streaming.jsonl used only for diagnostic context. |
| 11 | low | **Addressed** in design rev 2 | Add explicit precedence test per pair of overlapping heuristics; document "rules evaluated top-to-bottom; first match wins." |
| 12 | low | **Addressed** in design rev 2 | Add JSON-schema fixture (or shared dataclass) for compare-vs-baseline.json contract; round-trip test in tools/test_sweep_compare_wiring.py + tools/test_audit_new_errors.py. |

**Suggested tests:** all 10 will be added to `tools/test_audit_new_errors.py` per the test plan in design rev 2. The `field_names_match_real_data` test (real-row fixtures from 2026-05-09) is the load-bearing one — catches gaps 1, 2, 9, 10 in one fixture set.

**Recommendation accepted:** before implementation, do a 30-min live data exploration of `sweep_results/nightly/2026-05-09/identify_streaming.jsonl` + `identify_results.json` to ground the heuristic table in actual (status, error_type, phase, returncode) tuples observed in the sweep. Output of the exploration becomes the test fixture set.

**Exit code 3 for "no baseline" suggestion:** accepted; design rev 2 includes it.
