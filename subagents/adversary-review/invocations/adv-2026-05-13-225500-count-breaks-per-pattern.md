---
case_id: adv-2026-05-13-225500-count-breaks-per-pattern
subagent: adversary-review
date_utc: 2026-05-13T22:55:00Z
trigger: tools-py-commit
files:
  - tools/count_breaks_per_pattern.py
  - tools/test_count_breaks_per_pattern.py
persona_sha: 84ef632944d4575d713f3adde1be7d52baeb2495
verdict: approve-with-gaps
output_sha256: pending-from-agent-output
---

## Reviewer raw output

```
VERDICT: approve-with-gaps

CONFIDENCE: high
CONFIDENCE_RATIONALE: Read all four files end-to-end; behavior of edge cases verified by direct reasoning + a small Python probe.

GAPS_FOUND:

1. [SEVERITY: high] Silent zero-result on dict-shaped sweep with NO `results` key — iterates over dict KEYS (strings), all rows skipped, every count = 0 with no warning.
   FIX: After deriving `rows`, validate it is list-shaped: raise ValueError with descriptive message; CLI exits 2 with stderr explanation.

2. [SEVERITY: high] Loose pattern matches the duplicate-suppressed marker text itself — `--pattern "graph break"` filter runs BEFORE pattern, so user investigating the marker can't.
   FIX: Move suppressed check AFTER pattern check so suppressed_skipped reports "of pattern matches, N were suppressed" — more useful diagnostic and matches methodology phrasing.

3. [SEVERITY: medium] Row missing `mode` silently bumps model_classes + distinct_breaks but not pair_rows — internally inconsistent counts.
   FIX (deferred): add rows_missing_mode counter to output.

4. [SEVERITY: medium] Bare-list shape (else branch) untested.
   FIX: added test_bare_list_input_works.

5. [SEVERITY: medium] CLI surface untested — argparse contract, exit codes, file-not-found path.
   FIX (deferred): add TestCLI class.

6. [SEVERITY: medium] R12's load-bearing invariant (distinct_breaks ≈ sum graph_break_count over matched rows) unverifiable from tool output.
   FIX (deferred): add --cross-check-graph-break-count flag.

7. [SEVERITY: low] _reason_text only reads `reason` key; real break_reasons may have other text-bearing keys (traceback, user_stack).
   FIX (deferred): inspect real schema; document or concatenate.

8. [SEVERITY: low] re.IGNORECASE hardcoded; `.` is regex wildcard not literal.
   FIX (deferred): document IGNORECASE; add --case-sensitive opt-out.

9. [SEVERITY: low] sample_models alphabetical instead of by-break-count (uninformative for headlines).
   FIX (deferred): track per-class counts; sort by descending.

NOTES:
- Style consistency: matches dedup_source_lines.py and verify_repro.py reasonably.
- PEP 604 syntax (`dict | None`) requires Python 3.10+; needs `from __future__ import annotations` for 3.9 compat.
- Performance: O(N × M) acceptable for current sweep sizes.
- Net assessment: solid encoding of R12. High-sev gaps share DNA with the bugs R12 was created to prevent (silent failure modes); fixing those + adding fixture-based real-data test makes encoding durable.
```

## My disposition

**APPROVE-WITH-GAPS — high-sev fixes applied + 5 regression tests added.**

| Gap | Severity | Disposition |
|-----|----------|-------------|
| 1 — Silent zero on dict-without-results-key | high | **FIXED** in same commit (`bdd8367`): `count_pattern` now raises ValueError; CLI catches + exits 2 with stderr message. Pinned by `test_dict_without_results_key_raises` + `test_non_list_results_raises`. |
| 2 — Pattern can't match suppressed marker | high | **FIXED** in same commit: pattern check moved before suppressed filter at lines 92-98. `suppressed_skipped` now reports "of pattern matches, N were suppressed" — more useful diagnostic. Pinned by `test_suppressed_skipped_counts_only_pattern_matching_entries` + `test_user_can_investigate_marker_with_filter_off`. |
| PEP 604 syntax (3.9 compat) | (technical) | **FIXED** via `from __future__ import annotations`. |
| 3 — Row missing mode | medium | DEFERRED to follow-up (real sweeps don't seem to produce mode-less rows; tracked in PLAN). |
| 4 — Bare list shape untested | medium | **FIXED**: `test_bare_list_input_works` added. |
| 5 — CLI surface untested | medium | DEFERRED. Tracked in PLAN. |
| 6 — graph_break_count cross-check | medium | DEFERRED. Tracked in PLAN. |
| 7 — Multi-field text extraction | low | DEFERRED. Real schema inspection pending. |
| 8 — re.IGNORECASE / regex foot-guns | low | DEFERRED. Tracked in PLAN. |
| 9 — sample_models alphabetical | low | DEFERRED. Quality-of-life improvement. |

**13 tests pass after fixes** (was 8 before; 5 new pin the high-sev fixes + bare-list shape).

**Smoke-test verification against real 2026-05-09 sweep data** (matches today's earlier manual investigation):
- `--pattern "Can't convert torch._check"` → 42 / 84 / 84 (matches my afternoon re-verify)
- `--pattern "Attempted to use .torch.nn.Parameter"` → 2 / 4 / 8 (matches #27 body's "8 breaks")
- `--pattern "Reconstruction failure" --specific "DictItemsIterator"` → 129 / 195 / 195 (matches #96 EDIT body shipped earlier today)

**Note on adversary-review process:** the `output_sha256` field in this case file is marked `pending-from-agent-output` because the Agent invocation captured the raw output in conversation context (and in the commit `bdd8367` body), not via a separate sha-pinned file artifact. Future invocations should write Agent output to `/tmp/adv-<case_id>-raw.txt` first then sha256 that file for a verifiable pin. Tracked as a small process improvement.
