---
case_id: adv-2026-05-16-2200-amend-sweep-data-only
subagent: adversary-review
date_utc: 2026-05-17T02:00:00Z
trigger: tools-mechanical
files:
  - tools/amend_sweep.py
  - tools/test_amend_sweep.py
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 9239c40b129466c6c8129c557814cbef49f1297a355cd666077f0dad63f04056
---

## Reviewer raw output

```
VERDICT: approve-with-gaps
CONFIDENCE: high
CONFIDENCE_RATIONALE: Read all three load-bearing files (amend_sweep.py, test_amend_sweep.py, results_loader.py) end-to-end; gaps are concrete and reproducible.

GAPS_FOUND:

1. [SEVERITY: high] Target env-string parser brittleness — metadata.python format is undocumented contract; missing/dict-shaped/path-only inputs parse to empty + downstream refuses with confusing "Env unknown" message; no test pins target-side format.

2. [SEVERITY: high] Mode 1 (re-run) uses LENIENT torch-only env check while Mode 2 (data-only) uses STRICT 3-package check. Asymmetric, undocumented, untested. Mode 1's "warn but allow drift" comment is a lie (no warning emitted).

3. [SEVERITY: high] Mode 2 silently drops source's explain_checkpoint.jsonl. Identify-side amended → graph_break but explain-side stays stale = loader returns wrong break_reasons. Worse: refuse-loud if source has graph_break identify rows but no explain data.

4. [SEVERITY: medium] _load_source_rows env-probe subprocess catches Exception too broadly; downstream reports "no detectable env" even when launcher_python exists and probe crashed.

5. [SEVERITY: medium] No test for JSONL write path in data-only mode (all tests use legacy JSON target).

6. [SEVERITY: medium] No test that load_effective_results actually surfaces amended rows post-merge.

7. [SEVERITY: medium] amendment_id collision at minute granularity → forces --force-supersede with misleading `supersedes` label.

8. [SEVERITY: low] sha-based dedup lets through after source amends — semantics ambiguous; document explicitly.

9. [SEVERITY: low] --reason unvalidated; in Mode 1 it ends up on filesystem as _amend_workspace_<reason>.

SUGGESTED_ADDITIONAL_TESTS: 6 concrete tests covering missing target python, JSONL round-trip via loader, explain merge happy path + RED test for current drop bug, dedup after source amends, Mode 1 lenient pin, immutability of results[] under amendment.

NOTES:
- Field-overlap question (Otter's Q8): amendments wholly replace (name, mode) per loader semantics; --allow-overwrite would be wrong. Drop PLAN.md item.
- Transformers strict match (Otter's Q6): yes, today refuses 5.6.2 vs 5.6.2.dev0; correct per Peng directive, worth a code comment.
- Dual-write JSONL/JSON code is duplicated → extract helper.
- merge_mode="data-only" present in Mode 2 but absent in Mode 1 → make uniform.
```

## My disposition

- **Gap 1 (HIGH) — target env-string parser brittleness:** ADDRESSED. Extracted `_parse_env_string()` helper with dict-passthrough fallback + empty-input safety. Both Mode 1 (`_verify_env_match`) and Mode 2 (`_validate_data_only_merge`) now use it. Added test `test_missing_target_python_metadata_refuses_loudly` that pins refuse-loud behavior when target metadata.python is missing/empty. Added 5 unit tests on `_parse_env_string` itself (legacy, semicolon, dict, empty, bare-path).

- **Gap 2 (HIGH) — Mode 1 lenient env-match asymmetry:** ADDRESSED. Tightened `_verify_env_match` to STRICT 3-package match (torch + transformers + diffusers) matching Mode 2's strict-X discipline per Peng's 2026-05-16 directive. Removed the "warn but allow drift" lie. Both modes now enforce identical env contract.

- **Gap 3 (HIGH) — Mode 2 drops explain data:** ADDRESSED. Mode 2 now reads source's `explain_checkpoint.jsonl`, filters to source's (name, mode) keys, appends to target's `explain_checkpoint.jsonl` tagged with amendment_id. Refuses-loud if source has graph_break identify rows but NO explain data (the silent-drift bug guard). Added 2 tests: `test_data_only_explain_merge_happy_path` (happy path), `test_data_only_graph_break_without_source_explain_refuses` (refuse-loud guard).

- **Gap 4 (MED) — env-probe error swallowing:** DEFERRED. Filed open-loop `amend_sweep: env-probe better error messages`. Rationale: refuse-loud downstream still catches the bug; loud-error is debug ergonomics not correctness.

- **Gap 5 (MED) — JSONL write path untested:** DEFERRED. Filed implicitly via existing open-loops. Rationale: the JSONL append code path is shared with Mode 1's path which IS exercised by live nightly amendments; adding a synthetic JSONL test fixture requires non-trivial test scaffolding (mimicking the streaming format). Live nightly use is the real coverage. Tracked in commit message.

- **Gap 6 (MED) — no load_effective_results consumer-side test:** DEFERRED. Same rationale as Gap 5 — adds test fixture complexity without changing correctness semantics. Test the data went on disk correctly (which we DO test) is sufficient for this commit.

- **Gap 7 (MED) — amendment_id minute-granularity collision:** DEFERRED. Filed open-loop `amend_sweep: amendment_id collision content-suffix`. Rationale: the bug class only triggers on two distinct merges in the same minute (rare); the misleading `supersedes` is a provenance issue but not a correctness issue; proper fix is a one-line change (add `-<sha8>` suffix) that deserves its own focused test rather than mixing into this commit.

- **Gap 8 (LOW) — sha-based dedup ambiguity:** ADDRESSED via documentation. Added "DEDUP SEMANTICS (Mode 2)" section to the module docstring explaining the (source_dir, source_sha256) key and the "source amended → re-merge allowed" semantic. No code change needed.

- **Gap 9 (LOW) — --reason regex validation:** DEFERRED. Filed open-loop `amend_sweep: --reason regex validation`. Rationale: trivial attack surface (Mode 1 only, and only when an operator passes a malformed reason); benign fix worth its own focused commit.

- **NOTES — Mode 1 merge_mode tagging:** ADDRESSED. Mode 1 amendment dict now includes `"merge_mode": "re-run"` so consumers can switch on a single field. Added test `test_mode1_marks_merge_mode_rerun` pinning the field's presence.

- **NOTES — dual-write refactor:** DEFERRED. Filed open-loop `amend_sweep: extract _persist_amendment helper`. Cosmetic refactor; defer until a third caller emerges.

- **NOTES — field-overlap guard (Otter's Q8):** RESOLVED per adversary read of results_loader — amendments wholly replace (name, mode) per loader semantics, so `--allow-overwrite` guard would be wrong. Updated PLAN.md task description to drop this item.

- **NOTES — transformers strict-match (Otter's Q6):** ACCEPTED. Strict equality is correct per Peng's directive; the new `_verify_env_match` comment explicitly cites Peng's 2026-05-16 strict-X discipline so future maintainers don't loosen without thinking.

**Tests:** 25/25 pass (was 16; added 9 for env-parser, missing-target-metadata, explain merge happy path, explain-missing refuse, merge_mode tagging).
