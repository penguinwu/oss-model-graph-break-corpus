---
case_id: adv-2026-05-07-124100-cohort-regen-fix
original_case_id: 2026-05-07-124100-cohort-regen-fix
subagent: adversary-review
migrated_from: skills/adversary-review/reviews_log.md
migration_date: 2026-05-08
date_utc: "2026-05-07T16:41:00Z"
trigger: validator-code + tools-script (retrospective adversary review of already-shipped cohort-regen mitigation bundle)
files: tools/generate_cohort.py, tools/test_generate_cohort.py, skills/sweep_sanity_check.md, skills/sweep.md (Pre-flight + §8), sweep/run_sweep.py (cohort-loading lines ~617-790), experiments/configs/nested_gb_cohort_2026-05-06.json (_metadata + samples), experiments/2026-05-06-ngb-verify-postmortem.md (context)
persona_sha: b9a495616d731234299ad9bb6638418e328d8c2e
verdict: approve-with-gaps
output_sha256: 5f4a88d4957292c2b4f42b3807f73b68e3c0452bd2bd1d027a4cae3a0f8c81c8
---

> **Pre-migration entry.** `persona_sha` references the file at its pre-migration path `skills/adversary-review/persona.md@<sha>`. Use `git show <sha>:skills/adversary-review/persona.md` to retrieve.

| field | value |
|-------|-------|
| date_utc | 2026-05-07T16:41:00Z |
| trigger | validator-code + tools-script (retrospective adversary review of already-shipped cohort-regen mitigation bundle) |
| files | tools/generate_cohort.py, tools/test_generate_cohort.py, skills/sweep_sanity_check.md, skills/sweep.md (Pre-flight + §8), sweep/run_sweep.py (cohort-loading lines ~617-790), experiments/configs/nested_gb_cohort_2026-05-06.json (_metadata + samples), experiments/2026-05-06-ngb-verify-postmortem.md (context) |
| persona_sha | b9a495616d731234299ad9bb6638418e328d8c2e |
| verdict | approve-with-gaps |
| output_sha256 | 5f4a88d4957292c2b4f42b3807f73b68e3c0452bd2bd1d027a4cae3a0f8c81c8 |

**Reviewer raw output:** see `/tmp/raw_output_cohort_regen.txt` (full verbatim copy preserved at invocation time; hash matches above). Summary structure inline below for quick reference; full output is the source of truth.

**GAPS_FOUND summary:** 9 gaps total — 4 high, 4 medium, 1 low.
- HIGH 1: bare-list cohorts (no _metadata) only WARN, don't refuse → original 2026-05-06 failure shape STILL launches today
- HIGH 2: empty `source_versions: {}` silently bypasses version-compat (`if source_versions and version_info` short-circuits)
- HIGH 3: no round-trip generator/loader contract test → live drift caught (skill says `target_versions`, code uses `source_versions`)
- HIGH 4: 20-random sample-sweep gate has NO scripted sampler — agents pick manually, reproducing the original failure mode
- MED 5: cohort-mtime vs explain-pass freshness not enforced (mechanically closes "directive persistence" open mitigation)
- MED 6: generate_cohort doesn't validate torch/transformers/diffusers all present → partial-versions silently allow mismatches
- MED 7: sanity-check skill INV-A1/A3 invariants have no executors (markdown-only)
- MED 8: regression test for 2026-05-06 silently SKIPS when fixture pruned → load-bearing test can become no-op
- LOW 9: generate_cohort.py uses Python 3.9+ feature without runtime-version guard

**SUGGESTED_ADDITIONAL_TESTS:** 12 tests with full SETUP/ACTION/EXPECTED/DETECTS structure. Most-leveraged per reviewer: test 5 (stale-cohort mtime check) — only one that mechanically prevents the exact 2026-05-06 directive-loss failure.

**My disposition (Peng directed 2026-05-07 13:04: "Fix everything the agent has surfaced. Adding tests first makes sense."):**
- gap 1 (bare-list rejection): ADDRESSED — `sweep/cohort_validator.py` raises `BARE_LIST_REJECTED` unless `--allow-bare-cohort` opt-in; wired into `sweep/run_sweep.py`. Test: `test_loader_rejects_bare_list_cohort`.
- gap 2 (empty source_versions silent bypass): ADDRESSED — validator raises `EMPTY_SOURCE_VERSIONS` unless `--allow-empty-versions`. Test: `test_loader_rejects_empty_source_versions`.
- gap 3 (no round-trip test + skill drift `target_versions` vs `source_versions`): ADDRESSED — added `test_round_trip_generator_to_validator_version_mismatch` in `tools/test_generate_cohort.py`; corrected skill text in `skills/sweep_sanity_check.md` v3.1 with revision-log entry.
- gap 4 (no scripted sampler): ADDRESSED — built `tools/sample_cohort.py` with deterministic seed (sha256 of cohort path + mtime). 8 tests in `tools/test_sample_cohort.py` including `test_sampler_hits_poisoned_subset_with_reasonable_probability` (review test 6) and `test_sampler_deterministic_for_same_cohort` (review test 7).
- gap 5 (cohort-mtime freshness not enforced): ADDRESSED — validator raises `STALE_COHORT` when source mtime > cohort mtime, unless `--allow-stale-cohort`. Test: `test_loader_rejects_stale_cohort_vs_source_mtime`. **Mechanically closes the postmortem's "in-conversation directive persistence" mitigation-OPEN row.**
- gap 6 (partial source_versions silent allow): ADDRESSED on BOTH sides — generator (`tools/generate_cohort.py`) refuses unless `--allow-partial-versions`; validator raises `PARTIAL_SOURCE_VERSIONS` unless `--allow-partial-versions`. Tests on both sides.
- gap 7 (sanity-check invariants have no executor): ADDRESSED — built `tools/check_cohort_invariants.py` with pre-launch (A2/A3/A4) and post-sweep (A1/C1/C2/G1) modes. 8 tests in `tools/test_check_cohort_invariants.py` including `test_pre_launch_a3_strict_fail_on_extras_review_test_8` (would have caught 2026-05-06 pre-launch) and `test_post_sweep_a1_strict_fail_on_attribute_errors_review_test_9`.
- gap 8 (regression test silent skip): ADDRESSED — snapshotted 190 NGB explain-ok names to `tools/fixtures/ngb_2026-05-05_explain_ok_names.json`; updated `test_regression_2026_05_06_real_explain_pass` to FAIL LOUD if both snapshot and real artifact are missing (instead of silent skip).
- gap 9 (Python 3.9+ feature without runtime check): ADDRESSED — added `sys.version_info < (3, 9)` guard at top of `generate_cohort.py` and `sample_cohort.py` and `check_cohort_invariants.py` `main()`.

**Test coverage delivered:**
- 13 new tests in `sweep/test_cohort_validator.py` (validator invariants + opt-in overrides)
- 5 new + 1 modified test in `tools/test_generate_cohort.py` (generator hardening + round-trip)
- 8 new tests in `tools/test_sample_cohort.py` (sampler determinism + poisoned-subset hit-rate)
- 8 new tests in `tools/test_check_cohort_invariants.py` (mechanical invariant executor)
- 1 fail-loud-instead-of-silent-skip fix to `test_regression_2026_05_06_real_explain_pass`
- Total: 35 net-new tests; full suite 55/55 PASS

**Commit:** see commits referencing this case_id in repo log.


---
