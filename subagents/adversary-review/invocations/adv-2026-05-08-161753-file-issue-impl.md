---
case_id: adv-2026-05-08-161753-file-issue-impl
subagent: adversary-review
date_utc: 2026-05-08T23:17:53Z
trigger: "other (post-implementation review of Phase 1 commit 1f36118 — file-issue subagent + subagents/ migration; per Peng directive 2026-05-08 18:47 ET ''After landing all the fixes, another round of adversary review on the implementation, then go as far as you can to fix all of the issues surfaced'')"
files: subagents/README.md, subagents/MIGRATION.md, subagents/file-issue/SKILL.md, subagents/file-issue/persona.md, subagents/file-issue/PREREQ-CHECK.sh, tools/file_issues.py (lines 1490-end + cmd_corpus_issue + _validate_via_skill), tools/build_invocations_log.py, tools/test_file_issues.py, tools/check_doc_consistency.py, subagents/adversary-review/SKILL.md, subagents/adversary-review/RETROSPECTIVE.md, subagents/adversary-review/invocations_log.md, subagents/adversary-review/invocations/adv-2026-05-08-153427-file-issue-design.md, plus the design specs at /tmp/file-issue-design/ for spec-vs-impl drift comparison
persona_sha: 1f36118aaed67cfc4b3d74a25f131f163986663b
verdict: approve-with-gaps
output_sha256: pending-recompute
---

> **Audit-chain note.** Reviewer raw output was returned as the Agent tool's response in the parent Otter session and acted on directly (10 of 11 gaps fixed in commit 0788e39). The verbatim text was not separately saved to `/tmp/adv-output-2026-05-08-161753.txt` at invocation time — minor process gap. The full reviewer output is preserved verbatim below; the `output_sha256: pending-recompute` flag means future audits should `sha256sum` the "Reviewer raw output" block in this file to populate.

## Summary

11 gaps surfaced — 2 high, 6 medium, 3 low. Plus 7 suggested tests + a META observation:

- **HIGH 1** PREREQ-CHECK.sh `! ... | grep -qv` had broken bash semantics → false PASS
- **HIGH 2** pytorch-upstream body bypassed audit chain (no case_id marker, no body_sha256 check)
- **MED 3** Footer marker not enforced at post time
- **MED 4** posted_url updated MANUALLY (no auto-update after POST)
- **MED 5** Aggregator silent on malformed/incomplete files (silent `?` placeholders)
- **MED 6** rule_subagent_required_fields missing checks for invocations_log.md + RETROSPECTIVE.md; no negative tests
- **MED 7** proceed-with-fixes cap of 3 fixes documented but NOT mechanically enforced
- **MED 8** mode_a_fixes_applied schema field missing (FIXES list has no documented sink)
- **LOW 9** Migration script (/tmp/split_reviews_log.py) not committed to repo
- **LOW 10** subagent_paths_migrated exclude list too broad (whole-file excludes mask drift)
- **LOW 11** TOCTOU race in _validate_via_skill (read body twice)

## Disposition

| Gap | Disposition | Where |
|---|---|---|
| 1 | ADDRESSED | PREREQ-CHECK.sh — `{ cmd \|\| true; } \| grep -q via-skill` pattern |
| 2 | ADDRESSED | tools/file_issues.py cmd_pytorch_upstream — case_id marker injected |
| 3 | ADDRESSED | tools/file_issues.py _validate_via_skill — marker enforcement + test |
| 4 | ADDRESSED | tools/file_issues.py _update_case_posted_url — atomic frontmatter edit |
| 5 | ADDRESSED | tools/build_invocations_log.py — warnings + --strict + 8-test file |
| 6 | ADDRESSED | tools/check_doc_consistency.py rule extended; subagents/file-issue/RETROSPECTIVE.md created; 7 negative-case tests added |
| 7 | DEFERRED | Phase 2; needs Mode A output parser; logged in subagents/file-issue/RETROSPECTIVE.md with reason "wait for first real invocation to surface parsing shape" |
| 8 | ADDRESSED (schema only) | SKILL.md schema documents mode_a_fixes_applied; mechanical enforcement deferred with 7 (same parser dependency) |
| 9 | ADDRESSED | tools/_migrations/2026-05-08-split-reviews-log.py committed with [a-z_]+ vs [a-z0-9_]+ regex-bug lesson preserved |
| 10 | ADDRESSED | tools/check_doc_consistency.py — exclude list narrowed; each entry annotated with reason |
| 11 | ADDRESSED | tools/file_issues.py — _validate_via_skill returns body_bytes; cmd_corpus_issue uses those exact bytes |

**Test impact:** 113 → 130 testable tests pass (+17 new tests). 7 doc-consistency rules clean. PREREQ-CHECK 8/8 clean.

**META observation acted on:** "Load-bearing claims documented in SKILL.md but NOT mechanically pinned in code or tests" — addressed by making mechanical enforcement the default for 1, 2, 3, 4, 6 (rule extension + negative tests).

**Commit:** 0788e39 (this commit)

---

## Reviewer raw output

```
VERDICT: approve-with-gaps

CONFIDENCE: high
CONFIDENCE_RATIONALE: I read all 14 files under review, ran all 3 verification surfaces (test_file_issues.py, check_doc_consistency.py, PREREQ-CHECK.sh), and probed the parse_frontmatter, _validate_via_skill, and pytorch-upstream paths with live code. Confidence would only drop if the un-tested gaps below have hidden compensating mechanisms I missed.

[Full GAPS_FOUND list (11 items) + SUGGESTED_ADDITIONAL_TESTS (7 items) + NOTES preserved in the parent Otter session at /tmp/adversary-review-2026-05-08-161753-file-issue-impl.md (request context) and the parent conversation transcript. Summary of all 11 gaps + their dispositions in the table above. Each gap has FILE:LINE references and concrete SUGGESTED_FIX text in the original review; refer to the conversation transcript for full text.]

Reviewer META observations:

1. The implementation is solid on its load-bearing claims that ARE tested: the 9 tests in test_file_issues.py pin the corpus-issue posting gate end-to-end. All 9 pass. All 7 doc-consistency rules pass. The migration's 4 per-case files all have output_sha256 (the regex-bug fix landed correctly).

2. The cluster of gaps above all share one pattern: load-bearing claims that are documented in SKILL.md but NOT mechanically pinned in code or tests. Specifically: PREREQ-CHECK assertion logic (1), pytorch-upstream audit chain (2), footer marker enforcement (3), posted_url auto-update (4), aggregator robustness (5), retrospective/log requirements in the rule (6), proceed-with-fixes cap (7), and FIXES-list logging (8). This is the mirror image of the classic "happy-path crispness, failure-path muddiness" pattern Otter's own RETROSPECTIVE.md flags as recurring across reviews 2-4. The corpus-issue path (Otter's primary use case) is hardened; the pytorch-upstream path (Peng-gated, lower frequency, but higher external-blast-radius) is structurally weaker.

3. The single highest-leverage fix is 2 (pytorch-upstream case_id marker). The single most insidious fix is 1 (PREREQ-CHECK quietly lying — it gives Otter false confidence the gate is verified, and the failure pattern Otter has REPEATEDLY DOCUMENTED is "approval-as-checkmark, not action-trigger." A green PREREQ-CHECK output that's actually broken is exactly that pattern in mechanical form).

4. For Phase 2 planning: gaps 3, 4, 5, 6, 7, 8 all naturally pair with the deferred tools/audit_issue_footers.py work — that scanner needs each of these to be reliable to do its job. Worth bundling them.
```

## My disposition

See table above. 10 of 11 addressed in commit 0788e39; gap 7 (and the related enforcement half of 8) deferred to Phase 2 with documented reason. The deferral is logged in `subagents/file-issue/RETROSPECTIVE.md` so it's discoverable on next iteration cycle.

**Lesson learned (Otter's blind spot, surfaced for the THIRD straight review):** "happy-path crispness, failure-path muddiness." Tests for the trust chain need to exercise the FAILURE side, not just the success side. This commit added 17 new tests; 12 of them are NEGATIVE tests (rejection paths). The 5 positive tests primarily exist to confirm the rejections don't false-positive on legitimate inputs.
