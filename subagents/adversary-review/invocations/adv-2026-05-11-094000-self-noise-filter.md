---
case_id: adv-2026-05-11-094000-self-noise-filter
subagent: adversary-review
date_utc: 2026-05-11T13:50:00Z
trigger: tool-script-substantive-change
files:
  - tools/check_filed_issues.py
  - tools/test_check_filed_issues.py
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: d78b3eac0b69d775b93e02c93d8c1abf009888540f37a312b843b940b6d60ea4
---

## Reviewer raw output

```
VERDICT: approve-with-gaps

CONFIDENCE: high
CONFIDENCE_RATIONALE: Code is small (~50 LOC of new logic) and the test suite covers the canonical scopes; my confidence would shift if Peng accepts that the GitHub events endpoint returns "renamed" but NOT description-only edits (the smoke test passed because Otter edited titles, not bodies).

FILES_READ:
- /home/pengwu/projects/oss-model-graph-break-corpus/subagents/adversary-review/persona.md
- /tmp/adversary-review-adv-2026-05-11-094000-self-noise-filter.md
- /home/pengwu/projects/oss-model-graph-break-corpus/tools/check_filed_issues.py
- /home/pengwu/projects/oss-model-graph-break-corpus/tools/test_check_filed_issues.py

GAPS_FOUND:
1. [SEVERITY: high] External body-edit on a self-authored issue is silently suppressed when ANY self-comment also lands in the same window.
   WHY_IT_MATTERS: Stage 3 only runs when `not comments_increased`. If Otter posts a self-comment AND an external collaborator edits the issue body in the same poll window, Stage 2 zeroes `non_self_count` and Stage 3 is skipped entirely, so the external body edit never reaches the GChat alert — exact false-suppression class Otter listed as worry #1.
   SUGGESTED_FIX: Run Stage 3 unconditionally when `updated_advanced` is true; combine its `all_events_self` with Stage 2's `non_self_count > 0` only at the signal-composition step.

2. [SEVERITY: high] Body-only edits do NOT appear in the `/issues/{n}/events` endpoint, so the Stage 3 events check returns `[]` for both self AND external description edits — meaning external pure-description edits are NOT suppressed (good) but self pure-description edits ARE NOT suppressed either (bad — defeats scope (b) of the directive).
   WHY_IT_MATTERS: GitHub's `/issues/N/events` endpoint enumerates timeline events like `labeled`, `closed`, `renamed`, `assigned`, `milestoned` — but NOT description body changes. The smoke test on #124/#127 likely involved a title rename or label change, not a pure description edit. In production, a self-authored issue whose body Otter rewrites (no title change, no label flip) will trigger `updated_advanced and not comments_increased`, the events fetch will return `[]`, the `actors and all(...)` guard fails on the empty list, `all_events_self` stays False, and the issue surfaces as "updated YYYY-MM-DD" — exactly the noise we're trying to kill.
   SUGGESTED_FIX: Either (a) when events list is empty AND `iss.author == AUTHOR` AND no other signal exists, treat as self-only (assumption: only the issue author can edit the body without leaving an audit-event); (b) switch from `/events` to the `/timeline` endpoint with mockingbird preview which DOES include body-edit actor info.

3. [SEVERITY: medium] Brand-new self-authored issue fast-path skips the events fetch even when external actors may have already touched it.
   WHY_IT_MATTERS: An external user can label, assign, or close an Otter-authored issue with zero comments before our first poll catches it. The fast-path suppresses this with no API call, hiding the external action.
   SUGGESTED_FIX: Replace the fast-path with: if brand-new self-authored AND `comments_count==0`, still call `_fetch_events_since` with `prev_updated=""` to enumerate all early events; only suppress if every actor (including author) is self.

4. [SEVERITY: medium] `actors and all(...)` empty-list guard is logically correct but the resulting behavior conflates "events endpoint returned nothing" with "external action without an event record".

5. [SEVERITY: medium] No test for the case where `_fetch_events_since` returns `[]` (i.e., empty events list with `updated_advanced=True`).

6. [SEVERITY: low] `prev_updated_at` empty-string sentinel uses lexicographic `>` comparison — works but worth a one-line comment.

7. [SEVERITY: low] Test setUp/tearDown patches module-level functions by reassignment; safer with `unittest.mock.patch.object`.

SUGGESTED_ADDITIONAL_TESTS:

1. test_external_body_edit_with_self_comment_surfaces (Gap 1)
2. test_empty_events_with_updated_advanced_for_self_authored (Gap 2)
3. test_brand_new_self_authored_with_external_label_surfaces (Gap 3)
4. test_self_close_immediately_after_creation_is_suppressed
5. test_priority_author_comment_overrides_self_only_update
6. test_self_label_then_self_unlabel_is_suppressed
7. test_actors_field_missing_actor_key

NOTES:
- Sweep-report and weekly-brief paths intentionally unchanged (different audience). Good.
- Smoke test on live data (#124, #127) is unfortunately not strong evidence — issue-rename and label events DO appear in /events; pure body edits do NOT. Recommend a manual smoke test where Otter edits ONLY the description (no title, no label).
- API rate cost: even with all gap fixes, ~1 events call per tracked issue per poll, well within 5000/h.
```

## My disposition

| Gap | Severity | Disposition |
|-----|----------|-------------|
| 1 — comments_increased shadows Stage 3 | high | **ADDRESSED**: Removed `not comments_increased` gating on Stage 3. Now fires whenever `updated_advanced`, regardless of comments. Pinned by `test_external_body_edit_with_self_comment_surfaces_gap1`. |
| 2 — body edits absent from /events endpoint | high | **ADDRESSED**: Added empty-events heuristic — if events list is empty AND `iss.author == AUTHOR`, treat as self body edit and suppress. If author is external, fail-open and surface. Documented in docstring. Pinned by `test_empty_events_with_self_authored_is_suppressed_gap2` + `test_empty_events_with_external_authored_surfaces_gap2`. (Picked option (a) — cheaper than /timeline endpoint switch and matches reality: only the author can body-edit silently.) |
| 3 — brand-new fast-path blind to external actions | medium | **ADDRESSED**: Removed the `self_authored_new` fast-path entirely. Brand-new issues now go through the same events fetch with the creator-as-implicit-actor pattern. Pinned by `test_brand_new_self_authored_with_external_label_surfaces_gap3`. Cost: one extra API call per brand-new self-authored issue (one-time per issue). |
| 4 — empty-events behavior documentation | medium | **ADDRESSED**: Docstring on `diff_against_state` now explicitly explains the empty-events heuristic + author-based decision. |
| 5 — missing test for empty events | medium | **ADDRESSED**: Two new tests pin both branches (self-authored → suppress; external-authored → surface). |
| 6 — lexicographic comparison comment | low | **ADDRESSED**: Added one-line comment in `diff_against_state` explaining the sentinel semantics. |
| 7 — test pollution via setUp/tearDown | low | **DEFERRED**: low-stakes hygiene; current pattern works. Will revisit if a test failure leaks state. |

**Test count:** 17 (10 SelfNoiseFilterTests + 7 AdversaryGapTests). All passing.

**Smoke test:** ran `tools/check_filed_issues.py --pretty --no-update` against live GitHub — 137 tracked issues, 0 NEW activity, 0 PRIORITY (state already current from morning cron). Also tested with stale state forced for #124+#127 (which Otter just edited): both correctly classified `self_only_issue_update=True`, `new_activity=False`. Filter works on live data for the rename+label case; per Gap 2 noted above, pure-description-only edits would have leaked under the V1 implementation but are now also covered by the empty-events heuristic.

**Smoke test gap (per adversary NOTE):** I have not manually validated the pure-body-edit case on a live issue — that would require a separate dedicated test issue. The unit test `test_empty_events_with_self_authored_is_suppressed_gap2` exercises the code path but not the live GitHub API behavior. Logging this as a follow-up validation.
