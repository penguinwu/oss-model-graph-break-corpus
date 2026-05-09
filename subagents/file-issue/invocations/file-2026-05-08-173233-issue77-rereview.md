---
case_id: file-2026-05-08-173233-issue77-rereview
parent_case_id: file-2026-05-08-170223-issue77-review
date_utc: 2026-05-09T00:32:33Z
target_repo: penguinwu/oss-model-graph-break-corpus
issue_type: existing-issue-review
review_target_url: https://github.com/penguinwu/oss-model-graph-break-corpus/issues/77
persona_sha: ab15fd3c032df9ea08c84a73b7358dce773069ca
draft_path: /tmp/file-issue-file-2026-05-08-173233-issue77-rereview-draft.md
draft_sha256: 964defbeb3cf6e97865a091318f1256657c0d21d4e783623e349456a588f4b25
validation_file_path: /tmp/file-issue-file-2026-05-08-170223-issue77-review-validation.md
validation_sha256: 5a2e55ddb9aa6fe47589e2ae0ea3bab3efc32f3e8076022e5a7cde8c8459234b
mode_a_verdict: reframe
mode_a_sha256: d656d5e2e3ec1c588284a3ef5fe47ceaf73cd1e349e1acae64f8a7f7b028711e
mode_a_fixes_applied: null
mode_b_sha256: null
body_sha256: null
footer_marker: null
posted_url: blocked at Mode A (reframe — pending Peng decision on rewrite scope)
---

> **Second invocation on the same issue (#77).** Re-run after persona amendment in commit `ab15fd3` (criterion #4 redefined; fix-suggestion content forbidden). The chained `parent_case_id` links to the first invocation. This is the verification step for the Phase 1.2 amendment.

## Pre-registered predictions vs reality

| Predicted | Reality | ✓/✗ |
|---|---|---|
| Gap #1 from first invocation ("no `single_fix_claim`") DROPS | Confirmed dropped — the field was renamed to `repro_strategy` and the new gap is about repro feasibility, not fix-prescription | ✓ |
| Gap #5 from first invocation ("3-direction enumeration dilutes the ask") DROPS in its old form | Confirmed dropped — replaced by the new check-#8 catch which treats "Possible directions" as a HARD-BLOCK forbidden section, not a "dilutes" complaint | ✓ |
| New gap from check #8 (Possible directions = forbidden anti-pattern) APPEARS | Confirmed — gap #1 of this invocation, severity HIGH | ✓ |
| Gaps #2 / #3 / #4 / #6 / #7 from first invocation REMAIN | Confirmed — no MRE (gap #2), stale counts (gap #3), missing env (gap #4), PII (gap #5), title polish (gap #6+#7) | ✓ |
| Verdict stays `reframe` | Confirmed (7 gaps; well past the `proceed-with-fixes` cap of 3) | ✓ |
| Rewrite proposal does NOT include "Pick ONE direction" / "Recommendations" / any fix-suggestion content | Confirmed — Mode A self-noted: "I deliberately produced NO 'Pick ONE direction' content, NO 'Recommendations' content, NO triage prescription. The amendment took." | ✓ |

**6/6 predictions confirmed.** The persona amendment took effect cleanly.

## Diff vs first invocation

| | First invocation (pre-amendment) | Re-invocation (post-amendment) |
|---|---|---|
| Verdict | `reframe` | `reframe` |
| Total gaps | 7 | 7 |
| Top-priority gap | "No single_fix_claim" — Otter encouraged to pick a fix | "Body contains forbidden 'Possible directions' section" — explicit anti-pattern catch |
| Rewrite proposal includes a "Proposed fix" section | YES (recommended HF-side branchless rewrite as the primary fix) | NO — explicitly notes maintainer decides |
| Body skeleton recommended | 7 sections including "Proposed fix" + "Considered alternatives" | 6 sections (Symptom → Reproduction → Observed vs expected → Source → Environment → Affected models) — NO fix section |

The structural defect Peng caught (Mode A reinforcing the fix-suggestion anti-pattern) is no longer present.

## Mode A raw output

See `/tmp/file-issue-mode-a-output-2026-05-08-173233.txt` (sha256 matches frontmatter `mode_a_sha256` field).

## Disposition

**PENDING Peng decision** on whether to proceed with the rewrite. Same authority gate as first invocation (corpus-repo edit; Otter creates + surfaces, but editing an EXISTING live issue body is a distinct operation that probably needs a separate gate language — flagged in the first case file's "lessons" section).

If Peng approves the rewrite, the work is now well-scoped:
1. Delete the "Possible directions" section + the "(We can drive an HF PR...)" volunteer line (gap #1)
2. Construct a 5-15 line MRE based on Mode A's suggested skeleton (gap #2) — needs the actual Bart forward signature; will need to verify the MRE actually triggers the break before posting
3. Reconcile aggregate counts (gap #3 — option (b) "reduce specificity to what was validated" is cheaper)
4. Add Environment block (gap #4)
5. Scrub `/home/pengwu/...` paths in any quoted break_reason text (gap #5)
6. Tighten title (gap #6)
7. Trim verbose framing (gap #7)

Then invoke Mode B for the rewritten body (first ever Mode B invocation; will surface its own lessons).

## Lessons captured for next RETROSPECTIVE entry

1. **Persona amendment took effect cleanly on first re-invocation.** Pre-registering predictions made the verification rigorous — 6/6 confirmed. Worth doing every time persona is materially amended.
2. **Mode A's self-narration is useful.** It explicitly noted "I deliberately produced NO 'Pick ONE direction' content..." — meta-commentary that confirms the amendment is internalized, not just superficially dodged.
3. **The "no fix suggestion" rule is harder than it looks.** First-invocation Mode A clearly DID know fix-suggestions are dispreferred (it framed gap #1 as "no single_fix_claim" — i.e., it wanted MORE fix-claim, not less). The persona's notion of "actionable" was the load-bearing defect, NOT a missing rule against fix-suggestions per se. The fix had to redefine the criterion, not just add a new prohibition.
4. **The forbidden-headers list is the mechanical enforcement.** Mode A check #8's enumeration ("Proposed fix", "Possible directions", "Suggested fix", etc.) caught the existing body's section by exact-name match. Without that enumeration, the persona might have said "this proposes a fix" softly without flagging it as HARD-BLOCK. The lesson: enumerate forbidden patterns explicitly, don't rely on the persona inferring them from criterion redefinition alone.
