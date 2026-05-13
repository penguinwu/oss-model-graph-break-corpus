---
case_id: file-2026-05-13-204500-issue25-mre-fullgraph
subagent: file-issue
date_utc: 2026-05-13T21:00:00Z
persona_sha: 84ef632944d4575d713f3adde1be7d52baeb2495
target_repo: penguinwu/oss-model-graph-break-corpus
mode_a_verdict: proceed-with-fixes
mode_b_sha256: pending
body_sha256: cd337fbb326ffa6d20dde7e59f369c4043e918972b37cfb75cdc323110fda2c2
posted_issue_num: 25
target_issue_num: 25
cluster_id: single-file-2026-05-13-204500-issue25-mre-fullgraph
cluster_plan_path: subagents/file-issue/cluster-plans/single-manual-file-2026-05-13-204500-issue25-mre-fullgraph.yaml
mode_a_fixes_applied:
  - "FIX 1: appended explicit grep-hit line to validation file Evidence 1 confirming 'unsupported function.__get__ call' verbatim in MRE stderr (provenance from MRE run, not just sweep break_reason)"
  - "FIX 2: ran tools/dedup_source_lines.py --source-lines transformers/models/recurrent_gemma/modeling_recurrent_gemma.py:655,656 --exclude-issue 25 → 0 overlap across 54 open [dynamo] issues"
  - "FIX 3: title↔table coupling resolved via option (a) couple-update — refresh both title and Affected Models table to current 2026-05-09 sweep counts (2 models, 6 breaks; 4 + 2 by mode)"
disposition: pending — case file written; about to post EDIT to issue 25 via tools/file_issues.py corpus-issue --edit
---

## Mode A raw output

```
VERDICT: proceed-with-fixes

GAPS_FOUND:
1. [SEVERITY: medium] [CRITERION: 3] Validation file Evidence 1 shows "Verbatim stderr (last lines)" but the load-bearing fragment "unsupported function.__get__ call" is NOT in the captured snippet — only indirectly referenced via the gb9277 URL. The fragment IS present in Evidence 3 (sweep break_reason), so source-of-truth resolves, but the MRE's own stderr capture should directly contain the substring the verification signal greps for.
   FIX: re-grep the full Evidence 1 stderr for "unsupported function.__get__ call" and append the matching line to the validation file so Mode B's exit_nonzero+stderr_contains signal block has direct provenance from the MRE run, not just from sweep break_reason text.

2. [SEVERITY: medium] [CRITERION: other / dedup discipline check 12] Draft cites "dedup_search.py confirmed 0 candidates" but Mode A check 12 requires tools/dedup_source_lines.py --draft <path> --exclude-issue 25. The body cites transformers/models/recurrent_gemma/modeling_recurrent_gemma.py:655 — exactly the source-line citation pattern check 12 governs.
   FIX: run tools/dedup_source_lines.py --draft <path> --exclude-issue 25 and append the zero-overlap confirmation to the draft's dup_search section.

3. [SEVERITY: low] [CRITERION: 2] Title↔body coupling: title currently says "10 breaks"; CHANGE 3 (optional) refreshes to 6; body's existing Affected Models table sums to 10 (already stale vs Evidence 3's 6). If Mode B refreshes the table to 6 without retitling, body and title will disagree on a top-line scope number.
   FIX: couple the choice — either (a) retitle to "6 breaks" AND refresh the table to 2+1+2+1=6, OR (b) leave both at 10. Don't half-update.

FIXES:
1. Append to validation file Evidence 1 the explicit grep-hit line showing "unsupported function.__get__ call" appears in the MRE's actual stderr.
2. Run dedup_source_lines.py and add the result confirmation to the draft.
3. Decide title↔table coupling; resolved as (a) couple-update.

NOTES:
- CHANGES 1, 2, 4 clean; Evidence 1 confirms exit-nonzero hard error supporting CHANGE 2; Evidence 4 confirms tx 5.8.0 lines 655-656 unchanged supporting CHANGE 4.
- Repro status line should be refreshed to torch 2.13.0.dev20260507+cu126 + new UTC timestamp per Phase 3 v1.0 body shape rule.
- PII watch: Evidence 3 contains /home/pengwu/envs/modellibs/transformers-5.6.2/... — Mode B must not echo verbatim. Body should keep transformers/models/recurrent_gemma/...:655 form.
- Fix-suggestion anti-pattern (check 8) CLEAN.
- EDIT path exempt from Phase 3 v1.0 frontmatter check (check 11).
- Defensive-disclaimer pattern (check 8 broader): EDIT does NOT introduce any "What this issue is NOT" section. PASS.
```

## Mode B raw output

```
TITLE: [dynamo] function.__get__ descriptor binding not traceable (RecurrentGemma, 2 models, 6 breaks)

LABELS: for:dynamo-team

BODY:
[full body persisted at /tmp/file-issue-file-2026-05-13-204500-issue25-mre-fullgraph-body.md ; sha256 cd337fbb326ffa6d20dde7e59f369c4043e918972b37cfb75cdc323110fda2c2]

SELF_CHECK:
- [x] Self-contained (criterion 1) — MRE has zero transformers dependency; only import torch + stdlib + torch.nn; runs standalone.
- [x] Concise (criterion 2) — every section earns its place; no preamble or hedging. Word count: ~340 words.
- [x] Trustworthy (criterion 3) — every numeric/string claim cites the validation file: 6 breaks across 4 pairs (Evidence 3), torch 2.13.0.dev20260507+cu126 + UTC timestamp (validation header), transformers 5.6.2 (Evidence 3) + 5.8.0 lines 655-656 unchanged (Evidence 4), exit-nonzero verification fragment verbatim (Evidence 1 addendum).
- [x] Actionable / reproducible (criterion 4) — body contains repro + symptom + source only; NO fix-suggestion content; maintainer decides the fix.
```

## Disposition notes

EDIT to existing corpus issue 25 (function.__get__ in RecurrentGemma). Bug confirmed NOT fixed via three-way evidence: synthetic MRE under fullgraph=True hits gb9277 hard error; latest 2026-05-09 sweep shows all 4 RecurrentGemma pairs still graph_break with verbatim "unsupported function.__get__ call" at lines 655-656; transformers 5.8.0 source unchanged at the same lines.

EDIT scope: refresh MRE to use fullgraph=True (so failure is observable as hard error, not soft graph break); update verification signal kinds to exit_nonzero+stderr_contains; refresh Repro status to current torch + transformers; refresh Affected Models table to current sweep counts (6 breaks); refresh Source block to point at 2026-05-09 sweep instead of stale animesh-fullgraph-2026-04-28; add note on tx 5.8.0 source survival.

Mode A: proceed-with-fixes (3 fixes, at cap; all in-place applied without re-invoke).
Mode B: clean SELF_CHECK on all 4 criteria.

retrospective check: this is the first file-issue invocation since 2026-05-11 evening (#125 + #126 retro EDITs). RETROSPECTIVE.md cadence: still within "every 3 issues" — no mid-pipeline retrospective triggered.
