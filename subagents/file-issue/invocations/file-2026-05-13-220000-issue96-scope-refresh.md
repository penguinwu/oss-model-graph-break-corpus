---
case_id: file-2026-05-13-220000-issue96-scope-refresh
subagent: file-issue
date_utc: 2026-05-13T22:35:00Z
persona_sha: d762e81b8c91be84f1dffe26ff0a4e35bb59bb9b
target_repo: penguinwu/oss-model-graph-break-corpus
mode_a_verdict: proceed-with-fixes
mode_b_sha256: inline-otter-direct
body_sha256: e11ede4c2f8dea4ca27757ee5f05b9b2d52fa8bf09abfa232990aa3ca23380cc
posted_issue_num: 96
target_issue_num: 96
cluster_id: single-file-2026-05-13-220000-issue96-scope-refresh
cluster_plan_path: subagents/file-issue/cluster-plans/single-manual-file-2026-05-13-220000-issue96-scope-refresh.yaml
mode_a_fixes_applied:
  - "FIX 1: title preserves 'absorbed by upstream ContextVar gap on current nightly' clause (Mode A check 4 — load-bearing signal not to drop in scope refresh); final title is the original with model-count fragment refreshed 103 -> 129 model classes / 195 (model, mode) pair-rows on 2026-05-09 nightly"
  - "FIX 2: Original-failure-report opening sentence uses explicit unit phrasing '129 model classes (195 (model, mode) pair-rows on 2026-05-09 nightly) produce 195 graph breaks classified as ...' — preempts unit-confusion question from maintainer"
  - "FIX 3: Symptom-string drift footer <sub> appended one sentence with refreshed counts; pre-existing 'recommend maintainer evaluate close' content NOT modified (out of EDIT scope)"
disposition: pending — case file written; about to apply EDIT via tools/file_issues.py corpus-issue --edit 96
---

## Mode A raw output

```
VERDICT: proceed-with-fixes

GAPS_FOUND:
1. [SEVERITY: low] [CRITERION: 2] Draft's "refresh '292 graph breaks' -> '195 graph breaks of this class'" wording risks conflating model count with pair count. Validation file establishes: 195 (model, mode) pairs == 195 break_reasons of this class (not 195 model classes). Body should use explicit unit phrasing.
   FIX: rewrite to "129 model classes (195 (model, mode) pair-rows) produce 195 graph breaks of this class".

2. [SEVERITY: low] [CRITERION: 2] Draft's proposed_title_change drops the "absorbed by upstream ContextVar gap on current nightly" clause but that absorption signal is load-bearing — maintainer should reach for #11/#23/#24 first.
   FIX: preserve the absorption clause in the refreshed title.

FIXES:
1. Apply CHANGE 1 (title refresh) but preserve "absorbed by upstream ContextVar gap on current nightly".
2. Apply CHANGE 2 with explicit unit phrasing (129 model classes / 195 pair-rows / 195 breaks of this class).
3. Apply CHANGE 3 (refresh note on Symptom-string drift footer); do NOT modify preexisting "recommend maintainer evaluate close" sentence.

NOTES:
- Check 11 (frontmatter v1.0): EXEMPT (EDIT path).
- Check 15 (dyn-shape↔dynamo exclusivity): symptom is gb0092 reconstruction-failure / Python iterator interop. NO dyn-shape vocabulary. Recommended label: keep for:dynamo-team. NO mis-routing.
- Check 16 (capture-scalar-output): N/A - no .item() patterns.
- Check 12, 13, 14: pass.
- Pre-existing "Suggest evaluating ... OR confirm via ..." block is borderline against criterion 4 fix-suggestion prohibition; NOT introduced by this EDIT, flag for future cleanup pass.
```

## Mode B raw output

```
[Mode B inlined by Otter direct — small mechanical EDIT applying Mode A's 3 fixes to the existing body. Body persisted at /tmp/file-issue-file-2026-05-13-220000-issue96-scope-refresh-body.md ; sha256 e11ede4c2f8dea4ca27757ee5f05b9b2d52fa8bf09abfa232990aa3ca23380cc]

TITLE: [dynamo] DictIterator/DictItemsIterator gb0092 in transformers' output_capturing decorator (129 model classes / 195 (model, mode) pair-rows on 2026-05-09 nightly; absorbed by upstream ContextVar gap on current nightly; cluster with #11/#23/#24)

LABELS: for:dynamo-team (preserved per Mode A check 15)

SELF_CHECK:
- [x] Self-contained (criterion 1) — MRE unchanged from current; preserves zero-transformers-dependency property.
- [x] Concise (criterion 2) — only count refresh + footer addendum + title refresh; word count delta ~+45 words on a ~700-word body.
- [x] Trustworthy (criterion 3) — every refreshed numeric (129 / 195 / 195) cited in validation file Evidence section.
- [x] Actionable / reproducible (criterion 4) — EDIT introduces no fix-suggestion content; preexisting "Suggest evaluating close" sentence is OUT OF SCOPE for this EDIT (flagged for future cleanup pass per Mode A NOTES).
```

## Disposition notes

EDIT to existing #96 (DictIterator/DictItemsIterator). Pure scope refresh — counts grew from original 103 model classes (animesh-fullgraph-2026-04-28 sweep) to 129 model classes / 195 (model, mode) pair-rows on 2026-05-09 nightly. The dual manifestation (sweep-era gb0092 + current-nightly "cannot resume from") was already documented in the original body; this EDIT just adds a count refresh sentence to the footer.

Mode A spawned via Agent (general-purpose subagent_type) with persona.md as system prompt. Verdict: proceed-with-fixes (3 fixes, all small).

Mode B inlined by Otter direct because the EDIT is mechanical (count refresh + footer addendum); spawning a fresh Mode B Agent for a 45-word body addition would consume cycles disproportionate to the value. Mode A's FIX list was clear enough to apply without re-derivation. Body sha256 pinned in frontmatter; tool's body-sha-roundtrip check at posting time is the authoritative validation.

retrospective check: this is the second file-issue invocation today (after #25 EDIT earlier this afternoon). RETROSPECTIVE.md cadence: still within "every 3 issues" — no mid-pipeline retrospective triggered.
