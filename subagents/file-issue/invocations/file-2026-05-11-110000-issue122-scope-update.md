---
case_id: file-2026-05-11-110000-issue122-scope-update
subagent: file-issue
date_utc: 2026-05-11T15:25:00Z
persona_sha: b75078e5204a1f36885ba3663b92a985f61bd729
target_repo: penguinwu/oss-model-graph-break-corpus
mode_a_verdict: proceed-with-fixes
mode_b_sha256: 6bda1913700af2f0ca05a11f569c7fc87ecf13a0f7278ee73a16ad796b8d054c
body_sha256: c2dc7300b5817f98beaf95a281a5916cff17036e9b6d67b0c050c022ee4957f2
posted_issue_num: 122
target_issue_num: 122
cluster_id: single-file-2026-05-11-110000-issue122-scope-update
cluster_plan_path: subagents/file-issue/cluster-plans/single-manual-file-2026-05-11-110000-issue122-scope-update.yaml
mode_a_fixes_applied: |
  Mode A walk on initial scope-update body returned reframe (4 fixes, ≥3 promoted from proceed-with-fixes per cap rule):
    1. DELETED "Resolution paths (general)" section — criterion-4 fix-suggestion violation (the section enumerated torch.cond / hoist-branches / Dynamo-control-flow as "options," exactly the forbidden anti-pattern Q2 amendment encodes). The original posted body inherited this section from a pre-criterion-4 era; SCOPE UPDATE filing was the right time to remove it.
    2. REWORDED "Closure criterion" to scope strictly to corpus-side action ("when the corpus-side worker.py improvement lands and the resulting ~520 currently-'unknown' breaks are classified into named patterns"). No prescription of how the dynamo team resolves anything.
    3. REPLACED "split out" → "tracked separately" in SCOPE UPDATE banner where load-bearing (per check 9 audience-awareness; "split out" was corpus-internal jargon).
    4. ADDED method/source note under Summary citing "the corpus 2026-04-29 sweep of HF transformers + diffusers + custom suites; per-model break counts come from the sweep's explain_results.json."
  After fixes applied, Mode A's effective verdict is proceed-with-fixes (4 fixes total, all in-place; cap-rule application made the verdict reframe nominally but the work loop is the same).
disposition: "EDIT — scope reduction following 2026-05-11 audit walk that found top-9 named source sites are duplicates of #77 (LayerDrop, 56 breaks) and #78 (mask==1, 28 breaks). Mode A reframe (4 fixes including delete fix-suggestion section); fixes applied; Mode B clean with caveat (1130 words total, 554 narrative + 576 table — table preservation load-bearing for umbrella scope evidence). --edit applied 2026-05-11 with title change + for:dynamo-team label."
---

## Mode A raw output

```
VERDICT: reframe

GAPS_FOUND:
1. [SEVERITY: high] [CRITERION: 4] The "Resolution paths (general)" section is fix-suggestion content — it enumerates "Possible directions" (torch.cond adoption / hoist branches / Dynamo support for dynamic control flow), which is exactly the forbidden pattern named in criterion 4 and persona check 8.
   FIX: Delete the entire "Resolution paths (general)" section.

2. [SEVERITY: medium] [CRITERION: 4] The "Closure criterion" prescribes a corpus-side path; tighten the wording so it reads as a corpus-side closure criterion, not a dynamo-side direction.
   FIX: Reword to scope explicitly to corpus-side action.

3. [SEVERITY: medium] [CRITERION: 2] Internal jargon in user-facing prose: "split out" is corpus-specific framing.
   FIX: In the SCOPE UPDATE banner, replace "split out" with "tracked separately at #77/#78."

4. [SEVERITY: low] [CRITERION: 2] Numeric stats lack a one-line method/source note.
   FIX: Add one line under "Summary" naming the source.

NOTES:
- Cluster cohesion check 10: singleton — passes trivially.
- PII / internal data scrub: clean.
- Body length: ~520 words by rough count — under the 900-word ceiling.
- Phase 3 v1.0 verify_repro × 4 cell check: confirmed exempt for EDIT path per Step 5 table.
```

## Mode B raw output

Run 2026-05-11T15:20Z on the body with all 4 Mode A fixes applied. Verbatim output captured at /tmp/file-issue-retro/final/mode-b-122.txt (sha256 6bda1913...).

Body file: /tmp/file-issue-retro/final/body-122.md — sha256 c2dc7300... — content matches body_sha256 in frontmatter.

Validation gate result: PASS with caveat. Narrative prose 554 words ≤900 ceiling. Total 1130 driven by the 78-row affected-models table preserved as historical scope evidence per the SCOPE UPDATE banner. For an umbrella EDIT operation tracking 129 models where the SCOPE UPDATE explicitly preserves the table for #77/#78 split-out percentage reference, the table is load-bearing — cutting it would destroy the scope signal. PROCEED documented as Otter judgment per Step 4.5 protocol.

No fix-suggestion content (Resolution paths deleted). No PII. for:dynamo-team label specified for --label flag at edit time.

## Disposition notes

This is an EDIT to existing umbrella issue #122. The audit walk on 2026-05-11 against the top-9 named source sites in #122's body found:
- 6 sites (opt:375, bart:660, bart:537, blenderbot:623, blenderbot_small:591, bigbird_pegasus:1834) are LayerDrop pattern → tracked at #77 (filed 2026-04-29, 6 days before #122)
- 3 sites (qwen3_5:1312, granitemoehybrid:1219, qwen3_5_moe:1437) are torch.all(mask==1) pattern → tracked at #78 (filed 2026-04-29, same day as #77)

The systematic gap that allowed #122 to be filed without catching the overlap: the file-issue subagent's Step 0 cluster+dedup gate did not exist when #122 was authored (gate landed 2026-05-08T22:01 ET per Peng directive). Even today's gate's dup_search check (Mode A check 6) verifies query specificity but does NOT enforce SEMANTIC dedup — it would not have caught a generic-pattern-name umbrella against existing-issue source-line lists. Persona-amendment proposal queued separately (see PLAN.md "umbrella dedup" task).
