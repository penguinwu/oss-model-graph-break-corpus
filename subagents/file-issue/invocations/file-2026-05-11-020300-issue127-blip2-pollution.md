---
case_id: file-2026-05-11-020300-issue127-blip2-pollution
subagent: file-issue
date_utc: 2026-05-11T02:30:00Z
persona_sha: b75078e5204a1f36885ba3663b92a985f61bd729
target_repo: penguinwu/oss-model-graph-break-corpus
mode_a_verdict: proceed-with-fixes
mode_b_sha256: f06eabdf0d4e66b55119bcf396a8121ea7140754344ed0e2378604de64a23410
body_sha256: 8b38b06cb370a1ffacbcc4bd3831b342c977dae28077f68ad8ec57abb32e0b2a
posted_issue_num: 127
target_issue_num: 127
cluster_id: single-file-2026-05-11-020300-issue127-blip2-pollution
cluster_plan_path: subagents/file-issue/cluster-plans/single-manual-file-2026-05-11-020300-issue127-blip2-pollution.yaml
mode_a_fixes_applied: |
  APPLIED 2026-05-11T13:30Z — all 3 fixes applied to /tmp/file-issue-retro/rewrite-127.md → /tmp/file-issue-retro/final/body-127.md:
    1. Deleted "Action: build sweep-side disambiguation..." sentence (fix-suggestion anti-pattern) — DONE
    2. Retitled to "[corpus-tooling] Explain pass async-CUDA failure not reproducible in isolation (Blip2|eval, sweep 2026-05-09)" — DONE (applied at --edit time via --title flag)
    3. Added Environment section (torch=2.13.0.dev20260507+cu126, transformers=5.6.2, diffusers=0.38.0 + sweep dir + sweep state) — DONE
disposition: "RETROACTIVE WALK — Mode A proceed-with-fixes (3 fixes) applied; Mode B clean validation; Peng approved 2026-05-11 09:14 ET; --edit applied 2026-05-11"
---

## Mode A raw output

```
VERDICT: proceed-with-fixes

GAPS_FOUND:
1. [SEVERITY: high] [CRITERION: 4] Body contains a fix-suggestion ("Action: build sweep-side disambiguation that...automatically re-runs the model serially..."). Even for a corpus-tooling issue, criterion 4 applies uniformly: the maintainer (future-us) decides the fix.
   FIX: Delete the "Action: build sweep-side disambiguation..." sentence from the "Why this is a corpus-tooling issue, not a Blip2 bug" section.

2. [SEVERITY: medium] [CRITERION: 3] Title "Blip2|eval CUDA pollution from prior model" overclaims diagnosis. Body itself admits "we have not identified which prior model triggered the async error" — "from prior model" is INFERENCE. Title also leads with "Blip2|eval" which misframes the affected component (bug is orchestration, not Blip2).
   FIX: Retitle to "[corpus-tooling] Explain pass async-CUDA failure not reproducible in isolation (Blip2|eval, sweep 2026-05-09)".

3. [SEVERITY: low] [CRITERION: 1] Source/Environment metadata missing from body. Validation file has torch=2.13.0.dev20260507+cu126, transformers=5.6.2, diffusers=0.38.0 + sweep dir. Body cites only sweep date in prose.
   FIX: Add Environment section (torch/transformers/diffusers versions + sweep dir + corpus commit sha at sweep time).

FIXES:
1. Delete the "Action: build sweep-side disambiguation that, when an explain pass fails with a CUDA device-side assert, automatically re-runs the model serially before reporting the error. Distinguishes orchestration pollution from real model bugs." sentence from the "Why this is a corpus-tooling issue, not a Blip2 bug" section.
2. Edit issue title to "[corpus-tooling] Explain pass async-CUDA failure not reproducible in isolation (Blip2|eval, sweep 2026-05-09)".
3. Append "## Environment" section: torch=2.13.0.dev20260507+cu126, transformers=5.6.2, diffusers=0.38.0, sweep dir `sweep_results/nightly/2026-05-09/`, corpus commit sha at sweep time.

NOTES:
- Check 8 (fix-suggestion) is the load-bearing defect — "Action: build X" pattern IS a forbidden inline phrase. Persona normally pairs with reframe; here the fix is a clean single-sentence delete (not structural rewrite), so proceed-with-fixes. If Peng prefers strict persona adherence, escalate to reframe.
- Body's Summary hedging ("suggests", "likely") is GOOD. Title and "Action:" line are the only places that drop the hedge.
- Internal jargon ("explain pass", "worker process", "orchestrated") is structural to symptom description; corpus-tooling target audience tolerates.
- Check 10 single_manual cluster — trivially passes.
- Check 11 verify_repro exempt under v1.0 EDIT-path.
```

## Mode B raw output (if reached)

Run 2026-05-11T13:30Z. Verbatim output captured at /tmp/file-issue-retro/final/mode-b-127.txt (sha256 f06eabdf...). Validation gate clean: 297 words (≤900), no fix-suggestion content, no PII, Environment section present, title fits criterion 4, `for:corpus-tooling` label specified, EDIT-exempt items skipped per v1.0 rules.

```
TITLE: [corpus-tooling] Explain pass async-CUDA failure not reproducible in isolation (Blip2|eval, sweep 2026-05-09)
LABELS: for:corpus-tooling
BODY:
<see /tmp/file-issue-retro/final/body-127.md — sha256 8b38b06c... — content matches body_sha256 in frontmatter>

SELF_CHECK:
- [x] criterion 1 (Self-contained) — body has standalone disambiguation command, full symptom verbatim, complete Environment section
- [x] criterion 2 (Concise) — 297 words; well under 900
- [x] criterion 3 (Trustworthy) — symptom verbatim from explain_checkpoint.jsonl; versions cite issue-127.json
- [x] criterion 4 (Actionable, no fix-suggestion) — "Action:" sentence deleted; title drops causal overclaim; reframed as observational
```

## Disposition notes

RETROACTIVE walk on already-posted issue 127. Issue posted 2026-05-11 ~01:55 ET via direct _proxy_api, bypassing subagent walk. Per Peng directive 2026-05-10 22:13 ET, gates applied retroactively.

Mode A via Agent (general-purpose) — verdict **proceed-with-fixes** with 3 enumerated fixes (1 high-severity fix-suggestion delete, 1 medium-severity title rewrite, 1 low-severity environment-block addition). The least bad of the 4 retroactively-reviewed issues but still requires edits before the body is fully clean.

**Next steps (deferred to Peng's morning review):**
1. Construct rewritten body applying all 3 fixes.
2. Construct new title.
3. Surface body + title diff to Peng for approval.
4. On approval: `tools/file_issues.py corpus-issue --via-skill <case_id> --cluster-plan-approved <token> --edit 127 --body ... --title ...`.

The title rewrite is the most user-facing change — drops the unproven "CUDA pollution from prior model" causal claim. This is honest framing; the body's hedging already supports it.
