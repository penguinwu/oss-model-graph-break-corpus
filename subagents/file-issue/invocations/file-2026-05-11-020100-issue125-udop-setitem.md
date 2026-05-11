---
case_id: file-2026-05-11-020100-issue125-udop-setitem
subagent: file-issue
date_utc: 2026-05-11T02:30:00Z
persona_sha: b75078e5204a1f36885ba3663b92a985f61bd729
target_repo: penguinwu/oss-model-graph-break-corpus
mode_a_verdict: reframe
mode_b_sha256: null
body_sha256: null
posted_issue_num: 125
target_issue_num: 125
cluster_id: single-file-2026-05-11-020100-issue125-udop-setitem
cluster_plan_path: subagents/file-issue/cluster-plans/single-manual-file-2026-05-11-020100-issue125-udop-setitem.yaml
disposition: "RETROACTIVE WALK — issue already posted via direct _proxy_api bypassing the subagent walk; Mode A finds reframe — body needs structural rewrite for [dynamo] audience (no MRE, fix-suggestion in 'What this issue is NOT', pervasive corpus jargon); Peng approval required before --edit"
---

## Mode A raw output

```
VERDICT: reframe

GAPS_FOUND:
1. [SEVERITY: high] [CRITERION: 1] No standalone MRE — body's reproduction is `python sweep/worker.py --model-json '{"name":"UdopEncoderModel",...}'` followed by `tools/run_experiment.py explain`. Both require the corpus harness. For a [dynamo]-tagged issue PT2 maintainers will read, this fails self-contained.
   FIX: Construct standalone MRE using `transformers` directly + torch.compile, with version pins and smallest input shape that triggers FX setitem failure. Validate via verify_repro × 4 cells (the EDIT-path exemption was for the original posted body; structural MRE rewrite re-engages the standard gate).

2. [SEVERITY: high] [CRITERION: 4] Fix-suggestion anti-pattern in "What this issue is NOT" section: "The maintainer should determine whether this is a Dynamo gap in `setitem` with multi-dimensional scalar-tensor indices, or a transformers-side input-shape issue exposed during fake-tensor tracing." Even framed defensively, this enumerates two possible directions — exactly the "Possible directions" forbidden pattern. Negation does not exempt.
   FIX: Delete the entire "What this issue is NOT" section.

3. [SEVERITY: high] [CRITERION: 2] Pervasive internal corpus-tooling jargon in user-facing prose. Summary uses "explain pass"; Notes use "Identify pass succeeds" / "Explain pass crashes"; Reproduction cites `sweep/worker.py` + `tools/run_experiment.py explain` as load-bearing repro path; "2 pairs" / "modellibs" are internal terminology. For a [dynamo]-tagged issue, structural — not 1-2 strings to rephrase.
   FIX: Rewrite Summary as plain English. Replace "Identify/Explain pass" framing with "model loads + runs eager fine; torch.compile trace fails with the stderr above." Move corpus-internal repro commands to "Source" section; lead with standalone MRE.

4. [SEVERITY: medium] [CRITERION: 3] Bare ISO dates without regression context — "2026-05-03 + 2026-05-09 sweeps" in Summary and Notes. Stable failure across two sweeps; dates are scaffolding, not signal.
   FIX: Replace with "Stable failure observed across multiple recent sweeps" or drop. Move sweep-dir references to Source section.

5. [SEVERITY: medium] [CRITERION: 2] Truncated stderr in "Error" section ends mid-sentence. Maintainer wonders what's hidden.
   FIX: Either show full stderr in `<details>` block or truncate cleanly with "[truncated — N more entries]" note.

6. [SEVERITY: low] [CRITERION: 2] Title uses "2 pairs" — internal corpus terminology. PT2 maintainers won't parse "pair" as (model, mode).
   FIX: Retitle to "[dynamo] FX setitem on FakeTensor fails for UdopEncoderModel (eval + train)".

REFRAME_INSTRUCTION: (a) standalone MRE that imports transformers + torch.compile directly — no sweep/worker.py dependency in load-bearing repro; (b) delete "What this issue is NOT" section entirely; (c) strip corpus-internal jargon ("explain pass", "identify pass", "pairs", "modellibs", bare sweep dates) from Summary, Error, Notes — move corpus-internal repro commands and sweep dates to a "Source" section as paths/references. After rewrite, verify_repro × 4 cells. Re-invoke Mode A on revised draft before applying in-place edit.

NOTES:
- Dup-search distinction (closed #4 install-time vs this Dynamo-tracing-time) verified correct against live body's stderr. Dedup classification stands.
- "What this issue is NOT" section appears to be Otter's anticipatory defense against criterion 4. It IS the criterion-4 violation. Defensive framing that enumerates possible causes IS speculation, regardless of label.
- Repro reliance on sweep/worker.py is the most consequential defect — even before the [dynamo] tag forces strict criterion 1, a maintainer who clicks the issue cannot reproduce without joining a different project. Justifies reframe.
```

## Mode B raw output (if reached)

NOT REACHED — Mode A reframe blocks Mode B until draft is rewritten + verify_repro × 4 cells run + Mode A re-cleared.

## Disposition notes

RETROACTIVE walk on already-posted issue 125. Issue posted 2026-05-11 ~01:30 ET via direct _proxy_api, bypassing subagent walk. Peng directive 2026-05-10 22:13 ET requires the gates be applied retroactively.

Mode A via Agent (general-purpose) — verdict **reframe** with 6 gaps, 3 high-severity. The most consequential gap is no standalone MRE for a [dynamo]-tagged issue: PT2 maintainers cannot reproduce without cloning the corpus repo. The "What this issue is NOT" defensive-framing section is the second structural defect — it enumerates fix directions while claiming not to.

**Next steps (deferred to Peng's morning review):**
1. Construct standalone MRE — minimal `transformers` + `torch.compile` invocation that triggers FX setitem on FakeTensor for UdopEncoderModel.
2. Run verify_repro × 4 cells (current + nightly venvs × original_command + MRE).
3. Rewrite body addressing all 6 gaps.
4. Re-run Mode A on rewrite.
5. Run Mode B for final body.
6. Surface body + diff to Peng for approval.
7. On approval: `tools/file_issues.py corpus-issue --via-skill <case_id> --cluster-plan-approved <token> --edit 125 --body ...` (with title change).

The MRE construction is non-trivial — UdopEncoderModel may need specific input shapes / a particular forward path to trigger the FakeTensor setitem failure. Tomorrow: probe whether minimal `model = UdopEncoderModel(config); torch.compile(model, fullgraph=True)(...)` reproduces, or whether the failure is shape-specific.
