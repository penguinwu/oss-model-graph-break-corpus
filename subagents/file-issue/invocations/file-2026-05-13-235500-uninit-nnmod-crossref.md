---
case_id: file-2026-05-13-235500-uninit-nnmod-crossref
subagent: file-issue
date_utc: 2026-05-13T23:55:00Z
persona_sha: b8ffa9d3-jsonl-greppable-shipped
target_repo: penguinwu/oss-model-graph-break-corpus
mode_a_verdict: reframe
mode_b_sha256: null
body_sha256: null
posted_issue_num: null
target_issue_num: null
related_issue: 27
cluster_id: single-manual-file-2026-05-13-235500-uninit-nnmod-crossref
cluster_plan_path: subagents/file-issue/cluster-plans/single-manual-file-2026-05-13-235500-uninit-nnmod-crossref.yaml
disposition: ABANDONED — Q2(a) cross-ref filing not separable from #27 via MRE. The synthetic MRE under fullgraph=True HALTS on #27's primary break before reaching the target "Uninitialized nn.Module" downstream break. Downstream break only manifests in sweep (fullgraph=False) where multiple sequential breaks are logged. Surfaced to Peng for revised Q2 path decision.
---

## Mode A raw output

```
VERDICT: reframe

GAPS_FOUND:
1. [SEVERITY: high] [CRITERION: 4] MRE verification signal fingerprints WRONG break_reason. Body title is "Uninitialized nn.Module" but MRE's expected_signal keys off "torch.nn.Parameter() constructor with Dynamo" (#27's break). The MRE may not actually reach the Uninitialized nn.Module break — it might halt on the #27 break first.
   FIX: re-anchor the MRE expected_signal on the target break OR construct an MRE that bypasses #27's break to surface Uninitialized nn.Module in isolation.

2. [SEVERITY: high] [CHECK: 12] Per-source-line dedup tool was NOT run before Mode A invocation. Draft says "pending Mode A FIX". Per check 12, tool MUST be run BEFORE Mode A.

3. [SEVERITY: medium] [CRITERION: 3] Repro status line under-reports verification coverage.

NOTES:
- Check 13 deeper concern: if MRE can't reproduce target break in isolation, filing's separation from #27 is not justified mechanically; better disposition may be a comment on #27 rather than separate issue. Q2(a) approved separate filing assuming MRE would verify target signal; if it doesn't, escalate back to Peng.
```

## Disposition notes — Mode A finding investigated; ABANDON path chosen

Mode A surfaced the deep concern explicitly. I investigated by running the MRE manually with `~/envs/torch-nightly-cu126/bin/python3` + `fullgraph=True`:

```
$ python3 /tmp/mre_uninit_test.py
EXC: Unsupported: Attempted to use `torch.nn.Parameter()` constructor with Dynamo
  Hint: Try to construct `torch.nn.Parameter()` outside the compiled region.
```

**Confirmed:** the MRE halts on #27's primary nn.Parameter ctor break. It NEVER reaches the "Uninitialized nn.Module" downstream break under `fullgraph=True`. The downstream break only manifests in the **sweep** because the sweep uses `fullgraph=False` (graph_break=allowed) — under `fullgraph=False`, dynamo logs the #27 break, falls back to eager, then attempts to trace the same partially-constructed `BigBirdPegasusSelfAttention` instance and emits the Uninitialized nn.Module break as the second sequential break in the same trace.

This means the Q2(a) cross-ref filing as designed has no clean MRE-based isolation. The "Uninitialized nn.Module" break is **structurally inseparable from #27** in any synthetic MRE form — they are sequential consequences of the same root cause in the same compile-trace.

**Implications for Q2 decision (surfaced to Peng):**

The Q2 question Peng decided 2026-05-13 18:50 ET ("(a) strict per-rule REJECT [of #27 EDIT] + file separate 1-line cross-ref issue") assumed the cross-ref could be cleanly separated. Today's investigation shows it can't be — the cross-ref filing would carry the SAME MRE as #27 but claim a different target signal that the MRE doesn't actually reach standalone.

Three revised paths Peng could take:
- **(a-revised) Comment on #27** (NOT a separate issue): a 1-paragraph comment on #27's thread noting "models also hit Uninitialized nn.Module as a downstream consequence in the same trace; reproducible via the sweep's fullgraph=False mode but not via this issue's fullgraph=True MRE." This achieves the maintainer-discoverability goal (comment is searchable) without the check-13 violation OR the MRE-fingerprint-mismatch problem. **Requires SKILL extension** (file-issue subagent doesn't have a `--comment` mode today; tracked as Phase-2 RETROSPECTIVE candidate). Lean: this.
- **(b) Iterate check-13 carve-out** (the original Q2(b) path) — explicitly allow the EDIT to #27 with the dual-manifestation subsection under a strict shape (named as "Downstream observations in the same trace", no fix content, root-cause attribution to existing primary break). The persona-level work to amend check 13 is small (~1 paragraph addition). Re-walk #27 EDIT under the carve-out.
- **(c) Drop the cross-ref** entirely — the maintainer will discover the dual manifestation when investigating #27 (it's in the sweep evidence #27 already references). Cheapest path; loses some discoverability for someone searching "Uninitialized nn.Module" specifically.

Lean: **(a-revised)**. The information has value (helps a maintainer searching the symptom land at the right root cause) but the comment form is the right fit — separable from the MRE, discoverable by GitHub repo search, no check-13 tension. The SKILL extension to support comment-posting is real work but valuable for this and future similar cross-ref needs.

retrospective check: this is the 5th file-issue invocation today. The 4-invocation retrospective entry was added earlier; this 5th invocation's REFRAME-then-ABANDON is captured here for the next retrospective cycle. Key learning: **the Q2 decision pattern was structurally faulty — the proposed cross-ref filing assumed a separable MRE that doesn't exist for in-trace downstream breaks. Future similar Q2-style decisions need a "is the cross-ref MRE-isolable?" check BEFORE Peng commits to a path.**
