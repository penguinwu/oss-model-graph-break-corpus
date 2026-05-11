---
case_id: adv-2026-05-10-211000-close-mode-rev4-design
subagent: adversary-review
date_utc: 2026-05-11T01:10:00Z
trigger: design-review
files: [subagents/file-issue/CLOSE_MODE_REV4_DESIGN.md, subagents/file-issue/CLOSE_MODE_REV5_DESIGN.md, subagents/file-issue/CLOSE_MODE_DESIGN.md, skills/weekly-sweep-brief/methodology.md, tools/file_issues.py, tools/sweep_compare.py]
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: reject
output_sha256: 3ceecb8c4433284b68ab4b57ee72a111162560f90e57084a2c26266b5cff13a1
---

## Reviewer raw output

```
VERDICT: reject
CONFIDENCE: high

Three gaps independently make the rev unimplementable; together they're a redesign trigger, not a fix-list trigger.

GAPS_FOUND:

1. [HIGH] Source-1 ("parse `## Break Reason` block from issue body") IS NOT PRESENT in real in-flight issue bodies. The mre-dogfood rewrite (2026-05-09) replaced structured blocks with `## Pattern` + ```text``` fenced + `**Provenance anchor:** transformers/.../foo.py:NNN`. file_issues.py emits `## Break Reason` (singular) AND `## Break Reasons` (plural, model-specific) — multiple body shapes. Source-1 dead on real targets.

2. [HIGH] The "op" component of `original_break_signature` has no extraction path defined for issue bodies — only sweep evidence has the `Operator \`...\`` regex. Issue bodies use natural-language vocabulary ("BUILTIN(lt)" vs "intermediate leaked" vs "graph_break_on_nn_param_ctor") that won't round-trip through `sweep_compare.reason_signature`.

3. [HIGH] Line tolerance ±10 + 120-char op truncation is internally inconsistent with `sweep_compare.classify_shift` (sweep_compare.py:446-464), which uses STRICT signatures with NO line tolerance and Counter-multiset semantics. Two parallel systems, two answers, same question. Granite case (line 1219→1218) classified contradictorily.

4. [HIGH] `add-to-existing` action proposes commenting on existing issues, but file-issue subagent has NO comment-mode. Rev 4 punts to "when shipped." Without it, half of `close-relaxed-with-new-issues` proposals are unactionable; operators forced to silent-skip OR hand-edit outside audit chain.

5. [HIGH] `_do_close_op` rev 5 accepts `verdict == "all-clear"` ONLY. Rev 4's `close-relaxed-with-new-issues` conflates close-mode VERDICT with verify-JSON VERDICT. Either breaks rev-5's invariant OR requires schema bump that breaks existing rev-5 sha256 pin tests. Two verdict spaces collapsed into one sentence.

6. [MEDIUM] Source-path normalization undefined — needs closed-set enumeration (pip-installed / modellibs / future buck cell / unknown). NEW path appearing silently fails normalization → false moved-GB.

7. [MEDIUM] Issue #27's break is at `torch/nn/modules/linear.py` (torch source, not transformers). Rev 4 only describes transformers normalization. Torch-source-anchored issues fall outside scope OR silently misclassify.

8. [MEDIUM] dup-search bottleneck: `gh issue list --search` per signature group with NO budget cap, NO local-cache fallback, NO rate-limit handling. Could burn GitHub's 30 req/min secondary limit on dup-search alone.

9. [MEDIUM] No precedence pinning for "pair has BOTH original-persists AND new-GB-sites simultaneously."

10. [LOW] `parse_affected_models` falls back to `["eval", "train"]` for malformed Modes cells — silent overcounting → false negatives on close.

SUGGESTED_TESTS: 10 specific tests covering body-extraction round-trip, sweep_compare reuse, torch-source handling, dup-search caching, verdict-space disambiguation, precedence pinning, fail-loud parse warnings.

NOTES:
- Strongest argument for REJECT (vs approve-with-gaps): rev 4 introduces a parallel signature/normalization system that DOES NOT REUSE sweep_compare.reason_signature/classify_shift — the same tooling Otter built for the weekly brief. Two-source-of-truth on the same semantic question. Either reuse + extend (preferred), or document why existing code is unsuitable. Rev 4 currently does neither.
- This is the exact "new code path duplicates existing semantics in subtly-different way" pattern adversary-review is meant to catch.
- Probe (h) added 3 missing topics: (1) compile_kwargs interaction with moved-GB sites (new break may require different config to reach), (2) idempotency (rev 4 run twice → duplicate file-new proposals), (3) methodology R5 update strategy (R5 was sized for ~5 new issues/week, not 30).
```

## My disposition

**Verdict: REJECT acknowledged.** Rev 4 v1 design is unimplementable as written. Redesign required.

| # | Sev | Disposition | Action |
|---|---|---|---|
| 1 | high | **Redesign** | Multi-strategy body extractor: `## Break Reason` (singular), `## Break Reasons` (plural), `**Provenance anchor:**` regex, first ```text``` block under `## Pattern`. Closed-set + fail-loud on bodies whose structure isn't recognized (don't silent-fallback to `reframe`). |
| 2 | high | **Redesign** | Reuse `sweep_compare.reason_signature` directly. Extract from issue body's stored sweep evidence (the `<details><summary>Verification signal>` block with `fragment` field) — round-trips through the same function the brief uses. |
| 3 | high | **Redesign** | DROP rev-4 parallel signature system. Reuse `sweep_compare.classify_shift` as the single source of truth. Define rev 4 verdicts in terms of {NO_CHANGE, CATEGORIZATION_SHIFT, REAL_NEW} per pair. Resolves the line-tolerance question by deferring to existing tool's semantics. |
| 4 | high | **Scope reduction** | Restrict rev 4 v1 to `file-new` proposals only. Emit `add-to-existing` candidates as REFUSE-CLOSE with `verdict_reason="add-to-existing proposal exists; comment-mode not yet implemented"`. Comment-mode becomes a separate task BEFORE rev 4 v2. |
| 5 | high | **Redesign** | Disambiguate verdict spaces. Bump `verify_schema_version` to 2. Add new top-level `close_mode: {"verdict": ..., "moved_gb_analysis": {...}}` field. Keep rev-5's `verdict` field (verification-pass closed set) intact. Test both verdict spaces independently. |
| 6 | medium | **Addressed in redesign** | Normalization: strip prefix up to and including `transformers/` (or `torch/`). Fail loud on unknown prefix (raise; not silent default). Closed-set: pip-installed, modellibs, hypothetical buck cell, torch. |
| 7 | medium | **Addressed in redesign** | Extend normalization to torch-source paths explicitly. Test with #27's body. |
| 8 | medium | **Addressed in redesign** | Single bulk fetch + local cache: `gh issue list --state all --limit 500 --json` once per close-mode invocation, cached to `/tmp/close-mode-issues-cache-<hash>.json`. Hard cap on `len(new_issue_proposals)` (e.g., 20). Beyond cap: verdict `close-relaxed-too-many-moved-gbs`. |
| 9 | medium | **Addressed in redesign** | Add explicit precedence test: original-persists wins over moved-GB. |
| 10 | low | **Addressed in redesign** | Make `parse_affected_models` fail loud on unknown Modes tokens; emit to `parse_warnings[]`; close-mode refuses with verdict `parse-warning-needs-human-triage`. |

**Plus 3 missing topics from probe (h):**
- compile_kwargs interaction with moved-GB sites
- idempotency (single case_id pinning the proposal-disposition handoff)
- methodology R5 update for the new-issue-proposal provenance

**Resolution:** rev 4 v1 design REJECTED. Rev 2 redesign required before implementation. Surfaced to Peng — this is an architecture-level decision (single source of truth for signature comparison; comment-mode prerequisite; verdict-space disambiguation) that needs review beyond mechanical fixes.

**Sequencing implication:** modellibs upgrade (WS2-1) was originally recommended to depend on rev 4 to avoid manual triage of moved-GB cases. With rev 4 rejected and pending redesign + adversary-review + implementation (~2-3 sessions of work), the modellibs upgrade should EITHER (a) wait for rev 4 v2 to ship, OR (b) accept manual triage of moved-GB cases on the first post-upgrade weekly sweep. Need Peng's call.
