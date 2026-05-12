---
case_id: adv-2026-05-11-211500-close-mode-rev4-v2-design
subagent: adversary-review
date_utc: 2026-05-12T01:30:00Z
trigger: validator-design-substantive-change
files:
  - subagents/file-issue/CLOSE_MODE_REV4_V2_DESIGN.md
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: reject
output_sha256: b07a53b32b627e2ac871749ac887c215845086aafa60b74d4b93a63908bfca8a
---

## Reviewer raw output

```
VERDICT: reject

CONFIDENCE: high
CONFIDENCE_RATIONALE: The core architecture (reuse classify_shift, extend verdict space, re-run for missing data) is sound and resolves the rev 1 architecture-level rejections — but several mechanical specifics are inconsistent with the existing tools they propose to call, and one of the rev 1 high-sev findings is silently re-introduced. These are concrete, citable defects, not opinion.

GAPS_FOUND:

1. [SEVERITY: high] Source-1 body parser dead — rev 1 GAP #1 silently re-introduced. The `## Break Reason` block parser fails on post-2026-05-09 mre-dogfood body shapes (`## Pattern` + ```text``` + `**Provenance anchor:**`). Rev 4 v2 carries forward the SAME parser without extension or per-shape tests. Without fix, every rev 4 v2 close-mode invocation against an in-flight issue silently falls through to Source 2 — Source 1 path is dead code.
   FIX: Step 0 must enumerate supported body shapes + closed-set parser + per-shape unit tests + fail-loud on unknown shape (not silent fallback).

2. [SEVERITY: high] Verdict-space disambiguation asserted-resolved but mechanics not specified. `_do_close_op` (file_issues.py:1551+) has no entry point for `close-with-cascade`; line 1620-1628's `_per_mode_pass_check` will REJECT close-with-cascade cases at Step 6 before any cascade logic runs.
   FIX: Specify `_do_close_op` modification (new flag like `--accept-relaxed-criterion` OR new `_do_close_op_relaxed` path); replace/augment `_per_mode_pass_check` with "no pair hits tracked_pattern" check; explicit pin that strict path remains unchanged.

3. [SEVERITY: high] `dedup_search` proposed semantics inconsistent with `dedup_source_lines.py` actual semantics. Cascade payload's `query`/`existing_match` schema doesn't match the tool's CLI (takes `--draft` or `--source-lines`, returns LIST of overlaps not single integer, defaults to dynamo-only filter).
   FIX: Either (a) wrap dedup_source_lines in `tools/cascade_dedup.py` with closed-set output schema + precedence rules when N>1 issues match; OR (b) change cascade schema to match tool's actual output shape. Pin precedence in test.

4. [SEVERITY: high] `tracked_pattern` line tolerance (±2 lines) inconsistent with `classify_shift`'s no-tolerance Counter-multiset semantics — re-introduces rev 1 GAP #3. classify_shift (sweep_compare.py:446-464) uses STRICT location-string match; rev 4 v2's local logic uses ±2 line tolerance. Same input → contradictory results between rev 4 v2 local check and the source-of-truth tool the design claims to reuse. Q1 premise violated.
   FIX: Either (a) extend `sweep_compare.reason_signature` to optionally normalize lines within tolerance window (gated by flag) so both pipelines use the same rule; OR (b) drop line tolerance entirely and let line-shifts read as "no longer hitting" → close-with-cascade with the same pattern → dedup catches as near-duplicate.

5. [SEVERITY: medium] Re-run cost-cap of 20 pairs ungrounded; failure-mode for cap firing not specified. Large umbrellas (e.g., #122 with 18+ pairs) hit cap → `block-stale-rerun` permanently un-closeable.
   FIX: (a) Calibrate cap empirically against open-issues p95; (b) replace pair-cap with wall-time cap; (c) specify continuation: persist `re_runs_queued.json` for next-invocation pickup (idempotency).

6. [SEVERITY: medium] `known_errors.json` eager_error → "still missing" asymmetric with original-cohort logic. Pair that was IN cohort at issue-filing time but is NOW in known_errors is "still missing" (correct conservative); pair that was NEVER in cohort should be EXCLUDED entirely (rev 4 v2 wrongly counts it as missing → blocks close).
   FIX: Distinguish (a) at-filing-in-cohort + now-in-known_errors → reject-keep-open; (b) never-in-cohort → exclude entirely.

7. [SEVERITY: medium] `skip_models.json` bypass for re-run lacks audit trail + risks resource-exhaustion cascade. Hard-isolation per-pair re-runs (RLIMIT_AS, ulimit) needed; `re_runs[]` schema must record bypass authorization.
   FIX: Subprocess isolation + bypass audit fields + recorded-in-case-file test.

8. [SEVERITY: medium] Stale-venv detection mechanism hand-wavy + references non-existent `torch.version.published_nightlies()` API.
   FIX: Concrete options — (a) `pip show torch` Version: field date; (b) `pip install --pre --dry-run`; (c) `~/envs/.../`/`.installed_at` manifest. Pick ONE + add deterministic-mock test.

9. [SEVERITY: medium] No precedence pinning when `close-with-cascade` AND `block-stale-rerun` could BOTH apply.
   FIX: Explicit precedence in verdict table + pin test.

10. [SEVERITY: low] Cascade actions surfaced to Peng have no idempotency. Re-invocation produces duplicate ask asks or duplicate file-new issues.
    FIX: Persist disposed cascade case_ids in plan/sidecar; skip emission for already-disposed `pattern_id`.

NOTES:
- Architectural framing (reuse classify_shift; verdict extension; re-run for missing data) IS correct + resolves highest-level rev 1 rejections. Defects are mechanical + fixable in-design — "redesign Step 0/0.5/3 specifics + add consistency tests" exercise, not throw-away.
- Process gap: rev 4 v2 narrative addresses ~4 of rev 1's 10 gaps; the doc should explicitly cite which rev 1 gaps were carried forward with what changes vs actively dropped with what reasoning.
- 7-day venv staleness window + close-mode re-run on a venv that's older than the venv used for the original baseline sweep → comparison across two different toolchains. The "no longer hits tracked_pattern" verdict could be venv-version artifact rather than real fix. compiler-flags-match check (file_issues.py:1635) only checks `fullgraph`+`backend`; does NOT check torch.version.git_version match. **Surface as Q5 to Peng**: should re-run torch.git_version match original sweep's torch.git_version, OR is fresher torch acceptable (this might be intended — fresher torch closes issues that were fixed in newer nightlies)?
- Sequencing: gaps 1-4 (high) are blocking; 5-9 (medium) are fixable in-design before impl; 10 (low) deferrable.
```

## My disposition

**REV 4 V2 DESIGN — REJECTED. Needs rev 2.5 redesign before impl.**

| Gap | Severity | Disposition |
|-----|----------|-------------|
| 1 — Source-1 body parser dead | high | OPEN — must address in rev 2.5: enumerate body shapes + closed-set parser + per-shape tests + fail-loud on unknown |
| 2 — Verdict mechanics not specified | high | OPEN — must specify `_do_close_op` modification path + new flag + tests |
| 3 — dedup_search schema mismatch | high | OPEN — wrap `dedup_source_lines.py` in cascade-specific helper OR change schema to match tool output |
| 4 — Line tolerance inconsistent with classify_shift | high | OPEN — either extend reason_signature normalization (preferred) OR drop tolerance from rev 4 v2 |
| 5 — Re-run cost-cap ungrounded | medium | OPEN — calibrate empirically + add wall-time cap + specify continuation |
| 6 — known_errors timing distinction | medium | OPEN — distinguish at-filing-in-cohort vs never-in-cohort |
| 7 — skip_models bypass audit | medium | OPEN — subprocess isolation + audit fields |
| 8 — Stale-venv detection mechanism | medium | OPEN — pick concrete check + deterministic-mock test |
| 9 — Precedence not pinned | medium | OPEN — add to verdict table + test |
| 10 — Cascade idempotency | low | DEFER — fix in v2.5 if cheap, else follow-up task |

**Q5 surfaced for Peng (NEW architecture question, not in original Q1-Q4):**
> Re-run torch.git_version match policy: should the per-pair re-run venv's `torch.version.git_version` match the original sweep's `torch.version.git_version`, OR is using the freshest available nightly acceptable? The latter could spuriously close issues fixed only in newer nightlies than the original sweep — which might be intentional (we WANT issues to close when fixes land), or might be misleading (the close looks like "this week's torch fixed it" when actually it was fixed in a torch the original sweep didn't see).

**Rev 2.5 design + adversary + impl estimate:** ~4-5 sessions (was 3-4 for rev 4 v2; +1 session for the rev 2.5 fixes including Q5 resolution).

**Notes:** the architectural CORE is sound (reuse classify_shift, verdict extension, re-run for missing data) — Peng's Q1-Q4 decisions are correct. The mechanical defects are encoding gaps, not architecture gaps. Rev 2.5 is a design-tightening pass, not a re-architecture.
