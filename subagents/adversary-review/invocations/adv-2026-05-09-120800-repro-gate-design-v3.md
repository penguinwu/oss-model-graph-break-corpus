---
case_id: adv-2026-05-09-120800-repro-gate-design-v3
subagent: adversary-review
date_utc: 2026-05-09T12:08:00Z
persona_sha: 1f36118aaed67cfc4b3d74a25f131f163986663b
target: file-issue Phase 3 design v3 — repro verification gate
target_artifact: /tmp/file-issue-design-repro-gate-v3.md
parent_case_id: adv-2026-05-09-113538-repro-gate-design-v1
verdict: approve-with-gaps
confidence: high
gaps_found: 11
high_severity: 4
medium_severity: 6
low_severity: 1
suggested_tests: 10
disposition: design v3.1 in progress; all 11 gaps dispositioned in v3.1; commit 1 begins after v3.1 + Peng FYI
---

## Why this case file exists

Second adversary review on the repro-gate design lineage. v1 → 13 gaps → v2 (incorporated 5 high-severity fixes) → Peng iteration → v3 (incorporated BOTH-original-AND-MRE + sweep-cache + partial-fix EDIT) → this review on v3.

Adversary's own process observation worth elevating: "v1's 13 gaps got 100% disposition, but the disposition table didn't catch that v3's 'body-shape unification' reopened gap 1 in a new form. The pattern to encode: when a disposition uses fix (a) or fix (b), and a later iteration changes that decision, RE-RUN the v1 gap-detection logic against the new shape — don't trust the disposition table to remain valid across re-architectures."

## Files reviewed by the adversary

- `/tmp/file-issue-design-repro-gate-v3.md` (primary)
- `subagents/adversary-review/invocations/adv-2026-05-09-113538-repro-gate-design-v1.md` (parent — verify v3 preserves v1 fixes)
- `subagents/file-issue/SKILL.md`, `subagents/file-issue/persona.md`
- `tools/file_issues.py` (cmd_pytorch_upstream 1314-1433, cmd_corpus_issue 1718-1791, argparse 1844-1897)
- `sweep/orchestrator.py` (verified `run_sweep` / `run_explain` are NOT here — they're in sweep/run_sweep.py per file's own docstring)

## Verdict + confidence

`approve-with-gaps` (high confidence). Adversary's note: "v3 is measurably better than v1 — the 13 v1 gaps are all addressed in spirit, the body-shape unification is a real direction, the sweep cache reduces wall-clock cost, and the 5-commit split is appropriately small. The verdict is approve-with-gaps (NOT reject) because the gaps above are seam-level and addressable in the implementation phase without redesigning the architecture."

## Gaps + dispositions for v3.1

### High severity (must resolve BEFORE commit 1)

1. **Body-shape unification claim contradicts cmd_pytorch_upstream's actual assembly.** v3 says "corpus and upstream bodies are identical per Peng's A4" but cmd_pytorch_upstream (file_issues.py:1314-1433) assembles body at posting time from --script + --summary + per-venv probing — has "## Reproducer / ## Setup / ## Captured output" sections, NOT "## Original failure report / ## Minimal reproducer (MRE)". Adversary surfaced this as a gap-1 reopening (v1 fixed it via carve-out; v3 escalated to body-unification which silently grows commit 5).
   - **v3.1 disposition:** ADOPT FIX (b) — carve out upstream. For upstream: `--script` IS both original and MRE (the script is the canonical reproducer). The 4-cell matrix collapses to 2 cells (current MRE + nightly MRE). New flag `--expected-signal-json <path>` for upstream. Body assembler unchanged in v1.0 (rewrite deferred to a separate later commit). Document the asymmetry explicitly in SKILL.md.

2. **verify_repro --use-original cache-miss runs a full sweep subprocess but design hand-waves safety/duration/GPU coordination.** The corpus has check_gpu_health/kill_gpu_zombies in orchestrator.py:28+49 — verify_repro doesn't reference these. Cost note "30s-2min" is for MRE; cache-miss original could be 5-30 min.
   - **v3.1 disposition:** Specify the cache-miss path explicitly. (a) verify_repro shells out to `tools/run_experiment.py sweep` (inherits GPU coordination) with strict --workers 1 + per-call timeout. (b) Subprocess wrapped in same GPU-pre-check as orchestrator. (c) Tempdir cleanup on success/failure (try/finally with `shutil.rmtree`). (d) When the sweep subprocess itself errors (typo, missing model), classify as `different-failure` with stderr captured — not a separate `command-error` class.

3. **INDEX.json schema lacks cohort_sha256 + args_fingerprint.** Cached row from old cohort can satisfy new filing — soundness hole.
   - **v3.1 disposition:** ADOPT — INDEX.json adds `cohort_sha256` (sha of cohort file at sweep time), `args_fingerprint` (sha of canonicalized worker/mode/flag set), AND `sweep_kind` + `produced_fields` (per gap 9). lookup_sweep_evidence emits these into verification JSON.

4. **Layer 3 design points orchestrator hook at WRONG FILE.** `sweep/orchestrator.py` does NOT contain `run_sweep` or `run_explain`. Verified: those are in `sweep/run_sweep.py:510` (run_sweep) and `sweep/run_sweep.py:1261` (run_explain).
   - **v3.1 disposition:** Re-anchor Layer 3. Hook lives in `sweep/run_sweep.py` — append to INDEX.json at end of `run_sweep` (post-pass-completion) and end of `run_explain`. Idempotent by sweep_id. Test that exactly 1 row per sweep, not 1 per pass.

### High severity (must address but not blocking commit 1 since it's in commit 4)

5. **TOCTOU between verify-time body and post-time body — non-substantive whitespace changes can break sha equality.** Mode B emits final body AFTER verify_repro runs against draft body; whitespace alone could differ.
   - **v3.1 disposition:** Specify CANONICALIZATION — `extracted_bytes_sha256` is hashed over the TEXT INSIDE the fence/comment, leading/trailing whitespace stripped, line endings normalized to LF. NOT raw body bytes. Document in verify_repro.py spec. Test extracts same sha from bodies that differ only in surrounding whitespace.

### Medium severity (address in design but pin tests later)

6. **No escape valve for legitimate "nightly venv stale, can't produce JSONs" filings.** 4 required argparse flags + Layer 5 anomaly assumes JSONs exist. v2's --accept-nightly-anomaly was the valve; v3 dropped it.
   - **v3.1 disposition:** Add `--nightly-unavailable-reason "<text>"` flag. Presence makes the two nightly flags optional. Body's "Repro status:" line carries the reason text verbatim. Surface block to Peng requires explicit ack before posting (similar pattern to External Engagement).

7. **Mixed-current cells (current MRE reproduces but current original doesn't, or vice versa) — policy unspecified.**
   - **v3.1 disposition:** Hard-refuse on ANY current cell that's not `reproduces`. Both must independently reproduce on current. Specific error: "MRE does not reproduce on current venv but original does — MRE captures different/no bug; revise the MRE."

8. **Open-question 4's "lean: Mode B emits exact command verbatim" introduces a contract with no enforcement.** Layer 6 lists "matches a real sweep command" as gate item 3 but provides no implementation.
   - **v3.1 disposition:** Add `tools/run_experiment.py sweep --validate-args <command>` helper (or argparse-introspection equivalent). Mode B's pre-submission gate item 3 calls this; rejects with `ORIGINAL_REVISION_NEEDED` on parse error.

9. **INDEX.json doesn't record sweep_kind — cache hit can return wrong-shape evidence (e.g., graph-break-only sweep when MRE expects numeric_max_diff).**
   - **v3.1 disposition:** Adopted via gap 3 fix (sweep_kind + produced_fields).

10. **lookup_sweep_evidence doesn't accept --case-id, so cluster batch filings can't get distinct case_ids in their verification JSONs.**
    - **v3.1 disposition:** lookup_sweep_evidence takes `--case-id <id>`, stamps it into emitted JSON. verify_repro propagates.

### Low severity (pedantic — flag and move on)

11. **INDEX.json paths could be brittle under symlink/PARA moves.**
    - **v3.1 disposition:** Use absolute paths in INDEX.json. Document. Cheap.

## Process learning to encode

Adversary's strategic observation: "when a disposition uses fix (a) or fix (b), and a later iteration changes that decision, RE-RUN the v1 gap-detection logic against the new shape — don't trust the disposition table to remain valid across re-architectures."

Worth encoding into the file-issue RETROSPECTIVE.md as a permanent practice — when iterating a design across versions, the new version's gap-disposition table must explicitly re-check ALL prior gaps against the new shape, not just the gaps surfaced by the latest review.

## Suggested tests (10 — see verbatim adversary output for full setup→action→expected→detects)

1. test_verify_repro_use_original_cache_miss_invokes_real_sweep_subprocess (gap 2)
2. test_lookup_sweep_evidence_rejects_match_when_row_lacks_required_signal_field (gap 9)
3. test_extracted_bytes_sha256_invariant_to_body_prose_whitespace_changes (gap 5)
4. test_corpus_new_with_mixed_current_cells_is_hard_refused (gap 7)
5. test_orchestrator_index_writes_one_row_per_sweep_not_per_pass (gap 4)
6. test_pytorch_upstream_body_extraction_or_carve_out_path_is_pinned (gap 1)
7. test_cluster_batch_filings_each_get_distinct_case_id (gap 10)
8. test_corpus_new_with_nightly_unavailable_reason_proceeds_without_nightly_jsons (gap 6)
9. test_lookup_sweep_evidence_records_cohort_sha256_and_args_fingerprint (gap 3)
10. test_mode_b_original_command_must_parse_via_run_experiment_sweep_argparse (gap 8)

These will be folded into the per-commit test files when the corresponding feature is implemented.

## What this case file binds

This case file is the audit anchor for design v3 → v3.1. v3.1 (when written) will reference this case_id and confirm all 11 gap dispositions are reflected. If v3.1 silently drops a disposition, that's a regression of this audit chain.
