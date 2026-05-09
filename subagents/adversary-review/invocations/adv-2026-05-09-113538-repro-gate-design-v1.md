---
case_id: adv-2026-05-09-113538-repro-gate-design-v1
subagent: adversary-review
date_utc: 2026-05-09T11:35:38Z
persona_sha: 1f36118aaed67cfc4b3d74a25f131f163986663b
target: file-issue Phase 3 design — repro verification gate (V1)
target_artifact: /tmp/file-issue-design-repro-gate-v1.md
verdict: reject
confidence: high
gaps_found: 13
high_severity: 4
medium_severity: 6
low_severity: 3
suggested_tests: 8
disposition: design v2 in progress; v1 superseded; gaps 1-13 addressed in design v2 before any code lands
---

## Why this case file exists

Per RETROSPECTIVE.md "honest gap" entry (2026-05-09 V1 cluster+dedup ship): the V1 cluster+dedup design went through 2 verbal adversary-review iterations whose verbatim outputs were NOT captured. Future practice: write the case file BEFORE iterating, even if early case files are short.

This is the FIRST design adversary-review case file written before the design enters v2. The discipline being enforced.

## Files reviewed by the adversary

- `/tmp/file-issue-design-repro-gate-v1.md` (primary review target)
- `subagents/file-issue/SKILL.md` (existing skill the design extends)
- `subagents/file-issue/persona.md` (existing persona the design extends)
- `tools/file_issues.py` (`cmd_pytorch_upstream`, `_validate_via_skill`, `_validate_cluster_plan`, `cmd_corpus_issue`, argparse plumbing)
- `tools/cluster_failures.py` (header / line count only)

## Verdict + confidence

`reject` (high confidence). Rationale (verbatim from adversary): "The design has multiple structural contradictions with existing code (pytorch-upstream body assembly, --close having no --body, Mode A temporal ordering vs Mode B's MRE) that cannot be resolved by clarification — they require a redesign of the verification anchor. Reading verify_repro.py running locally would not change my view; these are spec-level issues."

## Gaps (verbatim adversary output)

### High severity (4)

1. **Design is structurally incoherent for `pytorch-upstream`** — the gate's MRE-extraction premise contradicts how upstream bodies are actually assembled.
   - WHY: Layer 1 requires extracting one `python repro=true` fence from a body file, hashing it as `mre_sha256`, and binding that hash to the JSON. But `cmd_pytorch_upstream` (file_issues.py:1314-1518) builds the body at *posting* time from `--script <path-to-py>` + `--summary <md>` + per-venv probing — there is no pre-existing body file to extract from, and there is no `python repro=true` fence convention in the assembled output.
   - FIX: Either (a) explicitly carve upstream out of the new gate in v1 with a written reason, OR (b) define how `verify_repro.py` consumes the upstream `--script` directly (bypassing body extraction) and the JSON `mre_sha256` is hashed over the script bytes.

2. **`corpus-issue --close` has no `--body`**, so `_validate_repro_evidence`'s mre_sha256 binding cannot fire — the design demands the gate but the operation has nothing to gate against.
   - WHY: The strongest validator condition (sha256 of body's MRE matches JSON's `mre_sha256`) has no body and no MRE in the close path. Same hole applies to `--comment` if the comment doesn't itself contain a `python repro=true` fence.
   - FIX: For `--close`/`--comment`, add an alternate binding: comment text MUST contain the verification's torch_version + git_version verbatim, and the validator extracts and matches.

3. **Mode A check 11 is temporally impossible** — Mode A runs BEFORE Mode B writes the body, so the MRE-sha that Mode A is asked to cross-check does not exist yet.
   - WHY: Design says Mode A "Cross-check each JSON: mre_sha256 matches the MRE Mode B will emit." Mode B has not been invoked at Mode A time. Either (a) MRE must be frozen in the draft, OR (b) check moves post-Mode-B, OR (c) verify_repro runs AFTER Mode B and BEFORE posting.
   - FIX: Add explicit Mode B contract: "MRE bytes from the draft (frozen at Step 1 / Step 2.5) MUST be embedded byte-for-byte. Mode B MAY NOT re-minimize or re-format." Drop the "Mode A check 11" framing; Mode A's check is just metadata-presence.

4. **Classification by `exit_code == 0` is wrong** for graph-break / numeric-divergence / fallback symptoms (the dominant corpus classes), which routinely exit 0 even when the symptom fires.
   - WHY: Graph-break MREs typically exit 0 (break observed via stderr/TORCH_LOGS); numeric-divergence MREs exit 0 unless they `assert close()`; fallback MREs run to completion. Strictly-exit-code classifier creates noisy false-alarm channel.
   - FIX: Make classification per-symptom-type. Require Mode B to emit `expected_signal` field (HTML comment in body): `<!-- expected_signal: {"kind": "stderr_contains", "fragment": "..."} -->`. verify_repro extracts this comment, NOT the body prose.

### Medium severity (6)

5. **`--accept-nightly-anomaly <reason>` fights V1's body_sha256 audit chain.** Layer 5's anomaly-variant Repro-status line is composed at posting time AFTER Mode B already wrote the body — either Mode B is invoked twice (breaks audit chain) or the line is patched at posting time (silently invalidates body_sha256).
   - FIX: On (a) FILE ANYWAY decision, Otter loops back to Step 4 with the Peng-supplied reason in the draft as `nightly_anomaly_reason: "..."`, Mode B re-emits with the line, new case file (chained `parent_case_id`), new sha256, new --via-skill.

6. **No mechanism prevents Mode B from silently altering MRE post-verification.** Mode B's MRE checklist explicitly invites re-minimization in Step 4. (Compounds gap 3.)
   - FIX: Same as gap 3 — explicit Mode B "MAY NOT alter MRE" contract.

7. **Venv-age computation in Open Q5 is inverted.** "Older treated as venv age" is wrong for freshness — most recent install activity is what matters.
   - FIX: Use the MAX (most recent) of {`pip show torch` install date, `~/envs/torch-nightly-cu128/.install_date`}. Better still: also check `torch.version.git_version` against pytorch nightly's recent commits.

8. **`--close` and `--comment` interact with `_validate_cluster_plan`'s case_id-must-appear constraint** but the design doesn't define how.
   - FIX: Explicitly state that `--close` and `--comment` operations always require a `single_manual` cluster plan (use `tools/cluster_failures.py single-manual <case_id>`).

9. **24h staleness on verification JSON understates cost** for slow Mode A → Mode B cycles.
   - FIX: Replace `wall_clock_utc within 24h` with `torch_git_version matches the current published nightly's git version`. Re-verify only when nightly has actually changed.

10. **Repro status: line as body's first line breaks backward-compat** with existing posted bodies (issue 77) AND the persona's pre-submission gate doesn't enforce its presence.
    - FIX: Add explicit pre-submission gate item with regex check. Add SKILL.md migration note for issue 77 EDIT.

### Low severity (3)

11. **"Load-bearing message fragment" extraction is fragile AND target-specific.** Corpus uses "## Observed vs expected"; pytorch-upstream uses "## Captured output."
    - FIX: Adopt structured `expected_signal` HTML comment (same fix as gap 4).

12. **Implementation order step 3 lumps 5 distinct changes into a single commit** — violates the small-commit + adversary-review-fired pattern.
    - FIX: Split step 3 into 3a (validator + flags), 3b (--close + --close-comment), 3c (--comment + --accept-nightly-anomaly). Each gets its own adversary review + commit.

13. **No design coverage of `cluster_id` for verification-only filings.** Mode A check 10 (cluster cohesion) would fire and demand a representative_case MRE for a `--comment` updating nightly status.
    - FIX: Same as gap 8 — `--close` and `--comment` use `single_manual` plans; check 10 passes trivially via the existing carve-out.

## Suggested tests (8 — see verbatim adversary output for full structure)

1. test_verify_repro_classifies_silent_graph_break_correctly (detects gap 4)
2. test_corpus_close_op_validates_without_body_path (detects gap 2)
3. test_pytorch_upstream_repro_gate_extraction_path (detects gap 1)
4. test_mode_b_cannot_alter_mre_after_verification (detects gap 6)
5. test_accept_nightly_anomaly_requires_chained_case_file (detects gap 5)
6. test_verify_repro_extracts_expected_signal_from_body_comment (detects gap 11)
7. test_venv_age_uses_max_not_min_of_install_dates (detects gap 7)
8. test_close_op_uses_single_manual_cluster_plan (detects gap 8)

## Adversary's strategic note (worth elevating)

> "A simpler v1 framing worth considering: Phase 3 v1 could ship ONLY the NEW-issue path (current + nightly verification, hard refusal on current-not-reproducing, surface on nightly-not-reproducing). Defer `--close` and `--comment` to Phase 3.5 once the NEW-issue path has proved out. This sidesteps gaps 2 + 8 + 13 entirely, ships the highest-leverage piece (NEW issues are the maintainer-flooding risk Peng cares most about), and lets the close/comment design get shaped by real surfaces."

## Gap dispositions (Otter's plan for design v2)

| Gap | Severity | Disposition in v2 |
|---|---|---|
| 1 | high | Adopt fix (b): verify_repro consumes `--script` directly for upstream; JSON `mre_sha256` hashed over script bytes. Two extraction modes. |
| 2 | high | Adopt suggested fix: comment text contains `torch_version + git_version` verbatim; validator extracts. New `_validate_close_comment_binding` helper. |
| 3 | high | Adopt fix: explicit Mode B "MAY NOT alter MRE" contract. Drop Mode A check 11; replace with pre-Mode-B Step 2.5 verification + post-Mode-B body-MRE-binding gate at posting time. |
| 4 | high | Adopt fix: `expected_signal` HTML comment in body. Mode B emits per symptom type. verify_repro classifies via this comment. |
| 5 | high | Adopt fix: chained case file pattern documented; `--accept-nightly-anomaly` flag binds to a fresh body with the anomaly-variant first line. |
| 6 | medium | Same fix as gap 3 (Mode B contract). |
| 7 | medium | Adopt fix: MAX, not MIN. Document why. |
| 8 | medium | Adopt fix: `--close` and `--comment` use `single_manual` plans. Document in authority gate table + SKILL.md. |
| 9 | medium | Adopt fix: replace 24h with `torch_git_version mismatch with published nightly` check. |
| 10 | medium | Adopt fix: pre-submission gate item with regex check. SKILL.md migration note for issue 77. |
| 11 | low | Adopted via gap 4 fix (structured signal). |
| 12 | low | Adopt fix: split impl order step 3 into 3a/3b/3c. |
| 13 | low | Adopted via gap 8 fix (single_manual plans). |

**Strategic adversary recommendation:** Otter's response — adopt a HYBRID. Ship NEW-issue path first (v1.0 = gaps 1, 3, 4, 6, 7, 9, 10 + impl steps 1-2, 3a). Then `--close` + `--comment` as v1.5 (gaps 2, 5, 8, 12, 13 + impl 3b/3c). Two ship cycles. Smaller adversary surface per cycle. Peng can see the NEW-issue path land + use it for a few real filings before v1.5 design solidifies.

## What this case file binds

This case file is the audit anchor for design v1 → v2. Design v2 (when written) will reference this case_id in its frontmatter / opening paragraph. The 13 gap dispositions above are the load-bearing claim that v2 will address. If v2 silently drops a disposition, that's a regression of this audit chain.

---

The full verbatim adversary output (with all 8 SUGGESTED_ADDITIONAL_TESTS in setup→action→expected→detects format, plus the NOTES section on V1↔V2 coherence + verify_repro cost + issue 77 backward-compat) is preserved in the agent invocation transcript at the timestamp on this case file. Did NOT inline the full verbatim text here because the gap summaries above + the disposition table cover all load-bearing claims; the test setups will be re-derived in `tools/test_verify_repro.py` when implementation begins.
