---
case_id: adv-2026-05-10-193500-close-mode-rev5-design
subagent: adversary-review
date_utc: 2026-05-10T23:35:00Z
trigger: design-review
files: [subagents/file-issue/CLOSE_MODE_REV5_DESIGN.md, subagents/file-issue/CLOSE_MODE_DESIGN.md, subagents/file-issue/persona.md, tools/verify_repro.py, sweep/worker.py, tools/file_issues.py, sweep/run_sweep.py, sweep/orchestrator.py]
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 72d45729f53611c468b404e6279dedaa57573143bea0bd0609d0d34318ac744f
---

## Reviewer raw output

```
VERDICT: approve-with-gaps
CONFIDENCE: high

GAPS_FOUND:

1. [HIGH] Worker CLI flag in design (`--pass 1`) does NOT match worker.py's `--pass-num` (worker.py:4442 + _handle_chaos:121). Every per-pair re-run will exit 2 (argparse error) → verdict=`run-failed`. Bug visible only when actually trying to close.

2. [HIGH] Hand-rolling subprocess.run instead of using existing `sweep.orchestrator.spawn_worker` (orchestrator.py:120). `_run_one_model` (referenced in design) does NOT exist — Otter's "Never State Mutable Facts from Memory" rule violation. spawn_worker provides: process-group setup (preexec_fn=os.setsid), CUSTOM_WORKER_SCRIPT vs WORKER_SCRIPT for source=='custom', LOCAL_KERNELS resolver via SWEEP_USE_KERNEL_RESOLVER, temp files (not PIPE) to avoid pipe-buffer deadlock. Hand-rolled subprocess.run silently loses ALL of these.

3. [HIGH] Status vocabulary mismatch + missing statuses. Design enumerates `full_graph|graph_break|eager_error|create_error|worker_error|timeout` but worker.py also emits `compile_error` (lines 3777, 4170, 4184, 4285), `error` (3789), `success` (3749), `explain_error` (4395). Plus design conflates worker.py status with `classify_close_candidate`'s `"fullgraph on current sweep"` model_disposition string — different namespaces. Future reader silently rejects.

4. [MEDIUM] Verify-json gate has no sha256 pin. file_issues.py rev 3 went to substantial trouble to mod-pin Mode B body via body_sha256 — same drift class. Anyone (or cron drift) can hand-craft a JSON with matching `case_id` + `verdict=="all-clear"` and bypass.

5. [MEDIUM] `venv-drifted` not in verdict enum (only all-clear|reproduces|partial-fix|run-failed). Design's own risk-table mitigation unenforceable. Plus `venv-versions-missing` case (old sweeps with empty versions blocks) isn't specified.

6. [MEDIUM] `partial-fix` is single bucket — operator can't tell WHICH side disagreed (MRE vs pairs). Should split into `partial-fix:mre-still-reproduces` / `partial-fix:pairs-still-broken` / `partial-fix:mre-different-error`.

7. [MEDIUM] MRE classification missing `import-error` / `setup-error` for cases where MRE crashes due to torch API drift unrelated to the bug. verify_repro.py returns `different-failure` — design's mapping to close-mode buckets is incomplete.

8. [LOW] No precedence test for unknown worker status. Future fail-open refactor would silently bypass the gate.

SUGGESTED_TESTS: 11 specific tests covering all 8 gaps + happy-path bias on subprocess timeout/crash + custom-source worker script.

NOTES:
- Probe (a) verified: subprocess interpretation is correct for cron-invokable CLI; but design needs to add explicit "Mode A_close is mechanical, no judgment — escalate human if MRE = different-error" rule.
- Probe (e) FAILED verify-first: `_run_one_model` does NOT exist; actual function is `sweep.orchestrator.spawn_worker`. The design should name actual paths, not hedge ("or equivalent").
- Probe (f) verified: `verify_repro.extract_mre_from_body` (line 99) is clean public API. But design also needs `extract_expected_signal_from_body` (line 138) — without it, classify() can't run.
- Per Otter's Test-Fix-Rerun Loop rule: when implementing and first integration hits the `--pass-num` flag bug (gap #1), DO NOT surface — fix it, add regression test, rerun.
```

## My disposition

| # | Sev | Disposition | Action |
|---|---|---|---|
| 1 | high | **Addressed** in rev 2 | `--pass-num 1` not `--pass 1`. Pinned in test_worker_invocation_uses_pass_num_not_pass. |
| 2 | high | **Addressed** in rev 2 | Import + call `sweep.orchestrator.spawn_worker` directly. Risk-table updated to name actual function. test_invocation_uses_orchestrator_spawn_worker pins it. |
| 3 | high | **Addressed** in rev 2 | Enumerate closed set: `{full_graph, graph_break, eager_error, create_error, compile_error, error, worker_error, timeout, explain_error, skipped}`. Raise ValueError on unknown status (don't silent-default). Add docstring distinguishing worker.py status vs model_disposition string namespaces. |
| 4 | medium | **Addressed** in rev 2 | Add `--close-verify-json-sha256` arg. verify_close_candidate.py records its own JSON sha256 in case file frontmatter (`verify_json_sha256:` field). file_issues.py recomputes and compares (matches body_sha256 pattern). Also: pin `verify_schema_version: 1` field. |
| 5 | medium | **Addressed** in rev 2 | Verdict enum extended: `all-clear / reproduces / partial-fix-* / run-failed / venv-drifted / venv-versions-missing`. Close-mode REFUSES on the new venv-* verdicts with actionable message. |
| 6 | medium | **Addressed** in rev 2 | Split: `partial-fix:mre-still-reproduces` / `partial-fix:pairs-still-broken` / `partial-fix:mre-different-error`. All 3 refused by close-mode. Test per sub-case. |
| 7 | medium | **Addressed** in rev 2 | Document explicit mapping `verify_repro.classify() → mre.result`. Add `import-error` sub-bucket (or document it as `different-error` with stderr-substring evidence). Test for each verify_repro classification result. |
| 8 | low | **Addressed** in rev 2 | Add precedence test: MRE clean + pair=`compile_error` (unknown) → MUST raise ValueError, NOT silent all-clear. |

All 11 SUGGESTED_TESTS to be implemented in `tools/test_verify_close_candidate.py` + the file_issues.py block.

**Process note:** the `_run_one_model` reference was the "Never State Mutable Facts from Memory" rule violation Otter's local CLAUDE.md warns about. Design rev 2 verifies the actual function name + line numbers before pinning them.
