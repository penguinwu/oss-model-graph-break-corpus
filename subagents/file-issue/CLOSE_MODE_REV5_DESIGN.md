# close-mode rev 5: spawn-agent MRE+model verification

**Author:** Otter
**Date:** 2026-05-10 19:30 ET (rev 2: 19:45 ET — adversary-driven changes per case adv-2026-05-10-193500)
**Status:** DESIGN APPROVED (adversary-review case `adv-2026-05-10-193500`); implementation pending
**WS1 task:** "close-mode rev 5: spawn-agent MRE+model verification"
**Related directive:** Peng 2026-05-10 18:34 ET — "Even if the sweep result says that an issue can be closed, the issue closer needs to spawn a new agent to verify the MRE and the models are indeed fixed; the new agent can reuse the nightly pytorch already built from the current sweep."

---

## What changes

Close-mode today (rev 3) accepts close on 5 mechanical checks against sweep evidence + the rev-4 compile_kwargs match. Rev 5 adds defense-in-depth: a SEPARATE verification pass that re-runs the issue's MRE + each affected (model, mode) pair fresh against the sweep's nightly venv. Both sweep evidence AND fresh re-verification must agree before close fires.

## "Spawn-agent" interpretation

Peng's directive uses "spawn a new agent" framing. Two interpretations:
- Literal: spawn an Anthropic Agent (only available inside an Otter conversation, not a CLI tool).
- Conceptual: spawn a verification SUBPROCESS that runs the MRE + models fresh.

Going with the conceptual interpretation because file_issues.py is a CLI tool invoked by cron and humans (cron cannot spawn Anthropic Agents). The "agent" is a mechanical extract → run → classify → emit-JSON subprocess. Otter (or a human) drives the workflow: invoke verify_close_candidate.py, review the JSON, invoke file_issues.py with `--close-verify-json`.

**Mode A_close is mechanical, no judgment.** When the verifier returns `partial-fix:*` or `*-different-error`, close-mode REFUSES and the human reviewer judges whether to investigate or override. No automatic agent reasoning over verification output.

## Mechanism — `tools/verify_close_candidate.py` (NEW)

Standalone verification script. Invoked BEFORE `tools/file_issues.py corpus-issue --close`. Output: `/tmp/file-issue-<case_id>-close-verify.json`.

```
python3 tools/verify_close_candidate.py \
    --case-id <case_id> \
    --issue-num <num> \
    --sweep-dir <sweep_dir> \
    [--mre-only | --models-only | --both]   # default: --both
    [--timeout 1800]
```

### Steps

1. **Locate venv:** read `<sweep_dir>/sweep_state.json` → `python` field. Verify file exists + executable.
2. **Probe venv versions:** subprocess `python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'`. Compare to `sweep_state.json[versions]`. If `versions` block empty → verdict `venv-versions-missing` (immediate refuse). If versions present but mismatch → verdict `venv-drifted` (immediate refuse).
3. **Fetch issue body:** GitHub API GET `/repos/.../issues/<num>` → body markdown.
4. **Extract MRE + expected_signal:** REUSE `verify_repro.extract_mre_from_body` (verify_repro.py:99) AND `verify_repro.extract_expected_signal_from_body` (verify_repro.py:138). Both are required to use verify_repro's `classify()`. If either missing in issue body → mark `mre.extracted = false` and proceed to model-only re-verification.
5. **Extract affected models:** REUSE `parse_affected_models` from file_issues.py:720.
6. **Run MRE:** subprocess the extracted MRE bytes via the sweep's venv. Capture stdout, stderr, exit_code, wall_time. Pass through `verify_repro.classify(result, expected_signal)` to get one of: `reproduces / does-not-reproduce / different-failure / timeout`.
7. **Run each (model, mode):** REUSE `sweep.orchestrator.spawn_worker(python_bin, spec, pass_num=1, device='cuda', mode=<m>, timeout_s=<timeout>, dynamic=False)` for each affected pair. **DO NOT hand-roll subprocess.run** — spawn_worker provides process-group setup, custom-source worker script selection, kernel resolver, and deadlock-safe pipes. Then call `harvest_worker` to collect results.
8. **Classify each cell** (CLOSED-SET vocabularies — raise on unknown):
   - **MRE result mapping** (verify_repro.classify() → close-mode bucket):
     - `reproduces` → `reproduces`
     - `does-not-reproduce` → `does-not-reproduce`
     - `different-failure` → `different-error` (or `import-error` sub-bucket if stderr starts with `ModuleNotFoundError` / `ImportError`)
     - `timeout` → `timeout`
   - **Per-pair status** (worker.py emits one of these — pinned closed set):
     `{full_graph, graph_break, eager_error, create_error, compile_error, error, worker_error, timeout, explain_error, skipped}`
     ANY status outside this set → raise `ValueError("unknown worker status: <status>")`. Do NOT silent-default.
9. **Emit JSON** with sha256 self-pin:
```json
{
  "verify_schema_version": 1,
  "case_id": "...",
  "issue_num": N,
  "sweep_dir": "...",
  "venv_python": "...",
  "verified_at_utc": "...",
  "venv_check": {"recorded": {"torch": "...", "transformers": "..."},
                 "runtime":  {"torch": "...", "transformers": "..."},
                 "match": true|false},
  "mre": {
    "extracted": true|false,
    "result": "reproduces|does-not-reproduce|different-error|import-error|timeout|skipped",
    "exit_code": N, "wall_time_s": N,
    "stdout_first_500": "...", "stderr_first_500": "..."
  },
  "models": [
    {"name": "...", "mode": "...", "status": "<closed-set worker status>",
     "wall_time_s": N, "error_first_line": "..."}
  ],
  "verdict": "<closed-set; see below>",
  "verdict_reason": "..."
}
```

### Verdict (closed-set, REFUSE on anything except `all-clear`)

| Verdict | When | close-mode action |
|---|---|---|
| `all-clear` | venv match AND MRE = `does-not-reproduce` (or `skipped`) AND every pair = `full_graph` | ALLOW close |
| `reproduces` | MRE = `reproduces` AND/OR any pair NOT `full_graph` (and MRE didn't disagree) | REFUSE; gap is real |
| `partial-fix:mre-still-reproduces` | MRE = `reproduces` BUT all pairs = `full_graph` | REFUSE; surface MRE-only inconsistency |
| `partial-fix:pairs-still-broken` | MRE = `does-not-reproduce` BUT some pairs NOT `full_graph` | REFUSE; surface pairs-only inconsistency |
| `partial-fix:mre-different-error` | MRE = `different-error` OR `import-error` (regardless of pair status) | REFUSE; human judges if MRE drifted |
| `venv-drifted` | venv runtime versions != sweep_state recorded versions | REFUSE; rebuild venv to match recorded versions |
| `venv-versions-missing` | sweep_state.json `versions` block empty (legacy sweep) | REFUSE; re-run sweep on fail-loud-versions venv |
| `run-failed` | verify_close_candidate subprocess crashed (worker.py error, MRE parse failure, etc.) | REFUSE with diagnostic |

Each verdict carries a one-line `verdict_reason` for operator triage.

## Mechanism — `tools/file_issues.py corpus-issue --close` (MODIFIED)

Two new required flags: `--close-verify-json <path>` AND `--close-verify-json-sha256 <hex>`. Validation (Step 7 in `_do_close_op`):
- File at `--close-verify-json` exists; sha256 == `--close-verify-json-sha256` (catches tampering)
- `verify_schema_version == 1`
- `case_id` matches `--via-skill`
- `verdict == "all-clear"`
- `venv_check.match == true`

Refuses otherwise with the verify_reason embedded in the error message. The case file frontmatter must record `verify_json_sha256:` (set by verify_close_candidate.py at write time, mirroring how Mode B body sha is recorded).

Step ordering in `_do_close_op` (rev 5 = 7 steps):
1. case file `mode_a_verdict='close'` (rev 3)
2. plan candidate exists with `classify_close_candidate=='auto-close'` (rev 3)
3. sweep age ≤ 10 days (rev 3)
4. no `.audit-rerun-required` marker (rev 3)
5. per-mode pre-flight from sweep (rev 3)
6. compile_kwargs match canonical (rev 4)
7. **NEW: --close-verify-json gate (rev 5)**

## Implementation scope

- `tools/verify_close_candidate.py` (NEW): ~350 lines (extract MRE + expected_signal, venv probe, MRE run via verify_repro classify, per-pair via spawn_worker, full closed-set classifier, sha256 self-pin, JSON emit)
- `tools/test_verify_close_candidate.py` (NEW): ~400 lines (11 tests below)
- `tools/file_issues.py`: ~70 lines added (Step 7 + 2 new args + sha256 recompute)
- `tools/test_file_issues.py`: ~150 lines added (5 tests covering verify-json gate + sha256 + verdict variants)
- `subagents/file-issue/persona.md`: ~30 lines added (Mode A_close addendum: "mechanical, no judgment; escalate human on partial-fix:* and venv-* verdicts")
- `subagents/file-issue/CLOSE_MODE_DESIGN.md` (rev 4): sync — add Step 7 to validation list

Estimated 8-10h. Implementation gated by adversary-review (DONE: case file `adv-2026-05-10-193500-close-mode-rev5-design`, all 8 gaps addressed in this rev 2).

## Test plan (11 tests per adversary)

For `tools/test_verify_close_candidate.py`:
1. `test_worker_invocation_uses_pass_num_not_pass` — mock subprocess; assert cmd contains `--pass-num` not `--pass`
2. `test_invocation_uses_orchestrator_spawn_worker` — monkeypatch spawn_worker; assert called with right kwargs
3. `test_custom_source_uses_custom_worker_script` — source='custom' spec → CUSTOM_WORKER_SCRIPT path
4. `test_unknown_worker_status_raises` — mock returns status='compile_error'; classifier raises ValueError
5. `test_verify_json_sha256_pinned_in_case_file` — mutate JSON post-record → close-mode refuses
6. `test_venv_drift_blocks_close` — runtime version != recorded → verdict='venv-drifted'; close refuses
7. `test_partial_fix_subcases_distinguished` — 3 synthetic JSONs → 3 distinct sub-verdicts; all refused
8. `test_mre_import_error_classified_distinct` — MRE with ModuleNotFoundError → mre.result='import-error'; verdict NOT all-clear
9. `test_mre_timeout_path` — MRE that sleeps 9999s with timeout=5 → mre.result='timeout'; process-group killed
10. `test_worker_crash_path` — mock worker exits 139 (segfault) → pair status=worker_error; verdict=run-failed
11. `test_versions_missing_block_close` — sweep_state.json `versions: {}` → verdict='venv-versions-missing'

Plus 5 tests for `tools/test_file_issues.py`:
- `test_close_requires_verify_json` — argparse refuses without
- `test_close_requires_verify_json_sha256` — argparse refuses without
- `test_close_refuses_verdict_not_all_clear` — verdict in {reproduces, partial-fix:*, run-failed, venv-*} → refuse
- `test_close_refuses_verify_json_case_id_mismatch` — JSON's case_id != --via-skill → refuse
- `test_close_passes_with_all_clear_verdict_and_matching_sha` — happy path DRY-RUN succeeds

## Risks + mitigations

| Risk | Mitigation |
|---|---|
| MRE re-run takes a long time per close | Default --timeout 1800s. Per-pair worker.py runs bounded by sweep's --timeout. |
| Sweep's nightly venv has been mutated since sweep | venv-drifted verdict (Step 2) blocks close mechanically. Operator must rebuild venv to match recorded versions before re-running verify_close_candidate. |
| Worker invocation differs from sweep invocation | `sweep.orchestrator.spawn_worker` is the canonical wrapper. Test #2 pins this. Hand-rolled subprocess.run forbidden. |
| MRE extraction differs from verify_repro.py | Reuse `verify_repro.extract_mre_from_body` + `extract_expected_signal_from_body` directly. |
| Verification adds 5-30 min per close | Acceptable for irreversible close decisions. Per-issue, not per-sweep. Cron prompt's wake step batches verification on close-candidates. |
| Old sweeps (pre-fail-loud) lack recorded versions | venv-versions-missing verdict blocks close. Forces re-run on a fail-loud-versions venv before close-mode can fire. |
| Operator tampers with verify JSON | sha256 pin in case file frontmatter — recomputed at close-time. Catches single-byte mutation. |

## Adversary-review case file

`subagents/adversary-review/invocations/adv-2026-05-10-193500-close-mode-rev5-design.md` — verdict approve-with-gaps; 8 gaps; all addressed in this rev 2.
