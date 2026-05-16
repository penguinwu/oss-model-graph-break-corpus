---
case_id: adv-2026-05-15-2025-watchdog-v4-generalize
subagent: adversary-review
date_utc: 2026-05-16T00:25:00Z
trigger: sweep-mechanical
files:
  - sweep/sweep_watchdog_cycle.sh
  - sweep/sweep_watchdog.py
  - sweep/run_sweep.py
  - sweep/WATCHDOG_DESIGN.md
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 3d71d73dce5d689e6fd064f3f92990d8c7e81e89c12e28635d7dde0ae35a0d19
---

## Reviewer raw output

```
VERDICT: approve-with-gaps
CONFIDENCE: high
CONFIDENCE_RATIONALE: Read the full cycle script, observer, and the early_state block; concerns are concrete (heredoc injection vector, fallback policy, edge cases in arg parsing).

FILES_READ:
- /home/pengwu/projects/oss-model-graph-break-corpus/sweep/sweep_watchdog_cycle.sh
- /home/pengwu/projects/oss-model-graph-break-corpus/sweep/sweep_watchdog.py
- /home/pengwu/projects/oss-model-graph-break-corpus/sweep/run_sweep.py (lines 580-660)
- /home/pengwu/projects/oss-model-graph-break-corpus/sweep/WATCHDOG_DESIGN.md

GAPS_FOUND:

1. [SEVERITY: high] Heredoc shell-injection vector via $STATE_FILE / $REPO_ROOT / $LAUNCHER_PYTHON in the auto-resume Python heredoc.
   The heredocs are unquoted (`python3 -c "..."`), so bash performs variable interpolation BEFORE Python sees the source. Paths with apostrophes break Python or worse — arbitrary code execution.
   SUGGESTED_FIX: pass values via env vars instead of splicing; use single-quoted heredoc (<<'PY') to disable bash interpolation entirely.

2. [SEVERITY: high] launcher_python fallback to system python3 is a silent torch-mismatch hazard for in-flight sweeps.
   For nightly sweeps in flight TODAY (state file written by old run_sweep.py), auto-resume on DEAD will execvp system python3 against tools/run_experiment.py — wrong torch silently corrupts the sweep.
   SUGGESTED_FIX: refuse to auto-resume when launcher_python is missing; GChat-page Peng instead of silent stderr WARN.

3. [SEVERITY: medium] --cron-job-id required even when caller has no cron to disable.
   SUGGESTED_FIX: make --cron-job-id optional in generic mode; if absent, skip the sqlite UPDATE.

4. [SEVERITY: medium] First-positional arg starting with `-` is silently treated as a flag; no POSIX `--` separator support.
   SUGGESTED_FIX: accept `--` as standard end-of-options sentinel.

5. [SEVERITY: medium] Observer COMPLETE-message: total=0 reads as `done=0/0` which looks like a broken sweep.
   SUGGESTED_FIX: distinct framing for empty sweeps.

6. [SEVERITY: medium] Observer fallback when identify_streaming.jsonl is missing on report/done phase reads `done=0/N` — the false-failure framing the fix was meant to eliminate.
   SUGGESTED_FIX: comment explaining why the streaming-file fallback is safe (status==done already pins the message).

7. [SEVERITY: low] WATCHDOG_DESIGN.md header says v3 but v4 section appended. Bash pseudocode still shows v3.
   SUGGESTED_FIX: bump header to v4; mark pseudocode as historical OR replace.

8. [SEVERITY: low] RUN_LABEL leak check — verified no leaks. All 5+ user-facing strings use $RUN_LABEL not $SWEEP_DATE.

SUGGESTED_ADDITIONAL_TESTS:
1. Heredoc injection: results-dir with apostrophe.
2. Old state-file (no launcher_python) auto-resume safety.
3. --cron-job-id requirement vs manual diagnostic run.
4. SWEEP_DATE that starts with `-` or contains `=`.
5. Empty-sweep COMPLETE message.
6. identify_streaming.jsonl missing during report/done phase.
7. Round-trip: launch a real (tiny) sweep with v4 launcher_python field, kill it, verify resume works. **MUST run before v4 ships (Gate 5).**

NOTES:
- Gap 1 (heredoc injection) and Gap 2 (silent torch-mismatch) are both high-severity, both should block until fixed.
- Bundling 4 changes in one commit makes bisecting harder.
- Test #7 (Gate 5) is MANDATORY — without it the orchestrator's args-capture path is unverified.
- WATCHDOG_DESIGN.md should bump version + mark v3 pseudocode as historical.
```

## My disposition

- **Gap 1 (HIGH) — heredoc injection:** ADDRESSED in same commit. Rewrote the auto-resume to use single-quoted heredoc (`<<'PY'`) with env-var passthrough. **Gate 5 verified the fix with a path containing an apostrophe (`/tmp/peng's-injection-test`) — auto-resume launched cleanly.**

- **Gap 2 (HIGH) — silent torch-mismatch fallback:** ADDRESSED in same commit. Pre-v4 state file (no `launcher_python`) now REFUSES auto-resume + posts a GChat page with the manual-resume command. Verified with synthetic pre-v4 state test.

- **Gap 3 (MED) — --cron-job-id requirement:** ADDRESSED. Made optional in generic mode; when absent, COMPLETE arm posts "no --cron-job-id (caller manages cron)" and skips the sqlite UPDATE.

- **Gap 4 (MED) — POSIX `--` separator:** DEFERRED. Filed as `watchdog-posix-double-dash-support` open-loop. Rationale: low-likelihood failure mode; the current "starts with `-`" heuristic works for all known use cases; proper fix is a `--` handler in the arg loop. Not block-worthy.

- **Gap 5 (MED) — empty-sweep COMPLETE:** ADDRESSED. Observer emits `COMPLETE pid={pid} phase={label} (empty sweep — 0 work items)` when `total == 0`. Verified with synthetic empty-sweep state.

- **Gap 6 (MED) — streaming-file missing on report/done:** ADDRESSED via inline comment in observer (status==done pin already handles the false-failure case; documented why). No code change needed beyond the comment + the explicit `done=total/total` pin from the original fix.

- **Gap 7 (LOW) — WATCHDOG_DESIGN.md version header:** ADDRESSED. Header bumped to "v4, 2026-05-15"; added an explicit note that the bash pseudocode is "v3 historical reference."

- **Gap 8 (LOW) — RUN_LABEL leak:** NO ACTION (reviewer's own verification was negative — no leaks found).

- **Tests 1-6 (synthetic):** all verified — manual test runs pasted into the commit-prep cycle. Test 7 (Gate 5, end-to-end real-launch) PASSED after fixing the heredoc `< /dev/null` bug discovered during the test.

- **NEW: heredoc `< /dev/null` bug.** Gate 5 found that the auto-resume command had `python3 - <<'PY' > LOG 2>&1 < /dev/null` — the `< /dev/null` REPLACED the heredoc stdin, so python read nothing and exited silently. This was NOT in the adversary's gap list (the adversary read the heredoc as if it would work). Fixed: removed `< /dev/null`, kept heredoc as the only stdin. End-to-end run after fix: PID 2496014 spawned the resumed sweep, completed BartModel/eval, produced explain_results.json, watchdog cron self-disabled (exit 99). **This is the strongest validation case for why Gate 5 is mandatory** — adversary-review alone wouldn't have caught it.
