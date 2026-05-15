---
case_id: adv-2026-05-15-1910-watchdog-heartbeat-on-alive
subagent: adversary-review
date_utc: 2026-05-15T23:15:00Z
trigger: sweep-mechanical
files:
  - sweep/sweep_watchdog_cycle.sh
  - sweep/WATCHDOG_DESIGN.md
persona_sha: 7645abb2a84d54c2b73edc7eceb5ee491c97b454
verdict: approve-with-gaps
output_sha256: 89d9847df2c00303533c02fa279eaf208732607f495f063dc50b91f8d26a5241
---

## Reviewer raw output

```
VERDICT: approve-with-gaps
CONFIDENCE: medium
CONFIDENCE_RATIONALE: Read-only review of a small, surgical bash change; the dispatch logic is straightforward but I found three concrete behavior gaps the change introduces or fails to address. Cannot run the script to confirm dynamically.
FILES_READ:
- /home/pengwu/projects/oss-model-graph-break-corpus/sweep/sweep_watchdog_cycle.sh
- /home/pengwu/projects/oss-model-graph-break-corpus/sweep/WATCHDOG_DESIGN.md
- /home/pengwu/projects/oss-model-graph-break-corpus/sweep/sweep_watchdog.py
- (cross-checked) /home/pengwu/projects/oss-model-graph-break-corpus/PLAN.md and the myclaw `jobs` table

GAPS_FOUND:

1. [SEVERITY: high] The unexpected-output `*)` arm now ALSO posts on every cycle, but it has NO suppression switch and the message includes raw observer stderr.
   The case-statement reaches `*)` whenever `WATCHDOG_OUT` does not contain any of {DEAD, STALLED, PHASE_TRANSITION, COMPLETE, ALIVE, MISSING_STATE} and is not empty. Because `WATCHDOG_OUT` captures `2>&1`, ANY Python warning, deprecation notice, traceback prefix, or stderr noise from `sweep_watchdog.py` (e.g., a one-time "Future warning: …" from a transitive import on the torch211 venv) will cause the observer's healthy state line to be PREFIXED with stderr, and may push the line into the `*)` branch (since the first 200 chars are emitted but the substring match still has to find `*ALIVE*` somewhere; if a multi-line stderr blob comes back the case match may still hit ALIVE — but if observer crashes outright with a Python traceback that contains none of the keywords, you get a per-cycle "unexpected output" page).
   WHY_IT_MATTERS: Pre-change, an observer crash paged Peng once per cycle but most cycles were silent so it stood out. Post-change, the reviewer is now habituated to ~96 messages/day per sweep; an observer-crash page will look like just another heartbeat in the stream and lose its alarm value. Worse, the `*)` arm has NO `WATCHDOG_HEARTBEAT_SUPPRESS` check, so the opt-out doesn't quiet the legitimate page either.
   SUGGESTED_FIX: Either (a) gate `*)` posting behind a separate `WATCHDOG_UNEXPECTED_SUPPRESS` env var with a clearly different default, OR (b) add a deduplication marker so identical "unexpected output" lines only post once per N cycles. At minimum, document that the `*)` arm intentionally pages every cycle and is NOT covered by the heartbeat suppress switch.

2. [SEVERITY: medium] The MISSING_STATE arm now fires loudly every cycle during the legitimate "sweep just launched, sweep_state.json not yet written" window.
   `sweep/sweep_watchdog.py:99-104` returns `MISSING_STATE: sweep_state.json not found at <path>` when the file is absent. The cycle script's `*MISSING_STATE*` glob hits the new ALIVE arm, so the message posts on every cron tick. There is a normal startup window (orchestrator launches, takes a few seconds → minutes to write sweep_state.json on first run; longer on cold venv) where this will spam Peng with `[🦦 watchdog] MISSING_STATE: sweep_state.json not found at …`. Pre-change, this branch was silent unless the observer's text happened to contain "since last check" — i.e., effectively silent at startup. The change inverts this without acknowledging that MISSING_STATE has a legitimate "not yet ready" interpretation distinct from "this sweep dir is broken."
   WHY_IT_MATTERS: First-cycle-after-launch noise will train the reviewer to ignore MISSING_STATE, which is the same anti-pattern (silence-vs-alarm conflation) the change is trying to FIX for ALIVE. If the reviewer ignores MISSING_STATE, a genuinely misconfigured sweep dir will be missed.
   SUGGESTED_FIX: Either (a) split MISSING_STATE out of the ALIVE arm and give it its own grace window similar to `MARKER_GRACE_MIN` (e.g., suppress for the first 10 minutes of a sweep dir's existence based on directory mtime), OR (b) explicitly document in the cycle script comment that MISSING_STATE will spam at sweep startup and is expected behavior.

3. [SEVERITY: medium] The docstring change to the `*ALIVE*|*MISSING_STATE*|""` arm claims "Suppress with env WATCHDOG_HEARTBEAT_SUPPRESS=1 if a caller really wants quiet" but the env var name is bare `WATCHDOG_HEARTBEAT_SUPPRESS` without a `SWEEP_` prefix and is checked unconditionally — meaning if a future caller exports it for a DIFFERENT watchdog (e.g., a hypothetical pt2-skill-discovery watchdog) the variable will silently apply here too.
   WHY_IT_MATTERS: Env-var collision is invisible; the script doesn't echo "heartbeat suppressed by env" so the reviewer would see weeks of silence and not know why. Given Peng's repeated "silence is indistinguishable from broken" framing, this exact failure mode is what the change is designed to PREVENT — but the suppress switch has no audit trail.
   SUGGESTED_FIX: When `WATCHDOG_HEARTBEAT_SUPPRESS=1` is set AND the arm would otherwise post, echo a one-line stderr breadcrumb (`echo "heartbeat suppressed by env" >&2`) and/or rename to `SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS` for namespace hygiene. Also: add the env var to the script's header doc so future maintainers see it without grepping.

SUGGESTED_ADDITIONAL_TESTS:

1. **MISSING_STATE-on-fresh-launch test** — verifies startup grace window
2. **Observer-crash unexpected-output dedup test** — verifies rate-limit on *) arm
3. **WATCHDOG_HEARTBEAT_SUPPRESS audit-trail test** — verifies stderr breadcrumb fires
4. **PHASE_TRANSITION not double-posted by ALIVE arm** — pins case-clause ordering
5. **Volume calibration test** — ~96 posts/day, all single-line

NOTES:
- Downstream consumer of `echo "$WATCHDOG_OUT"` at line 96 (cron log) unaffected — echo always fires.
- DEAD / STALLED / PHASE_TRANSITION / COMPLETE arms untouched. Risk surface is narrow.
- `pt212-sweep-watchdog` and `sweep-watchdog-2026-05-09` exist but enabled=0; no impact.
- `ngb-sample-watchdog` invokes sweep_watchdog.py DIRECTLY, not the cycle script; bypasses this change.
- WATCHDOG_DESIGN.md pseudo-code (line 75) only lists `*ALIVE*|""`, doesn't include `*MISSING_STATE*` — minor doc drift; worth aligning.
```

## My disposition

- **Gap 1 (HIGH) — `*)` arm habituation:** PARTIALLY addressed in same commit. Added `⚠️ UNEXPECTED OUTPUT` visual prefix to the `*)` arm message so it stands out from heartbeat stream. Rate-limit / dedup mechanism deferred — see OPEN-LOOPS entry `watchdog-unexpected-output-rate-limit`. Rationale for split: visual distinction is the lower-risk piece and addresses habituation directly; rate-limiting needs a state file + careful semantics (don't suppress legitimately new crash classes), which deserves its own design pass + adversary review.

- **Gap 2 (MEDIUM) — MISSING_STATE startup spam:** ADDRESSED in same commit. Split `MISSING_STATE` out of the `*ALIVE*` arm into its own case. Added `SWEEP_WATCHDOG_STARTUP_GRACE_MIN` (default 10) env knob; if sweep dir mtime < grace window, silent with stderr breadcrumb. After grace expires, post with `(dir age Nmin)` annotation so a true misconfiguration is surfaced as "MISSING_STATE despite dir being old."

- **Gap 3 (MEDIUM) — env var namespacing + audit:** ADDRESSED in same commit. Renamed `WATCHDOG_HEARTBEAT_SUPPRESS` → `SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS`. Added stderr breadcrumb (`heartbeat suppressed by SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1`) when suppression active. Also documented in script header docstring.

- **Suggested tests 1-5:** DEFERRED. The cycle script has no shell-test harness today (`bash -n` syntax check is the only mechanical gate). Filed OPEN-LOOPS entry `watchdog-cycle-script-shell-tests` to set up a `tests/shell/test_sweep_watchdog_cycle.sh` harness with the 5 reviewer-proposed scenarios + the new MISSING_STATE-grace and stderr-breadcrumb scenarios.

- **Notes — WATCHDOG_DESIGN.md MISSING_STATE drift:** ADDRESSED. Design doc pseudo-code now lists `*MISSING_STATE*` as its own arm matching the script.
