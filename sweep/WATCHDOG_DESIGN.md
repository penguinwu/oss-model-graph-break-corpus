# Sweep Watchdog — Current State + Proposed Redesign

**Status:** DRAFT for review (Peng). Implementation BLOCKED on Peng approval per directive 2026-05-10 10:19 ET ("watchdog logic seems to be very buggy ... slow down. Write down current design ... invoke adversarial agents ... then improve").

**Trust history:** the existing watchdog has falsely declared sweeps "DEAD" multiple times during legitimate setup phases (refresh-nightly, canary, orchestrator startup). Peng has lost trust. **Reliability matters more than feature richness.** This design prioritizes correctness + observable simplicity over breadth.

## Part 1 — Current State (what we have today)

### 1A. Components

| Component | Path | Role |
|---|---|---|
| **Observer** | `sweep/sweep_watchdog.py` | One-shot read of `sweep_state.json`; classifies as ALIVE / STALLED / DEAD / COMPLETE; posts to GChat once per state transition |
| **Cycle script** | `sweep/sweep_watchdog_cycle.sh` | Cron-callable bash wrapper: invokes observer, parses output, conditionally triggers auto-resume |
| **State file** | `<sweep_dir>/sweep_state.json` | Owned by orchestrator. Holds `pid`, `phase`, `started`, `total_work_items`. Watchdog reads. |
| **Observer state** | `<sweep_dir>/sweep_watchdog.json` | Owned by observer. Tracks `notified` flag, `last_observation`, `first_observation`. |
| **Resume state** | `<sweep_dir>/.watchdog_resume_state` | Owned by cycle script. Tracks `last_done`, `no_progress_count` (death-spiral guard). |
| **Cron** | `sqlite jobs table id=sweep-watchdog-<DATE>` | Per-sweep recurring cron, 10-min interval. Calls cycle script. Self-disables on completion. |

### 1B. Decision flow (current)

```
Cron fires (every 10 min)
  ↓
sweep_watchdog_cycle.sh <SWEEP_DATE>
  ↓
  Step 1: Check completion
    if explain_results.json or results.jsonl exists → exit 99 (caller disables cron)
  ↓
  Step 2: Run sweep_watchdog.py observer (one-shot)
    Reads sweep_state.json's pid + phase
    Reads identify_streaming.jsonl line count
    Classifies:
      - alive AND progress since last_observation → "+N since last check"
      - alive AND no progress for ≥ 30min → "STALLED"
      - dead (is_alive(pid) == False) → "DEAD"
      - phase=done OR identify-complete → "COMPLETE"
    Sets notified=True after posting DEAD/STALLED/COMPLETE (suppresses repeat alerts)
  ↓
  Step 2.5: Setup-in-progress guard (added 2026-05-09)
    If observer says DEAD AND a refresh-nightly OR resume-launcher process is alive
    → silent exit (the new orchestrator is still spawning)
  ↓
  Step 3: If observer said DEAD AND not in setup
    → check death-spiral state file
    → if no_progress_count < 3 → auto-resume
    → if ≥ 3 → escalate to Peng, no resume
  ↓
  Step 4: If observer said STALL → post message, no auto-action
  ↓
  Step 5: If healthy → silent
```

### 1C. Known bugs (observed in tonight's 2026-05-09 sweep)

**BUG 1 — `notified=True` permanently suppresses observer DEAD signal.**
After the watchdog observes DEAD once, it sets `notified=True` and exits silently on subsequent invocations (line 199 of `sweep_watchdog.py`). The cycle script then sees empty observer output → no "DEAD" string → no auto-resume. Result: tonight's sweep stayed dead from 01:53 to 09:42 (~7h) despite the watchdog cron firing every 10 min.

**Reset condition** (line 195-208): `notified` resets when `notified_pid != current_pid` (i.e., a new orchestrator wrote sweep_state.json with a fresh PID). But the cycle script's auto-resume launches a NEW orchestrator — that should change pid in sweep_state.json → reset notified. The bug is that the auto-resume itself didn't fire (because cycle script didn't see "DEAD" output) so the new orchestrator never came up.

This is a **classic chicken-and-egg deadlock**: notified suppression requires the new orchestrator to update pid; the new orchestrator requires the auto-resume; the auto-resume requires the observer to say DEAD; the observer is suppressed by notified.

**BUG 2 — Setup-phase race on PID.**
When auto-resume fires, the new orchestrator takes ~2-15 min to start (refresh-nightly + canary phases). Until then, sweep_state.json's `pid` field still points at the OLD dead orchestrator. Observer reads OLD pid → is_alive returns False → DEAD. The Step 2.5 setup-in-progress guard (added 2026-05-09) catches this case via pgrep matching — but pgrep matching is fragile (pattern dependent on command-line layout) and subject to false negatives.

**BUG 3 — STALLED threshold is high (30 min).**
With `--interval-min 10` and `STALLED_INTERVAL_MULT=3`, the observer waits 30 minutes of no-progress before declaring STALLED. For a hung worker subprocess, that's slow. But raising it produces false STALLEDs for slow models (some models take 15-30 min to compile cleanly).

**BUG 4 — STALLED has no auto-action.**
Observer posts a STALLED message but the cycle script has no auto-resume for STALLED — by design (the orchestrator IS alive, restart wouldn't help if a worker is hung). But if a worker is genuinely hung indefinitely, the only option is human intervention.

**BUG 5 — gchat send timeout in observer.**
Tonight's 03:55 EDT background-result showed: "watchdog ran but its internal `gchat send` timed out after 30s — the status message didn't deliver." The observer treats gchat as best-effort, but a hung gchat call wastes the watchdog's wall-clock. (Mitigated by 30s timeout.)

**BUG 6 — Cycle script's pgrep guard is fragile.**
The setup-in-progress guard uses `pgrep -af "envs/.*python.*run_experiment\.py refresh-nightly"`. If the launcher's command line varies (e.g., different venv path, additional flags), pgrep misses → false-DEAD → spurious auto-resume.

### 1D. What we got right

- **Death-spiral guard** (3-attempt no-progress counter in `.watchdog_resume_state`): correct mechanism for preventing infinite resume loops on a deterministic killer model. Worked correctly in past dogfoods.
- **Self-disable on completion** (cycle script exits 99 → cron prompt disables itself): correct end-state handling.
- **Per-sweep cron naming** (`sweep-watchdog-<DATE>`): allows multiple sweeps to coexist without cron-id collisions.
- **Observer keeps `gchat send` timeout-bounded**: prevents an indefinitely hung notification call from blocking the next watchdog tick.

---

## Part 2 — Proposed Redesign

### 2A. Design principles (priorities, in order)

1. **Single source of truth for "is the sweep alive?"** — one signal, not three (process aliveness + state file freshness + observer's notified flag).
2. **Bias against false-DEAD.** A false-DEAD spawns redundant resume processes, races, and alarming GChat noise. A true-DEAD missed for one cycle (10 min) costs only 10 min of latency. Asymmetric cost → asymmetric design.
3. **Observable simplicity.** Each watchdog invocation should print a single-line status that says exactly what it observed and what it decided. A human reading the cron log can immediately tell what happened.
4. **No notified-flag suppression.** The observer should produce the same output every time; the caller decides whether to act. Stateless observer + stateful caller is easier to reason about than stateful observer.

### 2B. Proposed mechanism

**Heartbeat-file-based** (instead of PID-based):

The orchestrator (already running) writes a heartbeat file `<sweep_dir>/.heartbeat` at every checkpoint (each completed work item). The file's mtime IS the canonical "is this sweep alive?" signal.

Watchdog rule:
```
heartbeat_age = now - mtime(.heartbeat)

if heartbeat_age < 5 min:        ALIVE — silent
elif heartbeat_age < 15 min:     SLOW — post once, no action
elif heartbeat_age >= 15 min:
  if any setup-in-progress process exists:  WAIT — silent
  else:                          DEAD — auto-resume (with death-spiral guard)
```

**Why heartbeat-file is more reliable than PID:**

- **Atomic.** A single `os.utime()` call updates mtime; no race between writing pid and the new pid being readable.
- **Survives orchestrator restarts.** When orchestrator dies + auto-resume launches a new one, the new one starts touching the same heartbeat file. No cross-PID confusion.
- **Sample rate is decoupled from cron interval.** Heartbeat updates every work-item completion (~30s for typical models). Cron fires every 10 min. This gives 20× over-sampling of the underlying signal — false-DEAD requires 15 min of zero progress, which only happens when (a) sweep is genuinely hung or dead, or (b) a single model is taking >15 min to process (rare; mostly large_models which use the per-tier extended timeout).

### 2C. Proposed components (simpler than current)

| Component | Path | Role |
|---|---|---|
| **Heartbeat writer** | added to `sweep/run_sweep.py` orchestrator main loop | After each work-item write, `pathlib.Path(".heartbeat").touch()` |
| **Watchdog CLI** | `sweep/sweep_watchdog.py` (rewritten) | Reads `.heartbeat` mtime, decides ALIVE/SLOW/DEAD, prints single-line status, exits with explicit exit code |
| **Cycle script** | `sweep/sweep_watchdog_cycle.sh` (much shorter) | Calls watchdog CLI, parses exit code, conditionally auto-resumes |
| **Resume state** | `<sweep_dir>/.watchdog_resume_state` | Same as today (death-spiral guard) |

### 2D. Watchdog CLI contract (new)

```
Usage: sweep_watchdog.py <sweep_dir> [--max-alive-min N] [--max-slow-min N]

Exits with:
  0  → ALIVE (heartbeat fresh, no action needed)
  1  → SLOW (heartbeat 5-15 min old, post warning, no action)
  2  → DEAD (heartbeat >15 min old, no setup-in-progress process)
  3  → SETUP_IN_PROGRESS (heartbeat stale BUT a launcher/refresh process is alive)
  4  → COMPLETE (results.jsonl or explain_results.json present)
  5  → MISSING_HEARTBEAT (sweep_dir or .heartbeat doesn't exist; fresh sweep that hasn't started yet)

Single-line output to stdout:
  [SWEEP_DATE] <state>: heartbeat age=<N>min phase=<P> done=<X>/<Y>

Optional --post-to flag posts the same single-line to GChat (best-effort; 30s timeout).
```

The cycle script then becomes:

```bash
#!/bin/bash
SWEEP_DATE=$1
sweep_watchdog.py $RESULTS_DIR
EXIT=$?
case $EXIT in
  0) ;;                          # ALIVE — silent
  1) sweep_watchdog.py $RESULTS_DIR --post-to spaces/AAQANraxXE4 ;;  # SLOW — alert
  2) auto_resume_with_guard ;;    # DEAD — resume
  3) ;;                          # SETUP_IN_PROGRESS — silent
  4) sqlite3 ... 'UPDATE jobs SET enabled=0 ...' ;;  # COMPLETE — disable cron
  5) sweep_watchdog.py $RESULTS_DIR --post-to spaces/AAQANraxXE4 ;;  # MISSING_HEARTBEAT — alert
esac
```

### 2E. What this fixes vs the current design

| Bug | Current behavior | New behavior |
|---|---|---|
| BUG 1 (notified suppression) | Observer goes silent forever after first DEAD | Observer is stateless — same output every call |
| BUG 2 (PID race) | Reads pid from state file; old pid → false-DEAD | Reads heartbeat mtime; new orchestrator touches same file |
| BUG 3 (STALLED at 30 min) | Hard-coded 30-min threshold | Tunable via CLI: `--max-alive-min` (default 5), `--max-slow-min` (default 15). Conservative defaults — ALIVE bias. |
| BUG 4 (STALLED no-action) | Posts message, no action | SLOW state posts a warning; DEAD triggers resume. Two distinct states with two distinct actions. |
| BUG 5 (gchat timeout in observer) | Timeout already mitigated | Same — preserved. |
| BUG 6 (pgrep fragility) | Uses pattern-matching pgrep | SETUP_IN_PROGRESS exit code shifts the pgrep complexity to the cycle script (kept as fallback); heartbeat-file IS the primary signal so pgrep fragility is reduced to "extra defense" rather than "load-bearing". |

### 2F. Heartbeat-file write — orchestrator-side

The orchestrator already writes to `identify_streaming.jsonl` after each work-item. Adding a heartbeat is one extra line in the same critical section:

```python
# In run_sweep.py, inside the per-work-item completion handler:
with open(streaming_path, 'a') as f:
    f.write(json.dumps(result) + '\n')
# NEW:
heartbeat_path = output_dir / '.heartbeat'
heartbeat_path.touch()  # atomic mtime update
```

For phases without per-item streaming (e.g., orchestrator setup, between-phase transitions), the orchestrator can touch the heartbeat at known checkpoints. Worst-case: heartbeat goes stale during a long single-model compile (some large_models take 5-10 min). The `--max-alive-min` default of 5 makes that produce a SLOW signal once, but `--max-slow-min` of 15 means we wait 15 min total before declaring DEAD. Well above worst-case single-model compile times.

### 2G. Out-of-scope (deferred)

These are real concerns but NOT in this v1 redesign:

- Multi-sweep coordination (today: per-sweep cron, no race; v2: maybe a sweep-manager daemon).
- Auto-fix loop (today: human triages DEAD; v2 could file an issue automatically).
- GPU memory pressure observability (today: silent OOM in workers; v2: read nvidia-smi alongside heartbeat).
- Worker subprocess hangs (today: orchestrator's own timeout catches; v2: per-worker heartbeats).

---

## Part 3 — Implementation Plan (NOT YET RUN)

After Peng approves the design + adversary feedback addressed:

1. Add heartbeat-file write to `sweep/run_sweep.py` orchestrator (~5 lines, after each `streaming_path.write` in the worker-result handler).
2. Rewrite `sweep/sweep_watchdog.py` to be heartbeat-based + stateless. Delete `sweep_watchdog.json`. Preserve `--post-to` for manual invocations.
3. Simplify `sweep/sweep_watchdog_cycle.sh` to a 20-line case statement (per 2D).
4. Update cron prompt to use the new exit-code-based flow.
5. Test: kill an orchestrator mid-sweep, verify watchdog declares DEAD within 15-25 min and triggers auto-resume; verify a slow-but-alive orchestrator doesn't trigger DEAD.
6. Document in `sweep/WATCHDOG.md` (user-facing readme; this file is the design rationale).

Total scope: ~150 lines net change, ~3-4h with testing.

---

## Part 4 — Adversary Review (PENDING)

This document is being submitted to an adversarial agent for review before implementation. Specific questions for the adversary:

1. **Heartbeat-file mtime — what failure modes does it have that pid-based doesn't?** (Filesystem clock skew across NFS? mtime granularity issues?)
2. **The 5-min/15-min thresholds — are they robust against long-compile models?** (Spot-check large_models.json for compile-time outliers.)
3. **The cycle script's exit-code dispatch — what happens if the watchdog CLI itself crashes (segfault, OOM)?** (Does the cycle script have a default that prevents incorrect action?)
4. **Setup-in-progress detection — do we still need pgrep, or is heartbeat-mtime sufficient?** (Can the orchestrator touch the heartbeat as soon as it starts, before refresh-nightly even runs?)
5. **The death-spiral guard — preserved as-is?** (Today it's a 3-attempt counter; should it be tightened/loosened?)

Adversary should also identify failure modes I haven't thought of.

---

## Part 5 — Adversary Review (2026-05-10)

Adversary identified 12 concerns, 4 BLOCKING. Verdict: **do NOT approve as-currently-designed**. Phase the work; don't ship a single big rewrite.

### BLOCKING concerns

**B1 — 15-min DEAD threshold is contradicted by checked-in data.**
`sweep/large_models.json` shows multiple `large` models routinely take 8-9 min wall (BltModel 493s, EdgeTamModel 570s, SwinModel 515s, GroundingDinoModel 450s). `very_large` retries can run 30-90 min for a single model. The doc's claim that 15-min is "well above worst-case single-model compile times" is **flatly wrong** — this is the kind of error the rewrite was supposed to restore trust around. Need tier-aware thresholds OR a timer-thread heartbeat (B2 below) decoupled from per-model wall-time.

**B2 — Heartbeat tied to `_on_result` callback has the same root cause as the bug we're fixing.**
The callback only fires when a worker subprocess returns. A single long worker holds the orchestrator's main loop; `_on_result` doesn't fire; heartbeat doesn't tick. This re-introduces the same "alive but no signal" failure mode under a new name. Fix: heartbeat from a separate `threading.Timer` daemon thread that ticks every ~30s independent of work-item granularity.

**B3 — Silent heartbeat-write failures (NFS hiccup, full disk).**
`pathlib.Path('.heartbeat').touch()` can no-op silently on some filesystems. On full disk, `utimes()` may succeed (no allocation) while real writes fail — orchestrator looks healthy from watchdog POV while actual results.jsonl writes are dropping. Fix: content-write (JSON line with timestamp + in-flight model + completed counter), not `touch()`. Then watchdog validates content (timestamp parseable + recent + in-flight model didn't repeat unchanged for >timeout).

**B4 — GIL deadlock / stuck-but-alive case unaddressed.**
If orchestrator main thread is blocked (deadlocked queue.get, pickle stuck on corrupt frame), pgrep says alive, callback never fires, heartbeat never advances. Same root cause as B2. Timer-thread heartbeat distinguishes "main thread stuck" (heartbeat thread alive, completed counter doesn't move) from "process dead" (no heartbeat at all).

### IMPORTANT concerns

**I5 — Race between cron tick + orchestrator restart.** Launcher should pre-touch heartbeat with mtime=now BEFORE forking the new orchestrator (claims responsibility, gets a grace window). Removes the load-bearing pgrep dependency.

**I6 — Watchdog-crash exit-code semantics undefined.** Cycle script's `case` has no default arm. Crashed watchdog (segfault=139, OOM=137, Python exception=1) silently no-ops. Worse: bash exit=1 collides with proposed SLOW=1 → misroutes "watchdog itself broken" → "post SLOW alert." Fix: reserve 10/20/30/40/50/60 for application states; ANY other exit pages Peng + takes no auto action.

**I7 — `.watchdog_resume_state` has the same chicken-and-egg as `notified`.** If new orchestrator dies during refresh-nightly before writing a heartbeat, `last_done` doesn't advance, `no_progress_count` increments to 3, death-spiral guard locks out further auto-resumes. Need TWO counters: "launches that never produced a first heartbeat" (different remediation) vs "launches that wrote heartbeats but didn't advance work-counter."

**I8 — Adding code where modifying suffices.** Minimum viable fix for BUG 1: delete the `if notified: sys.exit(0)` branch (2 lines). Heartbeat-file mechanism is a much larger surface (orchestrator changes + observer rewrite + new contracts) for a problem that already has a working-though-fragile mitigation. Phase the work.

**I9 — Deferred concerns that should NOT be deferred.** Worker-subprocess-hang is the same failure class as BUG 3 (STALLED at 30 min); without it, redesigned watchdog can't distinguish "alive, one worker hung" from "alive, just slow on a big model." Suppress SLOW posts during single-worker pass + timeout-retry pass at minimum.

### NICE-TO-FIX

**N10** — `.heartbeat` dotfile not greppable in routine `ls`. Use `heartbeat.json`.
**N11** — mtime introduces clock-source ambiguity. Content-write with ISO timestamp uses single clock for write+read.
**N12** — Test plan covers only one happy path. Convert each adversary concern to a regression test before merge.

### Adversary's overall verdict

> "Do not approve. Phase the work. Phase 1: minimal `notified` removal + tier-aware threshold, ship and watch for a week. Phase 2 (after Phase 1 stabilizes): heartbeat-from-timer-thread + content-write + launcher-pre-touch + explicit watchdog-crash defaults + every adversary concern as a regression test."

---

## Part 6 — Author Response (Otter)

The adversary is right. Specifically:

- B1 is a factual error in my doc. I claimed "well above worst-case" without checking large_models.json. Apologies.
- B2-B4 share a root cause: my heartbeat design is callback-driven, which inherits the failure modes of the current PID-based design. Timer-thread heartbeat is the correct pattern.
- I8 is the most important: ship the minimal fix first. The full rewrite was over-scoped for the trust-restoration goal.

**Revised proposal:**

**Phase 1 (this PR, ~30 lines, 1h):**
- Remove `if notified: sys.exit(0)` from `sweep_watchdog.py`. Move "don't double-post the same DEAD alert" logic to the cycle-script side (it already has state in `.watchdog_resume_state`).
- Make STALLED threshold tier-aware: when current phase is `auto_retry_timeout` or `auto_retry_errors` or single-worker pass, use 60-min threshold; otherwise 30-min.
- Add explicit `*)` default arm to `sweep_watchdog_cycle.sh` that posts a "watchdog itself errored, no auto-action taken" alert.
- Run for 1 week.

**Phase 2 (separate proposal, after Phase 1 settles):**
- Timer-thread heartbeat in orchestrator (write every 30s).
- Content-write `heartbeat.json` (JSON line with timestamp + in-flight model + completed counter).
- Watchdog reads + validates content.
- Launcher pre-touches heartbeat before forking.
- Two-counter death-spiral guard.
- Convert each B/I/N concern into a regression test.

This phasing matches the adversary's recommendation AND the "reliability matters more than feature richness" priority. Submitting Phase 1 design for approval; deferring Phase 2 until Phase 1 has flushed out other latent issues.
