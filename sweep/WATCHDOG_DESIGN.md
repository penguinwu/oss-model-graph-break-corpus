# Sweep Watchdog — Simple Design (v3, 2026-05-10)

**Status:** Approved scope per Peng directive 2026-05-10 10:38 ET ("WatchDog is such a small script, don't make it into a big multi-week project ... fix watchdog today, test right away").

**Design philosophy:** Simple > clever. Two mechanisms only:
1. **Watchdog interval > setup time.** 15-min interval >> 2-min typical resume setup → no race.
2. **Resume-in-flight marker.** When auto-resume fires, write `.resume_in_flight` marker; orchestrator removes on first work-item completion. Watchdog defers DEAD declaration if marker is fresh.

Tier-aware progress threshold (NOT cron interval) handles the very_large model case.

## Components

| Component | Path | Role |
|---|---|---|
| Observer | `sweep/sweep_watchdog.py` | Stateless one-shot read; prints state; exits with code |
| Cycle script | `sweep/sweep_watchdog_cycle.sh` | Cron wrapper; auto-resume on DEAD; writes resume marker |
| Cron | jobs table `sweep-watchdog-<DATE>` | Interval is caller-controlled via `interval_seconds`. Nightly default: 900s (15 min). Shorter for short-running experiments (e.g. sample sweeps at 600s). |

## State files

| File | Owner | Purpose |
|---|---|---|
| `sweep_state.json` | orchestrator | pid, phase, total_work_items |
| `identify_streaming.jsonl` (and per-phase equivalents) | orchestrator | progress = line count |
| `.resume_in_flight` | cycle script | Marker: "auto-resume just launched, give grace until orchestrator writes its first work-item" |
| `.watchdog_resume_state` | cycle script | Death-spiral guard (preserved as-is) |

## Decision tree (cycle script, ~30 lines)

```
SWEEP_DATE=$1
RESULTS_DIR=<repo>/sweep_results/nightly/$SWEEP_DATE
MARKER=$RESULTS_DIR/.resume_in_flight
MARKER_GRACE_MIN=20   # how long after auto-resume to defer DEAD declaration

# Step 1: completion check
if [[ -f $RESULTS_DIR/explain_results.json || -f $RESULTS_DIR/results.jsonl ]]; then
    exit 99   # caller disables cron
fi

# Step 2: check resume-in-flight grace window
if [[ -f $MARKER ]]; then
    age_min=$(( ($(date +%s) - $(stat -c %Y $MARKER)) / 60 ))
    if (( age_min < MARKER_GRACE_MIN )); then
        echo "SETUP_IN_PROGRESS: resume marker age=${age_min}min (<${MARKER_GRACE_MIN}min grace)"
        exit 0   # silent, no action
    fi
    # Marker stale — must have failed during setup; remove + fall through
    rm -f $MARKER
fi

# Step 3: invoke observer (stateless)
observer_out=$(sweep_watchdog.py $RESULTS_DIR)
echo "$observer_out"

# Step 4: dispatch on observer's verdict
case "$observer_out" in
    *DEAD*)
        # Death-spiral guard (existing — preserved as-is)
        check_no_progress_count $RESULTS_DIR
        if (( count >= 3 )); then
            post_to_gchat "Death-spiral: $count consecutive failed resumes at item $done. Stopping."
            exit 0
        fi
        # Auto-resume
        touch $MARKER
        launch_resume $RESULTS_DIR
        post_to_gchat "Auto-resumed sweep ($SWEEP_DATE), marker placed (grace ${MARKER_GRACE_MIN}min)"
        exit 1
        ;;
    *STALLED*)
        post_to_gchat "$observer_out"   # no auto-action; surface to Peng
        exit 0
        ;;
    *ALIVE*|"")
        # Post the observer line as heartbeat every cycle. Silence is
        # indistinguishable from a broken watchdog, so default is loud.
        # Suppress with env SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1 (script
        # echoes a stderr breadcrumb so the suppression is visible).
        if [ "${SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS:-0}" = "1" ]; then
            echo "heartbeat suppressed by SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1" >&2
        else
            post_to_gchat "$observer_out"
        fi
        exit 0
        ;;
    *MISSING_STATE*)
        # Silent during sweep dir's startup grace window
        # (SWEEP_WATCHDOG_STARTUP_GRACE_MIN, default 10min). After grace
        # expires, post — sweep dir is misconfigured or state file deleted.
        dir_age_min=$(dir_age_minutes "$RESULTS_DIR")
        if (( dir_age_min < ${SWEEP_WATCHDOG_STARTUP_GRACE_MIN:-10} )); then
            : # silent
        else
            post_to_gchat "$observer_out (dir age ${dir_age_min}min)"
        fi
        exit 0
        ;;
    *COMPLETE*)
        post_to_gchat "Sweep $SWEEP_DATE COMPLETE; cron disabled."
        disable_cron
        exit 99
        ;;
    *)
        # Unknown output — observer may have crashed. ⚠️ prefix makes
        # this visually distinct from the heartbeat stream.
        post_to_gchat "⚠️ UNEXPECTED OUTPUT — sweep $SWEEP_DATE observer returned: $observer_out"
        exit 0
        ;;
    *)
        # Observer crashed/unknown output — page Peng, NO auto-action
        post_to_gchat "Watchdog observer returned unexpected output for $SWEEP_DATE: $observer_out — taking no action"
        exit 0
        ;;
esac
```

## Observer logic (~50 lines, stateless)

```python
# Read sweep_state.json
state = json.load(open(f'{sweep_dir}/sweep_state.json'))
pid = state.get('pid')
phase = state.get('phase', 'unknown')
total = state.get('total_work_items', 0)

# Phase → progress file mapping (existing PHASE_FILES dict, preserved)
fname = PHASE_FILES.get(phase, ('identify_streaming.jsonl', phase))[0]
done = wc -l f'{sweep_dir}/{fname}'

# Read previous progress from .watchdog_progress (cycle-script-side state — NOT notified flag)
prev_done, prev_at = read_progress_state(sweep_dir)
write_progress_state(sweep_dir, done, now)

# Tier-aware threshold (in MINUTES of no-progress before declaring STALLED)
PHASE_THRESHOLDS = {
    'identify': 30,            # HF models max ~9 min, with internal retries up to 30
    'auto_retry_timeout': 90,  # very_large tier = 1620s = 27 min per attempt; up to 3 attempts
    'auto_retry_errors': 30,   # serial single-worker, ~10 min per model max
    'explain': 30,             # similar to identify
    'report': 5,
    'done': 0,
}
threshold_min = PHASE_THRESHOLDS.get(phase, 30)

# Determine state
alive = is_alive(pid)
progress_age_min = (now - prev_at) / 60 if prev_at else 0

if not alive:
    print(f"DEAD pid={pid} phase={phase} done={done}/{total}")
    sys.exit(0)
if done > prev_done:
    print(f"ALIVE pid={pid} phase={phase} done={done}/{total} +{done-prev_done} since last check")
    sys.exit(0)
if progress_age_min >= threshold_min:
    print(f"STALLED pid={pid} phase={phase} done={done}/{total} no progress for {progress_age_min:.0f}min (threshold {threshold_min})")
    sys.exit(0)
# Alive but no progress yet, under threshold
print(f"ALIVE pid={pid} phase={phase} done={done}/{total} (no progress for {progress_age_min:.0f}min, threshold {threshold_min})")
sys.exit(0)
```

**Removed:**
- `notified` flag suppression (BUG 1) — observer is stateless, prints same output every call.
- `sweep_watchdog.json` state file — replaced by `.watchdog_progress` (cycle-script-side, simpler).

**Preserved:**
- Tier-aware threshold (replaces hardcoded 30 min)
- Phase detection from sweep_state.json
- Death-spiral guard (cycle-script side, 3-attempt no-progress)
- Self-disable on completion (exit 99)

## How this addresses the adversary's BLOCKING concerns

| Adversary concern | This design's response |
|---|---|
| B1 (15-min DEAD threshold contradicted by data) | Tier-aware threshold: identify=30min, auto_retry_timeout=90min, accommodates very_large 27-min models. |
| B2 (heartbeat on `_on_result` is wrong granularity) | Not using heartbeat-file approach. Progress = streaming-jsonl line count, tier-aware threshold tolerates long single-model compiles. |
| B3 (silent heartbeat write failures) | N/A — not using heartbeat. Streaming jsonl is already validated by orchestrator's existing write path. |
| B4 (GIL deadlock) | If orchestrator main thread is stuck > tier threshold, STALLED message posts. No auto-action (orchestrator is alive but stuck — restart wouldn't help if root cause is data-driven). |
| I5 (cron tick + orchestrator restart race) | `.resume_in_flight` marker explicitly handles this: cycle writes marker BEFORE launching, orchestrator removes on first work-item, watchdog defers DEAD until marker stale (20 min grace). |
| I6 (watchdog crash exit-code semantics) | `*)` default arm in cycle script case statement — pages Peng, no auto-action. |
| I7 (death-spiral chicken-and-egg) | Marker mechanism gives 20 min grace per resume attempt; orchestrator must produce at least one work-item write to clear the marker. If 3 resumes in a row don't clear marker (orchestrator dies during setup), death-spiral guard fires. |
| I8 (rewrite vs minimal change) | Adopted: this is a minimal change. ~30 lines cycle script + ~50 lines observer + ~5 lines orchestrator (marker removal). No new heartbeat mechanism. |
| I9 (worker-hang detection deferred) | STALLED state covers it: orchestrator alive but no progress > tier threshold → STALLED post. Human intervenes. |

**Phasing:** This IS the only phase. No "ship + watch for a week" — Peng's directive is "fix today, test right away."

## Test plan

1. **Manual kill test**: launch sweep, kill orchestrator mid-flight, verify watchdog declares DEAD within 1 cron interval (15 min) AND auto-resume succeeds.
2. **Setup-grace test**: launch resume, verify watchdog during setup phase reports SETUP_IN_PROGRESS (marker present), does NOT trigger spurious resume.
3. **Slow-model test**: spot-check that watchdog doesn't false-STALLED during a known slow model (e.g., GroundingDinoModel ~7.5 min compile).
4. **Auto-retry-timeouts threshold test**: verify watchdog uses 90-min threshold during `auto_retry_timeout` phase.

## Implementation scope

- `sweep/sweep_watchdog.py`: rewrite (~50 lines) — stateless, tier-aware, prints state.
- `sweep/sweep_watchdog_cycle.sh`: rewrite (~30 lines) — case-statement dispatch + marker mechanism.
- `sweep/run_sweep.py`: add ~5 lines to remove `.resume_in_flight` marker on first work-item completion.
- Cron interval: 15 min (was 10).
- Total: ~90 lines net change. ~1.5h with tests.
