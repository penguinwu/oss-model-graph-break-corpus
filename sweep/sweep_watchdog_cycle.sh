#!/usr/bin/env bash
# sweep_watchdog_cycle.sh — one watchdog cycle (cron-callable)
#
# v3 design (2026-05-10): simple dispatch on observer's single-line output.
# See sweep/WATCHDOG_DESIGN.md.
#
# Mechanism:
#   1. Completion check (results.jsonl or explain_results.json present) → exit 99.
#   2. Resume-in-flight grace check (.resume_in_flight marker fresh < 20 min) → silent exit.
#   3. Run observer, dispatch on output:
#        DEAD → check death-spiral, write marker, launch resume.
#        STALLED → post observer message, no auto-action.
#        ALIVE / COMPLETE / "" → silent.
#        anything else → page Peng "watchdog returned unexpected output" — no auto-action.
#
# Exit codes:
#   0  — silent / alerted (no resume launched)
#   1  — auto-resume launched
#   99 — sweep complete (caller disables cron)

set -euo pipefail

SWEEP_DATE="${1:-}"
if [ -z "$SWEEP_DATE" ]; then
    echo "ERROR: SWEEP_DATE required (e.g. 2026-05-09)" >&2
    exit 2
fi

GCHAT_SPACE="spaces/AAQANraxXE4"
if [ "${2:-}" = "--gchat-space" ] && [ -n "${3:-}" ]; then
    GCHAT_SPACE="$3"
fi

REPO_ROOT="${REPO_ROOT:-/home/pengwu/projects/oss-model-graph-break-corpus}"
RESULTS_DIR="$REPO_ROOT/sweep_results/nightly/$SWEEP_DATE"
NIGHTLY_PYTHON="${NIGHTLY_PYTHON:-/home/pengwu/envs/torch-nightly-cu126/bin/python}"
TORCH211_PYTHON="${TORCH211_PYTHON:-/home/pengwu/envs/torch211/bin/python}"

MARKER="$RESULTS_DIR/.resume_in_flight"
MARKER_GRACE_MIN=20   # how long after auto-resume to defer DEAD declaration

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: sweep dir not found: $RESULTS_DIR" >&2
    exit 2
fi

# ── Step 1: completion check ────────────────────────────────────────────
if [ -f "$RESULTS_DIR/explain_results.json" ] || [ -f "$RESULTS_DIR/results.jsonl" ]; then
    echo "DISABLE_WATCHDOG: sweep $SWEEP_DATE complete (explain_results.json or results.jsonl present)"
    exit 99
fi

# ── Step 2: resume-in-flight grace check ────────────────────────────────
if [ -f "$MARKER" ]; then
    age_sec=$(( $(date +%s) - $(stat -c %Y "$MARKER") ))
    age_min=$(( age_sec / 60 ))
    if [ "$age_min" -lt "$MARKER_GRACE_MIN" ]; then
        echo "SETUP_IN_PROGRESS: $SWEEP_DATE resume marker age=${age_min}min (< ${MARKER_GRACE_MIN}min grace)"
        exit 0
    fi
    # Marker stale — orchestrator never wrote first work-item or died during setup.
    # Remove marker so Step 4's DEAD path can fire (and the death-spiral guard counts it).
    rm -f "$MARKER"
    echo "STALE_MARKER: removed .resume_in_flight (age=${age_min}min, grace=${MARKER_GRACE_MIN}min)"
fi

# ── Step 3: invoke observer ─────────────────────────────────────────────
WATCHDOG_OUT="$($TORCH211_PYTHON "$REPO_ROOT/sweep/sweep_watchdog.py" "$RESULTS_DIR" 2>&1 || true)"
echo "$WATCHDOG_OUT"

# ── Step 4: dispatch on observer's verdict ──────────────────────────────
RESUME_STATE="$RESULTS_DIR/.watchdog_resume_state"
MAX_RESUME_NO_PROGRESS=3

case "$WATCHDOG_OUT" in
    *DEAD*)
        # Compute current done-count from streaming jsonl for death-spiral check.
        DONE_NOW=0
        if [ -f "$RESULTS_DIR/identify_streaming.jsonl" ]; then
            DONE_NOW=$(wc -l < "$RESULTS_DIR/identify_streaming.jsonl")
        fi
        LAST_DONE=0
        NO_PROGRESS_COUNT=0
        if [ -f "$RESUME_STATE" ]; then
            LAST_DONE=$(awk -F= '/^last_done=/{print $2}' "$RESUME_STATE" 2>/dev/null || echo 0)
            NO_PROGRESS_COUNT=$(awk -F= '/^no_progress_count=/{print $2}' "$RESUME_STATE" 2>/dev/null || echo 0)
        fi
        if [ "$DONE_NOW" -gt "$LAST_DONE" ]; then
            NO_PROGRESS_COUNT=0
        else
            NO_PROGRESS_COUNT=$((NO_PROGRESS_COUNT + 1))
        fi
        if [ "$NO_PROGRESS_COUNT" -ge "$MAX_RESUME_NO_PROGRESS" ]; then
            gchat send "$GCHAT_SPACE" \
                "[🦦 watchdog] 🛑 Sweep $SWEEP_DATE death-spiral: $NO_PROGRESS_COUNT consecutive failed resumes at item $DONE_NOW. Stopping auto-resume." \
                --as-bot || true
            cat > "$RESUME_STATE" <<EOF
last_done=$DONE_NOW
no_progress_count=$NO_PROGRESS_COUNT
last_check_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
escalated=true
EOF
            exit 0
        fi
        # Persist counter, write marker, launch resume.
        cat > "$RESUME_STATE" <<EOF
last_done=$DONE_NOW
no_progress_count=$NO_PROGRESS_COUNT
last_check_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
        touch "$MARKER"
        RESUME_LOG="/tmp/nightly-sweep-resume-$(date +%Y%m%d-%H%M%S).log"
        setsid nohup "$NIGHTLY_PYTHON" "$REPO_ROOT/tools/run_experiment.py" nightly \
            --force --resume --output-dir "$RESULTS_DIR" \
            > "$RESUME_LOG" 2>&1 < /dev/null &
        disown
        gchat send "$GCHAT_SPACE" \
            "[🦦 watchdog] Sweep $SWEEP_DATE was DEAD — auto-resumed. log=$RESUME_LOG marker=$MARKER (grace ${MARKER_GRACE_MIN}min). attempt $((NO_PROGRESS_COUNT + 1))/$MAX_RESUME_NO_PROGRESS at item $DONE_NOW." \
            --as-bot || true
        exit 1
        ;;
    *STALLED*)
        gchat send "$GCHAT_SPACE" "[🦦 watchdog] $WATCHDOG_OUT" --as-bot || true
        exit 0
        ;;
    *COMPLETE*)
        # Observer reported COMPLETE but the file-based completion check (Step 1)
        # didn't catch it. Edge case — might be `phase=done` but results.jsonl
        # not yet written. Disable cron preemptively.
        sqlite3 /home/pengwu/.myclaw/spaces/AAQANraxXE4/myclaw.db \
            "UPDATE jobs SET enabled=0 WHERE id='sweep-watchdog-$SWEEP_DATE';" 2>&1 || true
        gchat send "$GCHAT_SPACE" "[🦦 watchdog] Sweep $SWEEP_DATE COMPLETE; cron disabled." --as-bot || true
        exit 99
        ;;
    *ALIVE*|*MISSING_STATE*|"")
        # Healthy or sweep-not-yet-started. Silent.
        exit 0
        ;;
    *)
        # Unknown output — observer may have crashed. Page Peng, NO auto-action.
        gchat send "$GCHAT_SPACE" \
            "[🦦 watchdog] Sweep $SWEEP_DATE: observer returned unexpected output, taking no action: ${WATCHDOG_OUT:0:200}" \
            --as-bot || true
        exit 0
        ;;
esac
