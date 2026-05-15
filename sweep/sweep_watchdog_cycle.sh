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
#        COMPLETE → post completion, disable cron.
#        ALIVE → post observer line as heartbeat EVERY cycle (silence is
#                indistinguishable from a broken watchdog; per-cycle posts
#                are the liveness signal). Suppress with env
#                SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1 if you really want
#                quiet (script echoes a breadcrumb to stderr so the
#                suppression is visible in cron logs).
#        MISSING_STATE → silent during a startup grace window
#                (sweep dir < SWEEP_WATCHDOG_STARTUP_GRACE_MIN old);
#                post once per cycle after that.
#        "" → post verbatim line (treated as unhealthy observer).
#        anything else → page Peng "⚠️ UNEXPECTED OUTPUT" — no auto-action.
#                The ⚠️ prefix makes the page visually distinct from
#                heartbeats so habituation doesn't dull the alarm.
#
# Interval is caller-controlled: set `interval_seconds` on the cron job that
# invokes this script. The script itself is single-shot.
#
# Exit codes:
#   0  — posted / silent (no resume launched)
#   1  — auto-resume launched
#   99 — sweep complete (caller disables cron)

set -euo pipefail

DEFAULT_INTERVAL_MIN=15
DEFAULT_GCHAT_SPACE="spaces/AAQANraxXE4"

usage() {
    cat <<EOF
Usage: sweep_watchdog_cycle.sh <SWEEP_DATE> [options]

One watchdog cycle for a nightly sweep. Reads sweep state, posts a
heartbeat / alert / completion message to GChat, and triggers auto-resume
on DEAD. Designed to be invoked repeatedly by cron — the script itself is
single-shot.

Positional:
  SWEEP_DATE              Sweep identifier, e.g. 2026-05-15. Locates the
                          sweep dir at sweep_results/nightly/<SWEEP_DATE>/.

Options:
  --interval-min N        Caller's cron interval, in minutes. Default: ${DEFAULT_INTERVAL_MIN}.
                          The interval is the cadence at which YOUR cron
                          invokes this script. The script itself is
                          single-shot. This flag is passed in so the
                          heartbeat message can tell the reviewer "next
                          check in ~Nmin", so a missing-heartbeat is
                          recognizable as broken-watchdog without
                          guessing the schedule.
  --gchat-space SPACE     GChat space to post to. Default: ${DEFAULT_GCHAT_SPACE}.
  -h, --help              Show this help and exit.

Environment variables:
  REPO_ROOT                          Corpus repo root. Default: /home/pengwu/projects/oss-model-graph-break-corpus
  NIGHTLY_PYTHON                     Python for auto-resume (nightly venv).
  TORCH211_PYTHON                    Python for the observer script.
  SWEEP_WATCHDOG_STARTUP_GRACE_MIN   Minutes of silence for MISSING_STATE
                                     when sweep dir is fresh. Default: 10.
  SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS  Set to 1 to silence the ALIVE
                                     heartbeat (echoes a stderr breadcrumb
                                     so the suppression is visible).

Exit codes:
  0    posted / silent (no resume launched)
  1    auto-resume launched
  2    setup error (missing SWEEP_DATE, sweep dir not found, bad arg)
  99   sweep complete (caller should disable cron)

Examples:
  # Default 15-min cadence:
  sweep_watchdog_cycle.sh 2026-05-15

  # Faster 5-min cadence for a short sample sweep:
  sweep_watchdog_cycle.sh 2026-05-15 --interval-min 5

  # Post to a different space:
  sweep_watchdog_cycle.sh 2026-05-15 --gchat-space spaces/OTHER

EOF
}

# ── Arg parsing ─────────────────────────────────────────────────────────
# First positional is SWEEP_DATE (required, kept as positional for back-
# compat with existing cron entries). Remaining args are named flags.
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    usage
    exit 0
fi

SWEEP_DATE="${1:-}"
if [ -z "$SWEEP_DATE" ]; then
    echo "ERROR: SWEEP_DATE required (e.g. 2026-05-09)" >&2
    echo "Run with --help for usage." >&2
    exit 2
fi
shift

GCHAT_SPACE="$DEFAULT_GCHAT_SPACE"
INTERVAL_MIN=$DEFAULT_INTERVAL_MIN

while [ $# -gt 0 ]; do
    case "$1" in
        --gchat-space)
            if [ -z "${2:-}" ]; then
                echo "ERROR: --gchat-space requires an argument" >&2
                exit 2
            fi
            GCHAT_SPACE="$2"
            shift 2
            ;;
        --interval-min)
            if [ -z "${2:-}" ] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --interval-min requires a positive integer" >&2
                exit 2
            fi
            INTERVAL_MIN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="${REPO_ROOT:-/home/pengwu/projects/oss-model-graph-break-corpus}"
RESULTS_DIR="$REPO_ROOT/sweep_results/nightly/$SWEEP_DATE"
NIGHTLY_PYTHON="${NIGHTLY_PYTHON:-/home/pengwu/envs/torch-nightly-cu126/bin/python}"
TORCH211_PYTHON="${TORCH211_PYTHON:-/home/pengwu/envs/torch211/bin/python}"

MARKER="$RESULTS_DIR/.resume_in_flight"
MARKER_GRACE_MIN=20   # how long after auto-resume to defer DEAD declaration
STARTUP_GRACE_MIN="${SWEEP_WATCHDOG_STARTUP_GRACE_MIN:-10}"   # MISSING_STATE silence window after sweep dir created

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: sweep dir not found: $RESULTS_DIR" >&2
    exit 2
fi

# ── Step 1: completion check ────────────────────────────────────────────
if [ -f "$RESULTS_DIR/explain_results.json" ] || [ -f "$RESULTS_DIR/results.jsonl" ]; then
    # Post completion notification BEFORE exiting (silent-completion bug fix
    # 2026-05-10 13:13 ET — previously the cron disabled itself silently and
    # the human reviewer got no signal). Use a marker file so we only post
    # ONCE even if cycle fires again before cron disables.
    COMPLETION_MARKER="$RESULTS_DIR/.completion_posted"
    if [ ! -f "$COMPLETION_MARKER" ]; then
        # Build a small status summary
        explain_size="(missing)"
        identify_size="(missing)"
        [ -f "$RESULTS_DIR/explain_results.json" ] && explain_size="$(stat -c %s "$RESULTS_DIR/explain_results.json" 2>/dev/null | numfmt --to=iec)B"
        [ -f "$RESULTS_DIR/identify_results.json" ] && identify_size="$(stat -c %s "$RESULTS_DIR/identify_results.json" 2>/dev/null | numfmt --to=iec)B"
        identify_lines="?"
        [ -f "$RESULTS_DIR/identify_streaming.jsonl" ] && identify_lines=$(wc -l < "$RESULTS_DIR/identify_streaming.jsonl")
        gchat send "$GCHAT_SPACE" \
            "[🦦 watchdog] ✅ Sweep $SWEEP_DATE COMPLETE. identify=$identify_size ($identify_lines rows), explain=$explain_size. Watchdog disabling itself. Results: $RESULTS_DIR" \
            --as-bot || true
        touch "$COMPLETION_MARKER"
    fi
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
    *PHASE_TRANSITION*)
        # Sweep moved to a new phase (identify → auto_retry_timeout → auto_retry_errors → explain).
        # Post the transition so reviewer has visible signal during long-running sweeps.
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
    *ALIVE*|"")
        # Healthy. Post the observer line EVERY cycle as the heartbeat
        # (silence is indistinguishable from a broken watchdog — the
        # reviewer can't tell "fine" from "cron stopped firing" if ALIVE
        # is silent). The observer line itself carries the progress
        # signal ("+N since last check" or "no progress for Mmin"), so
        # the heartbeat doubles as a progress report.
        # Suppress with env SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1 — script
        # echoes a stderr breadcrumb so the suppression is visible in
        # cron logs (silent suppression would be the same anti-pattern
        # this whole change is fixing).
        # Peng directive 2026-05-15: silent-on-ALIVE is the anti-pattern.
        if [ "${SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS:-0}" = "1" ]; then
            echo "heartbeat suppressed by SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1" >&2
        else
            gchat send "$GCHAT_SPACE" "[🦦 watchdog] $WATCHDOG_OUT (next check in ~${INTERVAL_MIN}min)" --as-bot || true
        fi
        exit 0
        ;;
    *MISSING_STATE*)
        # sweep_state.json not yet written. Two cases:
        #   (a) sweep just launched — orchestrator hasn't written state
        #       yet. Normal startup window. Silent for STARTUP_GRACE_MIN
        #       to avoid spamming "MISSING_STATE" every cycle from launch.
        #   (b) sweep dir is misconfigured / state file was deleted —
        #       legitimately broken; surface to reviewer.
        # Discriminate by sweep dir mtime (proxy for "how long ago was
        # this sweep launched"). If dir is newer than the grace window,
        # silent; otherwise post.
        dir_age_sec=$(( $(date +%s) - $(stat -c %Y "$RESULTS_DIR" 2>/dev/null || echo 0) ))
        dir_age_min=$(( dir_age_sec / 60 ))
        if [ "$dir_age_min" -lt "$STARTUP_GRACE_MIN" ]; then
            echo "MISSING_STATE suppressed: sweep dir age ${dir_age_min}min < ${STARTUP_GRACE_MIN}min grace" >&2
        else
            gchat send "$GCHAT_SPACE" "[🦦 watchdog] $WATCHDOG_OUT (dir age ${dir_age_min}min)" --as-bot || true
        fi
        exit 0
        ;;
    *)
        # Unknown output — observer may have crashed. Page Peng, NO
        # auto-action. The ⚠️ prefix makes this visually distinct from
        # the heartbeat stream so habituation to ~96 heartbeats/day
        # doesn't dull this alarm.
        gchat send "$GCHAT_SPACE" \
            "[🦦 watchdog] ⚠️ UNEXPECTED OUTPUT — sweep $SWEEP_DATE observer returned: ${WATCHDOG_OUT:0:200}" \
            --as-bot || true
        exit 0
        ;;
esac
