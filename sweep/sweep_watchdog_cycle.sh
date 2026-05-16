#!/usr/bin/env bash
# sweep_watchdog_cycle.sh — one watchdog cycle (cron-callable)
#
# v4 (2026-05-15): generalized to any sweep dir (not just nightly).
# See sweep/WATCHDOG_DESIGN.md.
#
# Mechanism:
#   1. Completion check (results.jsonl or explain_results.json present) → exit 99.
#   2. Resume-in-flight grace check (.resume_in_flight marker fresh < 20 min) → silent exit.
#   3. Run observer, dispatch on output:
#        DEAD → check death-spiral, write marker, launch resume (replays
#               state["args"] from sweep_state.json + --resume; works for
#               any subcommand: sweep / nightly / run).
#        STALLED → post observer message, no auto-action.
#        COMPLETE → post completion, disable cron.
#        ALIVE → post observer line as heartbeat EVERY cycle. Silence is
#                indistinguishable from a broken watchdog. Suppress with
#                env SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1 (echoes a stderr
#                breadcrumb so suppression is visible in cron logs).
#        MISSING_STATE → silent during startup grace window
#                        (SWEEP_WATCHDOG_STARTUP_GRACE_MIN, default 10min);
#                        post once per cycle after that.
#        "" → post verbatim line (treated as unhealthy observer).
#        anything else → page Peng "⚠️ UNEXPECTED OUTPUT" — no auto-action.
#
# Interval is caller-controlled: set `interval_seconds` on the cron job
# that invokes this script. The script itself is single-shot.
#
# Exit codes:
#   0    posted / silent (no resume launched)
#   1    auto-resume launched
#   2    setup error (missing/conflicting args, sweep dir not found)
#   99   sweep complete (caller should disable cron)

set -euo pipefail

DEFAULT_INTERVAL_MIN=15
DEFAULT_GCHAT_SPACE="spaces/AAQANraxXE4"

# REPO_ROOT derived from script location (script lives at
# $REPO_ROOT/sweep/sweep_watchdog_cycle.sh). No env var override —
# eliminating hidden config knobs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<EOF
Usage: sweep_watchdog_cycle.sh [SWEEP_DATE] [options]

One watchdog cycle for a sweep. Reads sweep state, posts a heartbeat /
alert / completion message to GChat, and triggers auto-resume on DEAD.
Designed to be invoked repeatedly by cron — the script itself is
single-shot.

Two modes (mutually exclusive — exactly one of SWEEP_DATE / --results-dir):

  Nightly mode (back-compat):
    Pass SWEEP_DATE as the first positional. Watches
    sweep_results/nightly/<SWEEP_DATE>/. Self-disables cron
    'sweep-watchdog-<SWEEP_DATE>' on COMPLETE.

  Generic mode (any sweep dir):
    Pass --results-dir DIR. Watches that exact dir. --cron-job-id
    OPTIONAL: when present, the watchdog disables that cron on
    COMPLETE; when absent (e.g., manual diagnostic, no cron), the
    COMPLETE arm just posts and exits. --run-label optional
    (defaults to basename of the dir).

Positional:
  SWEEP_DATE              Sweep identifier (e.g. 2026-05-15). Locates the
                          sweep dir at sweep_results/nightly/<SWEEP_DATE>/.
                          Mutually exclusive with --results-dir.

Options:
  --results-dir DIR       Direct path to the sweep dir. Use for non-nightly
                          sweeps. Mutually exclusive with SWEEP_DATE.
  --run-label STR         Label used in user-facing messages (e.g.
                          "NGB-sample"). Defaults to SWEEP_DATE if given,
                          else basename of --results-dir.
  --cron-job-id ID        myclaw jobs.id of the cron entry to disable on
                          COMPLETE. Defaults to sweep-watchdog-<SWEEP_DATE>
                          in nightly mode; OPTIONAL in --results-dir mode
                          (absent = skip cron-disable on COMPLETE).
  --interval-min N        Caller's cron cadence in minutes. Default: ${DEFAULT_INTERVAL_MIN}.
                          Used in the heartbeat message ("next check in
                          ~Nmin") so missing-heartbeat is recognizable
                          without guessing the schedule.
  --gchat-space SPACE     GChat space to post to. Default: ${DEFAULT_GCHAT_SPACE}.
  -h, --help              Show this help and exit.

Environment variables (all optional):
  SWEEP_WATCHDOG_STARTUP_GRACE_MIN   Minutes of silence for MISSING_STATE
                                     when sweep dir is fresh. Default: 10.
  SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS  Set to 1 to silence the ALIVE
                                     heartbeat (echoes a stderr breadcrumb
                                     so the suppression is visible).

Exit codes:
  0    posted / silent (no resume launched)
  1    auto-resume launched
  2    setup error (missing/conflicting args, sweep dir not found)
  99   sweep complete (caller should disable cron)

Examples:
  # Nightly mode, default 15-min cadence:
  sweep_watchdog_cycle.sh 2026-05-15

  # Generic mode for an ad-hoc sweep:
  sweep_watchdog_cycle.sh --results-dir /tmp/ngb-sample/ngb-on \\
                          --cron-job-id ngb-sample-watchdog \\
                          --run-label NGB-sample-on \\
                          --interval-min 10

EOF
}

# ── Arg parsing ─────────────────────────────────────────────────────────
# SWEEP_DATE is OPTIONAL first positional (back-compat with existing cron
# entries). If first arg doesn't look like a flag, it's SWEEP_DATE.
SWEEP_DATE=""
RESULTS_DIR=""
RUN_LABEL=""
CRON_JOB_ID=""
GCHAT_SPACE="$DEFAULT_GCHAT_SPACE"
INTERVAL_MIN=$DEFAULT_INTERVAL_MIN

if [ $# -gt 0 ] && [ "${1:0:1}" != "-" ]; then
    SWEEP_DATE="$1"
    shift
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --results-dir)
            if [ -z "${2:-}" ]; then echo "ERROR: --results-dir requires an argument" >&2; exit 2; fi
            RESULTS_DIR="$2"; shift 2
            ;;
        --run-label)
            if [ -z "${2:-}" ]; then echo "ERROR: --run-label requires an argument" >&2; exit 2; fi
            RUN_LABEL="$2"; shift 2
            ;;
        --cron-job-id)
            if [ -z "${2:-}" ]; then echo "ERROR: --cron-job-id requires an argument" >&2; exit 2; fi
            CRON_JOB_ID="$2"; shift 2
            ;;
        --gchat-space)
            if [ -z "${2:-}" ]; then echo "ERROR: --gchat-space requires an argument" >&2; exit 2; fi
            GCHAT_SPACE="$2"; shift 2
            ;;
        --interval-min)
            if [ -z "${2:-}" ] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --interval-min requires a positive integer" >&2; exit 2
            fi
            INTERVAL_MIN="$2"; shift 2
            ;;
        -h|--help)
            usage; exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Run with --help for usage." >&2
            exit 2
            ;;
    esac
done

# ── Validate mode + derive defaults ─────────────────────────────────────
if [ -n "$SWEEP_DATE" ] && [ -n "$RESULTS_DIR" ]; then
    echo "ERROR: SWEEP_DATE and --results-dir are mutually exclusive" >&2; exit 2
fi
if [ -z "$SWEEP_DATE" ] && [ -z "$RESULTS_DIR" ]; then
    echo "ERROR: pass either SWEEP_DATE (positional) or --results-dir DIR" >&2
    echo "Run with --help for usage." >&2; exit 2
fi

if [ -n "$SWEEP_DATE" ]; then
    # Nightly mode
    RESULTS_DIR="$REPO_ROOT/sweep_results/nightly/$SWEEP_DATE"
    RUN_LABEL="${RUN_LABEL:-$SWEEP_DATE}"
    CRON_JOB_ID="${CRON_JOB_ID:-sweep-watchdog-$SWEEP_DATE}"
else
    # Generic mode — --cron-job-id optional. When absent, COMPLETE arm
    # skips the sqlite UPDATE (caller has no cron, or manages it manually).
    RUN_LABEL="${RUN_LABEL:-$(basename "$RESULTS_DIR")}"
fi

MARKER="$RESULTS_DIR/.resume_in_flight"
MARKER_GRACE_MIN=20
STARTUP_GRACE_MIN="${SWEEP_WATCHDOG_STARTUP_GRACE_MIN:-10}"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: sweep dir not found: $RESULTS_DIR" >&2; exit 2
fi

# ── Step 1: completion check ────────────────────────────────────────────
if [ -f "$RESULTS_DIR/explain_results.json" ] || [ -f "$RESULTS_DIR/results.jsonl" ]; then
    COMPLETION_MARKER="$RESULTS_DIR/.completion_posted"
    if [ ! -f "$COMPLETION_MARKER" ]; then
        explain_size="(missing)"
        identify_size="(missing)"
        [ -f "$RESULTS_DIR/explain_results.json" ] && explain_size="$(stat -c %s "$RESULTS_DIR/explain_results.json" 2>/dev/null | numfmt --to=iec)B"
        [ -f "$RESULTS_DIR/identify_results.json" ] && identify_size="$(stat -c %s "$RESULTS_DIR/identify_results.json" 2>/dev/null | numfmt --to=iec)B"
        identify_lines="?"
        [ -f "$RESULTS_DIR/identify_streaming.jsonl" ] && identify_lines=$(wc -l < "$RESULTS_DIR/identify_streaming.jsonl")
        gchat send "$GCHAT_SPACE" \
            "[🦦 watchdog] ✅ Sweep $RUN_LABEL COMPLETE. identify=$identify_size ($identify_lines rows), explain=$explain_size. Watchdog disabling itself. Results: $RESULTS_DIR" \
            --as-bot || true
        touch "$COMPLETION_MARKER"
    fi
    echo "DISABLE_WATCHDOG: sweep $RUN_LABEL complete (explain_results.json or results.jsonl present)"
    exit 99
fi

# ── Step 2: resume-in-flight grace check ────────────────────────────────
if [ -f "$MARKER" ]; then
    age_sec=$(( $(date +%s) - $(stat -c %Y "$MARKER") ))
    age_min=$(( age_sec / 60 ))
    if [ "$age_min" -lt "$MARKER_GRACE_MIN" ]; then
        echo "SETUP_IN_PROGRESS: $RUN_LABEL resume marker age=${age_min}min (< ${MARKER_GRACE_MIN}min grace)"
        exit 0
    fi
    rm -f "$MARKER"
    echo "STALE_MARKER: removed .resume_in_flight (age=${age_min}min, grace=${MARKER_GRACE_MIN}min)"
fi

# ── Step 3: invoke observer ─────────────────────────────────────────────
# Observer is pure stdlib — any python3 works. No torch needed.
WATCHDOG_OUT="$(python3 "$REPO_ROOT/sweep/sweep_watchdog.py" "$RESULTS_DIR" 2>&1 || true)"
echo "$WATCHDOG_OUT"

# ── Step 4: dispatch on observer's verdict ──────────────────────────────
RESUME_STATE="$RESULTS_DIR/.watchdog_resume_state"
MAX_RESUME_NO_PROGRESS=3

case "$WATCHDOG_OUT" in
    *DEAD*)
        DONE_NOW=0
        [ -f "$RESULTS_DIR/identify_streaming.jsonl" ] && DONE_NOW=$(wc -l < "$RESULTS_DIR/identify_streaming.jsonl")
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
                "[🦦 watchdog] 🛑 Sweep $RUN_LABEL death-spiral: $NO_PROGRESS_COUNT consecutive failed resumes at item $DONE_NOW. Stopping auto-resume." \
                --as-bot || true
            cat > "$RESUME_STATE" <<EOF
last_done=$DONE_NOW
no_progress_count=$NO_PROGRESS_COUNT
last_check_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
escalated=true
EOF
            exit 0
        fi

        # ── Generalized auto-resume ─────────────────────────────────────
        # Read sweep_state.json for launcher_python + args. Works for any
        # subcommand (sweep / nightly / run).
        STATE_FILE="$RESULTS_DIR/sweep_state.json"
        if [ ! -f "$STATE_FILE" ]; then
            gchat send "$GCHAT_SPACE" \
                "[🦦 watchdog] ⚠️ Sweep $RUN_LABEL DEAD but sweep_state.json missing — cannot auto-resume. Manual intervention needed." \
                --as-bot || true
            exit 0
        fi

        # Read launcher_python via env-var passthrough to a single-quoted
        # heredoc — NO bash interpolation into the Python source, so paths
        # / args containing apostrophes can't inject. (adversary HIGH 1)
        LAUNCHER_PYTHON=$(STATE_FILE="$STATE_FILE" python3 - <<'PY'
import json, os, sys
try:
    s = json.load(open(os.environ['STATE_FILE']))
    print(s.get('launcher_python') or '')
except Exception as e:
    sys.stderr.write(f"failed to read launcher_python: {e}\n")
    print('')
PY
)
        if [ -z "$LAUNCHER_PYTHON" ]; then
            # Refuse to auto-resume — system python3 will almost certainly
            # be the wrong torch venv, silently corrupting the in-flight
            # sweep. Page Peng for manual resume. (adversary HIGH 2)
            FALLBACK_PY="$(command -v python3 || echo /usr/bin/python3)"
            gchat send "$GCHAT_SPACE" \
                "[🦦 watchdog] ⚠️ Sweep $RUN_LABEL DEAD; sweep_state.json missing launcher_python (likely pre-v4 state file). REFUSING auto-resume to avoid torch-mismatch corruption. Manual resume cmd: $FALLBACK_PY $REPO_ROOT/tools/run_experiment.py <orig-args> --resume" \
                --as-bot || true
            exit 0
        fi
        if [ ! -x "$LAUNCHER_PYTHON" ]; then
            gchat send "$GCHAT_SPACE" \
                "[🦦 watchdog] ⚠️ Sweep $RUN_LABEL DEAD; launcher_python ($LAUNCHER_PYTHON) not executable. Cannot auto-resume." \
                --as-bot || true
            exit 0
        fi

        cat > "$RESUME_STATE" <<EOF
last_done=$DONE_NOW
no_progress_count=$NO_PROGRESS_COUNT
last_check_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
        touch "$MARKER"
        RESUME_LOG="/tmp/sweep-resume-$(date +%Y%m%d-%H%M%S)-$$.log"
        # Single-quoted heredoc disables ALL bash interpolation; values
        # cross via env vars. Eliminates the entire injection class.
        # NOTE: do NOT add `< /dev/null` — it would replace the heredoc
        # stdin and python would read nothing. The heredoc IS the stdin;
        # tty-hang isn't a risk because no terminal is attached.
        setsid nohup env \
            STATE_FILE="$STATE_FILE" \
            LAUNCHER_PY="$LAUNCHER_PYTHON" \
            REPO_ROOT="$REPO_ROOT" \
            python3 - > "$RESUME_LOG" 2>&1 <<'PY' &
import json, os
state_file = os.environ['STATE_FILE']
launcher = os.environ['LAUNCHER_PY']
repo = os.environ['REPO_ROOT']
args = list(json.load(open(state_file)).get('args', []))
if '--resume' not in args:
    args.append('--resume')
os.execvp(launcher, [launcher, f'{repo}/tools/run_experiment.py'] + args)
PY
        disown
        gchat send "$GCHAT_SPACE" \
            "[🦦 watchdog] Sweep $RUN_LABEL was DEAD — auto-resumed. log=$RESUME_LOG marker=$MARKER (grace ${MARKER_GRACE_MIN}min). attempt $((NO_PROGRESS_COUNT + 1))/$MAX_RESUME_NO_PROGRESS at item $DONE_NOW." \
            --as-bot || true
        exit 1
        ;;
    *STALLED*)
        gchat send "$GCHAT_SPACE" "[🦦 watchdog] $WATCHDOG_OUT" --as-bot || true
        exit 0
        ;;
    *PHASE_TRANSITION*)
        gchat send "$GCHAT_SPACE" "[🦦 watchdog] $WATCHDOG_OUT" --as-bot || true
        exit 0
        ;;
    *COMPLETE*)
        # Observer reported COMPLETE but Step 1's file check didn't catch
        # it. Edge case (phase=done but results.jsonl not yet written).
        if [ -n "$CRON_JOB_ID" ]; then
            sqlite3 /home/pengwu/.myclaw/spaces/AAQANraxXE4/myclaw.db \
                "UPDATE jobs SET enabled=0 WHERE id='$CRON_JOB_ID';" 2>&1 || true
            gchat send "$GCHAT_SPACE" "[🦦 watchdog] Sweep $RUN_LABEL COMPLETE; cron $CRON_JOB_ID disabled." --as-bot || true
        else
            gchat send "$GCHAT_SPACE" "[🦦 watchdog] Sweep $RUN_LABEL COMPLETE; no --cron-job-id (caller manages cron)." --as-bot || true
        fi
        exit 99
        ;;
    *ALIVE*|"")
        if [ "${SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS:-0}" = "1" ]; then
            echo "heartbeat suppressed by SWEEP_WATCHDOG_HEARTBEAT_SUPPRESS=1" >&2
        else
            gchat send "$GCHAT_SPACE" "[🦦 watchdog] $WATCHDOG_OUT (next check in ~${INTERVAL_MIN}min)" --as-bot || true
        fi
        exit 0
        ;;
    *MISSING_STATE*)
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
        gchat send "$GCHAT_SPACE" \
            "[🦦 watchdog] ⚠️ UNEXPECTED OUTPUT — sweep $RUN_LABEL observer returned: ${WATCHDOG_OUT:0:200}" \
            --as-bot || true
        exit 0
        ;;
esac
