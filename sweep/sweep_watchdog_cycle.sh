#!/usr/bin/env bash
# sweep_watchdog_cycle.sh — one watchdog cycle (intended to be invoked by a recurring cron)
#
# Behavior:
#   1. If sweep is COMPLETE (explain_results.json or results.jsonl present) → print
#      DISABLE_WATCHDOG and exit 99. The caller (the recurring cron's prompt)
#      should disable the cron when it sees exit code 99.
#   2. Run sweep_watchdog.py one-shot (observer).
#   3. If output contains "DEAD" → auto-resume via `tools/run_experiment.py nightly --resume`.
#      Guard against double-launch with pgrep.
#   4. If output contains "STALL" or "stalled" → emit alert (no action; cron caller posts).
#   5. Healthy → silent, exit 0.
#
# Usage: sweep_watchdog_cycle.sh <SWEEP_DATE> [--gchat-space SPACE]
#
# Where:
#   <SWEEP_DATE> — e.g. 2026-05-09 (used to find the sweep dir + name resume logs)
#   --gchat-space — optional GChat space for alerts (default: spaces/AAQANraxXE4)
#
# Exit codes:
#   0  — healthy or alerted, no action needed
#   1  — auto-resume launched
#   99 — sweep complete, caller should disable itself
#
# Added 2026-05-09 (Peng directive): replaces the inline bash logic that lived
# in the cron prompt body. Centralizing here keeps the watchdog code in the
# project repo (versioned) instead of in agent state dirs (not versioned).

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

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: sweep dir not found: $RESULTS_DIR" >&2
    exit 2
fi

# ── Step 1: completion check ────────────────────────────────────────────
if [ -f "$RESULTS_DIR/explain_results.json" ] || [ -f "$RESULTS_DIR/results.jsonl" ]; then
    echo "DISABLE_WATCHDOG: sweep $SWEEP_DATE complete (explain_results.json or results.jsonl present)"
    exit 99
fi

# ── Step 2: run watchdog observer ───────────────────────────────────────
WATCHDOG_OUT="$($TORCH211_PYTHON "$REPO_ROOT/sweep/sweep_watchdog.py" "$RESULTS_DIR" 2>&1 || true)"
echo "$WATCHDOG_OUT"

# ── Step 2.5: setup-in-progress guard ──────────────────────────────────
# If watchdog says DEAD but a resume launcher OR refresh-nightly subprocess
# is alive, the orchestrator is not really dead — its PID just hasn't been
# updated in sweep_state.json yet (setup phase in progress). Treat as
# silent + don't auto-resume. The next watchdog cycle will see the new PID.
#
# Process-match patterns are narrow enough to exclude the watchdog cycle
# script itself (which would otherwise self-match its own shell command):
#   refresh-nightly subprocess: must include "--venv" arg
#   nightly launcher: must end with output-dir arg matching this sweep
SETUP_IN_PROGRESS=0
# Filter to actual python subprocesses (process name = python). Shell wrappers
# whose cmdline happens to mention these patterns won't match because their
# process name is "bash". This avoids the watchdog cycle script self-matching.
if pgrep -af "^python.*run_experiment\.py refresh-nightly" > /dev/null \
   || pgrep -af "^python.*tools/run_experiment\.py refresh-nightly" > /dev/null \
   || pgrep -af "envs/.*python.*run_experiment\.py refresh-nightly" > /dev/null; then
    SETUP_IN_PROGRESS=1
elif pgrep -af "envs/.*python.*run_experiment\.py nightly .* --resume .* $RESULTS_DIR" > /dev/null; then
    # Launcher cmd uses "nightly --resume ... <results_dir>".
    # Orchestrator uses "sweep ..." (different subcommand) — won't match.
    SETUP_IN_PROGRESS=1
fi

if echo "$WATCHDOG_OUT" | grep -q "DEAD" && [ "$SETUP_IN_PROGRESS" = "1" ]; then
    echo "SETUP_IN_PROGRESS: watchdog reported DEAD but a resume launcher is in setup phase — staying silent"
    exit 0
fi

# ── Step 3: auto-resume on DEAD (with progress + max-attempts guard) ────
# Track resume attempts in a state file in the sweep dir. If N attempts pass
# WITHOUT advancing past the last-known-dead item count, stop auto-resuming
# and escalate to Peng. Prevents death-spirals on a deterministically-killer
# model (e.g., BltForCausalLM at item 127 — needs very_large timeout but
# regular sweep uses --timeout 180).
RESUME_STATE="$RESULTS_DIR/.watchdog_resume_state"
MAX_RESUME_NO_PROGRESS=3

if echo "$WATCHDOG_OUT" | grep -q "DEAD"; then
    if pgrep -af "run_experiment.*nightly.*--resume.*$SWEEP_DATE" > /dev/null; then
        gchat send "$GCHAT_SPACE" \
            "[🦦 Otter watchdog]: Sweep $SWEEP_DATE watchdog says DEAD but a resume process is already running — taking no action." \
            --as-bot || true
        exit 0
    fi
    # Compute current done-count from streaming jsonl (heartbeat-of-record).
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
        # Made progress since last resume → reset counter
        NO_PROGRESS_COUNT=0
    else
        NO_PROGRESS_COUNT=$((NO_PROGRESS_COUNT + 1))
    fi
    if [ "$NO_PROGRESS_COUNT" -ge "$MAX_RESUME_NO_PROGRESS" ]; then
        gchat send "$GCHAT_SPACE" \
            "[🦦 Otter watchdog]: 🛑 Sweep $SWEEP_DATE death-spiral detected — $NO_PROGRESS_COUNT consecutive resumes without progress past item $DONE_NOW. Stopping auto-resume; needs human triage. Likely a deterministically-killer model at item $((DONE_NOW + 1)). Watchdog cron remains active (will keep observing) but will NOT auto-resume until done-count advances or watchdog is manually disabled." \
            --as-bot || true
        # Persist updated state but don't resume
        cat > "$RESUME_STATE" <<EOF
last_done=$DONE_NOW
no_progress_count=$NO_PROGRESS_COUNT
last_check_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
escalated=true
EOF
        exit 0
    fi
    # Persist progress + bump counter, then resume
    cat > "$RESUME_STATE" <<EOF
last_done=$DONE_NOW
no_progress_count=$NO_PROGRESS_COUNT
last_check_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
    RESUME_LOG="/tmp/nightly-sweep-resume-$(date +%Y%m%d-%H%M%S).log"
    setsid nohup "$NIGHTLY_PYTHON" "$REPO_ROOT/tools/run_experiment.py" nightly \
        --force --resume --output-dir "$RESULTS_DIR" \
        > "$RESUME_LOG" 2>&1 < /dev/null &
    disown
    sleep 5
    RESUME_PID="$(pgrep -f "run_experiment.*nightly.*--resume.*$SWEEP_DATE" | head -1 || echo unknown)"
    gchat send "$GCHAT_SPACE" \
        "[🦦 Otter watchdog]: Sweep $SWEEP_DATE was DEAD — auto-resumed (PID $RESUME_PID, log $RESUME_LOG, attempt $((NO_PROGRESS_COUNT + 1))/$MAX_RESUME_NO_PROGRESS at item $DONE_NOW)." \
        --as-bot || true
    exit 1
fi

# Healthy → reset no-progress counter (don't carry stale state forward)
if [ -f "$RESUME_STATE" ]; then
    DONE_NOW=$(wc -l < "$RESULTS_DIR/identify_streaming.jsonl" 2>/dev/null || echo 0)
    cat > "$RESUME_STATE" <<EOF
last_done=$DONE_NOW
no_progress_count=0
last_check_iso=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
EOF
fi

# ── Step 4: alert on STALL ──────────────────────────────────────────────
if echo "$WATCHDOG_OUT" | grep -qiE "stall"; then
    gchat send "$GCHAT_SPACE" \
        "[🦦 Otter watchdog]: $WATCHDOG_OUT" \
        --as-bot || true
fi

# ── Step 5: healthy / alerted, no action ────────────────────────────────
exit 0
