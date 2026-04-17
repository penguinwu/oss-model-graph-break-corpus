#!/bin/bash
# Auto-resume wrapper for run_experiment.py
# Usage: scripts/run_with_retry.sh CONFIG [--output DIR] [extra args...]
#
# Re-launches the experiment if it exits non-zero (killed, OOM, etc).
# Stops after MAX_RETRIES consecutive failures or when the run completes.

set -euo pipefail

MAX_RETRIES=${MAX_RETRIES:-3}
CONFIG="${1:?Usage: $0 CONFIG [--output DIR] [extra args...]}"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SWEEP_PYTHON:-$(head -1 "$CONFIG" >/dev/null 2>&1 && python3 -c "import json,sys; c=json.load(open('$CONFIG')); print(c.get('settings',{}).get('python_bin','python3'))")}"

# Find or create output dir from args
OUTPUT_DIR=""
EXTRA_ARGS=("$@")
for i in "${!EXTRA_ARGS[@]}"; do
    if [[ "${EXTRA_ARGS[$i]}" == "--output" ]] && (( i+1 < ${#EXTRA_ARGS[@]} )); then
        OUTPUT_DIR="${EXTRA_ARGS[$((i+1))]}"
        break
    fi
done

attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))

    if [[ -n "$OUTPUT_DIR" ]] && [[ -f "$OUTPUT_DIR/results.jsonl" ]] && [[ -s "$OUTPUT_DIR/results.jsonl" ]]; then
        echo "[retry] Attempt $attempt: resuming from $OUTPUT_DIR"
        "$PYTHON" "$SCRIPT_DIR/tools/run_experiment.py" run "$CONFIG" --resume "$OUTPUT_DIR" || {
            rc=$?
            echo "[retry] Exit code $rc. Retrying in 10s..."
            sleep 10
            continue
        }
    else
        echo "[retry] Attempt $attempt: starting fresh"
        "$PYTHON" "$SCRIPT_DIR/tools/run_experiment.py" run "$CONFIG" "$@" || {
            rc=$?
            echo "[retry] Exit code $rc. Retrying in 10s..."
            sleep 10
            continue
        }
    fi

    echo "[retry] Completed successfully."
    exit 0
done

echo "[retry] Failed after $MAX_RETRIES attempts."
exit 1
