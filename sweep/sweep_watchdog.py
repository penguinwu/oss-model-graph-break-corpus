#!/usr/bin/env python3
"""Sweep watchdog — monitors progress, reports status, auto-restarts on failure.

Designed to run via cron every 10 minutes. Reads the sweep state file and
checkpoint to determine progress. Outputs a status message to stdout
(delivered to GChat by MyClaw cron).

Behavior:
  - Sweep running + progress made → compact progress line
  - Sweep running + no progress since last check → silent (no output)
  - Sweep process dead + not complete → auto-restart with --resume, report
  - Sweep complete → report final summary, mark as notified
  - No active sweep → silent (no output)

Usage:
  # Check a specific sweep
  python sweep_watchdog.py /path/to/sweep_results/output_dir

  # Check the most recent sweep (finds latest sweep_state.json)
  python sweep_watchdog.py --latest
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SWEEP_DIR = Path(__file__).resolve().parent
RUN_SWEEP = SWEEP_DIR / "run_sweep.py"
RESULTS_DIR = SWEEP_DIR.parent / "sweep_results"
MAX_RESTARTS = 3

# Additional directories to search for sweep results (handles path mismatch
# between PARA copy and repo copy of this script)
ADDITIONAL_RESULTS_DIRS = [
    Path("/home/pengwu/projects/oss-model-graph-break-corpus/sweep_results"),
    Path("/home/pengwu/.myclaw/PARA/01_projects/oss-model-graph-break-corpus/sweep_results"),
]


def load_state(output_dir):
    """Load sweep state file."""
    state_file = Path(output_dir) / "sweep_state.json"
    if not state_file.exists():
        return None
    with open(state_file) as f:
        return json.load(f)


def save_state(output_dir, state):
    """Save sweep state file."""
    state_file = Path(output_dir) / "sweep_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def read_progress(output_dir):
    """Read checkpoint and return progress summary.

    Deduplicates by (model name, mode) — auto-retries append additional lines
    for the same work item, so we keep the last result per (name, mode) to avoid
    exceeding 100%.
    """
    ckpt = Path(output_dir) / "pass1_checkpoint.jsonl"
    if not ckpt.exists():
        return {"completed": 0, "by_status": {}}

    # Last result per (name, mode) wins (retries overwrite earlier entries)
    by_work_item = {}
    with open(ckpt) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["name"], r.get("mode", "eval"))
                by_work_item[key] = r.get("status", "unknown")
            except (json.JSONDecodeError, KeyError):
                continue

    by_status = {}
    for status in by_work_item.values():
        by_status[status] = by_status.get(status, 0) + 1

    return {"completed": len(by_work_item), "by_status": by_status}


def is_pid_alive(pid):
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def format_elapsed(started_str):
    """Format elapsed time from ISO timestamp."""
    try:
        started = time.mktime(time.strptime(started_str, "%Y-%m-%dT%H:%M:%S"))
        elapsed_s = time.time() - started
        hours = int(elapsed_s // 3600)
        minutes = int((elapsed_s % 3600) // 60)
        if hours > 0:
            return f"{hours}h{minutes}m"
        return f"{minutes}m"
    except (ValueError, TypeError):
        return "?"


def format_progress(progress, total, state):
    """Format a compact progress line."""
    completed = progress["completed"]
    by_status = progress["by_status"]
    pct = completed / total * 100 if total > 0 else 0
    elapsed = format_elapsed(state.get("started", ""))

    # Compact status counts
    parts = []
    for status in ["clean", "graph_break", "eager_error", "create_error", "timeout", "worker_error"]:
        count = by_status.get(status, 0)
        if count > 0:
            short = {
                "clean": "clean", "graph_break": "break",
                "eager_error": "eager_err", "create_error": "create_err",
                "timeout": "timeout", "worker_error": "worker_err",
            }.get(status, status)
            parts.append(f"{short}:{count}")

    status_str = ", ".join(parts) if parts else "starting..."
    bar_len = 20
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    return f"Sweep [{bar}] {completed}/{total} ({pct:.0f}%) | {status_str} | {elapsed}"


def restart_sweep(state, output_dir):
    """Restart the sweep with --resume."""
    args = state.get("args", [])
    if not args:
        return False, "No saved args to restart with"

    # Find python binary in args
    python_bin = sys.executable
    for i, arg in enumerate(args):
        if arg == "--python" and i + 1 < len(args):
            python_bin = args[i + 1]
            break

    cmd = [python_bin, str(RUN_SWEEP), "--resume"] + args
    log_file = Path(output_dir) / "sweep.log"

    with open(log_file, "a") as log_fh:
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    # Update state
    state["pid"] = proc.pid
    state["status"] = "running"
    state["restart_count"] = state.get("restart_count", 0) + 1
    state["last_restart"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_state(output_dir, state)

    return True, proc.pid


def find_latest_sweep():
    """Find the most recent sweep output directory with a state file.

    Searches RESULTS_DIR (relative to this script) plus ADDITIONAL_RESULTS_DIRS
    to handle the case where the watchdog and sweep run from different copies.
    """
    candidates = []
    search_dirs = [RESULTS_DIR] + ADDITIONAL_RESULTS_DIRS
    seen = set()
    for results_dir in search_dirs:
        results_dir = results_dir.resolve()
        if results_dir in seen or not results_dir.exists():
            continue
        seen.add(results_dir)
        # Check subdirectories
        for d in results_dir.iterdir():
            state_file = d / "sweep_state.json"
            if d.is_dir() and state_file.exists():
                candidates.append((state_file.stat().st_mtime, str(d)))
        # Also check the results dir itself (sweep may write directly here)
        top_state = results_dir / "sweep_state.json"
        if top_state.exists():
            candidates.append((top_state.stat().st_mtime, str(results_dir)))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def main():
    if len(sys.argv) < 2 and "--latest" not in sys.argv:
        print("Usage: sweep_watchdog.py <output_dir> | --latest", file=sys.stderr)
        sys.exit(1)

    if "--latest" in sys.argv:
        output_dir = find_latest_sweep()
        if not output_dir:
            sys.exit(0)  # No active sweeps, silent
    else:
        output_dir = sys.argv[1]

    state = load_state(output_dir)
    if not state:
        sys.exit(0)  # No state file, silent

    status = state.get("status", "unknown")
    total = state.get("total_work_items", 0)
    pid = state.get("pid")

    # Sweep is still initializing (loading models, enumerating)
    if status == "initializing":
        alive = is_pid_alive(pid) if pid else False
        if alive:
            print(f"Sweep initializing (PID {pid}, started {format_elapsed(state.get('started', ''))} ago)")
        else:
            print(f"Sweep DIED during initialization (PID {pid} gone). "
                  f"Check logs — sweep never started running.")
            state["status"] = "failed"
            save_state(output_dir, state)
        sys.exit(0)

    # Already done and notified
    if status == "done":
        if not state.get("notified"):
            progress = read_progress(output_dir)
            print(f"Sweep complete! {format_progress(progress, total, state)}")
            state["notified"] = True
            save_state(output_dir, state)
        sys.exit(0)

    # Already failed permanently
    if status == "failed":
        sys.exit(0)

    progress = read_progress(output_dir)
    alive = is_pid_alive(pid) if pid else False

    if alive:
        # Sweep is running — check if progress changed since last report
        last_reported = state.get("last_reported_count", -1)
        if progress["completed"] == last_reported:
            sys.exit(0)  # No new progress, stay silent

        # Report progress
        print(format_progress(progress, total, state))

        # Save last reported count
        state["last_reported_count"] = progress["completed"]
        save_state(output_dir, state)

    else:
        # Sweep process is dead
        if progress["completed"] >= total:
            # Actually completed but state wasn't updated
            print(f"Sweep complete! {format_progress(progress, total, state)}")
            state["status"] = "done"
            state["notified"] = True
            save_state(output_dir, state)
        else:
            # Died mid-run — attempt restart
            restart_count = state.get("restart_count", 0)
            if restart_count >= MAX_RESTARTS:
                print(f"Sweep FAILED — died {restart_count} times, giving up. "
                      f"Last progress: {format_progress(progress, total, state)}")
                state["status"] = "failed"
                save_state(output_dir, state)
            else:
                ok, info = restart_sweep(state, output_dir)
                if ok:
                    print(f"Sweep died at {progress['completed']}/{total} — "
                          f"auto-restarted (attempt {restart_count + 1}/{MAX_RESTARTS}, "
                          f"PID {info})")
                else:
                    print(f"Sweep died at {progress['completed']}/{total} — "
                          f"restart failed: {info}")


if __name__ == "__main__":
    main()
