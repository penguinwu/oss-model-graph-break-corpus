#!/usr/bin/env python3
"""Sweep watchdog check — pure decision script. NO destructive side effects.

Outputs a verdict (DONE | HEALTHY | INITIALIZING | HUNG | CRASHED) plus
fact-backed status fields (PID, etime, completed/total, last_progress_at).

The cron-driven watchdog calls this script, gets the verdict + facts, then
posts a fact-backed gchat message and (on HUNG/CRASHED) escalates to Peng.
*Never* auto-restarts the sweep — surfacing failures is human-triage's job.

Usage:
    python tools/sweep_watchdog_check.py <output_dir> [--state-file PATH]

Reads:
    <output_dir>/sweep_state.json            — pid, started_at, total_work_items
    <output_dir>/identify_checkpoint.jsonl   — completed work items (count)
    <output_dir>/identify_results.json       — DONE marker (presence check)

Writes (only if --state-file passed):
    <state_file> — script's own state for tracking last_count / last_progress_at
                   so we can detect "no new progress for N seconds" correctly.

Output (JSON to stdout):
    {
      "verdict": "HEALTHY" | "INITIALIZING" | "HUNG" | "CRASHED" | "DONE",
      "facts": {
        "pid": int | None,
        "pid_alive": bool,
        "proc_etime_s": int | None,
        "completed": int,
        "total": int,
        "completed_pct": float,
        "last_progress_at_epoch": int | None,
        "seconds_since_last_progress": int | None,
        "results_file_exists": bool,
        "ckpt_file_exists": bool
      },
      "explanation": str  // human-readable one-liner
    }

Exit code: always 0 (verdict in stdout). Prints to stderr on input errors.

Why decision logic is here, not in the cron prompt:
- The cron prompt fires a fresh claude session that has no state continuity.
  Putting decision logic in a deterministic script lets us unit-test it and
  guarantees consistent behavior across firings.
- The original watchdog bug (kill-and-restart loop) was caused by ad-hoc
  decision logic in the cron prompt that nobody tested. This script's
  test_sweep_watchdog_check.py exercises 9 scenarios including the bug case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# Verdict values
VERDICT_DONE = "DONE"
VERDICT_HEALTHY = "HEALTHY"
VERDICT_INITIALIZING = "INITIALIZING"
VERDICT_HUNG = "HUNG"
VERDICT_CRASHED = "CRASHED"

# Default 30-min init grace period. After process start, wait this long for
# the first new item to complete before classifying "no progress" as HUNG.
DEFAULT_INIT_GRACE_S = 30 * 60


# ──────────────────────────────────────────────────────────────────────
# Pure helpers (no side effects, easily testable)
# ──────────────────────────────────────────────────────────────────────

def is_pid_alive(pid: int | None) -> bool:
    """Return True iff PID is currently a live process. None → False."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def proc_etime_seconds(pid: int) -> int | None:
    """Read /proc/<pid>/stat, compute elapsed time in seconds. None on failure.

    /proc/<pid>/stat field 22 is starttime in clock ticks since boot.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        starttime_ticks = int(fields[21])
        ticks_per_sec = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        with open("/proc/uptime") as f:
            uptime_s = float(f.read().split()[0])
        proc_started_s_ago = uptime_s - (starttime_ticks / ticks_per_sec)
        return int(proc_started_s_ago)
    except (FileNotFoundError, IndexError, ValueError):
        return None


def count_completed_work_items(ckpt_path: Path) -> int:
    """Count unique (name, mode) entries in identify_checkpoint.jsonl.

    Retries append additional lines for the same work item — the last result
    per (name, mode) wins (matches Tiger's read_progress in sweep_watchdog.py).
    """
    if not ckpt_path.exists():
        return 0
    by_work_item: set[tuple[str, str]] = set()
    with open(ckpt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = (r["name"], r.get("mode", "eval"))
                by_work_item.add(key)
            except (json.JSONDecodeError, KeyError):
                continue
    return len(by_work_item)


def read_sweep_state(state_path: Path) -> dict:
    """Load sweep_state.json. Returns {} if missing/malformed."""
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return {}


def load_script_state(state_file: Path | None) -> dict:
    """Load script's own state (tracks last_count + last_progress_at_epoch)."""
    if state_file is None or not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text())
    except json.JSONDecodeError:
        return {}


def save_script_state(state_file: Path | None, state: dict) -> None:
    if state_file is None:
        return
    state_file.write_text(json.dumps(state, indent=2))


# ──────────────────────────────────────────────────────────────────────
# Decision logic (the load-bearing function — UNIT TESTED)
# ──────────────────────────────────────────────────────────────────────

def decide_verdict(
    *,
    pid_alive: bool,
    proc_etime_s: int | None,
    completed: int,
    last_completed_count: int | None,
    seconds_since_last_progress: int | None,
    results_file_exists: bool,
    init_grace_s: int = DEFAULT_INIT_GRACE_S,
) -> str:
    """Return one of: DONE | HEALTHY | INITIALIZING | HUNG | CRASHED.

    Decision tree (first match wins):
      1. results.json exists → DONE
      2. no PID alive → CRASHED
      3. completed > last_completed_count → HEALTHY (advanced since last tick)
      4. PID alive but no new completion AND seconds_since_last_progress < init_grace_s → INITIALIZING
      5. PID alive but no new completion AND seconds_since_last_progress >= init_grace_s → HUNG

    On first invocation (last_completed_count is None), we treat current
    completed count as a baseline. Process is INITIALIZING unless etime is
    already past grace.
    """
    # 1. Results file is the definitive completion signal
    if results_file_exists:
        return VERDICT_DONE

    # 2. No process alive → crashed
    if not pid_alive:
        return VERDICT_CRASHED

    # 3. Made progress since last tick → healthy
    if last_completed_count is not None and completed > last_completed_count:
        return VERDICT_HEALTHY

    # 4 + 5. No new progress; distinguish init from hung.
    # If we have seconds_since_last_progress (script has been tracking),
    # use that. Otherwise fall back to proc_etime_s (process age).
    grace_indicator = seconds_since_last_progress
    if grace_indicator is None:
        grace_indicator = proc_etime_s if proc_etime_s is not None else 0

    if grace_indicator < init_grace_s:
        return VERDICT_INITIALIZING
    else:
        return VERDICT_HUNG


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("output_dir", type=Path, help="Sweep output directory")
    p.add_argument("--state-file", type=Path, default=None,
                   help="Optional script-state file to track last_count + last_progress_at across ticks")
    p.add_argument("--init-grace-s", type=int, default=DEFAULT_INIT_GRACE_S,
                   help=f"Init grace period in seconds (default: {DEFAULT_INIT_GRACE_S})")
    args = p.parse_args()

    # Gather facts
    sweep_state = read_sweep_state(args.output_dir / "sweep_state.json")
    pid = sweep_state.get("pid")
    total = sweep_state.get("total_work_items", 0)

    pid_alive = is_pid_alive(pid)
    proc_etime = proc_etime_seconds(pid) if pid_alive else None

    ckpt_path = args.output_dir / "identify_checkpoint.jsonl"
    completed = count_completed_work_items(ckpt_path)
    results_path = args.output_dir / "identify_results.json"
    results_exists = results_path.exists()

    # Load script's state (last_count + last_progress_at_epoch)
    script_state = load_script_state(args.state_file)
    now = int(time.time())

    # PID-change reset: if the sweep PID has changed since the prior tick,
    # the prior run is dead/replaced. Clear progress-tracking state so
    # `seconds_since_last_progress` doesn't carry across launches and falsely
    # trip HUNG. Preserve `last_reported_*` so suppression-vs-transition still
    # decides correctly when the new run hits its first verdict.
    prior_pid = script_state.get("last_reported_pid")
    pid_changed = prior_pid is not None and pid != prior_pid
    if pid_changed:
        script_state.pop("last_count", None)
        script_state.pop("last_progress_at_epoch", None)

    last_count = script_state.get("last_count")
    last_progress_at = script_state.get("last_progress_at_epoch")
    seconds_since_last_progress = (now - last_progress_at) if last_progress_at else None

    verdict = decide_verdict(
        pid_alive=pid_alive,
        proc_etime_s=proc_etime,
        completed=completed,
        last_completed_count=last_count,
        seconds_since_last_progress=seconds_since_last_progress,
        results_file_exists=results_exists,
        init_grace_s=args.init_grace_s,
    )

    # Suppression: only post when (verdict, pid) transitions from last reported.
    # Prevents re-posting CRASHED every 10 min when nothing has changed.
    last_reported_verdict = script_state.get("last_reported_verdict")
    last_reported_pid = script_state.get("last_reported_pid")
    should_post = (verdict != last_reported_verdict) or (pid != last_reported_pid)

    # Update script state for next tick (if state file passed)
    if args.state_file is not None:
        new_state = dict(script_state)
        new_state["last_count"] = completed
        # Update last_progress_at_epoch only when count actually advanced
        if last_count is None or completed > last_count:
            new_state["last_progress_at_epoch"] = now
        elif "last_progress_at_epoch" not in new_state:
            # First tick — initialize
            new_state["last_progress_at_epoch"] = now
        # Record what we're about to post (so next tick can suppress repeats)
        if should_post:
            new_state["last_reported_verdict"] = verdict
            new_state["last_reported_pid"] = pid
        save_script_state(args.state_file, new_state)

    # Build explanation
    if verdict == VERDICT_DONE:
        expl = f"sweep complete: {completed}/{total}"
    elif verdict == VERDICT_HEALTHY:
        delta = completed - (last_count or 0)
        expl = f"PID {pid} alive, advanced +{delta} since last tick → {completed}/{total}"
    elif verdict == VERDICT_INITIALIZING:
        if proc_etime is not None:
            expl = f"PID {pid} alive {proc_etime//60}m, no progress yet, within {args.init_grace_s//60}m grace"
        else:
            expl = f"PID {pid} alive, no progress yet, within grace"
    elif verdict == VERDICT_HUNG:
        ssp = seconds_since_last_progress if seconds_since_last_progress is not None else (proc_etime or 0)
        expl = f"PID {pid} alive but no progress for {ssp//60}m (>= {args.init_grace_s//60}m grace) — HUMAN ATTENTION NEEDED"
    elif verdict == VERDICT_CRASHED:
        expl = f"no PID alive, results file missing, last completed: {completed}/{total} — HUMAN ATTENTION NEEDED"
    else:
        expl = "unknown"

    out = {
        "verdict": verdict,
        "should_post": should_post,
        "facts": {
            "pid": pid,
            "pid_alive": pid_alive,
            "proc_etime_s": proc_etime,
            "completed": completed,
            "total": total,
            "completed_pct": round(completed / total * 100, 1) if total else 0.0,
            "last_progress_at_epoch": last_progress_at,
            "seconds_since_last_progress": seconds_since_last_progress,
            "results_file_exists": results_exists,
            "ckpt_file_exists": ckpt_path.exists(),
        },
        "explanation": expl,
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
