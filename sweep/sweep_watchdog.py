#!/usr/bin/env python3
"""Sweep watchdog — observe progress, alert on issues. NO auto-restart.

Reports the state of a long-running sweep. Posts updates to a GChat space
when something changes (progress made, sweep stalled, sweep dead, sweep done).
Silent when nothing has changed.

Usage:
    sweep_watchdog.py <sweep_state_dir> [--post-to SPACE_ID] [--interval-min N]

Required:
    sweep_state_dir       Path to dir containing sweep_state.json + identify_streaming.jsonl
                          The caller MUST specify this — no auto-discovery.

Optional:
    --post-to SPACE_ID    GChat space for alerts (default: spaces/AAQANraxXE4)
    --interval-min N      How often this watchdog runs in minutes (default: 10).
                          Used to compute stalled threshold = 3 * interval.

Behavior (per tick):
    - Sweep alive + new progress      → POST progress line + update last_observation
    - Sweep alive + no progress > 3 intervals → POST stalled alert
    - Sweep alive + no progress < threshold   → silent
    - Sweep dead + complete           → POST "complete" once (idempotent via notified flag)
    - Sweep dead + incomplete         → POST "dead" once, with progress-this-run delta
                                        (flags ZERO progress as likely env issue)
    - State file missing              → exit 2 with clear error (no fallback)

State files:
    sweep_state.json     — owned by orchestrator; this script READS only
    sweep_watchdog.json  — owned by this script; tracks observations + notified flag

This is NOT an auto-restart watchdog. If the sweep dies, the watchdog reports it
and stops. A human (or separate launcher) decides whether to relaunch.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SPACE_DEFAULT = "spaces/AAQANraxXE4"
INTERVAL_DEFAULT_MIN = 10
STALLED_INTERVAL_MULT = 3  # stalled threshold = STALLED_INTERVAL_MULT × --interval-min


def is_alive(pid):
    """Return True if process with given pid is alive."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def count_completed(sweep_dir):
    """Authoritative completed-item count = lines in identify_streaming.jsonl."""
    f = sweep_dir / "identify_streaming.jsonl"
    if not f.exists():
        return 0
    return sum(1 for _ in f.open())


def fmt_elapsed(started_iso):
    """Format elapsed time since started_iso as 'XhYYm' or 'YYm'."""
    if not started_iso:
        return "?"
    try:
        # Accept both "Z" and naive timestamps
        ts = started_iso.replace("Z", "+00:00")
        try:
            s = datetime.fromisoformat(ts)
        except ValueError:
            s = datetime.strptime(started_iso, "%Y-%m-%dT%H:%M:%S")
        if s.tzinfo is None:
            s = s.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - s
        total = int(delta.total_seconds())
        h, rem = divmod(total, 3600)
        m = rem // 60
        return f"{h}h{m:02d}m" if h else f"{m}m"
    except Exception:
        return "?"


def post_gchat(space, message):
    """Post message to GChat space as bot."""
    subprocess.run(
        ["gchat", "send", space, message, "--as-bot"],
        capture_output=True, timeout=30,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sweep_state_dir", help="Dir containing sweep_state.json")
    p.add_argument("--post-to", default=SPACE_DEFAULT,
                   help=f"GChat space for alerts (default: {SPACE_DEFAULT})")
    p.add_argument("--interval-min", type=int, default=INTERVAL_DEFAULT_MIN,
                   help=f"Cron interval in minutes (default: {INTERVAL_DEFAULT_MIN}). "
                        f"Stalled threshold = {STALLED_INTERVAL_MULT}× this value.")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_state_dir).resolve()
    state_file = sweep_dir / "sweep_state.json"
    if not state_file.exists():
        print(f"ERROR: {state_file} not found. Pass the sweep dir explicitly.",
              file=sys.stderr)
        sys.exit(2)

    with state_file.open() as f:
        state = json.load(f)

    # Watchdog's own state (separate file — never touch sweep_state.json)
    wd_file = sweep_dir / "sweep_watchdog.json"
    if wd_file.exists():
        with wd_file.open() as f:
            wd = json.load(f)
    else:
        wd = {}

    pid = state.get("pid")
    total = state.get("total_work_items", 0)
    started = state.get("started", "")
    completed = count_completed(sweep_dir)
    last_obs = wd.get("last_observation", {})
    last_completed = last_obs.get("completed", -1)
    last_at = last_obs.get("at")
    notified = wd.get("notified", False)
    first_completed = wd.get("first_observation", {}).get("completed")
    if first_completed is None:
        # Initialize on first run — used to compute progress-this-run on death
        first_completed = completed

    # If we already announced a terminal state, stay silent
    if notified:
        sys.exit(0)

    elapsed = fmt_elapsed(started)
    alive = is_alive(pid)
    pct = (completed / total * 100) if total else 0
    msg = None
    set_notified = False

    if alive:
        # Running
        if completed > last_completed:
            delta = completed - last_completed if last_completed >= 0 else 0
            tag = f"+{delta} since last check" if last_completed >= 0 else "first observation"
            msg = (f"[🦦 sweep watchdog] {sweep_dir.name}: "
                   f"{completed}/{total} ({pct:.0f}%) | {tag} | {elapsed} elapsed")
        elif last_at:
            # No new progress — check stalled threshold
            try:
                t_iso = last_at.replace("Z", "+00:00") if last_at.endswith("Z") else last_at
                t = datetime.fromisoformat(t_iso)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                stuck_min = (datetime.now(timezone.utc) - t).total_seconds() / 60
                threshold_min = STALLED_INTERVAL_MULT * args.interval_min
                if stuck_min >= threshold_min:
                    msg = (f"[🦦 sweep watchdog] ⚠️ {sweep_dir.name}: STALLED — "
                           f"alive (PID {pid}) but no new progress for {int(stuck_min)} min. "
                           f"At {completed}/{total} ({pct:.0f}%).")
            except Exception:
                pass
        # else: no last_obs yet, skip
    else:
        # Dead
        if completed >= total and total > 0:
            msg = (f"[🦦 sweep watchdog] ✅ {sweep_dir.name}: COMPLETE — "
                   f"{completed}/{total} | {elapsed} total elapsed")
        else:
            progress_this_run = completed - first_completed
            if progress_this_run == 0:
                tag = "ZERO progress this run — likely env issue, review before relaunch"
            else:
                tag = f"+{progress_this_run} items this run"
            msg = (f"[🦦 sweep watchdog] ⚠️ {sweep_dir.name}: DEAD at "
                   f"{completed}/{total} ({pct:.0f}%) | {tag} | {elapsed} elapsed")
        set_notified = True

    # Persist watchdog state (always update last_observation)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wd["last_observation"] = {"at": now_iso, "completed": completed}
    if "first_observation" not in wd:
        wd["first_observation"] = {"at": now_iso, "completed": first_completed}
    if set_notified:
        wd["notified"] = True
    with wd_file.open("w") as f:
        json.dump(wd, f, indent=2)

    if msg:
        post_gchat(args.post_to, msg)
        print(msg)
    # else: silent — nothing to report this tick


if __name__ == "__main__":
    main()
