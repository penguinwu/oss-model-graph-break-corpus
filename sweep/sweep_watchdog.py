#!/usr/bin/env python3
"""Sweep watchdog — observe progress, alert on issues. NO auto-restart.

Reports the state of a long-running sweep across all phases (identify,
auto_retry_timeout, auto_retry_errors, explain, report). Posts updates
to a GChat space when something changes.

Usage:
    sweep_watchdog.py <sweep_state_dir> [--post-to SPACE_ID] [--interval-min N]

Required:
    sweep_state_dir       Path to dir containing sweep_state.json + per-phase
                          streaming/checkpoint files. The caller MUST specify
                          this — no auto-discovery.

Optional:
    --post-to SPACE_ID    GChat space for alerts (default: spaces/AAQANraxXE4)
    --interval-min N      How often this watchdog runs in minutes (default: 10).
                          Used to compute stalled threshold = 3 * interval.

Phase detection:
    Primary: sweep_state.json's `phase` and `phase_total` fields (written by
    a patched run_sweep.py).
    Fallback: filesystem inference for unpatched sweeps already running.

State files:
    sweep_state.json     — owned by orchestrator; READ-ONLY.
    sweep_watchdog.json  — owned by this script; tracks observations + notified.

This is NOT an auto-restart watchdog. If the sweep dies, the watchdog reports it
once. A human (or separate launcher) decides whether to relaunch.
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
STALLED_INTERVAL_MULT = 3

# Phase → (streaming/checkpoint file used to count progress, human label)
PHASE_FILES = {
    "identify": ("identify_streaming.jsonl", "identify"),
    "auto_retry_timeout": ("auto_retry_timeout_checkpoint.jsonl", "auto-retry (timeouts)"),
    "auto_retry_errors": ("auto_retry_errors_checkpoint.jsonl", "auto-retry (errors)"),
    "explain": ("explain_checkpoint.jsonl", "explain"),
    "report": (None, "report-gen"),
    "done": (None, "done"),
}


def is_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def line_count(path):
    if not path.exists():
        return 0
    return sum(1 for _ in path.open())


def infer_phase(sweep_dir, total_work_items):
    """Filesystem fallback when sweep_state.json lacks `phase` (unpatched
    sweep). Walks files in reverse-chronological phase order.
    """
    if (sweep_dir / "explain_checkpoint.jsonl").exists():
        return "explain"
    if (sweep_dir / "auto_retry_errors_checkpoint.jsonl").exists():
        return "auto_retry_errors"
    if (sweep_dir / "auto_retry_timeout_checkpoint.jsonl").exists():
        return "auto_retry_timeout"
    # Identify done but no auto-retry checkpoint → unpatched sweep in
    # auto-retry, or never had auto-retry. Report "post-identify" so we
    # don't falsely claim "stalled".
    identify_done = line_count(sweep_dir / "identify_streaming.jsonl")
    if total_work_items and identify_done >= total_work_items - 50:
        # within ~50 of total → identify essentially done
        return "auto_retry_or_explain_unpatched"
    return "identify"


def compute_progress(sweep_dir, state):
    """Return (phase, label, completed, total) for the current sweep state.
    Falls back to filesystem inference if state lacks phase info.
    """
    total_work = state.get("total_work_items", 0)
    phase = state.get("phase")
    phase_total = state.get("phase_total")

    if phase and phase in PHASE_FILES:
        fname, label = PHASE_FILES[phase]
        if fname is None:
            return phase, label, 0, 0
        completed = line_count(sweep_dir / fname)
        denom = phase_total if phase_total else total_work
        return phase, label, completed, denom

    # Fallback for unpatched sweeps
    inferred = infer_phase(sweep_dir, total_work)
    if inferred == "auto_retry_or_explain_unpatched":
        # We don't know which phase; report the most informative count we have
        for fname in ("explain_checkpoint.jsonl",
                      "auto_retry_errors_checkpoint.jsonl",
                      "auto_retry_timeout_checkpoint.jsonl"):
            n = line_count(sweep_dir / fname)
            if n > 0:
                return inferred, f"post-identify ({fname})", n, 0
        # No checkpoint files yet — sweep is in auto-retry but writing only to stdout
        return inferred, "post-identify (no streaming file yet — sweep predates phase tracking)", 0, 0

    fname, label = PHASE_FILES.get(inferred, ("identify_streaming.jsonl", "identify"))
    return inferred, label, line_count(sweep_dir / fname), total_work


def fmt_elapsed(started_iso):
    if not started_iso:
        return "?"
    try:
        ts = started_iso.replace("Z", "+00:00")
        try:
            s = datetime.fromisoformat(ts)
        except ValueError:
            s = datetime.strptime(started_iso, "%Y-%m-%dT%H:%M:%S")
        if s.tzinfo is None:
            # Legacy writer emits naive ISO in LOCAL time (no Z marker).
            # Treat naive timestamps as local — assigning UTC here would
            # over-state elapsed by the local-vs-UTC offset on non-UTC boxes
            # (e.g. +7h on a PDT host). Current writer (post-2026-05-06) emits
            # UTC with Z, which the fromisoformat path above handles correctly.
            s = s.astimezone()
        delta = datetime.now(timezone.utc) - s
        total = int(delta.total_seconds())
        h, rem = divmod(total, 3600)
        m = rem // 60
        return f"{h}h{m:02d}m" if h else f"{m}m"
    except Exception:
        return "?"


def post_gchat(space, message):
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
        print(f"ERROR: {state_file} not found.", file=sys.stderr)
        sys.exit(2)

    with state_file.open() as f:
        state = json.load(f)

    wd_file = sweep_dir / "sweep_watchdog.json"
    if wd_file.exists():
        with wd_file.open() as f:
            wd = json.load(f)
    else:
        wd = {}

    pid = state.get("pid")
    started = state.get("started", "")
    alive = is_alive(pid)

    # Reset notified flag when the sweep has a new identity (PID changed since
    # last notification). Covers two cases:
    #   1. Prior watchdog announced DEAD; sweep was resumed under a new PID.
    #   2. Same PID, still alive — reset only if it was the same PID we previously
    #      announced about (prevents spurious re-announcement loops).
    notified_pid = wd.get("notified_pid")
    if wd.get("notified") and notified_pid != pid:
        # Different PID than the one we last notified about → new sweep instance
        wd["notified"] = False
        wd.pop("first_observation", None)
    elif alive and wd.get("notified"):
        # Alive but notified about THIS pid — must have been a stalled/dead
        # alert that turned out wrong. Reset and re-evaluate.
        wd["notified"] = False
        wd.pop("first_observation", None)

    notified = wd.get("notified", False)
    if notified:
        sys.exit(0)

    phase, label, completed, total = compute_progress(sweep_dir, state)
    last_obs = wd.get("last_observation", {})
    last_completed_for_phase = last_obs.get("completed", -1) if last_obs.get("phase") == phase else -1
    last_at = last_obs.get("at")
    first_obs = wd.get("first_observation")
    # Reset first_observation when phase OR pid changes, so progress-this-run
    # always reflects the CURRENT sweep instance (not a stale baseline carried
    # across kills+relaunches).
    if (first_obs is None
            or first_obs.get("phase") != phase
            or first_obs.get("pid") != pid):
        first_obs = {"phase": phase, "completed": completed, "pid": pid,
                     "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        wd["first_observation"] = first_obs

    elapsed = fmt_elapsed(started)
    pct = (completed / total * 100) if total else 0
    msg = None
    set_notified = False

    if alive:
        # Detect phase transition or progress
        phase_changed = last_obs.get("phase") and last_obs["phase"] != phase
        if phase_changed:
            msg = (f"[🦦 sweep watchdog] {sweep_dir.name}: ENTERED {label.upper()} phase | "
                   f"{completed}/{total} ({pct:.0f}%) | {elapsed} elapsed")
        elif completed > last_completed_for_phase:
            delta = completed - last_completed_for_phase if last_completed_for_phase >= 0 else 0
            tag = f"+{delta} since last check" if last_completed_for_phase >= 0 else "first observation"
            denom = f"{total}" if total else "?"
            msg = (f"[🦦 sweep watchdog] {sweep_dir.name} ({label}): "
                   f"{completed}/{denom} ({pct:.0f}%) | {tag} | {elapsed} elapsed")
        elif last_at:
            # No progress — check stalled threshold
            try:
                t_iso = last_at.replace("Z", "+00:00") if last_at.endswith("Z") else last_at
                t = datetime.fromisoformat(t_iso)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                stuck_min = (datetime.now(timezone.utc) - t).total_seconds() / 60
                threshold_min = STALLED_INTERVAL_MULT * args.interval_min
                if stuck_min >= threshold_min:
                    msg = (f"[🦦 sweep watchdog] ⚠️ {sweep_dir.name}: STALLED in {label} — "
                           f"alive (PID {pid}) but no new progress for {int(stuck_min)} min. "
                           f"At {completed}/{total or '?'} ({pct:.0f}%).")
            except Exception:
                pass
    else:
        # Dead
        # "Complete" judgment: phase=done OR (identify-done AND no other streaming files growing)
        sweep_status = state.get("status", "")
        report_or_done = phase in ("report", "done") or sweep_status == "done"
        identify_complete = (phase == "identify" and total and completed >= total)
        if report_or_done or identify_complete:
            msg = (f"[🦦 sweep watchdog] ✅ {sweep_dir.name}: COMPLETE in {label} | "
                   f"{completed}/{total or '?'} | {elapsed} total elapsed")
        else:
            first_completed = first_obs.get("completed", 0)
            progress_this_run = completed - first_completed
            if progress_this_run == 0:
                tag = "ZERO progress this run — likely env issue, review before relaunch"
            else:
                tag = f"+{progress_this_run} items this run"
            msg = (f"[🦦 sweep watchdog] ⚠️ {sweep_dir.name}: DEAD in {label} at "
                   f"{completed}/{total or '?'} ({pct:.0f}%) | {tag} | {elapsed} elapsed")
        set_notified = True

    # Persist watchdog state
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    wd["last_observation"] = {"at": now_iso, "phase": phase, "completed": completed}
    if set_notified:
        wd["notified"] = True
        wd["notified_pid"] = pid
    with wd_file.open("w") as f:
        json.dump(wd, f, indent=2)

    if msg:
        post_gchat(args.post_to, msg)
        print(msg)


if __name__ == "__main__":
    main()
