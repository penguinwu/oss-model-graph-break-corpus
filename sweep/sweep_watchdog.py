#!/usr/bin/env python3
"""Sweep watchdog — stateless one-shot observer.

Reads sweep_state.json + per-phase progress files; prints single-line state.
The cycle-script (sweep_watchdog_cycle.sh) consumes the output and decides
whether to auto-resume.

Stateless: NO `notified` flag, NO `sweep_watchdog.json`. The same input state
always produces the same output. Caller-side state (`.watchdog_progress`)
tracks last-observed progress for STALLED detection.

Usage: sweep_watchdog.py <sweep_state_dir> [--post-to SPACE] [--no-write-state]

States printed:
  ALIVE pid=<P> phase=<PH> done=<X>/<Y> [+N since last check | no progress for Mmin, threshold Tmin]
  STALLED pid=<P> phase=<PH> done=<X>/<Y> no progress for Mmin (threshold Tmin)
  DEAD pid=<P> phase=<PH> done=<X>/<Y>
  COMPLETE pid=<P> phase=<PH> done=<X>/<Y>
  MISSING_STATE: <reason>

Exit code is always 0 unless misconfigured. The cycle script dispatches on
output substring (DEAD/STALLED/ALIVE/COMPLETE).

Design doc: sweep/WATCHDOG_DESIGN.md (v3, 2026-05-10).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Phase → (progress filename, human label)
PHASE_FILES = {
    "identify": ("identify_streaming.jsonl", "identify"),
    "auto_retry_timeout": ("auto_retry_timeout_checkpoint.jsonl", "auto-retry (timeouts)"),
    "auto_retry_errors": ("auto_retry_errors_checkpoint.jsonl", "auto-retry (errors)"),
    "explain": ("explain_checkpoint.jsonl", "explain"),
    "report": (None, "report-gen"),
    "done": (None, "done"),
}

# Phase → minutes of zero progress before declaring STALLED.
# Tuned per adversary review (2026-05-10):
#   identify: HF models max ~9 min (GroundingDino, EdgeTam, etc.); 30 buffers internal retries.
#   auto_retry_timeout: very_large tier = 1620s = 27 min per attempt; 90 covers up to ~3 attempts.
#   auto_retry_errors: serial 1-worker, ~10 min per model max; 30 is generous.
PHASE_THRESHOLDS_MIN = {
    "identify": 30,
    "auto_retry_timeout": 90,
    "auto_retry_errors": 30,
    "explain": 30,
    "report": 5,
    "done": 0,
}


def is_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def post_gchat(space: str, message: str):
    try:
        subprocess.run(
            ["gchat", "send", space, message, "--as-bot"],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass  # best-effort — don't let gchat failure block watchdog


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("sweep_state_dir")
    p.add_argument("--post-to", default=None,
                   help="GChat space — posts the state line. Default: don't post (cycle script decides).")
    p.add_argument("--no-write-state", action="store_true",
                   help="Don't update .watchdog_progress (used by cycle script for diagnostic re-runs).")
    args = p.parse_args()

    sweep_dir = Path(args.sweep_state_dir).resolve()
    state_file = sweep_dir / "sweep_state.json"
    if not state_file.exists():
        msg = f"MISSING_STATE: sweep_state.json not found at {state_file}"
        print(msg)
        if args.post_to:
            post_gchat(args.post_to, f"[🦦 watchdog] {msg}")
        sys.exit(0)

    with state_file.open() as f:
        state = json.load(f)

    pid = state.get("pid")
    phase = state.get("phase", "unknown")
    total = state.get("total_work_items", 0)

    fname, label = PHASE_FILES.get(phase, ("identify_streaming.jsonl", phase))
    if fname is None:
        # report/done phases have no per-phase progress file. Fall back to
        # identify_streaming.jsonl as the count of work items actually
        # processed by the sweep. Without this, COMPLETE reads "done=0/N"
        # which looks like a failure at first glance.
        done = line_count(sweep_dir / "identify_streaming.jsonl")
    else:
        done = line_count(sweep_dir / fname)

    # Check for completion BEFORE process aliveness.
    sweep_status = state.get("status", "")
    if sweep_status == "done" or phase == "done":
        # On COMPLETE, pin done=total/total so the message reads as
        # success rather than partial-progress. Empty sweeps (total=0,
        # e.g. filter matched 0 models) get a distinct framing so
        # "done=0/0" isn't ambiguous with "broken sweep, never started."
        if total == 0:
            msg = f"COMPLETE pid={pid} phase={label} (empty sweep — 0 work items)"
        else:
            msg = f"COMPLETE pid={pid} phase={label} done={total}/{total}"
        print(msg)
        if args.post_to:
            post_gchat(args.post_to, f"[🦦 watchdog] {msg}")
        sys.exit(0)

    # Read previous progress from cycle-script-side state file.
    progress_file = sweep_dir / ".watchdog_progress"
    prev_done = 0
    prev_at = 0
    prev_phase = None
    if progress_file.exists():
        try:
            with progress_file.open() as f:
                prev_state = json.load(f)
            prev_phase = prev_state.get("phase")
            if prev_phase == phase:  # same-phase comparison only for done-counter
                prev_done = prev_state.get("done", 0)
                prev_at = prev_state.get("at", 0)
        except Exception:
            pass  # corrupt → treat as fresh observation

    now = int(time.time())
    progress_age_min = (now - prev_at) / 60 if prev_at else 0
    threshold_min = PHASE_THRESHOLDS_MIN.get(phase, 30)

    alive = is_alive(pid)

    # Phase transition takes precedence — emit even if alive/dead determination
    # would otherwise be silent. Helps caller post a phase-change signal.
    phase_changed = (prev_phase is not None and prev_phase != phase)

    if not alive:
        msg = f"DEAD pid={pid} phase={label} done={done}/{total}"
    elif phase_changed:
        msg = (f"PHASE_TRANSITION pid={pid} phase={label} done={done}/{total} "
               f"(was {prev_phase})")
    elif done > prev_done:
        delta = done - prev_done
        msg = f"ALIVE pid={pid} phase={label} done={done}/{total} +{delta} since last check"
    elif progress_age_min >= threshold_min:
        msg = (f"STALLED pid={pid} phase={label} done={done}/{total} "
               f"no progress for {progress_age_min:.0f}min (threshold {threshold_min}min)")
    else:
        msg = (f"ALIVE pid={pid} phase={label} done={done}/{total} "
               f"(no progress for {progress_age_min:.0f}min, threshold {threshold_min}min)")

    print(msg)

    # Update progress state — write whenever phase changes (so phase-transition
    # is reported once and we move on) OR alive AND made progress (resets
    # no-progress timer) OR first observation. If alive but no progress in
    # same phase, keep prev_at to accumulate the no-progress window.
    if not args.no_write_state:
        if phase_changed:
            with progress_file.open("w") as f:
                json.dump({"phase": phase, "done": done, "at": now}, f)
        elif alive and done > prev_done:
            with progress_file.open("w") as f:
                json.dump({"phase": phase, "done": done, "at": now}, f)
        elif not progress_file.exists():
            with progress_file.open("w") as f:
                json.dump({"phase": phase, "done": done, "at": now}, f)
        elif prev_done == 0 and prev_at == 0:
            with progress_file.open("w") as f:
                json.dump({"phase": phase, "done": done, "at": now}, f)

    if args.post_to:
        post_gchat(args.post_to, f"[🦦 watchdog] {msg}")

    sys.exit(0)


if __name__ == "__main__":
    main()
