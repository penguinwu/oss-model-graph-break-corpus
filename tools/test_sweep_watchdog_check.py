#!/usr/bin/env python3
"""Unit tests for sweep_watchdog_check.decide_verdict.

Covers the bug case from 2026-04-30 (kill-and-restart loop) plus boundaries.

Run: python -m unittest tools.test_sweep_watchdog_check -v
  or: python tools/test_sweep_watchdog_check.py
"""
import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.sweep_watchdog_check import (
    decide_verdict,
    VERDICT_DONE, VERDICT_HEALTHY, VERDICT_INITIALIZING,
    VERDICT_HUNG, VERDICT_CRASHED,
    DEFAULT_INIT_GRACE_S,
)


class TestDecideVerdict(unittest.TestCase):

    def test_results_exists_returns_DONE(self):
        # results.json existing always wins, even if process is dead
        v = decide_verdict(
            pid_alive=False, proc_etime_s=None,
            completed=1456, last_completed_count=126,
            seconds_since_last_progress=10000,
            results_file_exists=True,
        )
        self.assertEqual(v, VERDICT_DONE)

    def test_results_takes_priority_over_hung_process(self):
        # Even if everything else looks like HUNG, results file = DONE
        v = decide_verdict(
            pid_alive=True, proc_etime_s=10000,
            completed=1456, last_completed_count=126,
            seconds_since_last_progress=10000,
            results_file_exists=True,
        )
        self.assertEqual(v, VERDICT_DONE)

    def test_no_pid_no_results_returns_CRASHED(self):
        v = decide_verdict(
            pid_alive=False, proc_etime_s=None,
            completed=126, last_completed_count=126,
            seconds_since_last_progress=600,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_CRASHED)

    def test_pid_alive_advanced_returns_HEALTHY(self):
        # Sweep made progress since last tick
        v = decide_verdict(
            pid_alive=True, proc_etime_s=600,
            completed=145, last_completed_count=126,
            seconds_since_last_progress=300,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_HEALTHY)

    def test_THE_BUG_CASE_just_relaunched_no_progress_yet(self):
        # *** This is the kill-and-restart loop bug from 2026-04-30 ***
        # Process just relaunched. heartbeat (count) is from prior run = 126.
        # last_completed_count is also 126 (script's previous tick saw same).
        # Process has been running 5 min — should be INITIALIZING, not HUNG.
        v = decide_verdict(
            pid_alive=True, proc_etime_s=300,  # 5 min since process start
            completed=126, last_completed_count=126,
            seconds_since_last_progress=300,  # 5 min since we last saw progress
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_INITIALIZING,
                         "BUG REGRESSION: just-relaunched process within init grace must NOT be classified HUNG")

    def test_pid_alive_no_progress_within_grace_returns_INITIALIZING(self):
        v = decide_verdict(
            pid_alive=True, proc_etime_s=900,  # 15 min
            completed=126, last_completed_count=126,
            seconds_since_last_progress=900,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_INITIALIZING)

    def test_pid_alive_no_progress_past_grace_returns_HUNG(self):
        v = decide_verdict(
            pid_alive=True, proc_etime_s=2700,  # 45 min
            completed=126, last_completed_count=126,
            seconds_since_last_progress=2700,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_HUNG)

    def test_grace_boundary_just_below_grace_INITIALIZING(self):
        v = decide_verdict(
            pid_alive=True, proc_etime_s=DEFAULT_INIT_GRACE_S - 1,
            completed=126, last_completed_count=126,
            seconds_since_last_progress=DEFAULT_INIT_GRACE_S - 1,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_INITIALIZING)

    def test_grace_boundary_at_grace_HUNG(self):
        v = decide_verdict(
            pid_alive=True, proc_etime_s=DEFAULT_INIT_GRACE_S,
            completed=126, last_completed_count=126,
            seconds_since_last_progress=DEFAULT_INIT_GRACE_S,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_HUNG)

    def test_first_tick_no_script_state_pid_alive_short_etime_INITIALIZING(self):
        # First tick after enabling watchdog: no script state, no last_count.
        # Process just started → INITIALIZING based on proc_etime.
        v = decide_verdict(
            pid_alive=True, proc_etime_s=120,  # 2 min
            completed=126, last_completed_count=None,
            seconds_since_last_progress=None,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_INITIALIZING)

    def test_first_tick_long_running_no_progress_HUNG(self):
        # First tick on a process that's been running 1 hour with no progress
        # → HUNG (no script state, fall back to proc_etime)
        v = decide_verdict(
            pid_alive=True, proc_etime_s=3600,  # 1 hour
            completed=126, last_completed_count=None,
            seconds_since_last_progress=None,
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_HUNG)

    def test_progress_tracked_via_seconds_since_last_progress_not_proc_etime(self):
        # seconds_since_last_progress reflects MORE recent progress.
        # Process running 1h but last progress was 5 min ago → INITIALIZING (still "fresh").
        # This handles the case where sweep stalled briefly within a single process lifetime.
        v = decide_verdict(
            pid_alive=True, proc_etime_s=3600,  # process been running 1h
            completed=200, last_completed_count=200,  # no NEW progress this tick
            seconds_since_last_progress=300,  # but progress 5 min ago counts
            results_file_exists=False,
        )
        self.assertEqual(v, VERDICT_INITIALIZING)

    def test_custom_grace_period(self):
        # Allow tuning grace period via parameter
        v = decide_verdict(
            pid_alive=True, proc_etime_s=600,  # 10 min
            completed=126, last_completed_count=126,
            seconds_since_last_progress=600,
            results_file_exists=False,
            init_grace_s=300,  # custom 5 min grace
        )
        self.assertEqual(v, VERDICT_HUNG, "with 5-min grace, 10 min etime is past grace")


class TestPidChangeReset(unittest.TestCase):
    """End-to-end test of the main()-level PID-change reset behavior.

    Bug case (2026-04-30 15:14 ET): old PID died, new PID launched, but
    `last_progress_at_epoch` from old PID's lifetime carried over. The new PID's
    `seconds_since_last_progress` was computed against the old timestamp,
    falsely tripping HUNG on a brand-new process within init grace.
    """

    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self.output_dir = Path(self._tmpdir) / "sweep"
        self.output_dir.mkdir()
        self.state_file = Path(self._tmpdir) / "state.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_sweep_state(self, pid):
        (self.output_dir / "sweep_state.json").write_text(
            f'{{"pid": {pid}, "total_work_items": 1478}}'
        )

    def _write_script_state(self, **fields):
        import json as _json
        self.state_file.write_text(_json.dumps(fields))

    def _run_main(self):
        import subprocess
        result = subprocess.run(
            [sys.executable,
             str(Path(__file__).resolve().parent / "sweep_watchdog_check.py"),
             str(self.output_dir),
             "--state-file", str(self.state_file)],
            capture_output=True, text=True,
        )
        import json as _json
        return _json.loads(result.stdout)

    def test_pid_change_resets_progress_tracking(self):
        # Simulate: prior PID 999999 (long dead), state file says no progress for 31m.
        # Now sweep_state.json points to current process (us, definitely alive).
        # Without reset → would compute seconds_since_last_progress from stale ts → HUNG.
        # With reset → fresh tracking → INITIALIZING.
        self._write_sweep_state(pid=os.getpid())
        self._write_script_state(
            last_count=126,
            last_progress_at_epoch=int(time.time()) - 1860,  # 31 min ago
            last_reported_verdict="CRASHED",
            last_reported_pid=999999,
        )
        result = self._run_main()
        self.assertEqual(result["verdict"], VERDICT_INITIALIZING,
            "PID change must reset progress tracking; stale ts must NOT trip HUNG")

    def test_same_pid_preserves_progress_tracking(self):
        # If PID hasn't changed, do NOT reset — we want to detect real stalls.
        self._write_sweep_state(pid=os.getpid())
        self._write_script_state(
            last_count=126,
            last_progress_at_epoch=int(time.time()) - 1860,  # 31 min ago
            last_reported_verdict="HEALTHY",
            last_reported_pid=os.getpid(),  # same as current
        )
        result = self._run_main()
        self.assertEqual(result["verdict"], VERDICT_HUNG,
            "same PID + 31m no progress = real HUNG, must surface")


if __name__ == "__main__":
    unittest.main(verbosity=2)
