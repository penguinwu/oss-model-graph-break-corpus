#!/usr/bin/env python3
"""Tests for sweep_compare wiring into run_experiment.py nightly pipeline.

Pins:
  - _find_prior_baseline returns the largest dated sibling preceding current
  - _find_prior_baseline skips dirs without identify_results.json
  - _find_prior_baseline skips dirs whose sweep_state.json has status != "done"
  - Returns None when no qualifying baseline found

Per sweep/SWEEP_COMPARE_WIRING_DESIGN.md.
Run: PYTHONPATH=$(pwd) ~/envs/torch211/bin/python tools/test_sweep_compare_wiring.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from run_experiment import _find_prior_baseline  # noqa: E402


def _make_sweep_dir(parent: Path, date: str, *, with_identify=True, status="done"):
    d = parent / date
    d.mkdir()
    if with_identify:
        (d / "identify_results.json").write_text("{}")
    if status is not None:
        (d / "sweep_state.json").write_text(json.dumps({"status": status}))
    return d


def test_finds_largest_preceding_dated_sibling():
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        _make_sweep_dir(parent, "2026-04-26")
        _make_sweep_dir(parent, "2026-05-03")  # this is the answer
        current = _make_sweep_dir(parent, "2026-05-09")
        result = _find_prior_baseline(current)
        assert result is not None, "expected a baseline"
        assert result.name == "2026-05-03", f"expected 2026-05-03, got {result.name}"


def test_skips_dirs_without_identify_results():
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        _make_sweep_dir(parent, "2026-04-26")
        _make_sweep_dir(parent, "2026-05-03", with_identify=False)  # incomplete
        current = _make_sweep_dir(parent, "2026-05-09")
        result = _find_prior_baseline(current)
        assert result is not None and result.name == "2026-04-26", \
            f"expected fallback to 2026-04-26, got {result}"


def test_skips_dirs_with_non_done_status():
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        _make_sweep_dir(parent, "2026-04-26")
        _make_sweep_dir(parent, "2026-05-03", status="running")  # in flight
        current = _make_sweep_dir(parent, "2026-05-09")
        result = _find_prior_baseline(current)
        assert result is not None and result.name == "2026-04-26", \
            f"expected fallback to 2026-04-26, got {result}"


def test_returns_none_when_no_qualifying_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        current = _make_sweep_dir(parent, "2026-05-09")
        result = _find_prior_baseline(current)
        assert result is None, f"expected None, got {result}"


def test_skips_current_dir_itself():
    """Even if current_dir would sort first, exclude it from candidates."""
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        current = _make_sweep_dir(parent, "2026-05-09")
        # No prior dirs
        result = _find_prior_baseline(current)
        assert result is None, f"current dir should be excluded; got {result}"


def test_ignores_non_dated_dirs():
    """Random sibling subdirs (e.g. _pre-hf-only-backup) must not match."""
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        (parent / "_pre-hf-only-backup").mkdir()
        (parent / "experiments").mkdir()
        _make_sweep_dir(parent, "2026-05-03")
        current = _make_sweep_dir(parent, "2026-05-09")
        result = _find_prior_baseline(current)
        assert result is not None and result.name == "2026-05-03", \
            f"expected 2026-05-03, got {result}"


def test_sweep_state_missing_treated_as_done():
    """Pre-2026-05 sweeps may lack sweep_state.json — accept them if identify_results exists."""
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        d = parent / "2026-05-03"
        d.mkdir()
        (d / "identify_results.json").write_text("{}")
        # No sweep_state.json
        current = _make_sweep_dir(parent, "2026-05-09")
        result = _find_prior_baseline(current)
        assert result is not None and result.name == "2026-05-03", \
            f"expected acceptance of pre-2026-05 sweep; got {result}"


def test_corrupt_sweep_state_skipped():
    """Corrupt sweep_state.json should be skipped, not crash."""
    with tempfile.TemporaryDirectory() as tmp:
        parent = Path(tmp)
        _make_sweep_dir(parent, "2026-04-26")
        d = parent / "2026-05-03"
        d.mkdir()
        (d / "identify_results.json").write_text("{}")
        (d / "sweep_state.json").write_text("{not json")
        current = _make_sweep_dir(parent, "2026-05-09")
        result = _find_prior_baseline(current)
        assert result is not None and result.name == "2026-04-26", \
            f"expected fallback over corrupt state; got {result}"


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failures.append(name)
        except Exception as e:
            print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
            failures.append(name)
    print()
    print(f"{len(tests) - len(failures)}/{len(tests)} passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
