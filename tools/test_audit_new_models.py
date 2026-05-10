#!/usr/bin/env python3
"""Tests for tools/audit_new_models.py.

9 tests addressing all 9 gaps from adversary case adv-2026-05-10-150000.

Run: PYTHONPATH=$(pwd) ~/envs/torch211/bin/python tools/test_audit_new_models.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import audit_new_models as anm  # noqa: E402
from audit_new_models import (  # noqa: E402
    classify_tier, classify_timeout_tier,
    collect_new_models, collect_removed_models,
    WALL_LARGE_S, WALL_VERY_LARGE_S,
)


def _make_sweep_dir(tmp: Path, *, compare=None, rows_jsonl=None, sister_audit=None):
    """Build a synthetic sweep dir with the given inputs."""
    sweep = tmp
    sweep.mkdir(parents=True, exist_ok=True)
    if compare is not None:
        (sweep / "compare-vs-baseline.json").write_text(json.dumps(compare))
    if rows_jsonl is not None:
        # results_loader expects identify_results.json with metadata header + rows
        lines = [json.dumps({"_record_type": "metadata", "pass": "identify"})]
        lines.extend(json.dumps(r) for r in rows_jsonl)
        (sweep / "identify_results.json").write_text("\n".join(lines) + "\n")
    if sister_audit is not None:
        (sweep / "audit-new-errors.json").write_text(json.dumps(sister_audit))
    (sweep / "sweep_state.json").write_text(json.dumps(
        {"versions": {"torch": "2.13.0.dev20260507+cu126"}}))
    return sweep


# Gap #4 + #6 — threshold constants pinned
def test_threshold_constants_pinned():
    """If you change these, update the design doc + run adversary-review."""
    assert WALL_LARGE_S == 60, f"WALL_LARGE_S changed to {WALL_LARGE_S} — update design + invoke adversary-review"
    assert WALL_VERY_LARGE_S == 300, f"WALL_VERY_LARGE_S changed to {WALL_VERY_LARGE_S} — update design + invoke adversary-review"


# Gap #5 — boundary tests
def test_boundary_wall_exactly_60s():
    """≤60s = regular, >60s = large."""
    assert classify_tier(60.0) == "regular"
    assert classify_tier(60.0001) == "large"
    assert classify_tier(300.0) == "large"
    assert classify_tier(300.0001) == "very_large"


def test_classify_timeout_tier():
    assert classify_timeout_tier("create") == "very_large"
    assert classify_timeout_tier("eager") == "large"
    assert classify_timeout_tier(None) == "large"


# Gap #1 — REMOVED via skip_listed
def test_cat5_skip_listed_path():
    compare = {
        "cat4": [], "cat5": [],
        "skip_listed": [
            {"key": ["FooModel", "eval"], "only_in": "baseline"},
            {"key": ["FooModel", "train"], "only_in": "baseline"},
        ],
    }
    removed = collect_removed_models(compare, skip_set={"FooModel"},
                                     known_error_models=set())
    assert len(removed) == 1
    assert removed[0]["name"] == "FooModel"
    assert removed[0]["classification"] == "intentional-skip"


# Gap #2 — REMOVED via known-error-evolution
def test_cat5_known_error_baseline_only_model():
    compare = {
        "cat4": [],
        "cat5": [{"key": ["BarModel", "eval"], "baseline_status": "skipped"}],
        "skip_listed": [],
    }
    removed = collect_removed_models(compare, skip_set=set(),
                                     known_error_models={"BarModel"})
    assert len(removed) == 1
    assert removed[0]["name"] == "BarModel"
    assert removed[0]["classification"] == "known-error-evolution"


def test_cat5_unexpected_removal_default():
    """cat5 row not covered by skip_models or known_errors → unexpected-removal."""
    compare = {
        "cat4": [],
        "cat5": [{"key": ["RefactoredModel", "eval"]}],
        "skip_listed": [],
    }
    removed = collect_removed_models(compare, skip_set=set(),
                                     known_error_models=set())
    assert len(removed) == 1
    assert removed[0]["classification"] == "unexpected-removal"


# Gap #3 — cat 4 dedupe by name + MAX(wall) across modes
def test_cat4_dedupe_by_name_picks_max_wall():
    with tempfile.TemporaryDirectory() as tmp:
        compare = {
            "cat4": [
                {"key": ["NewModel", "eval"], "current_status": "full_graph"},
                {"key": ["NewModel", "train"], "current_status": "full_graph"},
            ],
            "cat5": [], "skip_listed": [],
        }
        rows_jsonl = [
            {"_record_type": "row", "name": "NewModel", "mode": "eval",
             "source": "hf", "status": "full_graph", "wall_time_s": 45},
            {"_record_type": "row", "name": "NewModel", "mode": "train",
             "source": "hf", "status": "full_graph", "wall_time_s": 75},
        ]
        sweep = _make_sweep_dir(Path(tmp), compare=compare, rows_jsonl=rows_jsonl)
        # Force load
        from results_loader import load_effective_results
        rows = load_effective_results(sweep)
        new_models = collect_new_models(compare, rows, fixture_fix_map=None)
        assert len(new_models) == 1, f"expected 1 unique model, got {len(new_models)}"
        m = new_models[0]
        assert m["name"] == "NewModel"
        assert m["max_wall_s"] == 75.0  # MAX across modes
        assert m["proposed_tier"] == "large"  # 75s in (60, 300] → large
        assert sorted(m["modes"]) == ["eval", "train"]


# Gap #6 — branch precedence: error row routes to fixture-fix, not tier
def test_branch_precedence_error_with_high_wall():
    with tempfile.TemporaryDirectory() as tmp:
        compare = {
            "cat4": [{"key": ["ErrorModel", "eval"], "current_status": "eager_error"}],
            "cat5": [], "skip_listed": [],
        }
        rows_jsonl = [
            {"_record_type": "row", "name": "ErrorModel", "mode": "eval",
             "source": "hf", "status": "eager_error", "wall_time_s": 400},
        ]
        sweep = _make_sweep_dir(Path(tmp), compare=compare, rows_jsonl=rows_jsonl)
        from results_loader import load_effective_results
        rows = load_effective_results(sweep)
        new_models = collect_new_models(compare, rows, fixture_fix_map=None)
        assert len(new_models) == 1
        m = new_models[0]
        assert m["any_error"] is True
        assert m["proposed_tier"] is None  # fixture-fix needed first
        assert m["max_wall_s"] is None  # no valid wall (error row excluded)


# Gap #8 — graceful degradation when audit-new-errors.json absent
def test_audit_runs_when_audit_new_errors_sidecar_absent():
    with tempfile.TemporaryDirectory() as tmp:
        compare = {
            "cat4": [{"key": ["X", "eval"], "current_status": "eager_error"}],
            "cat5": [], "skip_listed": [],
        }
        rows_jsonl = [
            {"_record_type": "row", "name": "X", "mode": "eval", "source": "hf",
             "status": "eager_error", "wall_time_s": 30},
        ]
        sweep = _make_sweep_dir(Path(tmp), compare=compare, rows_jsonl=rows_jsonl)
        # No sister audit file
        rc = anm.run_audit(sweep)
        assert rc == 0, f"expected exit 0, got {rc}"
        report = (sweep / "audit-new-models.md").read_text()
        assert "fixture-fix link unavailable" in report.lower() or \
               "run audit_new_errors first" in report.lower(), \
               "report should explain missing sister output"


# Gap #8 (positive case) — sister sidecar IS present
def test_uses_sister_sidecar_when_present():
    with tempfile.TemporaryDirectory() as tmp:
        compare = {
            "cat4": [{"key": ["X", "eval"], "current_status": "eager_error"}],
            "cat5": [], "skip_listed": [],
        }
        rows_jsonl = [
            {"_record_type": "row", "name": "X", "mode": "eval", "source": "hf",
             "status": "eager_error", "wall_time_s": 30},
        ]
        sister = {
            "candidates": [
                {"name": "X", "mode": "eval", "triage_class": "fixture-bug"},
            ]
        }
        sweep = _make_sweep_dir(Path(tmp), compare=compare,
                                rows_jsonl=rows_jsonl, sister_audit=sister)
        rc = anm.run_audit(sweep)
        assert rc == 0
        report = (sweep / "audit-new-models.md").read_text()
        assert "fixture-bug" in report, "report should reference sister-tool's class"


# Gap-adjacent: degraded mode exit code
def test_compare_absent_exit_code():
    with tempfile.TemporaryDirectory() as tmp:
        rows_jsonl = [
            {"_record_type": "row", "name": "X", "mode": "eval", "source": "hf",
             "status": "full_graph", "wall_time_s": 30},
        ]
        sweep = _make_sweep_dir(Path(tmp), rows_jsonl=rows_jsonl)
        # No compare json
        rc = anm.run_audit(sweep)
        assert rc == 3, f"expected exit 3 (degraded), got {rc}"


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
