#!/usr/bin/env python3
"""Tests for tools/check_cohort_invariants.py.

Surfaced by adversary-review case_id 2026-05-07-124100-cohort-regen-fix:
- review test 8: test_sanity_check_invariant_a3_executor (would have caught 2026-05-06)
- review test 9: test_sanity_check_invariant_a1_executor_attribute_errors

Run: python3 tools/test_check_cohort_invariants.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.check_cohort_invariants import (  # noqa: E402
    check_pre_launch,
    check_post_sweep,
)


def _make_source(path: Path, statuses: dict):
    """Write a source results file. statuses = {name: status}."""
    rows = [{"name": n, "source": "hf", "status": s} for n, s in statuses.items()]
    path.write_text(json.dumps({"results": rows}))


def _make_cohort(path: Path, *, derived_from: str, filter_expr: str, names: list):
    payload = {
        "_metadata": {
            "derived_from": derived_from,
            "filter": filter_expr,
            "model_count": len(names),
            "source_versions": {"torch": "2.13.0", "transformers": "5.6.2", "diffusers": "0.38.0"},
        },
        "models": [{"name": n, "source": "hf"} for n in names],
    }
    path.write_text(json.dumps(payload))


# Pre-launch tests ────────────────────────────────────────────────────────

def test_pre_launch_passes_canonical_cohort():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        _make_source(source, {"A": "ok", "B": "ok", "C": "graph_break"})
        _make_cohort(cohort, derived_from=str(source.resolve()),
                     filter_expr="status == ok", names=["A", "B"])
        failures = check_pre_launch(cohort)
        # No strict failures (skip_models check may FLAG/PASS depending on repo state)
        strict = [f for f in failures if f.severity == "STRICT_FAIL"]
        assert not strict, f"unexpected strict failures: {[str(f) for f in strict]}"


def test_pre_launch_a2_strict_fail_on_bare_list():
    """A2: bare list (no _metadata) → STRICT_FAIL."""
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "bare.json"
        cohort.write_text(json.dumps([{"name": "A", "source": "hf"}]))
        failures = check_pre_launch(cohort)
        codes = [f.code for f in failures]
        assert "A2" in codes, f"expected A2 STRICT_FAIL; got codes {codes}"
        a2 = next(f for f in failures if f.code == "A2")
        assert a2.severity == "STRICT_FAIL"


def test_pre_launch_a3_strict_fail_on_extras_review_test_8():
    """Review test 8: synthetic source with 100 ok rows; cohort with 90 of those PLUS
    5 names not in source (mimicking the 17 contaminated extras from 2026-05-06).
    Expect A3 STRICT_FAIL with the 5 extras enumerated."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        # 100 ok models in source
        statuses = {f"OK_{i:03d}": "ok" for i in range(100)}
        _make_source(source, statuses)
        # Cohort: 90 from source + 5 contaminated extras
        names = [f"OK_{i:03d}" for i in range(90)] + [f"CONTAMINATED_{i}" for i in range(5)]
        _make_cohort(cohort, derived_from=str(source.resolve()),
                     filter_expr="status == ok", names=names)
        failures = check_pre_launch(cohort)
        a3_failures = [f for f in failures if f.code == "A3"]
        assert a3_failures, f"expected A3 failure; got {[f.code for f in failures]}"
        a3 = a3_failures[0]
        assert a3.severity == "STRICT_FAIL"
        assert "5" in a3.message, f"A3 message should mention count of 5; got: {a3.message}"
        # All 5 contaminated names should be in examples
        for i in range(5):
            name = f"CONTAMINATED_{i}"
            assert any(name in ex for ex in a3.examples), \
                f"{name} should appear in A3 examples: {a3.examples}"


def test_pre_launch_a3_passes_when_cohort_is_strict_subset():
    """A3 passes when every cohort name is in the filtered source."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        _make_source(source, {f"M{i}": "ok" for i in range(50)})
        _make_cohort(cohort, derived_from=str(source.resolve()),
                     filter_expr="status == ok", names=[f"M{i}" for i in range(20)])
        failures = check_pre_launch(cohort)
        a3 = [f for f in failures if f.code == "A3"]
        assert not a3, f"A3 should pass; got {[str(f) for f in a3]}"


# Post-sweep tests ────────────────────────────────────────────────────────

def test_post_sweep_a1_strict_fail_on_attribute_errors_review_test_9():
    """Review test 9: synthetic post-sweep results with 6 create_error rows whose
    error contains 'module X has no attribute Y'. Expect A1 STRICT_FAIL."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.json"
        rows = []
        # 50 ok rows
        rows += [{"name": f"OK_{i}", "status": "ok"} for i in range(50)]
        # 6 attribute-not-found create_errors (mimicking 2026-05-06 cohort drift)
        attr_errors = [
            ("MiniCPMV4_6", "MiniCPMV4_6"),
            ("MiniCPMV4_6Model", "MiniCPMV4_6Model"),
            ("PPFormulaNetForConditionalGeneration", "PPFormulaNet"),
            ("PPFormulaNetModel", "PPFormulaNet"),
            ("Phi5", "Phi5"),
            ("Mistral8x22B", "Mistral8x22B"),
        ]
        for name, attr in attr_errors:
            rows.append({
                "name": name, "status": "create_error",
                "error": f"module 'transformers' has no attribute '{attr}'",
            })
        results.write_text(json.dumps({"results": rows}))

        failures = check_post_sweep(results)
        codes = [f.code for f in failures]
        assert "A1" in codes, f"expected A1; got {codes}"
        a1 = next(f for f in failures if f.code == "A1")
        assert a1.severity == "STRICT_FAIL"
        assert "6" in a1.message, f"A1 should mention 6 rows; got: {a1.message}"


def test_post_sweep_c1_strict_fail_on_create_errors():
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.json"
        rows = [{"name": "A", "status": "ok"},
                {"name": "B", "status": "create_error", "error": "Hub network failure"}]
        results.write_text(json.dumps({"results": rows}))
        failures = check_post_sweep(results)
        c1 = [f for f in failures if f.code == "C1"]
        assert c1, "expected C1 STRICT_FAIL"


def test_post_sweep_c2_distinguishes_env_vs_non_env_eager_errors():
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.json"
        rows = [
            {"name": "A", "status": "ok"},
            {"name": "B", "status": "eager_error", "error": "CUDA out of memory"},  # ENV
            {"name": "C", "status": "eager_error", "error": "input shape mismatch"},  # NON-ENV
        ]
        results.write_text(json.dumps({"results": rows}))
        failures = check_post_sweep(results)
        c2 = [f for f in failures if f.code == "C2"]
        assert c2, "expected C2 STRICT_FAIL on non-env eager_error"
        assert "1" in c2[0].message, f"should report 1 non-env eager_error; got: {c2[0].message}"


def test_post_sweep_passes_clean_results():
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.json"
        rows = [{"name": f"M{i}", "status": "ok"} for i in range(20)]
        results.write_text(json.dumps({"results": rows}))
        failures = check_post_sweep(results)
        strict = [f for f in failures if f.severity == "STRICT_FAIL"]
        assert not strict, f"expected no strict failures on clean results; got {[str(f) for f in strict]}"


# Runner ───────────────────────────────────────────────────────────────────

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
