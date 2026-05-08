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


def test_post_sweep_handles_jsonl_format():
    """The harness writes identify_results.json as JSONL (line-delimited records,
    first line a metadata header). Caught by M2-customized gate 2026-05-07 17:05 ET
    when the tool failed with INVALID_JSON on real harness output despite the
    file being valid JSONL."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.json"
        # Mimic the actual harness output: metadata line, then result rows
        lines = [
            json.dumps({"_record_type": "metadata", "pass": "identify",
                        "device": "cuda", "modes": ["eval", "train"]}),
            json.dumps({"name": "ModelA", "mode": "eval", "status": "graph_break"}),
            json.dumps({"name": "ModelA", "mode": "train", "status": "graph_break"}),
            json.dumps({"name": "ModelB", "mode": "eval", "status": "full_graph"}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        strict = [f for f in failures if f.severity == "STRICT_FAIL"]
        assert not strict, f"JSONL parse should succeed; got {[str(f) for f in strict]}"


def test_post_sweep_sp1_passes_with_matching_spec():
    """SP1: results metadata header references a real spec file with matching sha256."""
    import hashlib
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        spec = tmp / "test-spec.json"
        spec.write_text('{"name": "test-spec"}')
        spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
        results = tmp / "results.jsonl"
        lines = [
            json.dumps({"_record_type": "metadata", "spec_path": str(spec),
                        "spec_sha256": spec_sha}),
            json.dumps({"name": "M1", "status": "success"}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        sp1 = [f for f in failures if f.code == "SP1"]
        assert not sp1, f"SP1 should pass with matching sha256; got {[str(f) for f in sp1]}"


def test_post_sweep_sp1_strict_fail_on_spec_drift():
    """SP1: spec file's sha256 differs from recorded — STRICT_FAIL."""
    import hashlib
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        spec = tmp / "test-spec.json"
        spec.write_text('{"name": "test-spec"}')
        results = tmp / "results.jsonl"
        lines = [
            json.dumps({"_record_type": "metadata", "spec_path": str(spec),
                        "spec_sha256": "0" * 64}),  # wrong sha
            json.dumps({"name": "M1", "status": "success"}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        sp1 = [f for f in failures if f.code == "SP1"]
        assert sp1, "SP1 should fire on sha mismatch"
        assert sp1[0].severity == "STRICT_FAIL"
        assert "drift" in sp1[0].message


def test_post_sweep_sp1_strict_fail_on_missing_spec_file():
    """SP1: spec file recorded in metadata doesn't exist — STRICT_FAIL."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        results = tmp / "results.jsonl"
        lines = [
            json.dumps({"_record_type": "metadata",
                        "spec_path": "/nonexistent/spec.json",
                        "spec_sha256": "0" * 64}),
            json.dumps({"name": "M1", "status": "success"}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        sp1 = [f for f in failures if f.code == "SP1"]
        assert sp1
        assert sp1[0].severity == "STRICT_FAIL"
        assert "not found" in sp1[0].message


def test_post_sweep_sp1_flag_on_missing_provenance_header():
    """SP1: older results files without metadata header — FLAG (not STRICT)."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.jsonl"
        # No metadata header — just data rows
        results.write_text(json.dumps({"name": "M1", "status": "success"}) + "\n")
        failures = check_post_sweep(results)
        sp1 = [f for f in failures if f.code == "SP1"]
        assert sp1
        assert sp1[0].severity == "FLAG"
        assert "provenance" in sp1[0].message.lower()


def test_post_sweep_d1_strict_fail_on_catastrophic_divergence():
    """D1: max_diff > 1e-3 = catastrophic divergence, STRICT_FAIL.
    Was missing from the executor; surfaced 2026-05-07 20:50 when NGB verify
    sample's HubertModel reproduced a known ~4.9 divergence and the executor
    silently reported 'all PASS' (because D1 wasn't checked)."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.jsonl"
        lines = [
            json.dumps({"_record_type": "metadata"}),
            json.dumps({"name": "GoodModel", "mode": "eval", "status": "success",
                        "numeric_status": "match", "numeric_max_diff": 0.0}),
            json.dumps({"name": "DivergentModel", "mode": "train", "status": "success",
                        "numeric_status": "divergence", "numeric_max_diff": 4.895}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        d1 = [f for f in failures if f.code == "D1"]
        assert d1, f"D1 should fire on max_diff=4.895; got codes {[f.code for f in failures]}"
        assert d1[0].severity == "STRICT_FAIL"
        assert "DivergentModel" in str(d1[0])


def test_post_sweep_d2_flag_on_low_magnitude_divergence():
    """D2: max_diff in (1e-7, 1e-3] = noise-floor divergence, FLAG (not STRICT).
    Catches the typical 1e-6 numerical noise from float32 op ordering without
    halting the sweep."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.jsonl"
        lines = [
            json.dumps({"_record_type": "metadata"}),
            json.dumps({"name": "NoiseModel", "mode": "eval", "status": "success",
                        "numeric_status": "divergence", "numeric_max_diff": 5e-6}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        d2 = [f for f in failures if f.code == "D2"]
        assert d2, "D2 should fire on max_diff=5e-6"
        assert d2[0].severity == "FLAG"
        # No STRICT failures should fire (D2 is FLAG-only)
        strict = [f for f in failures if f.severity == "STRICT_FAIL"]
        assert not strict, f"only FLAG expected; got STRICT: {[str(f) for f in strict]}"


def test_post_sweep_recognizes_success_status():
    """Regression for 2026-05-07 20:46: run_experiment.py 'run' subcommand emits
    status='success' (vs sweep's 'ok'/'full_graph'). check_cohort_invariants
    must recognize 'success' as a non-failure status; otherwise every row from
    a config-driven launch false-flags as G1 untriaged."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.jsonl"
        # Mimic actual run_experiment.py 'run' output: JSONL, status='success'
        lines = [
            json.dumps({"_record_type": "metadata", "pass": "identify"}),
            json.dumps({"name": "ModelA", "mode": "eval", "status": "success",
                        "numeric_status": "match", "numeric_max_diff": 0.0}),
            json.dumps({"name": "ModelA", "mode": "train", "status": "success",
                        "numeric_status": "match", "numeric_max_diff": 0.0}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        # G1 should NOT fire on success rows
        g1 = [f for f in failures if f.code == "G1"]
        assert not g1, f"G1 should not fire on status='success' rows; got {[str(f) for f in g1]}"
        strict = [f for f in failures if f.severity == "STRICT_FAIL"]
        assert not strict, f"no strict failures expected; got {[str(f) for f in strict]}"


def test_post_sweep_jsonl_with_create_error_still_caught():
    """Regression: ensure the JSONL fallback still applies the invariants
    (specifically C1 — create_error rows should STRICT_FAIL even via JSONL path)."""
    with tempfile.TemporaryDirectory() as d:
        results = Path(d) / "results.json"
        lines = [
            json.dumps({"_record_type": "metadata"}),
            json.dumps({"name": "A", "status": "ok"}),
            json.dumps({"name": "B", "status": "create_error", "error": "boom"}),
        ]
        results.write_text("\n".join(lines) + "\n")
        failures = check_post_sweep(results)
        codes = [f.code for f in failures]
        assert "C1" in codes, f"C1 should fire via JSONL path; got {codes}"


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
