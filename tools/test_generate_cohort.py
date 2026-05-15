#!/usr/bin/env python3
"""Tests for tools/generate_cohort.py.

Convention: every tool revision (especially bug fixes) adds tests proving it
works (see docs/testing.md). These tests are the proof that the cohort generator
works as expected — not a "claim it works."

Run: python3 tools/test_generate_cohort.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "generate_cohort.py"


def _make_source(rows, tmp: Path, with_versions: bool = True) -> Path:
    """Write a fake sweep-results JSON to tmp/source.json."""
    source = tmp / "source.json"
    payload = {
        "metadata": {"versions": {"torch": "2.13.0", "transformers": "5.6.2", "diffusers": "0.38.0"}}
                    if with_versions else {},
        "results": rows,
    }
    source.write_text(json.dumps(payload))
    return source


def _run(source: Path, filter_expr: str, output: Path, force: bool = False,
         extra_args: list = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--from", str(source), "--filter", filter_expr, "--output", str(output)]
    if force:
        cmd.append("--force")
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


# Test cases ────────────────────────────────────────────────────────────────

def test_status_eq_filter():
    """status == ok matches Python ground truth."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [
            {"name": "A", "source": "hf", "status": "ok"},
            {"name": "B", "source": "hf", "status": "graph_break"},
            {"name": "C", "source": "diffusers", "status": "ok"},
        ]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)
        assert r.returncode == 0, f"exited {r.returncode}: {r.stderr}"
        data = json.loads(out.read_text())
        names = sorted(m["name"] for m in data["models"])
        assert names == ["A", "C"], f"got {names}, expected ['A', 'C']"


def test_status_in_filter():
    """status in foo,bar matches the union."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": x, "source": "hf", "status": s}
                for x, s in [("A", "ok"), ("B", "graph_break"), ("C", "explain_error"), ("D", "ok")]]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status in ok,graph_break", out)
        assert r.returncode == 0
        names = sorted(m["name"] for m in json.loads(out.read_text())["models"])
        assert names == ["A", "B", "D"], f"got {names}"


def test_status_neq_filter():
    """status != foo matches the complement."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": x, "source": "hf", "status": s}
                for x, s in [("A", "ok"), ("B", "fail"), ("C", "ok")]]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status != ok", out)
        assert r.returncode == 0
        names = sorted(m["name"] for m in json.loads(out.read_text())["models"])
        assert names == ["B"]


def test_dedupes_across_modes():
    """A name appearing in both eval+train rows yields one cohort entry."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [
            {"name": "A", "source": "hf", "mode": "eval", "status": "ok"},
            {"name": "A", "source": "hf", "mode": "train", "status": "ok"},
            {"name": "B", "source": "hf", "mode": "eval", "status": "ok"},
        ]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)
        assert r.returncode == 0
        names = [m["name"] for m in json.loads(out.read_text())["models"]]
        assert names.count("A") == 1, f"A appeared {names.count('A')} times"


def test_metadata_block_present():
    """Output has _metadata with all required fields."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": "A", "source": "hf", "status": "ok"}]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)
        assert r.returncode == 0
        data = json.loads(out.read_text())
        assert "_metadata" in data
        meta = data["_metadata"]
        for k in ("derived_from", "derived_at", "source_versions", "filter", "model_count", "generated_by"):
            assert k in meta, f"missing _metadata.{k}"
        assert meta["filter"] == "status == ok"
        assert meta["model_count"] == 1
        assert meta["source_versions"]["torch"] == "2.13.0"


def test_metadata_versions_empty_source_refused_by_default():
    """Source without versions metadata REJECTED by default (gap 6, adversary-review case_id 2026-05-07-124100).

    Previously this test asserted empty source_versions was FINE — that test enforced the
    very bug that produced the 2026-05-06 incident. Inverted: empty must now require
    --allow-empty-versions to succeed.
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp, with_versions=False)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)
        assert r.returncode != 0, f"empty-versions cohort should be refused; got rc={r.returncode}"
        assert "PARTIAL" in r.stderr or "NO versions" in r.stderr or "empty" in r.stderr.lower(), \
            f"stderr should explain refusal; got: {r.stderr[:300]}"


def test_metadata_versions_empty_allowed_with_override():
    """--allow-empty-versions opts in to legacy behavior."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp, with_versions=False)
        out = tmp / "cohort.json"
        cmd = [sys.executable, str(TOOL), "--from", str(source), "--filter", "status == ok",
               "--output", str(out), "--allow-empty-versions"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"override should succeed; rc={r.returncode}, stderr={r.stderr[:300]}"
        data = json.loads(out.read_text())
        assert data["_metadata"]["source_versions"] == {}


def test_metadata_versions_partial_refused_by_default():
    """Source with only torch (no transformers/diffusers) is REJECTED by default (gap 6)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # Use _make_source but stub out transformers/diffusers
        source = tmp / "source.json"
        source.write_text(json.dumps({
            "metadata": {"versions": {"torch": "2.13.0"}},  # missing transformers + diffusers
            "results": [{"name": "A", "source": "hf", "status": "ok"}],
        }))
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)
        assert r.returncode != 0, f"partial-versions cohort should be refused; rc={r.returncode}"
        assert "PARTIAL" in r.stderr, f"stderr should mention PARTIAL; got: {r.stderr[:300]}"


def test_metadata_versions_partial_allowed_with_override():
    """--allow-partial-versions opts in."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        source.write_text(json.dumps({
            "metadata": {"versions": {"torch": "2.13.0"}},
            "results": [{"name": "A", "source": "hf", "status": "ok"}],
        }))
        out = tmp / "cohort.json"
        cmd = [sys.executable, str(TOOL), "--from", str(source), "--filter", "status == ok",
               "--output", str(out), "--allow-partial-versions"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"override should succeed; rc={r.returncode}"
        data = json.loads(out.read_text())
        assert data["_metadata"]["source_versions"] == {"torch": "2.13.0"}


def test_round_trip_generator_to_validator_version_mismatch():
    """Gap 3 / review test 4: generator writes cohort, validator reads it, version mismatch is caught.

    Catches future renames on either side (the live drift the reviewer found in INV-A2:
    skill said target_versions, code uses source_versions).
    """
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp,
                              with_versions=True)  # writes torch 2.13.0
        cohort = tmp / "cohort.json"
        r = _run(source, "status == ok", cohort)
        assert r.returncode == 0
        # Now load via validator with a DIFFERENT version_info — must reject as VERSION_MISMATCH
        sys.path.insert(0, str(REPO_ROOT))
        from sweep.cohort_validator import validate_cohort, CohortValidationError
        try:
            validate_cohort(cohort, version_info={
                "torch": "2.11.0", "transformers": "5.6.2", "diffusers": "0.38.0",
            })
        except CohortValidationError as e:
            assert e.code == "VERSION_MISMATCH", f"got code {e.code!r}"
            return
        assert False, "round-trip should have raised VERSION_MISMATCH"


def test_carries_forward_spec_fields():
    """hf_class, hf_config, variant, etc. are carried into the cohort spec."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{
            "name": "A", "source": "hf", "status": "ok",
            "hf_class": "AutoModel", "hf_config": "AutoConfig",
            "variant": "base", "input_type": "text",
        }]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)
        assert r.returncode == 0
        spec = json.loads(out.read_text())["models"][0]
        for k in ("hf_class", "hf_config", "variant", "input_type"):
            assert spec[k] == rows[0][k], f"{k} not carried forward"


def test_no_overwrite_without_force():
    """Existing output file is preserved unless --force is passed."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp)
        out = tmp / "cohort.json"
        out.write_text("EXISTING")
        r = _run(source, "status == ok", out, force=False)
        assert r.returncode != 0, "should refuse without --force"
        assert out.read_text() == "EXISTING"


def test_force_overwrites():
    """--force overwrites an existing file."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp)
        out = tmp / "cohort.json"
        out.write_text("EXISTING")
        r = _run(source, "status == ok", out, force=True)
        assert r.returncode == 0
        data = json.loads(out.read_text())
        assert data["_metadata"]["model_count"] == 1


def test_invalid_filter_exits_nonzero():
    """Syntactically malformed filter expressions fail loudly, not silently.
    Updated 2026-05-15 (Task 0.5): grammar generalized to any field name;
    the original "name == A" example is now valid (filters by name field).
    Test now uses an expression that doesn't match the field-op-value regex."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp)
        out = tmp / "cohort.json"
        r = _run(source, "this is not a valid filter expression", out)
        assert r.returncode != 0


def test_filter_on_unknown_field_exits_nonzero():
    """Per Task 0.5 typo guard: filter field absent from EVERY source row → exit nonzero."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp)
        out = tmp / "cohort.json"
        r = _run(source, "numric_status == divergence", out)  # typo: numric not numeric
        assert r.returncode != 0
        assert "not present in any" in r.stderr or "not present in any" in r.stdout


def test_missing_source_exits_nonzero():
    """Missing source file fails fast with a clear error."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        out = tmp / "cohort.json"
        r = _run(tmp / "nonexistent.json", "status == ok", out)
        assert r.returncode != 0


def test_idempotent():
    """Running twice with same input produces same model list (timestamps differ; that's expected)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": x, "source": "hf", "status": "ok"}
                               for x in ["A", "B", "C"]], tmp)
        out1 = tmp / "c1.json"
        out2 = tmp / "c2.json"
        _run(source, "status == ok", out1)
        _run(source, "status == ok", out2)
        d1 = json.loads(out1.read_text())
        d2 = json.loads(out2.read_text())
        assert d1["models"] == d2["models"], "model lists diverge across runs"


# Bug-fix regression test — guards against the failure mode that motivated this tool ────

def test_regression_2026_05_06_real_explain_pass():
    """Regression test for the 2026-05-06 NGB verify failure.

    The broken cohort file had 214 models built from nightly identify
    status=graph_break filter; the correct cohort is the explain pass
    status=ok subset. This test runs against a snapshotted name list
    captured 2026-05-07 (so the test never silently SKIPs after artifact
    pruning — that was gap 8 in adversary-review case_id 2026-05-07-124100).

    If the snapshot is missing, the test FAILS LOUD (not skip) so missing
    fixtures don't masquerade as PASS.
    """
    snapshot = REPO_ROOT / "tools" / "fixtures" / "ngb_2026-05-05_explain_ok_names.json"
    explain = REPO_ROOT / "sweep_results/experiments/nested-gb-2026-05-05-2026-05-05/explain_results.json"

    # Prefer the snapshot (always available, immune to artifact pruning).
    if snapshot.is_file():
        snap_data = json.loads(snapshot.read_text())
        expected_count = snap_data["model_count"]
        expected_names = set(snap_data["names"])

        # Test the tool can generate from a synthetic source built FROM the snapshot.
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            synthetic_source = tmp / "synthetic_explain.json"
            synthetic_source.write_text(json.dumps({
                "metadata": {"versions": snap_data.get("source_versions", {
                    "torch": "2.13.0.dev20260502+cu126",
                    "transformers": "5.6.2",
                    "diffusers": "0.38.0",
                })},
                "results": [{"name": n, "source": "hf", "status": "ok"} for n in expected_names],
            }))
            out = tmp / "cohort.json"
            # Pass --include-large to preserve the original-cohort regression assertion;
            # this test predates the Task 0.5 default-large-exclude.
            r = _run(synthetic_source, "status == ok", out, extra_args=["--include-large"])
            assert r.returncode == 0, f"exited {r.returncode}: {r.stderr}"
            data = json.loads(out.read_text())
            assert data["_metadata"]["model_count"] == expected_count, \
                f"regression: expected {expected_count}, got {data['_metadata']['model_count']}"
            for contaminated in ("MiniCPMV4_6ForConditionalGeneration", "MiniCPMV4_6Model",
                                 "PPFormulaNetForConditionalGeneration"):
                assert contaminated not in {m["name"] for m in data["models"]}, \
                    f"regression: contaminated extra {contaminated} should not be in result"
        return

    # Fallback: real explain artifact still present (one-shot before snapshot exists).
    if explain.is_file():
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "cohort.json"
            r = _run(explain, "status == ok", out)
            assert r.returncode == 0, f"exited {r.returncode}: {r.stderr}"
            data = json.loads(out.read_text())
            count = data["_metadata"]["model_count"]
            names = {m["name"] for m in data["models"]}
            print(f"  [INFO] real-artifact path: {count} models")
            print(f"  [INFO] consider snapshotting via:")
            print(f"  [INFO]   echo '{{\"model_count\": {count}, \"names\": [...]}}' > {snapshot}")
            for contaminated in ("MiniCPMV4_6ForConditionalGeneration", "MiniCPMV4_6Model",
                                 "PPFormulaNetForConditionalGeneration"):
                assert contaminated not in names, \
                    f"regression: {contaminated} reappeared in cohort"
        return

    # Both missing — FAIL LOUD instead of silent skip.
    raise AssertionError(
        f"regression test cannot run: neither snapshot ({snapshot}) nor real artifact "
        f"({explain}) is present. Snapshot the explain pass names into the fixtures dir to fix. "
        f"This is intentional fail-loud (gap 8 of adversary-review case_id 2026-05-07-124100) — "
        f"silent skips let load-bearing tests become no-ops invisibly."
    )


# Runner ────────────────────────────────────────────────────────────────────

def test_filter_on_generalized_field():
    """Task 0.5: filter grammar generalizes from 'status'-only to any field."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [
            {"name": "A", "source": "hf", "status": "ok", "numeric_status": "match"},
            {"name": "B", "source": "hf", "status": "ok", "numeric_status": "divergence"},
            {"name": "C", "source": "hf", "status": "ok", "numeric_status": "divergence"},
        ]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "numeric_status == divergence", out, extra_args=["--include-large"])
        assert r.returncode == 0, f"exited {r.returncode}: {r.stderr}"
        data = json.loads(out.read_text())
        names = {m["name"] for m in data["models"]}
        assert names == {"B", "C"}, f"expected {{B, C}}, got {names}"


def test_sample_n_deterministic():
    """Task 0.5: --n + --seed produces same output across two runs."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": f"M{i:02d}", "source": "hf", "status": "ok"} for i in range(20)]
        source = _make_source(rows, tmp)
        out_a = tmp / "a.json"; out_b = tmp / "b.json"
        ra = _run(source, "status == ok", out_a, extra_args=["--n", "5", "--seed", "7", "--include-large"])
        rb = _run(source, "status == ok", out_b, extra_args=["--n", "5", "--seed", "7", "--include-large"])
        assert ra.returncode == 0 and rb.returncode == 0
        names_a = sorted(m["name"] for m in json.loads(out_a.read_text())["models"])
        names_b = sorted(m["name"] for m in json.loads(out_b.read_text())["models"])
        assert names_a == names_b, f"non-deterministic: {names_a} != {names_b}"
        assert len(names_a) == 5


def test_sample_n_different_seed_yields_different_set():
    """Task 0.5: different --seed should generally yield different sample (smoke test)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": f"M{i:02d}", "source": "hf", "status": "ok"} for i in range(20)]
        source = _make_source(rows, tmp)
        out_a = tmp / "a.json"; out_b = tmp / "b.json"
        _run(source, "status == ok", out_a, extra_args=["--n", "5", "--seed", "1", "--include-large"])
        _run(source, "status == ok", out_b, extra_args=["--n", "5", "--seed", "2", "--include-large"])
        names_a = sorted(m["name"] for m in json.loads(out_a.read_text())["models"])
        names_b = sorted(m["name"] for m in json.loads(out_b.read_text())["models"])
        assert names_a != names_b, "seeds 1 and 2 happened to produce identical samples; pick different seeds in test"


def test_sample_n_larger_than_population_takes_all():
    """Task 0.5: --n > available population takes all (no error)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": f"M{i}", "source": "hf", "status": "ok"} for i in range(3)]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out, extra_args=["--n", "10", "--include-large"])
        assert r.returncode == 0, f"exited {r.returncode}: {r.stderr}"
        models = json.loads(out.read_text())["models"]
        assert len(models) == 3


def test_sample_n_negative_or_zero_errors():
    """Task 0.5: --n must be positive."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out, extra_args=["--n", "0", "--include-large"])
        assert r.returncode != 0


def test_default_excludes_large_models():
    """Task 0.5: default behavior excludes models in sweep/large_models.json."""
    # Use actual large_models.json — pick a name from it that we KNOW is in the file
    with open(REPO_ROOT / "sweep" / "large_models.json") as f:
        large_names = list(json.load(f).keys())
    if not large_names:
        return  # can't test without real large models
    real_large = large_names[0]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [
            {"name": real_large, "source": "hf", "status": "ok"},
            {"name": "TinyTestModelXYZ", "source": "hf", "status": "ok"},
        ]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)  # no --include-large
        assert r.returncode == 0
        names = {m["name"] for m in json.loads(out.read_text())["models"]}
        assert real_large not in names, f"large model {real_large} should be excluded by default"
        assert "TinyTestModelXYZ" in names


def test_include_large_keeps_listed_models():
    """Task 0.5: --include-large flips the default to keep large models."""
    with open(REPO_ROOT / "sweep" / "large_models.json") as f:
        large_names = list(json.load(f).keys())
    if not large_names:
        return
    real_large = large_names[0]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [
            {"name": real_large, "source": "hf", "status": "ok"},
            {"name": "TinyTestModelXYZ", "source": "hf", "status": "ok"},
        ]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out, extra_args=["--include-large"])
        assert r.returncode == 0
        names = {m["name"] for m in json.loads(out.read_text())["models"]}
        assert real_large in names, f"large model {real_large} should be included with --include-large"


def test_filter_then_large_exclude_then_sample_n_order():
    """Adversary gap 1: pin order-of-ops filter → large-exclude → sample-N.
    Real-large-name in source + filter that matches it + --n → sampled cohort
    must NEVER include the large name (would happen if sample ran first)."""
    with open(REPO_ROOT / "sweep" / "large_models.json") as f:
        large_names = list(json.load(f).keys())
    if len(large_names) < 3:
        return
    # Pick 3 real large names + 12 synthetic
    real_large = large_names[:3]
    synth = [f"Synth{i:02d}" for i in range(12)]
    rows = [{"name": n, "source": "hf", "status": "ok"} for n in real_large + synth]
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        # No --include-large; filter status=ok matches all 15
        r = _run(source, "status == ok", out, extra_args=["--n", "5", "--seed", "42"])
        assert r.returncode == 0, f"exited {r.returncode}: {r.stderr}"
        names = {m["name"] for m in json.loads(out.read_text())["models"]}
        assert len(names) == 5, f"expected 5 sampled; got {len(names)}"
        for ln in real_large:
            assert ln not in names, \
                f"large model {ln} leaked into sample (order swap?); got {names}"


def test_metadata_records_task05_fields():
    """Adversary gap 2 + gap 3 + gap 6: pin new metadata fields are recorded
    correctly when --n is set vs not set."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": f"Synth{i:02d}", "source": "hf", "status": "ok"} for i in range(10)]
        source = _make_source(rows, tmp)
        # Run 1: --n 3 with explicit --seed 99
        out_1 = tmp / "with_n.json"
        r = _run(source, "status == ok", out_1, extra_args=["--n", "3", "--seed", "99", "--include-large"])
        assert r.returncode == 0
        meta_1 = json.loads(out_1.read_text())["_metadata"]
        assert meta_1["sample_n"] == 3
        assert meta_1["sample_seed"] == 99
        assert meta_1["sample_seed_was_default"] is False
        assert meta_1["large_excluded"] is False
        assert meta_1["sample_python_version"] is not None
        # Run 2: no --n → all sample_* fields None
        out_2 = tmp / "no_n.json"
        r = _run(source, "status == ok", out_2, extra_args=["--include-large"])
        assert r.returncode == 0
        meta_2 = json.loads(out_2.read_text())["_metadata"]
        assert meta_2["sample_n"] is None
        assert meta_2["sample_seed"] is None
        assert meta_2["sample_seed_was_default"] is None
        assert meta_2["sample_python_version"] is None
        # Run 3: default seed (no --seed) → sample_seed_was_default True
        out_3 = tmp / "default_seed.json"
        r = _run(source, "status == ok", out_3, extra_args=["--n", "3", "--include-large"])
        assert r.returncode == 0
        meta_3 = json.loads(out_3.read_text())["_metadata"]
        assert meta_3["sample_seed"] == 42
        assert meta_3["sample_seed_was_default"] is True


def test_filter_field_partial_presence():
    """Adversary gap 4: filter on field present in some rows, absent in others.
    Pin current behavior: rows lacking the field are silently skipped."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # 10 rows: 5 have numeric_status, 5 don't
        rows = (
            [{"name": f"M{i}", "source": "hf", "status": "ok",
              "numeric_status": "divergence" if i < 3 else "match"} for i in range(5)] +
            [{"name": f"N{i}", "source": "hf", "status": "ok"} for i in range(5)]
        )
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "numeric_status == divergence", out, extra_args=["--include-large"])
        assert r.returncode == 0
        names = {m["name"] for m in json.loads(out.read_text())["models"]}
        assert names == {"M0", "M1", "M2"}, f"expected only M0/M1/M2; got {names}"


def test_empty_population_after_filter_exits_nonzero():
    """Adversary gap 5: empty cohort after filter+exclude → exit nonzero
    (was previously silent 0-model cohort)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        rows = [{"name": "A", "source": "hf", "status": "ok"}]
        source = _make_source(rows, tmp)
        out = tmp / "cohort.json"
        r = _run(source, "status == graph_break", out, extra_args=["--include-large"])
        assert r.returncode != 0
        assert "EMPTY" in r.stderr or "EMPTY" in r.stdout


def test_corrupted_large_models_json_fails_loud(monkeypatch=None):
    """Adversary gap 7: corrupted large_models.json → sys.exit (not silent
    fail-soft). Patches LARGE_MODELS_PATH to a bad file."""
    import importlib
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    import generate_cohort as gc
    importlib.reload(gc)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        bad = tmp / "bad_large_models.json"
        bad.write_text("{not valid json")
        # Monkey-patch the module-level path
        original = gc.LARGE_MODELS_PATH
        gc.LARGE_MODELS_PATH = bad
        try:
            try:
                gc.load_large_models()
                raise AssertionError("expected sys.exit on corrupted JSON")
            except SystemExit as e:
                assert e.code != 0
        finally:
            gc.LARGE_MODELS_PATH = original


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
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
