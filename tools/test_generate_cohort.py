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


def _run(source: Path, filter_expr: str, output: Path, force: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(TOOL), "--from", str(source), "--filter", filter_expr, "--output", str(output)]
    if force:
        cmd.append("--force")
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
    """Unsupported filter expressions fail loudly, not silently."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp)
        out = tmp / "cohort.json"
        r = _run(source, "name == A", out)  # name field not supported
        assert r.returncode != 0


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
            r = _run(synthetic_source, "status == ok", out)
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
