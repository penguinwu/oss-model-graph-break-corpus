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


def test_metadata_versions_empty_when_source_lacks():
    """Source without versions metadata produces empty source_versions (warning, not crash)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = _make_source([{"name": "A", "source": "hf", "status": "ok"}], tmp, with_versions=False)
        out = tmp / "cohort.json"
        r = _run(source, "status == ok", out)
        assert r.returncode == 0
        data = json.loads(out.read_text())
        assert data["_metadata"]["source_versions"] == {}


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
    status=ok subset (190 models). This test runs against the real explain
    pass results and asserts the tool produces 190 names — exactly the count
    that should have been generated last night.
    """
    explain = REPO_ROOT / "sweep_results/experiments/nested-gb-2026-05-05-2026-05-05/explain_results.json"
    if not explain.is_file():
        # Skip silently if the historical artifact has been pruned
        print("  SKIP test_regression_2026_05_06_real_explain_pass: source artifact missing")
        return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "cohort.json"
        r = _run(explain, "status == ok", out)
        assert r.returncode == 0, f"exited {r.returncode}: {r.stderr}"
        data = json.loads(out.read_text())
        assert data["_metadata"]["model_count"] == 190, \
            f"regression: expected 190 models from explain pass status==ok filter, got {data['_metadata']['model_count']}"
        # Confirm the broken cohort's contaminated extras are NOT in the regenerated cohort
        names = {m["name"] for m in data["models"]}
        for contaminated in ("MiniCPMV4_6ForConditionalGeneration", "MiniCPMV4_6Model",
                             "PPFormulaNetForConditionalGeneration"):
            assert contaminated not in names, \
                f"regression: {contaminated} reappeared in cohort (was filtered out 2026-05-07)"


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
