#!/usr/bin/env python3
"""Tests for tools/cluster_failures.py.

Pins the load-bearing claims of the V1 cluster+dedup gate (Peng directive
2026-05-08T22:01 ET):
- single-manual mode emits a 1-row plan (closes the dead-air window)
- numeric clusterer audit invariant: total_clustered_rows == sum(case_count)
- numeric clusterer counts only D1 candidates (NOT all rows in the file)
  — fixes the "total_failure_rows: 380 from 14 actual D1 cases" defect
- magnitude bucket is METADATA, NOT a primary clustering axis
  — the audio-encoder family does NOT split into 7+2 by magnitude
- fallback mode warns and emits 1 cluster per row

Run: python3 tools/test_cluster_failures.py
Exit non-zero on any failure.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "cluster_failures.py"
PLANS_DIR = REPO_ROOT / "subagents/file-issue/cluster-plans"
PYTHON = sys.executable


def _run(*args: str) -> tuple[int, str, str]:
    r = subprocess.run([PYTHON, str(TOOL), *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def _write_results_jsonl(rows: list[dict]) -> Path:
    """Write a synthetic results.jsonl into a tempfile, return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


def _load_plan(stdout: str) -> dict:
    """Find the 'Wrote <path>' line in stdout, load and return that plan."""
    for line in stdout.splitlines():
        if line.startswith("Wrote "):
            rel = line[len("Wrote "):].strip()
            return json.loads((REPO_ROOT / rel).read_bytes())
    raise AssertionError(f"no 'Wrote' line in stdout: {stdout!r}")


def test_single_manual_emits_one_row_plan():
    case_id = "file-2026-05-09-test-single"
    rc, out, err = _run("single-manual", case_id)
    try:
        assert rc == 0, f"expected 0; got rc={rc}, err={err}"
        plan = _load_plan(out)
        assert plan["single_manual"] is True
        assert plan["total_failure_rows"] == 1
        assert len(plan["clusters"]) == 1
        assert plan["clusters"][0]["case_count"] == 1
        assert plan["clusters"][0]["affected_cases"][0]["case_id"] == case_id
    finally:
        # Clean up the test plan
        for p in PLANS_DIR.glob(f"*{case_id}*.yaml"):
            p.unlink(missing_ok=True)


def test_numeric_clusterer_audit_invariant():
    """sum(case_count) == total_clustered_rows == n_failure_rows.

    Counts ONLY rows where numeric_max_diff > 1e-3 (not all rows in the
    file). Fixes the post-smoke defect where total_failure_rows reported
    all 380 NGB rows instead of the 14 D1 candidates.
    """
    rows = [
        # 9 audio-encoder rows that DO diverge (D1 candidates)
        {"name": "Wav2Vec2Model", "mode": "train", "numeric_max_diff": 5.5},
        {"name": "WavLMModel", "mode": "train", "numeric_max_diff": 5.2},
        {"name": "HubertModel", "mode": "train", "numeric_max_diff": 5.1},
        {"name": "UniSpeechModel", "mode": "train", "numeric_max_diff": 5.3},
        {"name": "UniSpeechSatModel", "mode": "train", "numeric_max_diff": 5.0},
        {"name": "Data2VecAudioModel", "mode": "train", "numeric_max_diff": 5.0},
        {"name": "Wav2Vec2ConformerModel", "mode": "train", "numeric_max_diff": 3.3},
        {"name": "SEWModel", "mode": "train", "numeric_max_diff": 2.4},
        {"name": "SpeechEncoderDecoderModel", "mode": "train", "numeric_max_diff": 4.7},
        # 1 reformer (other family) divergence
        {"name": "ReformerModel", "mode": "train", "numeric_max_diff": 4.7},
        # NOISE: 100 rows that DO NOT diverge (numeric_max_diff at floor)
        *[{"name": f"NoiseModel{i}", "mode": "eval",
           "numeric_max_diff": 1e-5} for i in range(100)],
    ]
    rp = _write_results_jsonl(rows)
    try:
        rc, out, err = _run("from-sweep", str(rp), "--cluster-type", "numeric")
        assert rc == 0, f"expected 0; got rc={rc}, err={err}"
        plan = _load_plan(out)
        # total_failure_rows counts only D1 candidates (10), not all 110 rows
        assert plan["total_failure_rows"] == 10, (
            f"D1-candidate count wrong: expected 10, got {plan['total_failure_rows']}"
        )
        # total_rows_in_sweep records the file-row count (separate field)
        assert plan["total_rows_in_sweep"] == 110, (
            f"sweep-row count wrong: expected 110, got {plan['total_rows_in_sweep']}"
        )
        assert plan["total_clustered_rows"] == 10
        assert sum(c["case_count"] for c in plan["clusters"]) == 10
    finally:
        rp.unlink(missing_ok=True)
        for p in PLANS_DIR.glob(f"*{rp.name}*.yaml"):
            p.unlink(missing_ok=True)
        # The plan file is named by sweep_ref. It uses the abs path of rp.
        # Do a glob cleanup just in case.
        for p in PLANS_DIR.glob("*tmp*.yaml"):
            p.unlink(missing_ok=True)


def test_numeric_clusterer_does_not_split_audio_family_by_magnitude():
    """The 9 audio-encoder rows MUST land in ONE cluster regardless of
    magnitude spread (2.4–5.5). Magnitude is METADATA, not a primary axis.

    This pins the post-smoke fix where the prior (family + mode + magnitude)
    grouping artificially split the audio family into 7+2 by magnitude.
    """
    rows = [
        # 9 audio-encoder rows with magnitudes spanning 2.0+ buckets
        {"name": "Wav2Vec2Model", "mode": "train", "numeric_max_diff": 5.5},
        {"name": "WavLMModel", "mode": "train", "numeric_max_diff": 5.2},
        {"name": "HubertModel", "mode": "train", "numeric_max_diff": 5.1},
        {"name": "UniSpeechModel", "mode": "train", "numeric_max_diff": 5.3},
        {"name": "UniSpeechSatModel", "mode": "train", "numeric_max_diff": 5.0},
        {"name": "Data2VecAudioModel", "mode": "train", "numeric_max_diff": 5.0},
        {"name": "Wav2Vec2ConformerModel", "mode": "train", "numeric_max_diff": 3.3},
        {"name": "SEWModel", "mode": "train", "numeric_max_diff": 2.4},
        {"name": "SpeechEncoderDecoderModel", "mode": "train", "numeric_max_diff": 4.7},
    ]
    rp = _write_results_jsonl(rows)
    try:
        rc, out, err = _run("from-sweep", str(rp), "--cluster-type", "numeric")
        assert rc == 0, f"expected 0; got rc={rc}, err={err}"
        plan = _load_plan(out)
        # All 9 audio rows should land in ONE cluster
        audio_clusters = [c for c in plan["clusters"]
                          if c["root_signal"].get("architecture_family") == "audio_encoder"]
        assert len(audio_clusters) == 1, (
            f"audio family should be 1 cluster (magnitude is metadata, not "
            f"a clustering axis); got {len(audio_clusters)}: "
            f"{[c['cluster_id'] for c in audio_clusters]}"
        )
        assert audio_clusters[0]["case_count"] == 9
        # magnitude_range must be present in root_signal as METADATA
        assert "magnitude_range" in audio_clusters[0]["root_signal"]
    finally:
        rp.unlink(missing_ok=True)
        for p in PLANS_DIR.glob("*tmp*.yaml"):
            p.unlink(missing_ok=True)


def test_fallback_mode_warns_and_one_cluster_per_row():
    """Unknown failure types fall back to one-cluster-per-row with a warning.

    The warning surface is critical — silent fallback would mean the
    operator wouldn't know clustering was bypassed.
    """
    rows = [
        {"name": "ModelA"},
        {"name": "ModelB"},
        {"name": "ModelC"},
    ]
    rp = _write_results_jsonl(rows)
    try:
        rc, out, err = _run("from-sweep", str(rp), "--cluster-type", "fallback")
        assert rc == 0, f"expected 0; got rc={rc}, err={err}"
        assert "WARNING" in out and "fallback" in out, \
            f"fallback should print a warning; got: {out}"
        plan = _load_plan(out)
        assert plan["clustering_method"] == "unknown_type_fallback"
        assert len(plan["clusters"]) == 3
        assert plan["total_clustered_rows"] == 3
    finally:
        rp.unlink(missing_ok=True)
        for p in PLANS_DIR.glob("*tmp*.yaml"):
            p.unlink(missing_ok=True)


def test_unknown_cluster_type_exits_nonzero():
    """Bad --cluster-type → tool refuses (argparse choices=[...] enforces)."""
    rp = _write_results_jsonl([{"name": "X"}])
    try:
        rc, out, err = _run("from-sweep", str(rp), "--cluster-type", "bogus")
        assert rc != 0, f"expected non-zero (argparse rejects); got rc={rc}"
    finally:
        rp.unlink(missing_ok=True)


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
