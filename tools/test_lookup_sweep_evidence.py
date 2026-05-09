#!/usr/bin/env python3
"""Tests for tools/lookup_sweep_evidence.py.

Pins the load-bearing claims (per Peng directives + adversary case file
adv-2026-05-09-120800):

- Filter by venv_name (gap mismatch in v3)
- Filter by within_days (10 days default per Peng 08:04 ET)
- Reject sweep whose produced_fields lacks the required signal field (gap 9)
- Stamp case_id into emitted JSON (gap 10)
- Carry cohort_sha256 + args_fingerprint + sweep_kind from INDEX (gap 3)
- No-match returns None / CLI exits non-zero
- Newest-first ordering when multiple sweeps match
- Absolute paths in INDEX (gap 11)

Run: python3 tools/test_lookup_sweep_evidence.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import lookup_sweep_evidence as lse  # noqa: E402


def _now_iso(offset_days: int = 0) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=offset_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _make_index_and_jsonl(rows_per_sweep: list[tuple[dict, list[dict]]]) -> Path:
    """Write an INDEX.json + per-sweep results.jsonl files in a tempdir.
    rows_per_sweep is a list of (sweep_metadata, list-of-result-rows).
    Returns path to INDEX.json.
    """
    td = Path(tempfile.mkdtemp(prefix="test_lookup_"))
    sweeps_meta = []
    for i, (meta, rows) in enumerate(rows_per_sweep):
        results_path = td / f"sweep_{i}" / "results.jsonl"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # Use absolute path (gap 11)
        meta = {**meta, "results_jsonl": str(results_path.resolve())}
        sweeps_meta.append(meta)
    idx_path = td / "INDEX.json"
    idx_path.write_text(json.dumps({"schema_version": 1, "sweeps": sweeps_meta}, indent=2))
    return idx_path


# ── Filter tests ──────────────────────────────────────────────────────────

def test_filter_by_venv_name():
    """Sweep with wrong venv_name is skipped."""
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "abc", "args_fingerprint": "def",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.4}],
        ),
    ])
    # Request nightly — sweep is current, should miss
    result = lse.lookup(
        model="X", mode="train", venv_name="nightly",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5.4"},
        case_id="case-1", index_path=idx,
    )
    assert result is None, f"expected None (venv mismatch); got: {result}"


def test_filter_by_within_days():
    """Sweep older than within_days is skipped."""
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-15),  # 15 days old
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "abc", "args_fingerprint": "def",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.4}],
        ),
    ])
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5.4"},
        case_id="case-1", within_days=10, index_path=idx,
    )
    assert result is None, f"expected None (15 days > 10 day threshold); got: {result}"


def test_within_days_default_is_10():
    """Default staleness threshold per Peng 2026-05-09 08:04 ET."""
    assert lse.STALENESS_DAYS_DEFAULT == 10


def test_filter_skips_sweep_lacking_required_signal_field():
    """Gap 9 anchor: graph-break-only sweep can't satisfy a numeric_max_diff request."""
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["status", "graph_break_count"],  # NO numeric_max_diff
                "cohort_sha256": "abc", "args_fingerprint": "def",
                "sweep_kind": "graph-break",
            },
            [{"name": "X", "mode": "train", "status": "graph_break", "graph_break_count": 7}],
        ),
    ])
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        # Signal demands numeric_max_diff field which this sweep doesn't produce
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5.4"},
        case_id="case-1", index_path=idx,
    )
    assert result is None, f"expected None (sweep lacks required field); got: {result}"


# ── Hit tests ──────────────────────────────────────────────────────────────

def test_returns_freshest_matching_sweep_when_multiple_match():
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "old", "venv_name": "current",
                "completed_utc": _now_iso(-5),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "old_cohort", "args_fingerprint": "old_args",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.0}],
        ),
        (
            {
                "sweep_id": "new", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "new_cohort", "args_fingerprint": "new_args",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.4}],
        ),
    ])
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5"},
        case_id="case-1", index_path=idx,
    )
    assert result is not None
    # Newest-first: should pick the 1-day-old sweep with new_cohort
    assert result["cohort_sha256"] == "new_cohort", f"got: {result}"
    assert result["args_fingerprint"] == "new_args"
    assert result["sweep_age_days"] <= 1


def test_stamps_case_id_into_emitted_json():
    """Gap 10 anchor: cluster batches need distinct case_ids on shared sweep rows."""
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "x", "args_fingerprint": "y",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.4}],
        ),
    ])
    for case_id in ["case-A", "case-B", "case-C"]:
        result = lse.lookup(
            model="X", mode="train", venv_name="current",
            expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5"},
            case_id=case_id, index_path=idx,
        )
        assert result is not None
        assert result["case_id"] == case_id, f"case_id not stamped for {case_id}: {result}"


def test_carries_cohort_and_args_fingerprint_and_sweep_kind():
    """Gap 3 anchor: stale-cohort soundness."""
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "cohort_abc123",
                "args_fingerprint": "args_def456",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.4}],
        ),
    ])
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5"},
        case_id="case-1", index_path=idx,
    )
    assert result["cohort_sha256"] == "cohort_abc123"
    assert result["args_fingerprint"] == "args_def456"
    assert result["sweep_kind"] == "ngb-verify"


def test_emits_evidence_source_sweep_results():
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "x", "args_fingerprint": "y",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.4}],
        ),
    ])
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5"},
        case_id="case-1", index_path=idx,
    )
    assert result["evidence_source"] == "sweep_results"
    assert result["elapsed_s"] == 0.0  # cached, no run cost


def test_sweep_path_is_absolute():
    """Gap 11 anchor."""
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "x", "args_fingerprint": "y",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "X", "mode": "train", "numeric_max_diff": 5.4}],
        ),
    ])
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5"},
        case_id="case-1", index_path=idx,
    )
    assert Path(result["sweep_path"]).is_absolute()


def test_no_match_when_model_mode_not_in_results_jsonl():
    idx = _make_index_and_jsonl([
        (
            {
                "sweep_id": "s1", "venv_name": "current",
                "completed_utc": _now_iso(-1),
                "produced_fields": ["numeric_max_diff"],
                "cohort_sha256": "x", "args_fingerprint": "y",
                "sweep_kind": "ngb-verify",
            },
            [{"name": "OtherModel", "mode": "eval", "numeric_max_diff": 5.4}],
        ),
    ])
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5"},
        case_id="case-1", index_path=idx,
    )
    assert result is None


def test_missing_index_returns_none():
    result = lse.lookup(
        model="X", mode="train", venv_name="current",
        expected_signal={"kind": "stdout_contains", "fragment": "numeric_max_diff=5"},
        case_id="case-1", index_path=Path("/tmp/does-not-exist-INDEX.json"),
    )
    assert result is None


def test_signal_required_fields_recognizes_numeric_max_diff():
    fields = lse._signal_required_fields(
        {"kind": "stdout_contains", "fragment": "numeric_max_diff=5.4"}
    )
    assert "numeric_max_diff" in fields


def test_signal_required_fields_recognizes_runtime_error():
    fields = lse._signal_required_fields(
        {"kind": "exit_nonzero+stderr_contains", "fragment": "RuntimeError: aten.bincount"}
    )
    # Maps to status/error fields
    assert any(f in fields for f in ("status", "error"))


def test_signal_required_fields_returns_empty_for_unknown_pattern():
    """Unknown signal shape: don't filter (caller's responsibility)."""
    fields = lse._signal_required_fields(
        {"kind": "stdout_contains", "fragment": "some_unknown_keyword_pattern_xyz"}
    )
    assert fields == []


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
