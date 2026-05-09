#!/usr/bin/env python3
"""Tests for sweep/sweep_index.py.

Pins INDEX.json semantics per Peng directive 2026-05-09 07:55 ET +
adversary case file adv-2026-05-09-120800 (gaps 3, 4, 9, 10, 11):

- Idempotent by sweep_id (gap 4: 1 row per sweep, NOT 1 per pass)
- Records cohort_sha256 + args_fingerprint (gap 3 soundness)
- Records sweep_kind + produced_fields (gap 9: lookup filter prerequisite)
- Absolute paths (gap 11)
- Atomic write (concurrent safety)
- Failure swallow (sweep completion > index bookkeeping)

Run with PYTHONPATH=$(pwd):
  PYTHONPATH=$(pwd) python3 sweep/test_sweep_index.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sweep"))

import sweep_index  # noqa: E402


def _make_results_jsonl(rows: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    f.write(json.dumps({"_record_type": "metadata", "ran_at": "2026-05-09"}) + "\n")
    for r in rows:
        f.write(json.dumps({"_record_type": "row", **r}) + "\n")
    f.close()
    return Path(f.name)


def _make_idx_path() -> Path:
    return Path(tempfile.NamedTemporaryFile(suffix="-INDEX.json", delete=False).name)


def test_creates_index_when_missing():
    idx = Path(tempfile.mkdtemp()) / "subdir" / "INDEX.json"
    rows = [{"name": "X", "mode": "train", "status": "graph_break"}]
    rj = _make_results_jsonl(rows)
    try:
        sweep_index.append_to_index(
            sweep_id="s1", results_jsonl=str(rj),
            sweep_kind="identify", python_bin="/usr/bin/python3",
            index_path=idx,
        )
        assert idx.is_file(), f"index not created at {idx}"
        idx_data = json.loads(idx.read_bytes())
        assert idx_data["schema_version"] == 1
        assert len(idx_data["sweeps"]) == 1
        assert idx_data["sweeps"][0]["sweep_id"] == "s1"
    finally:
        rj.unlink(missing_ok=True)
        if idx.exists():
            idx.unlink()


def test_appends_to_existing_index():
    idx = _make_idx_path()
    idx.write_text(json.dumps({"schema_version": 1, "sweeps": [
        {"sweep_id": "old", "venv_name": "current"}
    ]}))
    rows = [{"name": "X", "mode": "train"}]
    rj = _make_results_jsonl(rows)
    try:
        sweep_index.append_to_index(
            sweep_id="new", results_jsonl=str(rj),
            sweep_kind="identify", python_bin="/usr/bin/python3",
            index_path=idx,
        )
        idx_data = json.loads(idx.read_bytes())
        sweep_ids = [s["sweep_id"] for s in idx_data["sweeps"]]
        assert "old" in sweep_ids
        assert "new" in sweep_ids
        assert len(idx_data["sweeps"]) == 2
    finally:
        idx.unlink(missing_ok=True)
        rj.unlink(missing_ok=True)


def test_idempotent_by_sweep_id():
    """Gap 4 anchor: re-running with same sweep_id REPLACES, doesn't duplicate.

    The original adversary concern: a sweep with multiple passes (identify +
    explain) might emit multiple INDEX rows. Idempotence-by-sweep_id ensures
    1 row per sweep regardless of how many passes call append_to_index for it.
    """
    idx = _make_idx_path()
    rows = [{"name": "X", "mode": "train", "status": "graph_break"}]
    rj = _make_results_jsonl(rows)
    try:
        # First call
        sweep_index.append_to_index(
            sweep_id="duplicate-id", results_jsonl=str(rj),
            sweep_kind="identify", python_bin="/usr/bin/python3",
            index_path=idx,
        )
        # Second call with SAME sweep_id (different sweep_kind to confirm replacement)
        sweep_index.append_to_index(
            sweep_id="duplicate-id", results_jsonl=str(rj),
            sweep_kind="explain", python_bin="/usr/bin/python3",
            index_path=idx,
        )
        idx_data = json.loads(idx.read_bytes())
        matching = [s for s in idx_data["sweeps"] if s["sweep_id"] == "duplicate-id"]
        assert len(matching) == 1, f"expected 1 row for duplicate-id; got {len(matching)}"
        # Latest call wins (sweep_kind=explain)
        assert matching[0]["sweep_kind"] == "explain"
    finally:
        idx.unlink(missing_ok=True)
        rj.unlink(missing_ok=True)


def test_records_cohort_sha256_when_cohort_provided():
    """Gap 3 anchor: cache freshness binds to cohort content, not just path."""
    idx = _make_idx_path()
    cohort = Path(tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False).name)
    cohort.write_text('{"models": ["X"]}')
    rj = _make_results_jsonl([{"name": "X", "mode": "train"}])
    try:
        sweep_index.append_to_index(
            sweep_id="s1", results_jsonl=str(rj),
            sweep_kind="identify", python_bin="/usr/bin/python3",
            cohort=str(cohort), index_path=idx,
        )
        idx_data = json.loads(idx.read_bytes())
        row = idx_data["sweeps"][0]
        assert row["cohort_sha256"] != ""
        # Verify it actually matches the cohort content
        import hashlib
        expected = hashlib.sha256(cohort.read_bytes()).hexdigest()
        assert row["cohort_sha256"] == expected
    finally:
        idx.unlink(missing_ok=True)
        cohort.unlink(missing_ok=True)
        rj.unlink(missing_ok=True)


def test_records_args_fingerprint_from_args_dict():
    """Gap 3 anchor: args_fingerprint records the worker/mode/flag canonicalization."""
    idx = _make_idx_path()
    rj = _make_results_jsonl([{"name": "X", "mode": "train"}])
    try:
        # Same args -> same fingerprint
        for _ in range(2):
            sweep_index.append_to_index(
                sweep_id=f"s_{_}", results_jsonl=str(rj),
                sweep_kind="identify", python_bin="/usr/bin/python3",
                args_dict={"workers": 4, "mode": "train"},
                index_path=idx,
            )
        idx_data = json.loads(idx.read_bytes())
        rows = idx_data["sweeps"]
        assert len(rows) == 2
        assert rows[0]["args_fingerprint"] == rows[1]["args_fingerprint"]

        # Different args -> different fingerprint
        sweep_index.append_to_index(
            sweep_id="s_diff", results_jsonl=str(rj),
            sweep_kind="identify", python_bin="/usr/bin/python3",
            args_dict={"workers": 8, "mode": "train"},  # workers differs
            index_path=idx,
        )
        idx_data = json.loads(idx.read_bytes())
        diff_row = next(s for s in idx_data["sweeps"] if s["sweep_id"] == "s_diff")
        assert diff_row["args_fingerprint"] != rows[0]["args_fingerprint"]
    finally:
        idx.unlink(missing_ok=True)
        rj.unlink(missing_ok=True)


def test_records_sweep_kind_and_produced_fields():
    """Gap 9 anchor: lookup needs sweep_kind + produced_fields to filter."""
    idx = _make_idx_path()
    rj = _make_results_jsonl([
        {"name": "X", "mode": "train", "status": "graph_break", "graph_break_count": 7},
        {"name": "Y", "mode": "train", "status": "full_graph", "graph_break_count": 0},
    ])
    try:
        sweep_index.append_to_index(
            sweep_id="s1", results_jsonl=str(rj),
            sweep_kind="ngb-verify", python_bin="/usr/bin/python3",
            index_path=idx,
        )
        idx_data = json.loads(idx.read_bytes())
        row = idx_data["sweeps"][0]
        assert row["sweep_kind"] == "ngb-verify"
        assert "status" in row["produced_fields"]
        assert "graph_break_count" in row["produced_fields"]
        assert "name" in row["produced_fields"]
        assert "mode" in row["produced_fields"]
    finally:
        idx.unlink(missing_ok=True)
        rj.unlink(missing_ok=True)


def test_paths_are_absolute():
    """Gap 11 anchor."""
    idx = _make_idx_path()
    rj = _make_results_jsonl([{"name": "X", "mode": "train"}])
    try:
        sweep_index.append_to_index(
            sweep_id="s1", results_jsonl=str(rj),
            sweep_kind="identify", python_bin="/usr/bin/python3",
            index_path=idx,
        )
        idx_data = json.loads(idx.read_bytes())
        row = idx_data["sweeps"][0]
        assert Path(row["results_jsonl"]).is_absolute()
        assert Path(row["venv_path"]).is_absolute()
    finally:
        idx.unlink(missing_ok=True)
        rj.unlink(missing_ok=True)


def test_failure_swallowed_does_not_raise():
    """Sweep completion > index bookkeeping. If index can't be written
    (permissions, disk full, etc.), the sweep should still succeed.
    """
    # Pass an unwritable parent path
    idx = Path("/proc/1/cannot-write-here.json")  # typically not writable
    rj = _make_results_jsonl([{"name": "X", "mode": "train"}])
    try:
        # Must not raise even with bogus index_path
        result = sweep_index.append_to_index(
            sweep_id="s1", results_jsonl=str(rj),
            sweep_kind="identify", python_bin="/usr/bin/python3",
            index_path=idx,
        )
        # Returns the path it tried to write; presence of the file isn't required
        assert result == idx
    finally:
        rj.unlink(missing_ok=True)


def test_atomic_write_via_tempfile():
    """Atomic via tempfile + rename: no partial INDEX.json on concurrent crashes.

    We can't easily test concurrent behavior; verify the write doesn't leave
    a half-written file when interrupted (pin: tempfile pattern is used).
    """
    idx = _make_idx_path()
    rj = _make_results_jsonl([{"name": "X", "mode": "train"}])
    try:
        # Inspect that source uses tempfile.NamedTemporaryFile + os.replace
        import inspect
        src = inspect.getsource(sweep_index.append_to_index)
        assert "tempfile.NamedTemporaryFile" in src
        assert "os.replace" in src
    finally:
        idx.unlink(missing_ok=True)
        rj.unlink(missing_ok=True)


def test_detects_venv_name_nightly_from_path():
    name = sweep_index._venv_name_from_path("/home/x/envs/torch-nightly-cu128/bin/python")
    assert name == "nightly"


def test_detects_venv_name_current_for_non_nightly():
    name = sweep_index._venv_name_from_path("/home/x/envs/torch211/bin/python")
    assert name == "current"


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
