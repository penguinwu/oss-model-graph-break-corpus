#!/usr/bin/env python3
"""Tests for sweep/cohort_validator.py.

Surfaced by adversary-review case_id 2026-05-07-124100-cohort-regen-fix.
Each test maps to one or more proposed tests in the review:

    test_loader_rejects_bare_list_cohort                 → review test 1 (gap 1)
    test_loader_rejects_empty_source_versions            → review test 2 (gap 2)
    test_loader_rejects_partial_source_versions_default  → review test 3 (gap 6)
    test_loader_rejects_stale_cohort_vs_source_mtime     → review test 5 (gap 5)
    test_loader_treats_missing_metadata_keys_as_failure  → review test 11 (gaps 1, 2)
    test_loader_accepts_canonical_cohort                 → happy path
    test_allow_bare_overrides_rejection                  → opt-in path
    test_allow_empty_versions_overrides_rejection        → opt-in path
    test_allow_partial_versions_overrides_rejection      → opt-in path
    test_version_mismatch_rejected                       → existing behavior, preserved
    test_allow_version_mismatch_overrides_rejection      → existing behavior, preserved

Run: python3 sweep/test_cohort_validator.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

# Make sure sweep/ is importable when test is run directly
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sweep.cohort_validator import (  # noqa: E402
    CohortValidationError,
    REQUIRED_METADATA_KEYS,
    REQUIRED_VERSION_KEYS,
    validate_cohort,
)


def _canonical_metadata(*, source_versions=None, derived_from="some/source.json"):
    sv = source_versions if source_versions is not None else {
        "torch": "2.13.0", "transformers": "5.6.2", "diffusers": "0.38.0",
    }
    return {
        "derived_from": derived_from,
        "filter": "status == ok",
        "model_count": 1,
        "source_versions": sv,
        "derived_at": "2026-05-07T16:00:00Z",
        "generated_by": "tools/generate_cohort.py",
    }


def _write_cohort(path: Path, *, models=None, metadata=None, bare=False):
    payload_models = models if models is not None else [{"name": "Bart", "source": "hf"}]
    if bare:
        path.write_text(json.dumps(payload_models))
        return
    payload = {"_metadata": metadata or _canonical_metadata(), "models": payload_models}
    path.write_text(json.dumps(payload))


# Loader-rejection tests ────────────────────────────────────────────────────

def test_loader_rejects_bare_list_cohort():
    """gap 1: bare list (no _metadata) → BARE_LIST_REJECTED unless allow_bare."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bare.json"
        _write_cohort(p, bare=True)
        try:
            validate_cohort(p)
        except CohortValidationError as e:
            assert e.code == "BARE_LIST_REJECTED", f"got code {e.code!r}"
            return
        assert False, "expected CohortValidationError(BARE_LIST_REJECTED), got success"


def test_loader_rejects_empty_source_versions():
    """gap 2: empty source_versions → EMPTY_SOURCE_VERSIONS unless allow_empty_versions."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cohort.json"
        _write_cohort(p, metadata=_canonical_metadata(source_versions={}))
        try:
            validate_cohort(p)
        except CohortValidationError as e:
            assert e.code == "EMPTY_SOURCE_VERSIONS", f"got code {e.code!r}"
            return
        assert False, "expected CohortValidationError(EMPTY_SOURCE_VERSIONS), got success"


def test_loader_rejects_partial_source_versions_default():
    """gap 6: partial source_versions (only torch, no transformers/diffusers) → PARTIAL_SOURCE_VERSIONS."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cohort.json"
        _write_cohort(p, metadata=_canonical_metadata(source_versions={"torch": "2.13.0"}))
        try:
            validate_cohort(p)
        except CohortValidationError as e:
            assert e.code == "PARTIAL_SOURCE_VERSIONS", f"got code {e.code!r}"
            assert "transformers" in e.details["missing"]
            assert "diffusers" in e.details["missing"]
            return
        assert False, "expected CohortValidationError(PARTIAL_SOURCE_VERSIONS), got success"


def test_loader_treats_missing_metadata_keys_as_failure():
    """gap 11: missing required _metadata key → MISSING_METADATA_KEY."""
    for missing_key in REQUIRED_METADATA_KEYS:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cohort.json"
            md = _canonical_metadata()
            del md[missing_key]
            _write_cohort(p, metadata=md)
            try:
                validate_cohort(p)
            except CohortValidationError as e:
                assert e.code == "MISSING_METADATA_KEY", \
                    f"missing {missing_key!r}: got code {e.code!r}"
                assert missing_key in e.details["missing"], \
                    f"missing {missing_key!r}: details.missing={e.details!r}"
                continue
            assert False, f"expected MISSING_METADATA_KEY for {missing_key!r}, got success"


def test_loader_rejects_stale_cohort_vs_source_mtime():
    """gap 5: cohort mtime older than source mtime → STALE_COHORT."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        # Write source first, then cohort, then touch source newer
        source.write_text('{"results": []}')
        time.sleep(0.05)
        _write_cohort(cohort, metadata=_canonical_metadata(derived_from=str(source.resolve())))
        time.sleep(0.05)
        # Touch source to a newer mtime
        new_mtime = cohort.stat().st_mtime + 10  # 10 seconds in the future
        import os
        os.utime(source, (new_mtime, new_mtime))
        try:
            validate_cohort(cohort)
        except CohortValidationError as e:
            assert e.code == "STALE_COHORT", f"got code {e.code!r}"
            assert e.details["source_mtime"] > e.details["cohort_mtime"]
            return
        assert False, "expected STALE_COHORT, got success"


# Happy-path tests ─────────────────────────────────────────────────────────

def test_loader_accepts_canonical_cohort():
    """Canonical cohort (full metadata, all 3 versions, fresh) loads cleanly."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        source.write_text('{"results": []}')
        _write_cohort(cohort, metadata=_canonical_metadata(derived_from=str(source.resolve())))
        result = validate_cohort(cohort)
        assert not result.is_bare
        assert result.source_versions["torch"] == "2.13.0"
        assert len(result.specs) == 1
        assert result.specs[0]["name"] == "Bart"


# Opt-in override tests ────────────────────────────────────────────────────

def test_allow_bare_overrides_rejection():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bare.json"
        _write_cohort(p, bare=True)
        result = validate_cohort(p, allow_bare=True)
        assert result.is_bare
        assert len(result.specs) == 1


def test_allow_empty_versions_overrides_rejection():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        source.write_text('{}')
        _write_cohort(cohort, metadata=_canonical_metadata(
            source_versions={}, derived_from=str(source.resolve())))
        result = validate_cohort(cohort, allow_empty_versions=True)
        assert result.source_versions == {}


def test_allow_partial_versions_overrides_rejection():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        source.write_text('{}')
        _write_cohort(cohort, metadata=_canonical_metadata(
            source_versions={"torch": "2.13.0"}, derived_from=str(source.resolve())))
        result = validate_cohort(cohort, allow_partial_versions=True)
        assert "transformers" not in result.source_versions


# Version-mismatch tests ──────────────────────────────────────────────────

def test_version_mismatch_rejected():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        source.write_text('{}')
        _write_cohort(cohort, metadata=_canonical_metadata(derived_from=str(source.resolve())))
        try:
            validate_cohort(cohort, version_info={
                "torch": "2.11.0", "transformers": "5.6.2", "diffusers": "0.38.0"
            })
        except CohortValidationError as e:
            assert e.code == "VERSION_MISMATCH", f"got code {e.code!r}"
            assert e.details["mismatches"][0]["package"] == "torch"
            return
        assert False, "expected VERSION_MISMATCH, got success"


def test_allow_version_mismatch_overrides_rejection():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        source = tmp / "source.json"
        cohort = tmp / "cohort.json"
        source.write_text('{}')
        _write_cohort(cohort, metadata=_canonical_metadata(derived_from=str(source.resolve())))
        result = validate_cohort(cohort, version_info={
            "torch": "2.11.0", "transformers": "5.6.2", "diffusers": "0.38.0"
        }, allow_version_mismatch=True)
        assert result.source_versions["torch"] == "2.13.0"


def test_file_not_found():
    p = Path("/tmp/nonexistent_cohort_xyz123.json")
    if p.exists():
        p.unlink()
    try:
        validate_cohort(p)
    except CohortValidationError as e:
        assert e.code == "FILE_NOT_FOUND"
        return
    assert False, "expected FILE_NOT_FOUND"


def test_invalid_json():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        p.write_text("{not json")
        try:
            validate_cohort(p)
        except CohortValidationError as e:
            assert e.code == "INVALID_JSON"
            return
        assert False, "expected INVALID_JSON"


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
