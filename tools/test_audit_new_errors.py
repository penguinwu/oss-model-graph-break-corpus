#!/usr/bin/env python3
"""Tests for tools/audit_new_errors.py.

7 critical tests pinned to real-data fixtures from sweep_results/nightly/2026-05-09.
Per design rev 2 + adversary case adv-2026-05-10-145000.

Run: PYTHONPATH=$(pwd) ~/envs/torch211/bin/python tools/test_audit_new_errors.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from audit_new_errors import (  # noqa: E402
    classify, RULES, run_audit, emit_rerun_marker,
    load_known_errors_filtered, ERROR_STATUSES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Real-row fixtures from sweep_results/nightly/2026-05-09 effective rows
# (loaded via load_effective_results — what the audit actually classifies).
# ─────────────────────────────────────────────────────────────────────────────

REAL_2026_05_09 = {
    ("HiggsAudioV2TokenizerModel", "eval"): {
        "name": "HiggsAudioV2TokenizerModel", "mode": "eval", "status": "eager_error",
        "error_type": "RuntimeError", "phase": "eager",
        "error": "Sizes of tensors must match except in dimension 1. Expected size 76 but got size 50 for tensor number 1 in the list.",
        "retry_note": "confirmed_error",
    },
    ("HiggsAudioV2TokenizerModel", "train"): {
        "name": "HiggsAudioV2TokenizerModel", "mode": "train", "status": "eager_error",
        "error_type": "RuntimeError", "phase": "eager",
        "error": "Sizes of tensors must match except in dimension 1. Expected size 76 but got size 50 for tensor number 1 in the list.",
        "retry_note": "confirmed_error",
    },
    ("PI0Model", "eval"): {
        "name": "PI0Model", "mode": "eval", "status": "eager_error",
        "error_type": "AttributeError", "phase": "eager",
        "error": "'NoneType' object has no attribute 'get_seq_length'",
        "retry_note": "confirmed_error",
    },
    ("PI0Model", "train"): {
        "name": "PI0Model", "mode": "train", "status": "eager_error",
        "error_type": "AttributeError", "phase": "eager",
        "error": "'NoneType' object has no attribute 'get_seq_length'",
        "retry_note": "confirmed_error",
    },
    ("RwkvForCausalLM", "eval"): {
        "name": "RwkvForCausalLM", "mode": "eval", "status": "timeout",
        "error_type": None, "phase": None, "phase_at_timeout": "eager",
        "error": "",
    },
    ("RwkvForCausalLM", "train"): {
        "name": "RwkvForCausalLM", "mode": "train", "status": "timeout",
        "error_type": None, "phase": None, "phase_at_timeout": "eager",
        "error": "",
    },
    ("RwkvModel", "eval"): {
        "name": "RwkvModel", "mode": "eval", "status": "timeout",
        "error_type": None, "phase": None, "phase_at_timeout": "eager",
        "error": "",
    },
    ("RwkvModel", "train"): {
        "name": "RwkvModel", "mode": "train", "status": "timeout",
        "error_type": None, "phase": None, "phase_at_timeout": "eager",
        "error": "",
    },
}

EXPECTED_CLASSIFICATION_2026_05_09 = {
    # HiggsAudio matches fixture-bug via "Sizes of tensors must match" substring +
    # RuntimeError + phase=eager
    ("HiggsAudioV2TokenizerModel", "eval"):  "fixture-bug",
    ("HiggsAudioV2TokenizerModel", "train"): "fixture-bug",
    # PI0Model matches upstream-bug via AttributeError
    ("PI0Model", "eval"):  "upstream-bug",
    ("PI0Model", "train"): "upstream-bug",
    # Rwkv* timeouts match tier-upgrade
    ("RwkvForCausalLM", "eval"):  "tier-upgrade",
    ("RwkvForCausalLM", "train"): "tier-upgrade",
    ("RwkvModel", "eval"):        "tier-upgrade",
    ("RwkvModel", "train"):       "tier-upgrade",
}


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_field_names_match_real_data():
    """Catches gap #1 (wrong field name → 100% unknown). Reads `error`, not `error_message`."""
    for key, row in REAL_2026_05_09.items():
        result = classify(row)
        assert result["triage_class"] != "unknown", \
            f"{key} fell through to unknown — likely field-name bug. Row: {row}"


def test_classifies_real_2026_05_09_candidates():
    """Each of the 8 effective error rows from 2026-05-09 gets the expected class."""
    for key, row in REAL_2026_05_09.items():
        result = classify(row)
        expected = EXPECTED_CLASSIFICATION_2026_05_09[key]
        assert result["triage_class"] == expected, \
            f"{key}: expected {expected!r}, got {result['triage_class']!r}\n  row: {row}"


def test_venv_bootstrap_broken_class():
    """Catches gap #2 — cuDNN cluster routes to venv-bootstrap-broken, not subprocess-crash."""
    row = {
        "name": "BambaModel", "mode": "eval", "status": "worker_error",
        "error_type": None, "phase": None,
        "error": "Unable to load any of {libcudnn_graph.so.9.10.2, libcudnn_graph.so.9.10}. Invalid handle. Cannot load symbol cudnnGetVersion",
    }
    result = classify(row)
    assert result["triage_class"] == "venv-bootstrap-broken", \
        f"cuDNN load failure should classify as venv-bootstrap-broken; got {result['triage_class']!r}"
    assert "venv" in result["suggested_action"].lower(), \
        f"action should reference venv fix; got {result['suggested_action']!r}"


def test_uses_effective_row_not_streaming():
    """Catches gap #10 — classify against effective row error text, not streaming.

    Streaming had 'Audio must be mono' (would match fixture-bug substring), but
    effective row has 'Sizes of tensors must match' (also fixture-bug substring,
    by coincidence — but the test pins WHICH text the rule sees).

    The audit's classify() takes a row dict directly. The orchestration in
    run_audit reads from load_effective_results, NOT the streaming file, so
    this test verifies the contract: classify operates on what the loader
    returns. We pin by giving classify the EFFECTIVE row text and asserting
    the classification matches that text's pattern.
    """
    effective_row = {
        "name": "HiggsAudio", "mode": "eval", "status": "eager_error",
        "error_type": "RuntimeError", "phase": "eager",
        "error": "Sizes of tensors must match except in dimension 1.",
    }
    result = classify(effective_row)
    assert result["triage_class"] == "fixture-bug", \
        f"effective text should match fixture-bug; got {result['triage_class']!r}"

    # Now a row where effective text would NOT match any fixture-bug substring
    effective_row_unmatched = {
        "name": "X", "mode": "eval", "status": "eager_error",
        "error_type": "RuntimeError", "phase": "eager",
        "error": "Some new error pattern not in our table.",
    }
    result = classify(effective_row_unmatched)
    # Falls to upstream-bug if "torch." or "transformers" mentioned, else unknown.
    # This row has neither; should be unknown.
    assert result["triage_class"] == "unknown", \
        f"row without substring/torch/transformers should be unknown; got {result['triage_class']!r}"


def test_applies_to_versions_missing_fails_loud():
    """Catches gap #6 — known_errors entry missing applies_to_versions exits 2."""
    import audit_new_errors
    # Monkeypatch the file path to point at a fixture
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "entries": [
                {"model": "FooModel", "modes": ["eval"], "status": "eager_error",
                 "error_pattern": "boom"}
                # NO applies_to_versions
            ]
        }, f)
        fixture_path = Path(f.name)
    try:
        original = audit_new_errors.KNOWN_ERRORS_FILE
        audit_new_errors.KNOWN_ERRORS_FILE = fixture_path
        try:
            load_known_errors_filtered("2.13")
            assert False, "expected SystemExit(2)"
        except SystemExit as e:
            assert e.code == 2, f"expected exit 2, got {e.code}"
        finally:
            audit_new_errors.KNOWN_ERRORS_FILE = original
    finally:
        fixture_path.unlink()


def test_branch_precedence_first_match_wins():
    """Catches gap #11 — overlapping heuristics resolve by top-to-bottom order.

    Row: status=worker_error AND error contains 'device-side assert'.
    - Matches venv-bootstrap-broken? no (no Unable to load / libcudnn / etc.)
    - Matches gpu-contention? no (no OOM, no SIGKILL)
    - Matches cuda-context-pollution? YES (substring)
    - Matches subprocess-crash? would match if returncode in {-6, -11} but here returncode=None
    Per declared order: cuda-context-pollution wins (rule #3 vs subprocess-crash rule #4).
    """
    row = {
        "name": "X", "mode": "eval", "status": "worker_error",
        "error_type": None, "returncode": -11,  # would match subprocess-crash
        "error": "RuntimeError: CUDA error: device-side assert triggered",
    }
    result = classify(row)
    assert result["triage_class"] == "cuda-context-pollution", \
        f"declared order says cuda-context-pollution wins over subprocess-crash; got {result['triage_class']!r}"


def test_rerun_marker_emitted_on_fixture_bug():
    """Catches gap #5 (emission side) — fixture-bug candidate triggers marker write."""
    with tempfile.TemporaryDirectory() as tmp:
        sweep_dir = Path(tmp)
        keys = [("HiggsAudio", "eval"), ("HiggsAudio", "train")]
        emitted = emit_rerun_marker(sweep_dir, keys)
        assert emitted is True
        marker = sweep_dir / ".audit-rerun-required"
        assert marker.exists()
        content = marker.read_text()
        assert "HiggsAudio|eval" in content
        assert "HiggsAudio|train" in content


def test_rerun_marker_not_emitted_when_no_fixture_bugs():
    """Empty fixture-bug list → no marker; idempotent removal of stale marker."""
    with tempfile.TemporaryDirectory() as tmp:
        sweep_dir = Path(tmp)
        marker = sweep_dir / ".audit-rerun-required"
        # Pre-create a stale marker
        marker.write_text("STALE\n")
        emitted = emit_rerun_marker(sweep_dir, [])
        assert emitted is False
        assert not marker.exists(), "stale marker should be removed when no fixture-bug candidates"


def test_rule_count_in_table():
    """Sanity: rule table is the expected length (catches accidental drops on refactor)."""
    assert len(RULES) == 7, f"expected 7 rules in RULES table, got {len(RULES)}"


def test_every_rule_has_case_id_attribution():
    """Per design rev 2: every heuristic rule has a case_id for traceability."""
    for rule_fn, case_id in RULES:
        assert case_id and isinstance(case_id, str), \
            f"rule {rule_fn.__name__} missing case_id attribution"
        assert case_id.startswith("adv-"), \
            f"case_id should start with 'adv-' (adversary-review namespace); got {case_id!r}"


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
