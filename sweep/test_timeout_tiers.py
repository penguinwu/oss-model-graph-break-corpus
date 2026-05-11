#!/usr/bin/env python3
"""Tests for sweep/timeout_tiers.py.

Pins the tier multipliers (large=3×, very_large=9×) per
sweep/TIMEOUT_PROPAGATION_DESIGN.md.

Run: PYTHONPATH=$(pwd)/sweep ~/envs/torch211/bin/python sweep/test_timeout_tiers.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "sweep"))

import timeout_tiers as tt  # noqa: E402


def test_multipliers_pinned():
    """Tier multipliers are load-bearing — pinned by literal-value asserts.

    Changing these requires updating the design doc + invoking adversary-review.
    """
    assert tt.TIER_MULTIPLIERS["large"] == 3, \
        f"large multiplier changed to {tt.TIER_MULTIPLIERS['large']} — update design + invoke adversary-review"
    assert tt.TIER_MULTIPLIERS["very_large"] == 9, \
        f"very_large multiplier changed to {tt.TIER_MULTIPLIERS['very_large']} — update design + invoke adversary-review"


def test_load_per_model_timeouts_with_synthetic_registry():
    """Synthetic large_models.json fixture: one of each tier."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({
            "BigModel": {"timeout_tier": "very_large", "source": "hf"},
            "MediumModel": {"timeout_tier": "large", "source": "hf"},
            "ImplicitLarge": {"source": "hf"},  # no tier field → defaults to large
        }, f)
        fixture = Path(f.name)
    original = tt.LARGE_MODELS_FILE
    tt.LARGE_MODELS_FILE = fixture
    try:
        result = tt.load_per_model_timeouts(180)
        assert result == {
            "BigModel": 180 * 9,        # 1620
            "MediumModel": 180 * 3,     # 540
            "ImplicitLarge": 180 * 3,   # 540 (default tier)
        }, f"got {result}"
    finally:
        tt.LARGE_MODELS_FILE = original
        fixture.unlink()


def test_load_per_model_timeouts_missing_file_returns_empty():
    original = tt.LARGE_MODELS_FILE
    tt.LARGE_MODELS_FILE = Path("/tmp/definitely-does-not-exist.json")
    try:
        assert tt.load_per_model_timeouts(180) == {}
    finally:
        tt.LARGE_MODELS_FILE = original


def test_load_per_model_timeouts_real_file_has_blt_at_very_large():
    """The real sweep/large_models.json: BLT models are very_large = 1620s."""
    result = tt.load_per_model_timeouts(180)
    assert "BltForCausalLM" in result, "BltForCausalLM should be in the registry"
    assert result["BltForCausalLM"] == 1620, \
        f"BltForCausalLM expected 1620s (very_large × 180), got {result['BltForCausalLM']}"
    assert "BltModel" in result
    assert result["BltModel"] == 1620


def test_summarize_overrides_empty():
    s = tt.summarize_overrides({}, 180)
    assert "(none)" in s
    assert "180s" in s


def test_summarize_overrides_with_counts():
    overrides = {
        "A": 540,  # large
        "B": 540,  # large
        "C": 1620, # very_large
    }
    s = tt.summarize_overrides(overrides, 180)
    assert "2 'large' tier" in s
    assert "1 'very_large' tier" in s
    assert "540s" in s
    assert "1620s" in s


def test_unknown_tier_defaults_to_no_override():
    """Unknown tier value should NOT silently get default treatment."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"WeirdModel": {"timeout_tier": "ultra_mega", "source": "hf"}}, f)
        fixture = Path(f.name)
    original = tt.LARGE_MODELS_FILE
    tt.LARGE_MODELS_FILE = fixture
    try:
        result = tt.load_per_model_timeouts(180)
        # Unknown tier → multiplier 1 → no override emitted
        assert "WeirdModel" not in result, \
            f"unknown tier should NOT silently get a multiplier; got {result}"
    finally:
        tt.LARGE_MODELS_FILE = original
        fixture.unlink()


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
