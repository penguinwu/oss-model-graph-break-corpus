#!/usr/bin/env python3
"""Tests for tools/sample_cohort.py.

Surfaced by adversary-review case_id 2026-05-07-124100-cohort-regen-fix:
- review test 6: test_sampler_hits_poisoned_subset
- review test 7: test_sampler_deterministic_for_same_cohort

Run: python3 tools/test_sample_cohort.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.sample_cohort import sample_cohort, _seed_from_cohort  # noqa: E402


def _make_cohort(path: Path, n: int, *, poison_prefix: str = "POISON_", poison_count: int = 0):
    """Write a synthetic cohort with `n` models; first `poison_count` are poisoned."""
    models = []
    for i in range(poison_count):
        models.append({"name": f"{poison_prefix}{i:04d}", "source": "hf"})
    for i in range(n - poison_count):
        models.append({"name": f"Model{i:04d}", "source": "hf"})
    payload = {
        "_metadata": {
            "derived_from": "synthetic",
            "filter": "all",
            "model_count": n,
            "source_versions": {"torch": "2.13.0", "transformers": "5.6.2", "diffusers": "0.38.0"},
        },
        "models": models,
    }
    path.write_text(json.dumps(payload))


# Tests ─────────────────────────────────────────────────────────────────────

def test_sampler_returns_correct_size():
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "cohort.json"
        _make_cohort(cohort, 200)
        sample = sample_cohort(cohort, 20)
        assert len(sample) == 20


def test_sampler_returns_all_when_n_exceeds_cohort():
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "cohort.json"
        _make_cohort(cohort, 5)
        sample = sample_cohort(cohort, 20)
        assert len(sample) == 5


def test_sampler_deterministic_for_same_cohort():
    """Review test 7: same cohort + same mtime → identical sample."""
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "cohort.json"
        _make_cohort(cohort, 200)
        s1 = sample_cohort(cohort, 20)
        s2 = sample_cohort(cohort, 20)
        names1 = [m["name"] for m in s1]
        names2 = [m["name"] for m in s2]
        assert names1 == names2, f"sampler not deterministic: {names1[:5]} vs {names2[:5]}"


def test_sampler_seed_changes_with_cohort_mtime():
    """Same cohort path, different mtime → different sample.

    This guards against "always-the-same-seed" regression: if we accidentally
    computed seed only from cohort path, two regenerations of the same cohort
    would always pick the same sample, masking new contamination.
    """
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "cohort.json"
        _make_cohort(cohort, 200)
        s1 = sample_cohort(cohort, 20)
        # Touch to a different mtime
        new_mtime = cohort.stat().st_mtime + 100
        os.utime(cohort, (new_mtime, new_mtime))
        s2 = sample_cohort(cohort, 20)
        names1 = sorted(m["name"] for m in s1)
        names2 = sorted(m["name"] for m in s2)
        assert names1 != names2, f"different mtime should give different sample (got identical)"


def test_sampler_hits_poisoned_subset_with_reasonable_probability():
    """Review test 6: synthetic cohort of 200, 10 poisoned. Vary the seed and check hit rate.

    Theoretical hit rate (any of n=20 draws hits one of 10/200 poisoned):
        P(at least one) = 1 - C(190, 20) / C(200, 20) ≈ 0.668

    With 100 trials we accept hit_count >= 50 (gives wide margin for sampler RNG variance).
    Also asserts at least one trial draws zero poisoned (proving non-trivial randomness).
    """
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "cohort.json"
        _make_cohort(cohort, 200, poison_count=10)

        hit_count = 0
        zero_count = 0
        trials = 100
        for trial_seed in range(trials):
            sample = sample_cohort(cohort, 20, seed=trial_seed)
            poisoned_in_sample = sum(1 for m in sample if m["name"].startswith("POISON_"))
            if poisoned_in_sample > 0:
                hit_count += 1
            if poisoned_in_sample == 0:
                zero_count += 1

        # Lower bound: 50/100 (theoretical ~67%, allow margin for finite-trial variance)
        assert hit_count >= 50, \
            f"sampler hit poisoned subset only {hit_count}/100 times (expected ~67)"
        # Upper bound: not 100/100 (would suggest the sampler is biased toward including
        # the prefix or otherwise tautological). Theoretical ~33% miss rate.
        assert zero_count >= 5, \
            f"sampler always hit poisoned ({100 - zero_count}/100); suspicious — check for bias"


def test_explicit_seed_overrides_default():
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "cohort.json"
        _make_cohort(cohort, 200)
        s1 = sample_cohort(cohort, 20, seed=42)
        s2 = sample_cohort(cohort, 20, seed=42)
        s3 = sample_cohort(cohort, 20, seed=43)
        names1 = [m["name"] for m in s1]
        names2 = [m["name"] for m in s2]
        names3 = [m["name"] for m in s3]
        assert names1 == names2, "same explicit seed should give same sample"
        assert names1 != names3, "different seeds should give different samples"


def test_seed_from_cohort_changes_with_path():
    """Different cohort paths produce different seeds even with same mtime."""
    with tempfile.TemporaryDirectory() as d:
        c1 = Path(d) / "c1.json"
        c2 = Path(d) / "c2.json"
        _make_cohort(c1, 10)
        _make_cohort(c2, 10)
        # Force same mtime
        mt = c1.stat().st_mtime
        os.utime(c2, (mt, mt))
        seed1 = _seed_from_cohort(c1)
        seed2 = _seed_from_cohort(c2)
        assert seed1 != seed2, "different paths should hash to different seeds"


def test_sampler_handles_bare_list_cohort():
    """Sampler accepts both dict-with-_metadata and bare-list cohorts (since it samples,
    not validates — validation is the loader's job)."""
    with tempfile.TemporaryDirectory() as d:
        cohort = Path(d) / "bare.json"
        cohort.write_text(json.dumps([{"name": f"M{i}", "source": "hf"} for i in range(50)]))
        sample = sample_cohort(cohort, 10)
        assert len(sample) == 10


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
