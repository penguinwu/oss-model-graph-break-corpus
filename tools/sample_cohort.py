#!/usr/bin/env python3
"""Sample N models from a cohort for the pre-launch sample-sweep gate.

Surfaced by adversary-review case_id 2026-05-07-124100-cohort-regen-fix gap 4:
the v3 sanity-check skill says "20 random models from the planned full cohort"
but no scripted sampler existed — agents picked manually, reproducing the
class of bug that produced the 2026-05-06 broken-cohort failure (humans pick
"first 20" or other biased samples that miss contamination).

Determinism contract:
- Default seed = sha256(cohort_path + cohort_mtime). Same cohort → same sample.
  Different cohort (or modified cohort) → different sample.
- --seed <int> overrides with an explicit seed (for repro across cohort
  regenerations).

Usage:
    python3 tools/sample_cohort.py <cohort.json> --n 20 --output /tmp/sample.json
    python3 tools/sample_cohort.py <cohort.json> --n 20 --seed 42 --output /tmp/sample.json

The output is a flat list of {name, source, ...} specs. Pass it to
`run_experiment.py sweep --models <sample.json> --allow-bare-cohort`
(the bare-list override is intentional here — it's a sample, not a canonical
cohort, and the parent cohort's metadata is preserved separately).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def _seed_from_cohort(cohort_path: Path) -> int:
    """Deterministic seed: sha256(absolute_path || mtime_ns)."""
    h = hashlib.sha256()
    h.update(str(cohort_path.resolve()).encode())
    h.update(str(cohort_path.stat().st_mtime_ns).encode())
    # Take first 8 bytes as a 64-bit int
    return int.from_bytes(h.digest()[:8], "big")


def sample_cohort(cohort_path: Path, n: int, seed: int | None = None) -> list:
    """Sample n models from a cohort file. Returns a list of model specs.

    If the cohort has fewer than n models, returns all of them.
    """
    raw = json.loads(cohort_path.read_text())
    if isinstance(raw, dict) and "models" in raw:
        all_models = raw["models"]
    elif isinstance(raw, list):
        all_models = raw
    else:
        sys.exit(f"ERROR: unrecognized cohort shape in {cohort_path}")

    if seed is None:
        seed = _seed_from_cohort(cohort_path)

    rng = random.Random(seed)
    if n >= len(all_models):
        return list(all_models)
    return rng.sample(all_models, n)


def main() -> int:
    if sys.version_info < (3, 9):
        sys.exit(f"ERROR: tools/sample_cohort.py requires Python >= 3.9.")

    p = argparse.ArgumentParser(
        description="Sample N models from a cohort for the pre-launch sample-sweep gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("cohort", type=Path, help="Path to cohort file (canonical or bare-list)")
    p.add_argument("--n", type=int, default=20, help="Sample size (default: 20)")
    p.add_argument("--seed", type=int, default=None,
                   help="Override the default seed (sha256 of cohort path + mtime). "
                        "Use this only for cross-cohort repro experiments.")
    p.add_argument("--output", "-o", type=Path, default=None,
                   help="Write sample to this path (JSON list). If omitted, prints to stdout.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite output file if it exists")
    args = p.parse_args()

    if not args.cohort.is_file():
        sys.exit(f"ERROR: cohort file not found: {args.cohort}")
    if args.output and args.output.exists() and not args.force:
        sys.exit(f"ERROR: {args.output} exists. Pass --force to overwrite.")
    if args.n <= 0:
        sys.exit(f"ERROR: --n must be positive (got {args.n})")

    sample = sample_cohort(args.cohort, args.n, seed=args.seed)
    payload = json.dumps(sample, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
        used_seed = args.seed if args.seed is not None else _seed_from_cohort(args.cohort)
        print(f"Wrote {args.output} ({len(sample)} models, seed={used_seed})")
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    sys.exit(main())
