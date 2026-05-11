"""Per-model timeout tiering — single source of truth.

Reads `sweep/large_models.json` and computes per-model timeout overrides
based on the entry's `timeout_tier` field.

Tier multipliers (matches the existing logic at
`sweep/run_sweep.py::_timeout_for`):
  - large       → base_timeout × 3
  - very_large  → base_timeout × 9

Both sweep entry points (deprecated `sweep/run_sweep.py` + current
`tools/run_experiment.py nightly`) import + use this helper. Without it,
the current entry point uses a single `--timeout 180` for all models, and
very_large-tier models (BLT, Gemma3n, etc.) deterministically time out at
180s before they finish compiling.

Per `sweep/TIMEOUT_PROPAGATION_DESIGN.md` (approved 2026-05-10 20:12 ET).
"""
from __future__ import annotations

import json
from pathlib import Path

LARGE_MODELS_FILE = Path(__file__).parent / "large_models.json"

TIER_MULTIPLIERS: dict[str, int] = {
    "large": 3,
    "very_large": 9,
}


def load_per_model_timeouts(base_timeout_s: int) -> dict[str, int]:
    """Return {model_name: timeout_s} for models with a tier override.

    Reads sweep/large_models.json. Each entry has a `timeout_tier` field
    (defaults to 'large' if dict but field absent). Models not in
    large_models.json get no override (caller uses base_timeout_s).
    """
    if not LARGE_MODELS_FILE.exists():
        return {}
    with open(LARGE_MODELS_FILE) as f:
        registry = json.load(f)
    overrides: dict[str, int] = {}
    for name, entry in registry.items():
        if isinstance(entry, dict):
            tier = entry.get("timeout_tier", "large")
        else:
            tier = "large"
        multiplier = TIER_MULTIPLIERS.get(tier, 1)
        if multiplier > 1:
            overrides[name] = base_timeout_s * multiplier
    return overrides


def summarize_overrides(overrides: dict[str, int], base_timeout_s: int) -> str:
    """Return a printable summary line for launch logs.

    Per design § "Validation gate at sweep launch" — makes the wiring
    observable. If a regression removes the per-tier path, the launch log
    shows '(none)' instead of the tier counts — easy to catch.
    """
    if not overrides:
        return f"Per-model timeout overrides: (none); all models use base {base_timeout_s}s"
    n_large = sum(1 for v in overrides.values() if v == base_timeout_s * 3)
    n_very_large = sum(1 for v in overrides.values() if v == base_timeout_s * 9)
    return (f"Per-model timeout overrides: {n_large} 'large' tier "
            f"({base_timeout_s * 3}s) + {n_very_large} 'very_large' tier "
            f"({base_timeout_s * 9}s)")
