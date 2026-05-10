# Per-Model Timeout Propagation — Design Proposal

**Status:** Proposed 2026-05-09; pending implementation tomorrow morning.

**Trigger:** 2026-05-03 AND 2026-05-09 nightly sweeps both died at item 127 (BltForCausalLM/eval). Both BLT models are in `large_models.json` with `timeout_tier: "very_large"` (needs 1620s = 9× base timeout) but the regular nightly sweep uses `--timeout 180` and never reads `large_models.json`.

## Root cause: bifurcated entry points, partial wiring

The corpus has TWO sweep entry points:

| Entry point | Reads `large_models.json`? | Per-tier timeout? |
|---|---|---|
| `sweep/run_sweep.py` (DEPRECATED) | ✅ yes (line 836) | ✅ yes — `large=3×`, `very_large=9×` |
| `tools/run_experiment.py nightly` (CURRENT) | ❌ no | ❌ no — hardcoded `--timeout 180` everywhere |

The deprecation moved the entry point but didn't move the tier-aware timeout logic. `run_experiment.py nightly` calls into the orchestrator with `timeout=180` directly (no per-model override), so very_large-tier models hit the 180s wall before they finish compiling.

## Design — three changes (small, focused)

### Change 1: Lift `_timeout_for(name)` into a shared helper

Move the existing logic from `sweep/run_sweep.py:836-855` to `sweep/timeout_tiers.py` (new module):

```python
# sweep/timeout_tiers.py
from pathlib import Path
import json

LARGE_MODELS_FILE = Path(__file__).parent / "large_models.json"

def load_per_model_timeouts(base_timeout_s: int) -> dict[str, int]:
    """Read large_models.json and compute per-model timeout overrides.

    Returns {model_name: timeout_s} for models with a tier override.
    Tier multipliers: large=3×, very_large=9× (matches the existing logic
    in the deprecated run_sweep.py).
    """
    if not LARGE_MODELS_FILE.exists():
        return {}
    with open(LARGE_MODELS_FILE) as f:
        registry = json.load(f)
    overrides = {}
    for name, entry in registry.items():
        tier = entry.get("timeout_tier", "large") if isinstance(entry, dict) else "large"
        if tier == "very_large":
            overrides[name] = base_timeout_s * 9
        elif tier == "large":
            overrides[name] = base_timeout_s * 3
    return overrides
```

Both entry points import + use this helper.

### Change 2: Plumb per-model timeout through the orchestrator

`sweep/orchestrator.py` already takes `timeout_s` per worker (see `spawn_worker` line 120). Currently `tools/run_experiment.py` passes a single value. Update to:

```python
# tools/run_experiment.py — sweep launch path
from sweep.timeout_tiers import load_per_model_timeouts

per_model_timeouts = load_per_model_timeouts(args.timeout)
# When spawning each worker:
timeout_s = per_model_timeouts.get(spec["name"], args.timeout)
spawn_worker(..., timeout_s=timeout_s, ...)
```

### Change 3: Validation gate at sweep launch

Before workers spawn, the orchestrator should print + log:

```
Per-model timeout overrides:
  BltForCausalLM:    1620s  (very_large; default 180s)
  BltModel:          1620s  (very_large; default 180s)
  Gemma3nModel:      1620s  (very_large; default 180s)  [also in skip_models.json — overrides apply only if NOT skipped]
  ... 12 large-tier models at 540s
```

This makes the wiring observable. If a regression removes the per-tier path, the launch log shows `Per-model timeout overrides: (none)` instead of the list — easy to catch.

## Prevention: launch-command lint

To prevent the "single `--timeout 180` across all models" regression class going forward, add a launch-side check in `tools/run_experiment.py nightly`:

```python
# Before launching the sweep workers
overrides = load_per_model_timeouts(args.timeout)
if not overrides:
    print("ERROR: large_models.json is non-empty but no per-model timeout overrides "
          "were derived. Either large_models.json is missing OR the per-tier "
          "wiring is broken. Refusing to launch — would deterministically kill "
          "very_large-tier models.", file=sys.stderr)
    sys.exit(2)
```

This makes the launch FAIL LOUDLY if the per-tier wiring is missing — instead of silently running with a single timeout that kills very_large models.

## Test plan

1. Add a test `tools/test_timeout_propagation.py`:
   - Invoke `tools/run_experiment.py sweep --timeout 180 --dry-run` (need to add `--dry-run`)
   - Capture the per-worker timeout map
   - Assert `BltForCausalLM` gets 1620s, `Gemma3nModel` gets 1620s (when not skipped), `BloomModel` gets 180s
2. Add a test `sweep/test_timeout_tiers.py`:
   - Mock `large_models.json` with one of each tier
   - Assert `load_per_model_timeouts(180)` returns the correct dict

Both tests prevent regression of the wiring at the unit + end-to-end level.

## Migration

After Changes 1+2+3 land:
- Remove `BltForCausalLM`, `BltModel` from `skip_models.json` (they should now run cleanly with 1620s timeout).
- Re-run weekly sweep, verify they complete.
- Remove the duplicate timeout logic from `sweep/run_sweep.py` (the deprecated entry point), since it now imports from `sweep/timeout_tiers.py`.

## Estimated scope

- New file: `sweep/timeout_tiers.py` (~30 lines)
- Edit: `tools/run_experiment.py` (~10 lines added, +load_per_model_timeouts call + spawn_worker plumbing)
- Edit: `sweep/run_sweep.py` (~20 lines removed, replaced by import)
- Tests: `tools/test_timeout_propagation.py`, `sweep/test_timeout_tiers.py` (~50 lines each)

Total: ~150 lines net add, ~3 hours implementation + test.
