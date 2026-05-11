"""Single source of truth for reading sweep/skip_models.json.

Supports BOTH legacy + current schema during the migration window:

- Legacy (flat string array):
    ["ModelA", "ModelB", ...]
- Current (dict-of-objects with per-entry metadata):
    {
      "ModelA": {"reason": "...", "follow_up_task": "...", "added": "YYYY-MM-DD"},
      "ModelB": {...},
      ...
    }

All callers should use this loader instead of reading the file directly.
That way, the schema upgrade only requires updating ONE place in the future.

Per audit_new_models adversary case `adv-2026-05-10-150000` gap #7
(deferred to dedicated tool); shipped 2026-05-10 21:00 ET.
"""
from __future__ import annotations

import json
from pathlib import Path

SKIP_MODELS_FILE = Path(__file__).parent / "skip_models.json"


def load_skip_models(path: Path | None = None) -> set[str]:
    """Return the set of model names that should be skipped from sweeps.

    Accepts either the legacy flat-string-array format OR the new
    dict-of-objects format. Returns just the names — discards metadata
    (reason / follow_up_task / added) at this layer; tooling that needs
    them can re-read the raw file via load_skip_models_raw().
    """
    p = path or SKIP_MODELS_FILE
    if not p.exists():
        return set()
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return set(data)
    if isinstance(data, dict):
        return set(data.keys())
    return set()


def load_skip_models_raw(path: Path | None = None) -> dict | list:
    """Return the raw parsed contents (dict or list) for tooling that needs metadata."""
    p = path or SKIP_MODELS_FILE
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)
