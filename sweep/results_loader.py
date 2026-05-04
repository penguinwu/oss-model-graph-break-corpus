"""Canonical reader for sweep result data, with amendment merging.

Single source of truth for "what does the sweep say about (model, mode)?"
Every consumer of identify_results.json MUST go through load_effective_results
or load_amendments_metadata. Direct json.load() of identify_results.json is
prohibited (enforced by sweep/test_results_loader.py).

Schema (identify_results.json):
    {
      "metadata": {...},
      "results": [<row>, ...],         # ORIGINAL — immutable after sweep
      "amendments": [                   # OPTIONAL — chronological order
        {
          "amendment_id": "<slug>",
          "applied_at": "<ISO timestamp>",
          "fix_commit": "<sha>",
          "fix_description": "<text>",
          "trigger": "<text>",
          "env_constraints": {"torch": "<ver>", "transformers": "<ver>", ...},
          "rows": [<row>, ...]
        }
      ]
    }

Each row in `amendments[*].rows` has the same shape as a `results[*]` row
(name, mode, status, numeric_status, ...).
"""
import json
from pathlib import Path
from typing import Any


def _resolve(path_or_dir: Path | str) -> Path:
    """Accept either a sweep dir or a direct path to identify_results.json."""
    p = Path(path_or_dir)
    if p.is_file():
        return p  # direct file path
    if p.is_dir():
        return p / "identify_results.json"
    # Fall back: maybe a non-existent path; return as-is to surface clear error
    if p.suffix == ".json":
        return p
    return p / "identify_results.json"


def load_raw(path_or_dir: Path | str) -> dict:
    """Read identify_results.json verbatim. Prefer load_effective_results."""
    return json.loads(_resolve(path_or_dir).read_text())


def load_effective_results(path_or_dir: Path | str) -> dict[tuple[str, str], dict]:
    """Return effective (name, mode) → result, applying amendments in order.

    Each row gets a `result_source` field:
      - "original"               — from results[]
      - "amended:<amendment_id>" — from amendments[].rows (last write wins)

    Use this in EVERY consumer that reads sweep results. Direct json.load
    of identify_results.json is the wrong access path.
    """
    data = load_raw(path_or_dir)
    effective: dict[tuple[str, str], dict] = {}
    for r in data.get("results", []):
        key = (r["name"], r["mode"])
        effective[key] = {**r, "result_source": "original"}
    for amendment in data.get("amendments", []):
        aid = amendment["amendment_id"]
        for row in amendment.get("rows", []):
            key = (row["name"], row["mode"])
            effective[key] = {**row, "result_source": f"amended:{aid}"}
    return effective


def load_amendments_metadata(path_or_dir: Path | str) -> list[dict]:
    """Return amendment metadata (no row data) in chronological order.

    Use for rendering the "Amendments applied" section in reports.
    """
    data = load_raw(path_or_dir)
    out = []
    for a in data.get("amendments", []):
        out.append({
            k: v for k, v in a.items() if k != "rows"
        } | {"row_count": len(a.get("rows", []))})
    return out


def load_results_list(path_or_dir: Path | str) -> list[dict]:
    """Return effective results as a flat list (with result_source field).

    Convenience for consumers that prefer list shape over dict-keyed-by-key.
    """
    return list(load_effective_results(path_or_dir).values())


def _resolve_explain(path_or_dir: Path | str) -> Path:
    p = Path(path_or_dir)
    if p.is_file():
        return p
    if p.is_dir():
        return p / "explain_checkpoint.jsonl"
    if p.name.endswith(".jsonl"):
        return p
    return p / "explain_checkpoint.jsonl"


def load_effective_explain(path_or_dir: Path | str) -> dict[tuple[str, str], dict]:
    """Return effective (name, mode) → explain entry, last-write-wins.

    Reads explain_checkpoint.jsonl and returns one entry per (name, mode).
    When amend_sweep.py amends an explain pass, it APPENDS new lines for the
    affected (name, mode) keys; this loader returns the LAST entry per key,
    so amended entries supersede originals automatically.

    Each returned entry has a `result_source` field:
      - "original"               — first entry seen for this key
      - "amended:<amendment_id>" — later entry overriding it (carries the
                                   amendment_id field set by amend_sweep)

    Use this in EVERY consumer that reads explain data alongside identify.
    Direct line-by-line read of explain_checkpoint.jsonl is the wrong access
    path — it would silently use stale pre-amendment break_reasons for
    models whose harness was fixed after the original sweep.
    """
    p = _resolve_explain(path_or_dir)
    if not p.exists():
        return {}
    effective: dict[tuple[str, str], dict] = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "name" not in row or "mode" not in row:
                continue
            key = (row["name"], row["mode"])
            # last-write-wins; amended entries carry amendment_id which we
            # surface as result_source for symmetry with load_effective_results
            aid = row.get("amendment_id")
            tagged = {**row, "result_source": f"amended:{aid}" if aid else "original"}
            effective[key] = tagged
    return effective
