"""Canonical reader for sweep result data, with amendment merging.

Single source of truth for "what does the sweep say about (model, mode)?"
Every consumer of identify_results.json MUST go through load_effective_results
or load_amendments_metadata. Direct json.load() of identify_results.json is
prohibited (enforced by sweep/test_results_loader.py).

Two on-disk formats are supported (auto-detected on read):

Format A — JSON object (legacy, pre-migration):
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

Format B — JSONL (current, written by run_sweep.py after migration):
    {"_record_type": "metadata", ...metadata fields...}
    {"_record_type": "row", ...row fields...}
    {"_record_type": "row", ...row fields...}
    {"_record_type": "amendment", ...amendment fields...}

    Properties:
    - True append-only: amendments are a single appended line (no rewrite)
    - Crash-tolerant: last incomplete line is dropped on read
    - Each line is valid JSON; whole file is valid JSONL

load_raw() returns the same canonical dict shape for both formats:
    {"metadata": {...}, "results": [...], "amendments": [...]}

Each row in `amendments[*].rows` has the same shape as a `results[*]` row
(name, mode, status, numeric_status, ...).
"""
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Format detection and JSONL parser
# ---------------------------------------------------------------------------

def _is_jsonl_file(path: Path) -> bool:
    """Return True if the file is in JSONL format (first line has _record_type)."""
    try:
        with open(path) as f:
            first_line = f.readline().strip()
        if not first_line:
            return False
        obj = json.loads(first_line)
        return isinstance(obj, dict) and "_record_type" in obj
    except (json.JSONDecodeError, OSError):
        return False


def _load_jsonl(path: Path) -> dict:
    """Parse JSONL identify_results file into canonical dict shape.

    Tolerates a last incomplete line (crash-safe: last incomplete line is dropped).
    The _record_type field is stripped from all returned records.
    """
    metadata: dict = {}
    results: list[dict] = []
    amendments: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate last incomplete line on mid-write crash
            rt = record.get("_record_type")
            row = {k: v for k, v in record.items() if k != "_record_type"}
            if rt == "metadata":
                metadata = row
            elif rt == "row":
                results.append(row)
            elif rt == "amendment":
                amendments.append(row)
    out: dict = {"metadata": metadata, "results": results}
    if amendments:
        out["amendments"] = amendments
    return out


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

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
    """Read identify_results file. Supports both JSON object and JSONL formats.

    Returns canonical dict shape: {metadata, results, amendments?}
    regardless of the file's on-disk format.
    Prefer load_effective_results() for analysis; use load_raw() when you need
    the metadata or amendment structs directly.
    """
    p = _resolve(path_or_dir)
    if _is_jsonl_file(p):
        return _load_jsonl(p)
    return json.loads(p.read_text())


def is_jsonl_format(path_or_dir: Path | str) -> bool:
    """Return True if the identify_results file is in JSONL format.

    Used by amend_sweep.py to choose between append (JSONL) and
    atomic-rewrite (JSON legacy) write strategies.
    """
    return _is_jsonl_file(_resolve(path_or_dir))


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
