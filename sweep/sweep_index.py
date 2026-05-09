"""sweep_index — write/append to experiments/results/INDEX.json after each sweep.

Per Peng directive 2026-05-09 07:55 ET (sweep ↔ filing relationship) +
adversary case file adv-2026-05-09-120800 (gaps 3, 4, 9, 10, 11).

Used by:
- sweep/run_sweep.py:run_sweep (post-completion hook)
- sweep/run_sweep.py:run_explain (post-completion hook)

Read by:
- tools/lookup_sweep_evidence.py (verify_repro's cache-first ORIGINAL path)

INDEX.json shape (one entry per sweep pass):
{
  "schema_version": 1,
  "sweeps": [
    {
      "sweep_id": "<unique id>",
      "results_jsonl": "<absolute path>",
      "venv_name": "current" | "nightly",
      "venv_path": "<absolute path to python interpreter>",
      "torch_version": "2.13.0.dev20260502",
      "torch_git_version": "<git sha>",
      "started_utc": "<ISO 8601 Z>",
      "completed_utc": "<ISO 8601 Z>",
      "cohort": "<path to cohort source>",
      "cohort_sha256": "<sha of cohort file content>",   (gap 3)
      "args_fingerprint": "<sha of canonicalized args>", (gap 3)
      "sweep_kind": "identify" | "explain" | "correctness" | "ngb-verify" | ..., (gap 9)
      "produced_fields": ["status", "numeric_max_diff", ...]                     (gap 9)
    }
  ]
}

Write semantics:
- Idempotent by sweep_id (re-running with same id replaces the row, not duplicates)
- Atomic via tempfile + rename (concurrent safety)
- All paths absolute (gap 11)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = REPO_ROOT / "experiments" / "results" / "INDEX.json"
SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_sha256(path: Path) -> str:
    """sha256 of file content; empty string on read error."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return ""


def _canonicalize_args(args_dict: dict) -> str:
    """Stable, deterministic JSON-serialize args for fingerprint computation."""
    return json.dumps(args_dict, sort_keys=True, separators=(",", ":"))


def _args_fingerprint(args_dict: dict) -> str:
    canonical = _canonicalize_args(args_dict)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scan_produced_fields(results_jsonl: Path, max_rows: int = 50) -> list[str]:
    """Sample up to max_rows rows from results.jsonl; return union of field names.

    Skip _record_type=metadata rows. Return sorted list for determinism.
    """
    fields: set[str] = set()
    if not results_jsonl.is_file():
        return []
    with open(results_jsonl) as f:
        for i, line in enumerate(f):
            if i >= max_rows + 50:  # extra slack for metadata + amendments
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_record_type") == "metadata":
                continue
            for k in row.keys():
                if k != "_record_type":
                    fields.add(k)
    return sorted(fields)


def _venv_torch_info(python_bin: str) -> tuple[str, str]:
    """Return (torch.__version__, torch.version.git_version) by invoking python_bin.

    Returns ("", "") on probe failure (not fatal — INDEX still written).
    """
    import subprocess
    try:
        r = subprocess.run(
            [python_bin, "-c",
             "import torch; print(torch.__version__); print(torch.version.git_version)"],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode != 0:
            return "", ""
        lines = r.stdout.strip().splitlines()
        return lines[0], (lines[1] if len(lines) > 1 else "")
    except Exception:
        return "", ""


def _venv_name_from_path(python_bin: str) -> str:
    """Heuristic: 'nightly' if the venv path contains 'nightly', else 'current'.

    Matches the convention in tools/file_issues.py:_venv_name_from_python.
    """
    p = python_bin.lower()
    if "nightly" in p:
        return "nightly"
    return "current"


def _abs(path: str | Path) -> str:
    """Ensure absolute path (gap 11)."""
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    return str(p)


def append_to_index(
    *,
    sweep_id: str,
    results_jsonl: str | Path,
    sweep_kind: str,
    python_bin: str,
    started_utc: str | None = None,
    completed_utc: str | None = None,
    cohort: str | None = None,
    args_dict: dict | None = None,
    index_path: Path | None = None,
) -> Path:
    """Append (or replace) a sweep entry in INDEX.json.

    Idempotent by sweep_id: if a row with the same sweep_id exists, it's
    replaced (not duplicated). Returns the index_path.

    Safe to call from inside run_sweep / run_explain at completion.
    Errors are logged but never raised — sweep completion takes priority
    over index bookkeeping.
    """
    idx_path = index_path or DEFAULT_INDEX_PATH

    try:
        results_abs = _abs(results_jsonl)
        cohort_abs = _abs(cohort) if cohort else ""
        cohort_sha = _file_sha256(Path(cohort_abs)) if cohort_abs else ""
        args_fp = _args_fingerprint(args_dict or {})
        torch_v, torch_git = _venv_torch_info(python_bin)
        produced = _scan_produced_fields(Path(results_abs))
        venv_name = _venv_name_from_path(python_bin)

        new_row = {
            "sweep_id": sweep_id,
            "results_jsonl": results_abs,
            "venv_name": venv_name,
            "venv_path": _abs(python_bin),
            "torch_version": torch_v,
            "torch_git_version": torch_git,
            "started_utc": started_utc or _now_iso(),
            "completed_utc": completed_utc or _now_iso(),
            "cohort": cohort_abs,
            "cohort_sha256": cohort_sha,
            "args_fingerprint": args_fp,
            "sweep_kind": sweep_kind,
            "produced_fields": produced,
        }

        # Read-modify-write atomically (tempfile + rename)
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        if idx_path.exists():
            try:
                idx = json.loads(idx_path.read_bytes())
            except json.JSONDecodeError:
                idx = {"schema_version": SCHEMA_VERSION, "sweeps": []}
        else:
            idx = {"schema_version": SCHEMA_VERSION, "sweeps": []}

        sweeps = idx.get("sweeps", [])
        # Idempotence: replace any existing row with same sweep_id
        sweeps = [s for s in sweeps if s.get("sweep_id") != sweep_id]
        sweeps.append(new_row)
        idx["sweeps"] = sweeps
        idx["schema_version"] = SCHEMA_VERSION

        # Atomic write
        with tempfile.NamedTemporaryFile(
            "w", dir=idx_path.parent, prefix=".INDEX-", suffix=".tmp", delete=False
        ) as tf:
            json.dump(idx, tf, indent=2)
            tf.write("\n")
            tmp_name = tf.name
        os.replace(tmp_name, idx_path)
        return idx_path

    except Exception as e:
        # Sweep completion takes priority over index bookkeeping.
        # Print a warning; do not raise.
        print(f"[sweep_index] WARNING: failed to append to INDEX.json: {e}")
        return idx_path
