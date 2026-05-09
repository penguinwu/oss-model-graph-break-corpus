#!/usr/bin/env python3
"""lookup_sweep_evidence - find fresh cached sweep evidence for a (model, mode).

Library + CLI. Reads experiments/results/INDEX.json (commit 3 writes it),
filters to fresh+matching sweeps, returns the per-(model, mode) row's
symptom data in a verify_repro-compatible shape.

Per Peng directives 2026-05-09 07:55 / 08:04 ET (sweep ↔ filing relationship,
10-day staleness) + adversary case file adv-2026-05-09-120800 (gaps 3, 9, 10
for cohort/args fingerprint, sweep_kind+produced_fields filter, case_id stamping).

Behavior:
1. Load INDEX.json. Filter to rows where venv_name matches AND
   completed_utc is within `within_days` days.
2. (gap 9) Filter to rows whose `produced_fields` includes the field the
   expected_signal needs. A graph-break-only sweep cannot satisfy a request
   whose signal demands `numeric_max_diff`.
3. Sort newest-first.
4. For each, check if results.jsonl contains the (model, mode) row.
5. First hit → emit a verify_repro-compatible JSON snippet with
   evidence_source="sweep_results", sweep_path, sweep_age_days, plus the
   matched row's per-symptom fields, plus cohort_sha256 + args_fingerprint
   (gap 3).
6. (gap 10) Stamp the requested case_id into the emitted JSON.
7. No hit → return None (CLI exits non-zero with explanation).

Requires Python 3.9+.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.version_info < (3, 9):
    sys.exit("ERROR: lookup_sweep_evidence.py requires Python 3.9+")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INDEX_PATH = REPO_ROOT / "experiments" / "results" / "INDEX.json"
STALENESS_DAYS_DEFAULT = 10  # Per Peng 2026-05-09 08:04 ET


# Map an expected_signal to the field name(s) we need in a sweep row.
# Conservative — extends as new signal patterns appear.
_SIGNAL_TO_FIELDS = {
    # numeric divergences
    "numeric_max_diff": ["numeric_max_diff"],
    "numeric_status": ["numeric_status"],
    # graph break / status
    "graph_break": ["status", "break_reasons"],
    "graph_break_count": ["graph_break_count", "status"],
    # eager errors
    "eager_error": ["status", "eager_error"],
    "RuntimeError": ["status", "error"],
}


def _signal_required_fields(expected_signal: dict) -> list[str]:
    """Best-effort: which sweep-row field(s) the expected_signal probably needs.

    The signal's `fragment` is matched against stdout/stderr, but the SOURCE
    of that text is the sweep row's symptom field. If the row doesn't expose
    the field, the signal can't fire.

    Returns a list of field names; lookup filters to sweeps whose
    `produced_fields` includes at least one of these. Empty list = no
    filter (accept any sweep).
    """
    frag = expected_signal.get("fragment", "")
    for keyword, fields in _SIGNAL_TO_FIELDS.items():
        if keyword in frag:
            return fields
    return []  # Unknown signal shape → don't filter (caller decides)


def _parse_iso_utc(s: str) -> datetime:
    """Parse an ISO 8601 UTC timestamp (with or without trailing Z)."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def _sweep_age_days(completed_utc: str) -> int:
    completed = _parse_iso_utc(completed_utc)
    now = datetime.now(timezone.utc)
    return int((now - completed).total_seconds() / 86400)


def _find_row_in_results_jsonl(jsonl_path: Path, model: str, mode: str) -> dict | None:
    """Linear scan; returns first matching (model, mode) row or None."""
    if not jsonl_path.is_file():
        return None
    with open(jsonl_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("_record_type") == "metadata":
                continue
            if row.get("name") == model and row.get("mode") == mode:
                return row
    return None


def lookup(
    *,
    model: str,
    mode: str,
    venv_name: str,
    expected_signal: dict,
    case_id: str,
    within_days: int = STALENESS_DAYS_DEFAULT,
    index_path: Path | None = None,
) -> dict | None:
    """Library entry point. Returns a verify_repro-shaped dict or None on miss.

    Returned dict shape (subset compatible with verify_repro's
    VerificationResult — caller wraps it):
        evidence_source: "sweep_results"
        sweep_path: <abs path to results.jsonl>
        sweep_age_days: <int>
        cohort_sha256: <str> (gap 3)
        args_fingerprint: <str> (gap 3)
        sweep_kind: <str> (gap 9)
        case_id: <stamped from caller> (gap 10)
        exit_code, stdout, stderr: synthesized from the row's symptom data
        elapsed_s: 0 (cached, no run cost)
    """
    idx_path = index_path or DEFAULT_INDEX_PATH
    if not idx_path.is_file():
        return None

    try:
        idx = json.loads(idx_path.read_bytes())
    except json.JSONDecodeError:
        return None

    sweeps = idx.get("sweeps", [])
    required_fields = _signal_required_fields(expected_signal)

    # Filter: venv_name match + within_days + has required field if known
    candidates = []
    for s in sweeps:
        if s.get("venv_name") != venv_name:
            continue
        completed = s.get("completed_utc")
        if not completed:
            continue
        age = _sweep_age_days(completed)
        if age > within_days:
            continue
        if required_fields:
            produced = set(s.get("produced_fields") or [])
            if not produced.intersection(required_fields):
                continue  # gap 9: row can't produce the signal we need
        candidates.append((age, s))

    # Newest-first
    candidates.sort(key=lambda x: x[0])

    for age, s in candidates:
        results_path = Path(s["results_jsonl"])
        if not results_path.is_absolute():
            results_path = REPO_ROOT / results_path
        row = _find_row_in_results_jsonl(results_path, model, mode)
        if row is None:
            continue

        # Synthesize stdout/stderr from the row's symptom data so verify_repro's
        # classify() can run against it. Strategy: serialize the row's load-bearing
        # symptom fields as a single line each in stdout (per produced_fields).
        stdout_parts = []
        stderr_parts = []
        for fname in (s.get("produced_fields") or []):
            if fname in row and row[fname] is not None:
                # numeric / status fields → stdout; error/break fields → stderr
                if fname in ("error", "eager_error", "break_reasons"):
                    stderr_parts.append(f"{fname}={row[fname]}")
                else:
                    stdout_parts.append(f"{fname}={row[fname]}")
        # Always include name+mode for context
        stdout_parts.insert(0, f"name={row.get('name')} mode={row.get('mode')}")

        # exit_code: 0 if status looks like success/graph_break (run-completed),
        # non-zero otherwise (error)
        status = row.get("status", "")
        if status in ("full_graph", "graph_break", ""):
            exit_code = 0
        else:
            exit_code = 1

        return {
            "evidence_source": "sweep_results",
            "sweep_path": str(results_path),
            "sweep_age_days": age,
            "cohort_sha256": s.get("cohort_sha256", ""),
            "args_fingerprint": s.get("args_fingerprint", ""),
            "sweep_kind": s.get("sweep_kind", ""),
            "case_id": case_id,
            "exit_code": exit_code,
            "stdout": "\n".join(stdout_parts),
            "stderr": "\n".join(stderr_parts),
            "elapsed_s": 0.0,
        }

    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--venv-name", required=True, choices=["current", "nightly"])
    parser.add_argument("--within-days", type=int, default=STALENESS_DAYS_DEFAULT)
    parser.add_argument("--expected-signal-json", required=True, type=Path,
                        help="JSON file with kind+fragment")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--index-path", type=Path, default=None,
                        help=f"Default: {DEFAULT_INDEX_PATH}")
    args = parser.parse_args()

    expected_signal = json.loads(args.expected_signal_json.read_text())
    result = lookup(
        model=args.model, mode=args.mode, venv_name=args.venv_name,
        expected_signal=expected_signal, case_id=args.case_id,
        within_days=args.within_days, index_path=args.index_path,
    )
    if result is None:
        print(
            f"NO MATCH: no sweep within {args.within_days} days for "
            f"({args.model}, {args.mode}, venv={args.venv_name}) "
            f"with required fields for signal kind={expected_signal.get('kind')!r}",
            file=sys.stderr,
        )
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
