#!/usr/bin/env python3
"""Mechanical executor for the sanity-check skill invariants A1-A4 + C1-C2 + G1.

Surfaced by adversary-review case_id 2026-05-07-124100-cohort-regen-fix gap 7:
the v3 sanity-check skill (skills/sweep_sanity_check.md) defines invariants
A1-A4 (cohort), C1-C2 (status), G1 (hygiene), etc. as MARKDOWN TEXT — humans
were expected to read and apply them mentally. Under launch pressure this gets
skipped (exactly the 2026-05-06 failure path).

This tool mechanically checks the most actionable invariants. Two modes:

    PRE-LAUNCH (operates on cohort + source files):
        python3 tools/check_cohort_invariants.py <cohort.json>
        Checks A2 (metadata block present), A3 (no extras vs source), A4 (no skip_models).

    POST-SWEEP (operates on results files):
        python3 tools/check_cohort_invariants.py --post-sweep <results.json>
        Checks A1 (no attribute-not-found create_errors), C1 (no create_error),
        C2 (no non-env eager_error), G1 (every non-success row triaged).

Exit codes:
    0 = all checked invariants PASS
    1 = at least one STRICT_FAIL invariant
    2 = tool error (file not found, etc.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

# Patterns for A1: attribute-not-found create_errors that signal version contamination.
ATTR_ERROR_PATTERN = re.compile(
    r"module ['\"]([\w.]+)['\"] has no attribute ['\"]([\w]+)['\"]"
)


class InvariantFailure:
    def __init__(self, code: str, severity: str, message: str, *, examples: Optional[list] = None):
        self.code = code
        self.severity = severity
        self.message = message
        self.examples = examples or []

    def __str__(self):
        s = f"  [{self.severity}] {self.code}: {self.message}"
        if self.examples:
            for ex in self.examples[:5]:
                s += f"\n      • {ex}"
            if len(self.examples) > 5:
                s += f"\n      • ... ({len(self.examples) - 5} more)"
        return s


def check_pre_launch(cohort_path: Path) -> list:
    """Apply A2-A4 to a cohort file. Returns list of InvariantFailure (empty on PASS)."""
    failures = []

    if not cohort_path.is_file():
        return [InvariantFailure("FILE_MISSING", "STRICT_FAIL",
                                 f"cohort file not found: {cohort_path}")]

    try:
        loaded = json.loads(cohort_path.read_text())
    except json.JSONDecodeError as e:
        return [InvariantFailure("INVALID_JSON", "STRICT_FAIL",
                                 f"cohort file not parseable: {e}")]

    # A2 — cohort file has _metadata block
    if isinstance(loaded, list):
        failures.append(InvariantFailure(
            "A2", "STRICT_FAIL",
            "cohort is a flat list with no _metadata block (the 2026-05-06 failure shape)",
        ))
        # Can't apply A3/A4 without metadata
        return failures

    if not isinstance(loaded, dict) or "_metadata" not in loaded or not loaded["_metadata"]:
        failures.append(InvariantFailure(
            "A2", "STRICT_FAIL",
            "cohort dict missing or empty _metadata block",
        ))
        return failures

    metadata = loaded["_metadata"]
    cohort_models = loaded.get("models", [])
    cohort_names = {m.get("name") for m in cohort_models if m.get("name")}

    # A3 — every cohort model exists in declared source per declared filter
    derived_from = metadata.get("derived_from")
    filter_expr = metadata.get("filter")
    if derived_from and filter_expr and derived_from != "synthetic":
        # Resolve source path (relative to repo root or absolute)
        source_path = Path(derived_from)
        if not source_path.is_absolute():
            source_path = REPO_ROOT / source_path
        if source_path.is_file():
            try:
                source_raw = json.loads(source_path.read_text())
                source_results = (source_raw.get("results", []) if isinstance(source_raw, dict)
                                  else source_raw)
                source_names_per_filter = _apply_filter(source_results, filter_expr)
                extras = sorted(cohort_names - source_names_per_filter)
                if extras:
                    failures.append(InvariantFailure(
                        "A3", "STRICT_FAIL",
                        f"{len(extras)} cohort model(s) not in declared source per declared filter "
                        f"({filter_expr!r} on {derived_from})",
                        examples=extras,
                    ))
            except (json.JSONDecodeError, KeyError) as e:
                failures.append(InvariantFailure(
                    "A3", "FLAG",
                    f"could not verify A3: source file unparseable ({e})",
                ))
        else:
            failures.append(InvariantFailure(
                "A3", "FLAG",
                f"could not verify A3: declared source not found at {source_path}",
            ))

    # A4 — no cohort model in skip_models.json
    skip_models_path = REPO_ROOT / "sweep" / "skip_models.json"
    if skip_models_path.is_file():
        try:
            skip_data = json.loads(skip_models_path.read_text())
            # skip_models.json shape: {"models": [...]} OR {"<name>": {...}, ...}
            if isinstance(skip_data, dict) and "models" in skip_data:
                skip_set = set(skip_data["models"])
            elif isinstance(skip_data, dict):
                skip_set = set(skip_data.keys())
            else:
                skip_set = set()
            in_skip = sorted(cohort_names & skip_set)
            if in_skip:
                failures.append(InvariantFailure(
                    "A4", "STRICT_FAIL",
                    f"{len(in_skip)} cohort model(s) appear in skip_models.json",
                    examples=in_skip,
                ))
        except (json.JSONDecodeError, KeyError):
            pass  # Skip silently if skip_models.json is malformed; user can debug separately.

    return failures


def check_post_sweep(results_path: Path) -> list:
    """Apply A1 + C1-C2 + G1 to a results file. Returns list of InvariantFailure.

    Accepts BOTH formats produced by the sweep harness:
      - JSON: {metadata: ..., results: [...]} or top-level list
      - JSONL: line-delimited records (first line typically a metadata record
        with `_record_type: "metadata"`, subsequent lines are result rows).
    The harness's `identify_results.json` is JSONL despite the .json suffix.
    """
    failures = []

    if not results_path.is_file():
        return [InvariantFailure("FILE_MISSING", "STRICT_FAIL",
                                 f"results file not found: {results_path}")]

    text = results_path.read_text()
    rows = None

    # Try JSON first (single object or list)
    try:
        raw = json.loads(text)
        rows = raw.get("results", []) if isinstance(raw, dict) else raw
    except json.JSONDecodeError:
        # Fall back to JSONL
        try:
            parsed = []
            for line in text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict) and rec.get("_record_type") == "metadata":
                    continue  # skip metadata header
                parsed.append(rec)
            rows = parsed
        except json.JSONDecodeError as e:
            return [InvariantFailure("INVALID_JSON", "STRICT_FAIL",
                                     f"results file not parseable as JSON or JSONL: {e}")]

    if not isinstance(rows, list):
        return [InvariantFailure("INVALID_RESULTS", "STRICT_FAIL",
                                 "results does not contain a list of rows")]

    # A1 — no attribute-not-found create_errors
    a1_examples = []
    for r in rows:
        if r.get("status") == "create_error":
            err = r.get("error", "") or ""
            if ATTR_ERROR_PATTERN.search(err):
                a1_examples.append(f"{r.get('name', '<unknown>')}: {err.strip().splitlines()[0][:120]}")
    if a1_examples:
        failures.append(InvariantFailure(
            "A1", "STRICT_FAIL",
            f"{len(a1_examples)} attribute-not-found create_error row(s) (cohort drift / version contamination)",
            examples=a1_examples,
        ))

    # C1 — no create_error at all (curated cohort default)
    c1_count = sum(1 for r in rows if r.get("status") == "create_error")
    if c1_count > 0:
        failures.append(InvariantFailure(
            "C1", "STRICT_FAIL",
            f"{c1_count} create_error row(s) (cohort/loader/network bug)",
            examples=[r.get("name", "<unknown>") for r in rows if r.get("status") == "create_error"][:10],
        ))

    # C2 — non-env eager_error
    # Env-induced patterns we accept: CUDA OOM, cudnn, contention.
    env_patterns = [
        re.compile(r"out of memory", re.I),
        re.compile(r"CUDA error", re.I),
        re.compile(r"CUDNN", re.I),
        re.compile(r"contention", re.I),
    ]
    non_env_eager_errors = []
    for r in rows:
        if r.get("status") != "eager_error":
            continue
        err = r.get("error", "") or ""
        if not any(p.search(err) for p in env_patterns):
            non_env_eager_errors.append(f"{r.get('name', '<unknown>')}: {err.strip().splitlines()[0][:120]}")
    if non_env_eager_errors:
        failures.append(InvariantFailure(
            "C2", "STRICT_FAIL",
            f"{len(non_env_eager_errors)} non-env eager_error row(s) (harness bug, not env)",
            examples=non_env_eager_errors,
        ))

    # G1 — every non-success row is triaged
    # We can only check this if known_errors.json + skip_models.json exist.
    # For this lightweight executor, we report the COUNT of untriaged candidates
    # — the harness's --strict-known-errors mode is the canonical enforcer.
    # Status vocabulary varies by launch path:
    # - sweep subcommand (via run_sweep.py): "ok", "full_graph", "graph_break"
    # - run subcommand (via run_experiment.py + orchestrator): "success" (when
    #   compile succeeded with non-fullgraph mode), plus the others
    # Both paths emit the worker's status field; we accept all success synonyms.
    # (Caught missing 2026-05-07 20:46 ET when NGB verify gate via `run` produced
    # status="success" for all rows and check_cohort_invariants false-flagged them
    # as untriaged.)
    success_statuses = {"ok", "full_graph", "graph_break", "success"}
    non_success = [r for r in rows if r.get("status") not in success_statuses]
    if non_success:
        # Cross-check against known_errors.json/skip_models.json
        known_set, skip_set = _load_triaged_sets()
        untriaged = []
        for r in non_success:
            name = r.get("name")
            if not name:
                untriaged.append("<unnamed row>")
                continue
            if name in skip_set:
                continue
            err = r.get("error", "") or ""
            triaged_by_pattern = any(pat in err for pat in known_set if isinstance(pat, str))
            if name not in skip_set and not triaged_by_pattern:
                untriaged.append(f"{name} (status={r.get('status')})")
        if untriaged:
            failures.append(InvariantFailure(
                "G1", "STRICT_FAIL",
                f"{len(untriaged)} non-success row(s) not triaged in known_errors.json or skip_models.json",
                examples=untriaged,
            ))

    return failures


def _apply_filter(rows: list, filter_expr: str) -> set:
    """Apply a sanity-check-skill-shaped filter to source rows. Returns set of names."""
    s = filter_expr.strip()
    m = re.match(r'^status\s+in\s+(.+)$', s)
    if m:
        values = {v.strip() for v in m.group(1).split(',') if v.strip()}
        return {r["name"] for r in rows if r.get("status") in values and "name" in r}
    m = re.match(r'^status\s*(==|!=)\s*(.+)$', s)
    if m:
        op, val = m.group(1), m.group(2).strip()
        if op == "==":
            return {r["name"] for r in rows if r.get("status") == val and "name" in r}
        return {r["name"] for r in rows if r.get("status") != val and "name" in r}
    if s in ("(none)", "all", ""):
        return {r["name"] for r in rows if "name" in r}
    # Unknown filter — return all names to avoid false positives, with a print
    print(f"WARN: unknown filter expression {filter_expr!r}; A3 check is non-strict",
          file=sys.stderr)
    return {r["name"] for r in rows if "name" in r}


def _load_triaged_sets() -> tuple:
    """Load known_errors.json patterns + skip_models.json names."""
    known = set()
    skip = set()
    ke_path = REPO_ROOT / "sweep" / "known_errors.json"
    if ke_path.is_file():
        try:
            ke = json.loads(ke_path.read_text())
            # known_errors.json: {"errors": [{"error_pattern": "..."}, ...]} OR list-of-dicts
            errors = ke.get("errors", []) if isinstance(ke, dict) else ke
            for e in errors:
                if isinstance(e, dict) and "error_pattern" in e:
                    known.add(e["error_pattern"])
        except (json.JSONDecodeError, KeyError):
            pass
    sm_path = REPO_ROOT / "sweep" / "skip_models.json"
    if sm_path.is_file():
        try:
            sm = json.loads(sm_path.read_text())
            if isinstance(sm, dict) and "models" in sm:
                skip = set(sm["models"])
            elif isinstance(sm, dict):
                skip = set(sm.keys())
            elif isinstance(sm, list):
                skip = set(sm)
        except (json.JSONDecodeError, KeyError):
            pass
    return known, skip


def main() -> int:
    if sys.version_info < (3, 9):
        sys.exit(2)

    p = argparse.ArgumentParser(
        description="Mechanical executor for sanity-check skill invariants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("file", type=Path, help="Cohort JSON file (pre-launch) OR results JSON (--post-sweep)")
    p.add_argument("--post-sweep", action="store_true",
                   help="Treat <file> as a sweep results file; apply A1 + C1-C2 + G1.")
    args = p.parse_args()

    if args.post_sweep:
        failures = check_post_sweep(args.file)
        mode = "POST-SWEEP"
    else:
        failures = check_pre_launch(args.file)
        mode = "PRE-LAUNCH"

    print(f"[check_cohort_invariants] {mode} on {args.file}")
    if not failures:
        print(f"  ✓ all checked invariants PASS")
        return 0

    strict = [f for f in failures if f.severity == "STRICT_FAIL"]
    flags = [f for f in failures if f.severity == "FLAG"]
    print(f"  {len(strict)} STRICT_FAIL, {len(flags)} FLAG")
    print()
    for f in failures:
        print(str(f))
    print()
    return 1 if strict else 0


if __name__ == "__main__":
    sys.exit(main())
