#!/usr/bin/env python3
"""count_breaks_per_pattern - count distinct break_reasons by pattern across a sweep's explain results.

Encodes the unit-hierarchy + duplicate-suppressed-dedup rules from
methodology R12 (skills/weekly-sweep-brief/methodology.md).

Three units reported per pattern, all distinct:
  * model_classes   - len(set(r['name']))               -- maintainer-meaningful unit
  * pair_rows       - len(set((r['name'], r['mode'])))  -- internal harness unit
  * distinct_breaks - count of non-suppressed break_reason matches

The --filter-duplicate-suppressed flag (default ON) skips entries whose reason
text contains "suppressed due to duplicate graph break" - those are dynamo's
own dedupe-trace markers, not new distinct breaks. A model row may have 15
break_reasons[] entries with graph_break_count: 3 because dynamo logs duplicates.
Counting all 15 over-counts by ~5x.

Pattern matching:
  * --pattern <regex>     - required loose filter (matched against full reason text)
  * --specific <regex>    - optional additional narrow filter (also matched)
  * Both must match for an entry to count

Usage:

    # Loose: any "Reconstruction failure" break
    python3 tools/count_breaks_per_pattern.py \\
        --pattern "Reconstruction failure"

    # Specific: only Reconstruction failure tied to DictItemsIterator (issue 96 scope)
    python3 tools/count_breaks_per_pattern.py \\
        --pattern "Reconstruction failure" \\
        --specific "DictItemsIterator"

    # Custom sweep + JSON output for piping
    python3 tools/count_breaks_per_pattern.py \\
        --pattern "Can't convert torch._check" \\
        --sweep-results sweep_results/nightly/2026-05-09/explain_results.json \\
        --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_SWEEP = (
    Path(__file__).parent.parent / "sweep_results" / "nightly" / "2026-05-09" / "explain_results.json"
)
DUPLICATE_SUPPRESSED_MARKER = "suppressed due to duplicate graph break"


def _reason_text(br: Any) -> str:
    """Extract the reason text from a break_reason entry (dict or string)."""
    if isinstance(br, dict):
        return br.get("reason", "") or ""
    return str(br)


def count_pattern(
    sweep_results: dict,
    pattern_re: re.Pattern,
    specific_re: re.Pattern | None = None,
    filter_duplicate_suppressed: bool = True,
    max_sample_models: int = 8,
) -> dict:
    """Count distinct breaks matching a pattern across a sweep's explain results.

    Returns dict with: model_classes, pair_rows, distinct_breaks, sample_models.
    """
    if isinstance(sweep_results, dict):
        if "results" in sweep_results:
            rows = sweep_results["results"]
        else:
            raise ValueError(
                f"sweep_results dict has no 'results' key (keys: {sorted(sweep_results.keys())[:8]}); "
                f"either pass a dict with 'results' or a bare list of row dicts"
            )
    else:
        rows = sweep_results
    if not isinstance(rows, list):
        raise ValueError(f"expected list of result rows, got {type(rows).__name__}")
    classes: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    distinct_breaks = 0
    suppressed_skipped = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = r.get("name")
        mode = r.get("mode")
        if not name:
            continue
        for br in r.get("break_reasons") or []:
            text = _reason_text(br)
            # Pattern check FIRST (so suppressed_skipped reports relative to matches, not all rows).
            # Per adversary review 2026-05-13: this also lets users investigate the marker
            # itself via --pattern "suppressed" + --no-filter-duplicate-suppressed.
            if not pattern_re.search(text):
                continue
            if specific_re is not None and not specific_re.search(text):
                continue
            if filter_duplicate_suppressed and DUPLICATE_SUPPRESSED_MARKER in text:
                suppressed_skipped += 1
                continue
            classes.add(name)
            if mode is not None:
                pairs.add((name, mode))
            distinct_breaks += 1
    sample_models = sorted(classes)[:max_sample_models]
    return {
        "model_classes": len(classes),
        "pair_rows": len(pairs),
        "distinct_breaks": distinct_breaks,
        "suppressed_skipped": suppressed_skipped,
        "sample_models": sample_models,
        "filter_duplicate_suppressed": filter_duplicate_suppressed,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See methodology R12 for the unit-hierarchy + dedup rationale.",
    )
    ap.add_argument("--pattern", required=True, help="Loose regex (matched against reason text); required")
    ap.add_argument("--specific", default=None, help="Optional narrow regex (also matched); both must match")
    ap.add_argument(
        "--sweep-results",
        type=Path,
        default=DEFAULT_SWEEP,
        help=f"Path to explain_results.json (default: {DEFAULT_SWEEP})",
    )
    ap.add_argument(
        "--no-filter-duplicate-suppressed",
        dest="filter_duplicate_suppressed",
        action="store_false",
        default=True,
        help="Disable the default duplicate-suppressed filter (rarely useful; over-counts)",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON to stdout instead of human-readable")
    ap.add_argument("--max-sample-models", type=int, default=8, help="Sample models to include in output")
    args = ap.parse_args(argv)

    if not args.sweep_results.exists():
        print(f"ERROR: sweep_results not found: {args.sweep_results}", file=sys.stderr)
        return 2
    try:
        pattern_re = re.compile(args.pattern, re.IGNORECASE)
    except re.error as e:
        print(f"ERROR: --pattern is not a valid regex: {e}", file=sys.stderr)
        return 2
    specific_re = None
    if args.specific is not None:
        try:
            specific_re = re.compile(args.specific, re.IGNORECASE)
        except re.error as e:
            print(f"ERROR: --specific is not a valid regex: {e}", file=sys.stderr)
            return 2

    sweep = json.loads(args.sweep_results.read_text())
    try:
        result = count_pattern(
            sweep,
            pattern_re,
            specific_re,
            filter_duplicate_suppressed=args.filter_duplicate_suppressed,
            max_sample_models=args.max_sample_models,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    result["pattern"] = args.pattern
    result["specific"] = args.specific
    result["sweep_results"] = str(args.sweep_results)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"Sweep: {args.sweep_results}")
    print(f"Pattern: {args.pattern!r}" + (f" + specific: {args.specific!r}" if args.specific else ""))
    print(f"Filter duplicate-suppressed: {args.filter_duplicate_suppressed}")
    print()
    print(f"  model_classes:     {result['model_classes']:>4}  (maintainer-meaningful unit; for headlines)")
    print(f"  pair_rows:         {result['pair_rows']:>4}  (internal harness unit; sub-detail only)")
    print(f"  distinct_breaks:   {result['distinct_breaks']:>4}  (after dup-suppressed filter)")
    print(f"  suppressed_skipped:{result['suppressed_skipped']:>4}  (dynamo's own dedupe-trace markers)")
    print()
    if result["sample_models"]:
        print(f"  sample_models (first {len(result['sample_models'])} alphabetically):")
        for m in result["sample_models"]:
            print(f"    - {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
