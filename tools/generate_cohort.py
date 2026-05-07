#!/usr/bin/env python3
"""Generate a sweep cohort from a prior sweep's results, with provenance metadata.

Standalone version of the `--save-cohort` mechanism in `tools/run_experiment.py sweep`.
Use this when you want to BUILD a cohort file without ALSO running the sweep.

Why this exists:
    The save-cohort path inside `sweep` couples cohort-generation with sweep launch.
    Hand-building a cohort by other means produces flat lists with no _metadata,
    no provenance, and no version-stack assertion — which is exactly the bug
    pattern that produced nested_gb_cohort_2026-05-06.json (cohort built by hand
    from nightly graph_break filter instead of explain ok subset; no metadata;
    failed sanity check INV-A1 + A2 + A3).

Usage:
    python3 tools/generate_cohort.py \\
        --from sweep_results/experiments/<source_run>/explain_results.json \\
        --filter 'status in ok,graph_break' \\
        --output experiments/configs/<cohort_name>.json

The output file is a dict with two keys:
    "_metadata": { derived_from, derived_at, source_versions, filter, model_count }
    "models":    [ {name, source, hf_class?, hf_config?, variant?, ...}, ... ]

The launcher (`run_experiment.py sweep --models <cohort.json>`) accepts both this
dict-with-_metadata format and the legacy bare-list format. Always emit the dict
form so downstream sanity checks (INV-A2) pass.

Filter syntax (mirrors `--save-cohort`):
    status in foo,bar,baz   →  record's status in {foo, bar, baz}
    status == foo
    status != foo
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_filter_predicate(expr: str):
    """Parse a filter expression into a callable predicate(record) -> bool.

    Same grammar as sweep/run_sweep.py:_parse_filter_predicate. Kept in sync
    by convention; if the sweep grammar grows, mirror the change here.
    """
    s = expr.strip()
    m = re.match(r'^status\s+in\s+(.+)$', s)
    if m:
        values = {v.strip() for v in m.group(1).split(',') if v.strip()}
        if not values:
            sys.exit(f"ERROR: --filter {expr!r} has empty value list")
        return lambda r: r.get('status') in values
    m = re.match(r'^status\s*(==|!=)\s*(.+)$', s)
    if m:
        op, val = m.group(1), m.group(2).strip()
        return (lambda r: r.get('status') == val) if op == '==' else \
               (lambda r: r.get('status') != val)
    sys.exit(
        f"ERROR: --filter {expr!r} is not a supported expression. "
        f"Supported: 'status in foo,bar,...', 'status == foo', 'status != foo'."
    )


def load_source(path: Path):
    """Load a sweep results file. Supports both:
       - identify_results.json / explain_results.json: dict with metadata + results
       - identify_streaming.jsonl: line-delimited
    Returns (results_list, source_versions_dict)
    """
    if path.suffix == '.jsonl':
        results = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        # No metadata in jsonl; check sibling versions.json
        return results, _load_sibling_versions(path)

    raw = json.loads(path.read_text())
    metadata = raw.get('metadata', {}) if isinstance(raw, dict) else {}
    versions = metadata.get('versions', {}) if isinstance(metadata, dict) else {}
    if isinstance(raw, dict):
        results = raw.get('results', [])
    elif isinstance(raw, list):
        results = raw
    else:
        sys.exit(f"ERROR: unrecognized results format in {path}")

    # Fallback to sibling versions.json (older nightly formats)
    if not versions:
        versions = _load_sibling_versions(path)
    return results, versions


def _load_sibling_versions(path: Path) -> dict:
    sibling = path.parent / 'versions.json'
    if not sibling.is_file():
        return {}
    try:
        sv = json.loads(sibling.read_text())
        out = {
            'torch': sv.get('pytorch') or sv.get('torch'),
            'transformers': sv.get('transformers'),
            'diffusers': sv.get('diffusers'),
        }
        return {k: v for k, v in out.items() if v}
    except Exception as e:
        print(f"WARN: sibling versions.json unparseable: {e}", file=sys.stderr)
        return {}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate a sweep cohort from a prior sweep's results, with provenance metadata.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument('--from', dest='source', required=True, type=Path,
                   help='Source sweep results file (identify_results.json or explain_results.json)')
    p.add_argument('--filter', dest='filter_expr', required=True,
                   help='Filter expression. Examples: "status in ok,graph_break", "status == graph_break"')
    p.add_argument('--output', '-o', required=True, type=Path,
                   help='Output cohort path (typically experiments/configs/<name>.json)')
    p.add_argument('--force', action='store_true',
                   help='Overwrite output file if it exists')
    args = p.parse_args()

    if args.output.exists() and not args.force:
        sys.exit(f"ERROR: {args.output} exists. Pass --force to overwrite.")

    if not args.source.is_file():
        sys.exit(f"ERROR: source file not found: {args.source}")

    print(f"Source: {args.source}")
    results, source_versions = load_source(args.source)
    print(f"  loaded {len(results)} result rows")
    if source_versions:
        print(f"  source versions: {source_versions}")
    else:
        print(f"  WARNING: no source_versions found in metadata or sibling versions.json")

    predicate = parse_filter_predicate(args.filter_expr)
    matching_names = {r['name'] for r in results if predicate(r) and 'name' in r}
    print(f"Filter: {args.filter_expr!r} → {len(matching_names)} matching model names")

    seen = set()
    specs = []
    for r in results:
        name = r.get('name')
        if not name or name in seen or name not in matching_names:
            continue
        seen.add(name)
        spec = {'name': name, 'source': r.get('source', 'hf')}
        for k in ('hf_class', 'hf_config', 'variant', 'input_type',
                  'constructor_args', 'inputs'):
            if k in r:
                spec[k] = r[k]
        specs.append(spec)

    payload = {
        '_metadata': {
            'derived_from': str(args.source.resolve().relative_to(REPO_ROOT))
                            if args.source.resolve().is_relative_to(REPO_ROOT)
                            else str(args.source.resolve()),
            'derived_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'source_versions': source_versions,
            'filter': args.filter_expr,
            'model_count': len(specs),
            'generated_by': 'tools/generate_cohort.py',
        },
        'models': specs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + '\n')
    print(f"\nWrote {args.output} ({len(specs)} models)")
    print(f"  _metadata.source_versions: {source_versions}")
    print(f"  _metadata.derived_from: {payload['_metadata']['derived_from']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
