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

    Grammar (extended 2026-05-15 per Peng directive — generalized from
    `status`-only to any field name; needed for `numeric_status == divergence`
    NGB triage workflow):
        <field> in v1,v2,...   →  record.get(<field>) in {v1, v2, ...}
        <field> == val         →  record.get(<field>) == val
        <field> != val         →  record.get(<field>) != val

    `<field>` is `[A-Za-z_][A-Za-z0-9_]*`. Predicate caller is responsible for
    checking field presence (see main() for typo-detection guard).

    Same grammar as sweep/run_sweep.py:_parse_filter_predicate (status-only).
    Kept in sync by convention; if sweep grammar gets the same generalization,
    mirror the change here.
    """
    s = expr.strip()
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+)$', s)
    if m:
        field = m.group(1)
        values = {v.strip() for v in m.group(2).split(',') if v.strip()}
        if not values:
            sys.exit(f"ERROR: --filter {expr!r} has empty value list")
        return field, (lambda r: r.get(field) in values)
    m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=)\s*(.+)$', s)
    if m:
        field, op, val = m.group(1), m.group(2), m.group(3).strip()
        if op == '==':
            return field, (lambda r: r.get(field) == val)
        return field, (lambda r: r.get(field) != val)
    sys.exit(
        f"ERROR: --filter {expr!r} is not a supported expression. "
        f"Supported: '<field> in foo,bar,...', '<field> == foo', '<field> != foo'. "
        f"Field is any [A-Za-z_][A-Za-z0-9_]* (e.g. 'status', 'numeric_status')."
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


REQUIRED_VERSION_KEYS = ("torch", "transformers", "diffusers")
LARGE_MODELS_PATH = REPO_ROOT / "sweep" / "large_models.json"


def load_large_models() -> set:
    """Return set of model names listed in sweep/large_models.json (any tier).
    File missing → return empty set (fail-soft for partial checkouts).
    File present but corrupted → sys.exit (per adversary gap 7: corruption is
    more dangerous than absence and silent fail-soft hides bugs).
    """
    if not LARGE_MODELS_PATH.is_file():
        return set()
    try:
        return set(json.loads(LARGE_MODELS_PATH.read_text()).keys())
    except Exception as e:
        sys.exit(
            f"ERROR: {LARGE_MODELS_PATH} exists but is unparseable: {e}\n"
            f"  Fix the file or remove it. Do NOT silently default to "
            f"'include all large models' — that would burn sweep time."
        )


def main() -> int:
    # Adversary-review case_id 2026-05-07-124100 gap 9: explicit Python-version
    # guard. Path.is_relative_to() is 3.9+; print a useful error instead of an
    # AttributeError if someone runs from an older venv.
    if sys.version_info < (3, 9):
        sys.exit(f"ERROR: tools/generate_cohort.py requires Python >= 3.9 "
                 f"(got {sys.version_info.major}.{sys.version_info.minor}). "
                 f"Use ~/envs/torch211/bin/python or newer.")

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
    # Adversary-review case_id 2026-05-07-124100 gap 6:
    # default fail-loud on partial/empty source_versions. Overrides require
    # explicit opt-in so the generator and loader stay symmetric.
    p.add_argument('--allow-empty-versions', action='store_true',
                   help='Allow generating a cohort whose source has NO versions metadata. '
                        'Default: REFUSED (the resulting cohort cannot be version-checked).')
    p.add_argument('--allow-partial-versions', action='store_true',
                   help='Allow generating a cohort whose source has only some of '
                        'torch/transformers/diffusers in metadata.versions. Default: REFUSED '
                        '(cross-version mismatches in the unchecked library silently corrupt results).')
    # Sample-and-exclude flags (added 2026-05-15 per Peng directive — avoid one-off scripts for sample sweeps).
    p.add_argument('--n', type=int, default=None, metavar='N',
                   help='Random-sample N rows after filter + large-exclude. Deterministic given --seed. '
                        'If N >= matching count, takes all (no error).')
    p.add_argument('--seed', type=int, default=42,
                   help='Sampling seed (default: 42). Pass an explicit value for reproducible cohorts.')
    p.add_argument('--include-large', action='store_true',
                   help='Include models listed in sweep/large_models.json (any tier). Default: EXCLUDE '
                        'them (sample-sweep workflows usually want fast turnaround). Set this for '
                        'experiments that specifically need large/very_large coverage.')
    args = p.parse_args()

    if args.output.exists() and not args.force:
        sys.exit(f"ERROR: {args.output} exists. Pass --force to overwrite.")

    if not args.source.is_file():
        sys.exit(f"ERROR: source file not found: {args.source}")

    print(f"Source: {args.source}")
    results, source_versions = load_source(args.source)
    print(f"  loaded {len(results)} result rows")
    if source_versions:
        present = sorted(source_versions.keys())
        missing = [k for k in REQUIRED_VERSION_KEYS if k not in source_versions]
        print(f"  source versions: {source_versions}")
        if missing and not args.allow_partial_versions:
            sys.exit(
                f"ERROR: source versions are PARTIAL (missing: {', '.join(missing)}; "
                f"present: {', '.join(present)}).\n"
                f"  Cross-version mismatches in unchecked libraries (e.g. transformers) silently\n"
                f"  corrupt sweep results — this is the run-1 failure shape from 2026-05-06.\n"
                f"  Pass --allow-partial-versions to override (records partial state in cohort)."
            )
    else:
        if not args.allow_empty_versions:
            sys.exit(
                f"ERROR: source has NO versions metadata (and no sibling versions.json).\n"
                f"  The resulting cohort would have no version-compat assertion.\n"
                f"  Pass --allow-empty-versions to override (records empty state in cohort)."
            )
        print(f"  WARNING: no source_versions found in metadata or sibling versions.json (--allow-empty-versions set)")

    field, predicate = parse_filter_predicate(args.filter_expr)
    # Typo guard: if field is absent from EVERY non-metadata row, reject early.
    field_seen = sum(1 for r in results if field in r)
    if field_seen == 0:
        sys.exit(
            f"ERROR: --filter field {field!r} not present in any of the "
            f"{len(results)} source rows. Likely typo. Sample row keys: "
            f"{sorted(results[0].keys()) if results else '<no rows>'}"
        )
    matching_names = {r['name'] for r in results if predicate(r) and 'name' in r}
    print(f"Filter: {args.filter_expr!r} → {len(matching_names)} matching model names")

    # Large-model exclusion (default ON). Per Peng directive 2026-05-15:
    # sample-sweep workflows usually want fast turnaround; skip the large/very_large
    # tier unless --include-large is set.
    if not args.include_large:
        large = load_large_models()
        before_excl = len(matching_names)
        matching_names = {n for n in matching_names if n not in large}
        excluded = before_excl - len(matching_names)
        if excluded > 0:
            print(f"  excluded {excluded} large-tier model(s) (sweep/large_models.json) "
                  f"— pass --include-large to keep them")
    else:
        print(f"  --include-large set; keeping any large-tier models in cohort")

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

    # Random-sample (deterministic given --seed). Sort first so input order
    # doesn't affect output across machines/runs.
    if args.n is not None:
        if args.n <= 0:
            sys.exit(f"ERROR: --n must be positive (got {args.n})")
        specs.sort(key=lambda s: s['name'])
        if args.n >= len(specs):
            print(f"  --n {args.n} >= {len(specs)} candidates; taking all "
                  f"(no error, but may indicate too-strict filter)")
        else:
            import random
            rng = random.Random(args.seed)
            specs = sorted(rng.sample(specs, args.n), key=lambda s: s['name'])
            print(f"  --n {args.n} sampled (seed={args.seed}; deterministic)")

    # Per adversary gap 5: empty cohort after filter+exclude is almost certainly
    # a mistake. Fail loud at generation time instead of letting downstream
    # sweep launch produce a confusing "no models matched" error.
    if len(specs) == 0:
        sys.exit(
            f"ERROR: cohort is EMPTY after filter + large-exclude.\n"
            f"  Filter matched {len(matching_names)} name(s); large-exclude removed "
            f"{0 if args.include_large else 'some'}; final cohort 0 models.\n"
            f"  Loosen the filter, or pass --include-large if exclusion is the cause."
        )

    # Adversary gap 3: distinguish user-pinned seed from default-accepted seed
    # so future audits can trace cohort provenance correctly.
    seed_was_default = ('--seed' not in sys.argv)
    payload = {
        '_metadata': {
            'derived_from': str(args.source.resolve().relative_to(REPO_ROOT))
                            if args.source.resolve().is_relative_to(REPO_ROOT)
                            else str(args.source.resolve()),
            'derived_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'source_versions': source_versions,
            'filter': args.filter_expr,
            'sample_n': args.n,
            'sample_seed': args.seed if args.n is not None else None,
            'sample_seed_was_default': seed_was_default if args.n is not None else None,
            # Adversary gap 6: cross-Python-version determinism not guaranteed by
            # CPython for random.sample; record the runtime version so audits
            # know which Python produced the sample.
            'sample_python_version': (
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                if args.n is not None else None),
            'large_excluded': not args.include_large,
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
