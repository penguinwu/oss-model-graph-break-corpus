"""Aggregate per-config result.json files into one experiment summary.

Glob `<experiment_dir>/<config_id>/result.json` for all configs; compute
distributions; write `summary.md` (human-readable) + `summary.json`
(machine-readable).

Usage:
    python -m discovery.merge_results \\
        --in-dir /tmp/runs/v8-parallel-test \\
        --out /tmp/runs/v8-parallel-test/summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _load_results(experiment_dir: Path) -> dict[str, dict]:
    """Return {config_id: result_dict}. Missing results show up as {} marker."""
    results = {}
    for cfg_dir in sorted(experiment_dir.iterdir()):
        if not cfg_dir.is_dir():
            continue
        rj = cfg_dir / "result.json"
        if rj.exists():
            try:
                results[cfg_dir.name] = json.loads(rj.read_text())
            except json.JSONDecodeError as e:
                results[cfg_dir.name] = {"_error": f"JSON decode: {e}"}
        else:
            results[cfg_dir.name] = {"_error": "result.json not found"}
    return results


def _row(label: str, r: dict) -> str:
    """One-line summary row for the markdown table."""
    if "_error" in r:
        return f"| {label} | — | — | — | (no result) | {r['_error'][:40]} |"
    val = r.get("validation") or {}
    fs = val.get("fix_status", "?")
    det = val.get("details") or {}
    gb_count = det.get("gb_under_canonical_inputs", "?")
    gb_sites = det.get("gb_call_sites") or []
    gb_types = sorted({s.get("type") for s in gb_sites if s.get("file") and s.get("type")})
    gb_type_str = ",".join(gb_types) if gb_types else "(none)"
    perf_t1 = (r.get("perf") or {}).get("perf_shape_sanity", "—")
    perf_t2 = (r.get("perf_tier2") or {}).get("perf_shape_sanity", "—")
    fsp = r.get("fix_survives_perf")
    fsp_str = "True" if fsp is True else "False" if fsp is False else "None"
    speedup_t1 = (r.get("perf") or {}).get("speedup")
    speedup_str = f"{speedup_t1:.2f}x" if speedup_t1 and isinstance(speedup_t1, (int, float)) and speedup_t1 == speedup_t1 else "—"
    return f"| {label} | {fs} | {fsp_str} | {gb_count} | {gb_type_str} | {perf_t1}/{perf_t2} | {speedup_str} |"


def _build_summary(results: dict) -> tuple[str, dict]:
    """Build markdown + JSON summaries."""
    n = len(results)
    n_with_result = sum(1 for r in results.values() if "_error" not in r)
    n_failed = n - n_with_result

    # Distributions (over configs that have a result)
    fix_status_dist = Counter(
        (r.get("validation") or {}).get("fix_status", "?")
        for r in results.values() if "_error" not in r
    )
    fix_survives_perf_dist = Counter(
        r.get("fix_survives_perf") for r in results.values() if "_error" not in r
    )
    gb_type_dist: Counter = Counter()
    for r in results.values():
        if "_error" in r:
            continue
        sites = ((r.get("validation") or {}).get("details") or {}).get("gb_call_sites") or []
        for s in sites:
            if s.get("file") and s.get("type"):
                gb_type_dist[s["type"]] += 1

    # Markdown
    lines = ["# Experiment Summary", ""]
    lines.append(f"**Configs total:** {n}")
    lines.append(f"**Configs with result:** {n_with_result}")
    lines.append(f"**Configs failed:** {n_failed}")
    lines.append("")
    lines.append("## Per-config table")
    lines.append("")
    lines.append("| config_id | fix_status | fsp | gb_count | gb_types | sanity_t1/t2 | speedup_t1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for cfg_id, r in sorted(results.items()):
        lines.append(_row(cfg_id, r))
    lines.append("")
    lines.append("## Distributions")
    lines.append("")
    lines.append("### fix_status")
    for fs, c in fix_status_dist.most_common():
        lines.append(f"- {fs}: {c}")
    lines.append("")
    lines.append("### fix_survives_perf")
    for fsp, c in fix_survives_perf_dist.most_common():
        lines.append(f"- {fsp!s}: {c}")
    lines.append("")
    lines.append("### gb_call_site types (across all configs)")
    for t, c in gb_type_dist.most_common():
        lines.append(f"- {t}: {c}")
    lines.append("")
    if n_failed:
        lines.append("## Failed configs")
        lines.append("")
        for cfg_id, r in sorted(results.items()):
            if "_error" in r:
                lines.append(f"- {cfg_id}: {r['_error']}")
        lines.append("")
    md = "\n".join(lines)

    # JSON
    js = {
        "n_total": n,
        "n_with_result": n_with_result,
        "n_failed": n_failed,
        "fix_status_distribution": dict(fix_status_dist),
        "fix_survives_perf_distribution": {str(k): v for k, v in fix_survives_perf_dist.items()},
        "gb_type_distribution": dict(gb_type_dist),
        "configs": {
            cfg_id: {
                "fix_status": (r.get("validation") or {}).get("fix_status") if "_error" not in r else None,
                "fix_survives_perf": r.get("fix_survives_perf") if "_error" not in r else None,
                "error": r.get("_error"),
            }
            for cfg_id, r in results.items()
        },
    }
    return md, js


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, type=Path,
                   help="experiment dir (contains per-config dirs)")
    p.add_argument("--out", required=True, type=Path,
                   help="output markdown summary path; .json companion is auto-derived")
    args = p.parse_args()

    if not args.in_dir.is_dir():
        print(f"ERROR: not a dir: {args.in_dir}", file=sys.stderr)
        return 2

    results = _load_results(args.in_dir)
    md, js = _build_summary(results)

    args.out.write_text(md)
    js_path = args.out.with_suffix(".json")
    js_path.write_text(json.dumps(js, indent=2))
    print(f"wrote {args.out} + {js_path}", file=sys.stderr)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
