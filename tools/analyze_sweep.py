#!/usr/bin/env python3
"""Analyze sweep results — status breakdown by source, variant, and mode.

Usage:
  python analyze_sweep.py sweep_results/v2.10_full/identify_results.json
  python analyze_sweep.py sweep_results/v2.10_full/identify_checkpoint.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_results(path):
    """Load results from JSON or JSONL checkpoint file."""
    path = Path(path)
    if path.suffix == ".jsonl":
        results = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
    else:
        data = json.load(open(path))
        return data.get("results", data)


def analyze(results):
    """Produce analysis tables from sweep results."""
    # Deduplicate: keep latest result per (name, mode)
    seen = {}
    for r in results:
        key = (r["name"], r.get("mode", "eval"))
        seen[key] = r
    results = list(seen.values())

    print(f"Total results: {len(results)} (deduplicated by name+mode)\n")

    # --- Overall status breakdown ---
    statuses = Counter(r.get("status", "unknown") for r in results)
    print("=" * 60)
    print("OVERALL STATUS")
    print("=" * 60)
    for status, count in statuses.most_common():
        pct = count / len(results) * 100
        print(f"  {status:20s} {count:5d}  ({pct:5.1f}%)")
    print()

    # --- By source ---
    by_source = defaultdict(Counter)
    for r in results:
        by_source[r.get("source", "unknown")][r.get("status", "unknown")] += 1

    print("=" * 60)
    print("BY SOURCE")
    print("=" * 60)
    for source in sorted(by_source):
        total = sum(by_source[source].values())
        fg = by_source[source].get("full_graph", 0)
        gb = by_source[source].get("graph_break", 0)
        print(f"\n  {source} ({total} results):")
        for status, count in by_source[source].most_common():
            pct = count / total * 100
            print(f"    {status:20s} {count:5d}  ({pct:5.1f}%)")

    # --- By variant (HF only) ---
    by_variant = defaultdict(Counter)
    for r in results:
        if r.get("source") == "hf":
            variant = r.get("variant", "base")
            by_variant[variant][r.get("status", "unknown")] += 1

    if by_variant:
        print("\n" + "=" * 60)
        print("HF MODELS BY VARIANT")
        print("=" * 60)
        for variant in ["base", "causal_lm", "conditional_generation"]:
            if variant not in by_variant:
                continue
            total = sum(by_variant[variant].values())
            fg = by_variant[variant].get("full_graph", 0)
            pct = fg / total * 100 if total else 0
            print(f"\n  {variant} ({total} results, {pct:.1f}% full_graph):")
            for status, count in by_variant[variant].most_common():
                spct = count / total * 100
                print(f"    {status:20s} {count:5d}  ({spct:5.1f}%)")

    # --- By mode ---
    by_mode = defaultdict(Counter)
    for r in results:
        by_mode[r.get("mode", "unknown")][r.get("status", "unknown")] += 1

    print("\n" + "=" * 60)
    print("BY MODE (eval vs train)")
    print("=" * 60)
    for mode in ["eval", "train"]:
        if mode not in by_mode:
            continue
        total = sum(by_mode[mode].values())
        fg = by_mode[mode].get("full_graph", 0)
        pct = fg / total * 100 if total else 0
        print(f"\n  {mode} ({total} results, {pct:.1f}% full_graph):")
        for status, count in by_mode[mode].most_common():
            spct = count / total * 100
            print(f"    {status:20s} {count:5d}  ({spct:5.1f}%)")

    # --- Variant × Mode cross-tab (HF only) ---
    cross = defaultdict(lambda: defaultdict(Counter))
    for r in results:
        if r.get("source") == "hf":
            variant = r.get("variant", "base")
            mode = r.get("mode", "unknown")
            cross[variant][mode][r.get("status", "unknown")] += 1

    if cross:
        print("\n" + "=" * 60)
        print("HF VARIANT × MODE CROSS-TAB (full_graph / total)")
        print("=" * 60)
        print(f"  {'variant':30s} {'eval':>15s} {'train':>15s}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")
        for variant in ["base", "causal_lm", "conditional_generation"]:
            if variant not in cross:
                continue
            cells = []
            for mode in ["eval", "train"]:
                total = sum(cross[variant][mode].values())
                fg = cross[variant][mode].get("full_graph", 0)
                cells.append(f"{fg}/{total}" if total else "—")
            print(f"  {variant:30s} {cells[0]:>15s} {cells[1]:>15s}")

    # --- Top graph_break models ---
    breaks = [r for r in results if r.get("status") == "graph_break"]
    if breaks:
        print("\n" + "=" * 60)
        print(f"GRAPH BREAK MODELS ({len(breaks)} total)")
        print("=" * 60)
        # Just list unique model names
        break_names = sorted(set(r["name"] for r in breaks))
        print(f"  {len(break_names)} unique models with graph breaks")
        if len(break_names) <= 30:
            for name in break_names:
                print(f"    - {name}")

    # --- Errors summary ---
    errors = [r for r in results if r.get("status") in ("eager_error", "create_error", "worker_error")]
    if errors:
        print("\n" + "=" * 60)
        print(f"ERROR MODELS ({len(errors)} total)")
        print("=" * 60)
        by_err_type = Counter(r.get("status") for r in errors)
        for etype, count in by_err_type.most_common():
            print(f"  {etype}: {count}")

    # --- Numeric correctness (eager vs compiled, identify-pass) ---
    # Added 2026-05-01: identify pass now runs a strict-determinism numeric
    # check on full_graph models. numeric_status is absent on results from
    # sweeps run before that change.
    numeric_results = [r for r in results if "numeric_status" in r]
    numeric_summary = {}
    if numeric_results:
        numeric_statuses = Counter(r["numeric_status"] for r in numeric_results)
        print("\n" + "=" * 60)
        print(f"NUMERIC CORRECTNESS ({len(numeric_results)} models with numeric_status)")
        print("=" * 60)
        for status, count in numeric_statuses.most_common():
            pct = count / len(numeric_results) * 100
            print(f"  {status:20s} {count:5d}  ({pct:5.1f}%)")

        # Skip-reason breakdown (helps distinguish "skipped because graph_break"
        # from "skipped because custom_compile_kwargs" etc.)
        skipped = [r for r in numeric_results if r["numeric_status"] == "skipped"]
        if skipped:
            skip_reasons = Counter(r.get("numeric_skip_reason", "unknown") for r in skipped)
            print(f"\n  Skip reasons:")
            for reason, count in skip_reasons.most_common():
                print(f"    {reason:25s} {count:5d}")

        # Top divergent rows by severity_ratio (max_diff / atol)
        divergent = [r for r in numeric_results
                     if r["numeric_status"] in ("divergence", "nan_inf_introduced",
                                                 "shape_mismatch", "dtype_mismatch")]
        if divergent:
            print(f"\n  TOP DIVERGENT (sorted by severity_ratio):")
            divergent.sort(key=lambda r: r.get("numeric_severity_ratio", 0), reverse=True)
            print(f"    {'name':40s} {'status':22s} {'max_diff':>12s} {'severity':>10s}  first_divergence")
            for r in divergent[:15]:
                name = (r.get("variant", "") + ":" + r["name"]) if r.get("variant") else r["name"]
                fd = r.get("numeric_first_divergence", "—")
                if isinstance(fd, str) and len(fd) > 50:
                    fd = fd[:47] + "..."
                print(f"    {name[:40]:40s} {r['numeric_status']:22s} "
                      f"{r.get('numeric_max_diff', 0):>12.2e} "
                      f"{r.get('numeric_severity_ratio', 0):>10.1f}  {fd}")

        numeric_summary = {
            "models_with_numeric_status": len(numeric_results),
            "statuses": dict(numeric_statuses),
            "divergent_count": len(divergent),
            "top_divergent": [
                {"name": r["name"], "variant": r.get("variant"),
                 "status": r["numeric_status"],
                 "max_diff": r.get("numeric_max_diff"),
                 "severity_ratio": r.get("numeric_severity_ratio"),
                 "first_divergence": r.get("numeric_first_divergence")}
                for r in divergent[:25]
            ],
        }

    # --- JSON summary for programmatic use ---
    summary = {
        "total_results": len(results),
        "statuses": dict(statuses),
        "by_source": {s: dict(c) for s, c in by_source.items()},
        "by_variant": {v: dict(c) for v, c in by_variant.items()},
        "by_mode": {m: dict(c) for m, c in by_mode.items()},
        "numeric_correctness": numeric_summary,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("results", help="Path to identify_results.json or checkpoint.jsonl")
    parser.add_argument("--json", action="store_true", help="Output JSON summary")
    args = parser.parse_args()

    results = load_results(args.results)
    summary = analyze(results)

    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
