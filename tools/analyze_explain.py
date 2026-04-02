#!/usr/bin/env python3
"""Analyze explain (pass 2) results — graph break counts, reasons, and taxonomy.

Usage:
    python tools/analyze_explain.py                           # Default: sweep_results/explain/pass2_results.json
    python tools/analyze_explain.py results.json              # Custom input
    python tools/analyze_explain.py --json                    # Machine-readable output
    python tools/analyze_explain.py --top-reasons 20          # Top N break reasons
    python tools/analyze_explain.py --model BartModel         # Detail for one model
    python tools/analyze_explain.py --csv breaks.csv          # Export to CSV
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_INPUT = os.path.join(REPO_ROOT, "sweep_results", "explain", "pass2_results.json")


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", [])


def classify_reason(reason_str):
    """Classify a break reason string into a taxonomy category."""
    r = reason_str.lower()
    patterns = [
        ("copy.deepcopy", "copy.deepcopy()"),
        ("deepcopy", "copy.deepcopy()"),
        ("data-dependent", "Data-dependent branching"),
        ("depends on the value", "Data-dependent guard"),
        ("backed symint", "Data-dependent guard"),
        ("guard", "Data-dependent guard"),
        ("logger", "logging.Logger"),
        ("logging", "logging.Logger"),
        ("skip function", "Skipped function call"),
        ("skipped call", "Skipped function call"),
        ("forbidden callable", "Skipped function call"),
        ("builtin callable", "Builtin callable"),
        ("nn module method", "Skipped function call"),
        ("proxy", "Proxy conversion failure"),
        ("faketensor", "Fake tensor error"),
        ("fake tensor", "Fake tensor error"),
        ("requires_grad", "Tensor requires_grad mutation"),
        ("constraint violation", "Constraint violation"),
        ("mark_dynamic", "Constraint violation"),
        ("non-tensor", "Non-Tensor return"),
        ("observed exception", "Observed exception"),
    ]
    for pattern, category in patterns:
        if pattern in r:
            return category
    return "Other"


def analyze(results, args):
    # Filter to successful explain results
    ok_results = [r for r in results if r.get("status") == "ok"]
    error_results = [r for r in results if r.get("status") != "ok"]

    if not ok_results:
        print("No successful explain results found.")
        if error_results:
            print(f"{len(error_results)} results had errors.")
        return

    # ── Summary stats ──
    break_counts = [r.get("graph_break_count", 0) for r in ok_results]
    graph_counts = [r.get("graph_count", 0) for r in ok_results]
    unique_models = set(r["name"] for r in ok_results)

    total_breaks = sum(break_counts)
    max_breaks = max(break_counts)
    avg_breaks = total_breaks / len(break_counts) if break_counts else 0

    print("=" * 70)
    print("EXPLAIN ANALYSIS — Graph Break Deep Dive")
    print("=" * 70)
    print(f"\nModels analyzed: {len(unique_models)}")
    print(f"Results (model × mode): {len(ok_results)} ok, {len(error_results)} errors")
    print(f"\nTotal graph breaks across all models: {total_breaks}")
    print(f"Average breaks per model-mode: {avg_breaks:.1f}")
    print(f"Max breaks in a single model-mode: {max_breaks}")
    print(f"Average subgraphs per model-mode: {sum(graph_counts)/len(graph_counts):.1f}")

    # ── Distribution of break counts ──
    print(f"\n{'─' * 50}")
    print("GRAPH BREAK COUNT DISTRIBUTION")
    print(f"{'─' * 50}")
    count_dist = Counter()
    for c in break_counts:
        if c == 0:
            count_dist["0 (clean)"] += 1
        elif c == 1:
            count_dist["1"] += 1
        elif c <= 3:
            count_dist["2-3"] += 1
        elif c <= 5:
            count_dist["4-5"] += 1
        elif c <= 10:
            count_dist["6-10"] += 1
        else:
            count_dist["11+"] += 1

    bucket_order = ["0 (clean)", "1", "2-3", "4-5", "6-10", "11+"]
    print(f"\n{'Breaks':>12}  {'Count':>6}  {'%':>6}")
    for bucket in bucket_order:
        n = count_dist.get(bucket, 0)
        pct = n / len(break_counts) * 100 if break_counts else 0
        bar = "█" * int(pct / 2)
        print(f"{bucket:>12}  {n:>6}  {pct:>5.1f}%  {bar}")

    # ── Models with most breaks ──
    print(f"\n{'─' * 50}")
    print("TOP MODELS BY GRAPH BREAK COUNT")
    print(f"{'─' * 50}")
    model_breaks = defaultdict(lambda: {"eval": 0, "train": 0, "eval_graphs": 0, "train_graphs": 0})
    for r in ok_results:
        mode = r.get("mode", "eval")
        mb = model_breaks[r["name"]]
        mb[mode] = r.get("graph_break_count", 0)
        mb[f"{mode}_graphs"] = r.get("graph_count", 0)

    # Sort by total breaks
    ranked = sorted(model_breaks.items(), key=lambda x: x[1]["eval"] + x[1]["train"], reverse=True)

    n_show = args.top_models if hasattr(args, "top_models") and args.top_models else 20
    print(f"\n{'Model':<40} {'eval':>6} {'train':>6} {'total':>6} {'graphs':>7}")
    for name, mb in ranked[:n_show]:
        total = mb["eval"] + mb["train"]
        graphs = mb["eval_graphs"] + mb["train_graphs"]
        print(f"{name:<40} {mb['eval']:>6} {mb['train']:>6} {total:>6} {graphs:>7}")

    if len(ranked) > n_show:
        print(f"  ... and {len(ranked) - n_show} more models")

    # ── Break reasons taxonomy ──
    print(f"\n{'─' * 50}")
    print("ROOT CAUSE TAXONOMY (all breaks, not just first)")
    print(f"{'─' * 50}")
    reason_counter = Counter()
    reason_models = defaultdict(set)
    raw_reasons = []

    for r in ok_results:
        reasons = r.get("break_reasons", [])
        for br in reasons:
            reason_str = br.get("reason", "unknown")
            category = classify_reason(reason_str)
            reason_counter[category] += 1
            reason_models[category].add(r["name"])
            raw_reasons.append({
                "model": r["name"],
                "mode": r.get("mode", "?"),
                "category": category,
                "reason": reason_str,
                "file": br.get("file", ""),
                "line": br.get("line", 0),
            })

    print(f"\n{'#':>3}  {'Root Cause':<35} {'Breaks':>7} {'Models':>7} {'%':>6}")
    for i, (category, count) in enumerate(reason_counter.most_common(), 1):
        n_models = len(reason_models[category])
        pct = count / sum(reason_counter.values()) * 100
        print(f"{i:>3}  {category:<35} {count:>7} {n_models:>7} {pct:>5.1f}%")

    print(f"\n     {'Total':<35} {sum(reason_counter.values()):>7}")

    # ── Top raw reasons (deduplicated) ──
    n_reasons = args.top_reasons if hasattr(args, "top_reasons") and args.top_reasons else 15
    print(f"\n{'─' * 50}")
    print(f"TOP {n_reasons} RAW BREAK REASONS")
    print(f"{'─' * 50}")
    raw_counter = Counter()
    for rr in raw_reasons:
        # Normalize: take first 100 chars of reason for grouping
        short = rr["reason"][:100]
        raw_counter[short] += 1

    for i, (reason, count) in enumerate(raw_counter.most_common(n_reasons), 1):
        print(f"\n{i:>3}. [{count}x] {reason}")

    # ── Per-model detail ──
    if args.model:
        print(f"\n{'─' * 50}")
        print(f"DETAIL: {args.model}")
        print(f"{'─' * 50}")
        model_results = [r for r in ok_results if r["name"] == args.model]
        if not model_results:
            print(f"  No results for '{args.model}'")
            # Fuzzy match
            close = [n for n in unique_models if args.model.lower() in n.lower()]
            if close:
                print(f"  Did you mean: {', '.join(sorted(close))}")
        for r in model_results:
            mode = r.get("mode", "?")
            print(f"\n  Mode: {mode}")
            print(f"  Graph breaks: {r.get('graph_break_count', '?')}")
            print(f"  Subgraphs: {r.get('graph_count', '?')}")
            if r.get("ops_per_graph"):
                print(f"  Ops per graph: {r['ops_per_graph']}")
            for br in r.get("break_reasons", []):
                cat = classify_reason(br.get("reason", ""))
                loc = ""
                if br.get("file"):
                    loc = f" ({os.path.basename(br['file'])}:{br.get('line', '?')})"
                print(f"    → [{cat}] {br['reason'][:120]}{loc}")

    # ── JSON output ──
    if args.json:
        output = {
            "summary": {
                "models_analyzed": len(unique_models),
                "total_breaks": total_breaks,
                "avg_breaks_per_model_mode": round(avg_breaks, 1),
                "max_breaks": max_breaks,
            },
            "taxonomy": {
                cat: {"breaks": count, "models": len(reason_models[cat])}
                for cat, count in reason_counter.most_common()
            },
            "per_model": {
                name: {"eval_breaks": mb["eval"], "train_breaks": mb["train"],
                       "total": mb["eval"] + mb["train"]}
                for name, mb in ranked
            },
            "distribution": dict(count_dist),
        }
        print(f"\n{'─' * 50}")
        print("JSON OUTPUT")
        print(json.dumps(output, indent=2))

    # ── CSV export ──
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "mode", "graph_break_count", "graph_count", "break_reasons"])
            for r in ok_results:
                reasons = "; ".join(
                    classify_reason(br.get("reason", ""))
                    for br in r.get("break_reasons", [])
                )
                w.writerow([
                    r["name"], r.get("mode", "?"),
                    r.get("graph_break_count", 0), r.get("graph_count", 0),
                    reasons,
                ])
        print(f"\nExported to {args.csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze explain (pass 2) results — graph break taxonomy and stats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                        help=f"Pass 2 results JSON (default: {DEFAULT_INPUT})")
    parser.add_argument("--json", action="store_true", help="Include JSON output")
    parser.add_argument("--csv", help="Export per-model data to CSV")
    parser.add_argument("--model", help="Show detail for a specific model")
    parser.add_argument("--top-reasons", type=int, default=15,
                        help="Number of top raw reasons to show (default: 15)")
    parser.add_argument("--top-models", type=int, default=20,
                        help="Number of top models to show (default: 20)")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: File not found: {args.input}")
        print("Run the explain sweep first:")
        print("  python sweep/run_sweep.py --pass2-from <pass1_results.json> --device cuda --skip-traces")
        sys.exit(1)

    results = load_results(args.input)
    analyze(results, args)


if __name__ == "__main__":
    main()
