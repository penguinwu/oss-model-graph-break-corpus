#!/usr/bin/env python3
"""Analyze graph break trends across PyTorch versions.

Compares identify (model counts) and explain (total break counts) across multiple
PyTorch versions. Normalizes by testable models to avoid confounding from
changing eager_error/create_error rates.

Usage:
    # Default: compare v2.8, v2.9, v2.10
    python tools/analyze_trend.py

    # Custom version dirs
    python tools/analyze_trend.py sweep_results/v2.8 sweep_results/v2.9 --labels "2.8" "2.9"

    # Include train mode
    python tools/analyze_trend.py --train

    # JSON output
    python tools/analyze_trend.py --json

    # Show per-model details for fixed/regressed models
    python tools/analyze_trend.py --details
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def load_identify(path):
    """Load identify results into {name: {mode: status}}."""
    with open(path) as f:
        data = json.load(f)

    d = defaultdict(dict)
    if isinstance(data, dict) and "models" in data:
        # corpus.json format
        for m in data["models"]:
            for mode in ("eval", "train"):
                if mode in m:
                    d[m["name"]][mode] = m[mode].get("status", "unknown")
    else:
        # identify_results.json format (flat list or {results: [...]})
        results = data if isinstance(data, list) else data.get("results", [])
        for r in results:
            d[r["name"]][r.get("mode", "eval")] = r.get("status", "unknown")
    return d


def load_explain(path):
    """Load explain results into {name: {mode: result_dict}}."""
    if not os.path.exists(path):
        return None

    with open(path) as f:
        data = json.load(f)

    d = defaultdict(dict)
    results = data if isinstance(data, list) else data.get("results", [])
    for r in results:
        d[r["name"]][r.get("mode", "eval")] = r
    return d


def find_version_dirs(args):
    """Find version directories and their data files."""
    versions = []

    if args.dirs:
        # Explicit directories provided
        for i, d in enumerate(args.dirs):
            label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(d)
            p1 = os.path.join(d, "identify_results.json")
            p2 = os.path.join(d, "explain_results.json")
            if not os.path.exists(p1):
                print(f"ERROR: {p1} not found", file=sys.stderr)
                sys.exit(1)
            versions.append({"label": label, "identify": p1, "explain": p2, "dir": d})
    else:
        # Auto-discover: sweep_results/v2.* directories + main corpus
        sweep_dir = os.path.join(REPO_ROOT, "sweep_results")
        version_dirs = sorted(
            d for d in os.listdir(sweep_dir)
            if d.startswith("v") and os.path.isfile(os.path.join(sweep_dir, d, "identify_results.json"))
        )
        for vd in version_dirs:
            vpath = os.path.join(sweep_dir, vd)
            versions.append({
                "label": vd,
                "identify": os.path.join(vpath, "identify_results.json"),
                "explain": os.path.join(vpath, "explain_results.json"),
                "dir": vpath,
            })
        # Add main corpus as latest version
        corpus_path = os.path.join(REPO_ROOT, "corpus", "corpus.json")
        explain_path = os.path.join(sweep_dir, "explain", "explain_results.json")
        if os.path.exists(corpus_path):
            # Try to detect version from versions.json or metadata
            label = "v2.10"
            versions.append({
                "label": label,
                "identify": corpus_path,
                "explain": explain_path,
                "dir": sweep_dir,
            })

    return versions


def analyze_mode(versions_data, mode, common_models):
    """Analyze a single mode (eval or train) across versions."""
    results = []

    for vd in versions_data:
        p1 = vd["identify_data"]
        p2 = vd.get("explain_data")

        # Identify stats
        statuses = Counter(p1[n].get(mode, "missing") for n in common_models)
        testable = statuses.get("clean", 0) + statuses.get("graph_break", 0)

        # Explain stats (if available)
        total_breaks = 0
        explain_ok = 0
        explain_error = 0
        broken_with_breaks = 0

        if p2:
            for n in common_models:
                if n in p2 and mode in p2[n]:
                    r = p2[n][mode]
                    if r.get("status") == "ok":
                        explain_ok += 1
                        bc = r.get("graph_break_count", 0)
                        total_breaks += bc
                        if bc > 0:
                            broken_with_breaks += 1
                    elif r.get("status") == "explain_error":
                        explain_error += 1

        results.append({
            "label": vd["label"],
            "mode": mode,
            "total_models": len(common_models),
            "clean": statuses.get("clean", 0),
            "graph_break": statuses.get("graph_break", 0),
            "eager_error": statuses.get("eager_error", 0),
            "create_error": statuses.get("create_error", 0),
            "timeout": statuses.get("timeout", 0),
            "testable": testable,
            "total_graph_breaks": total_breaks if p2 else None,
            "explain_ok": explain_ok if p2 else None,
            "explain_error": explain_error if p2 else None,
            "avg_breaks_per_broken": round(total_breaks / statuses.get("graph_break", 1), 1) if statuses.get("graph_break", 0) > 0 else None,
            "has_explain": p2 is not None,
        })

    return results


def find_transitions(versions_data, mode, common_models):
    """Find models that changed status between first and last version."""
    if len(versions_data) < 2:
        return {}

    first = versions_data[0]["identify_data"]
    last = versions_data[-1]["identify_data"]

    transitions = defaultdict(list)
    for n in sorted(common_models):
        s_first = first[n].get(mode, "missing")
        s_last = last[n].get(mode, "missing")
        if s_first != s_last:
            # Get intermediate statuses
            path = [versions_data[i]["identify_data"][n].get(mode, "?") for i in range(len(versions_data))]
            transitions[f"{s_first} → {s_last}"].append((n, path))

    return transitions


def find_apples_to_apples(versions_data, mode, common_models):
    """Find models broken in ALL versions for normalized break count comparison."""
    broken_in_all = set(common_models)
    for vd in versions_data:
        broken_in_all = {n for n in broken_in_all if vd["identify_data"][n].get(mode) == "graph_break"}
    return broken_in_all


def print_report(versions_data, modes, common_models, show_details=False):
    """Print the full trend report."""
    labels = [vd["label"] for vd in versions_data]
    has_any_explain = any(vd.get("explain_data") for vd in versions_data)

    for mode in modes:
        stats = analyze_mode(versions_data, mode, common_models)

        print(f"\n{'='*70}")
        print(f"VERSION TREND — {mode.upper()} MODE ({len(common_models)} models)")
        print(f"{'='*70}\n")

        # Header
        col_w = max(len(l) for l in labels) + 2
        header = f"{'Metric':<35}" + "".join(f"{l:>{col_w}}" for l in labels)
        print(header)
        print("-" * len(header))

        # Rows
        rows = [
            ("Clean", "clean"),
            ("Graph break", "graph_break"),
            ("Eager error", "eager_error"),
            ("Create error", "create_error"),
            ("Timeout", "timeout"),
        ]
        for row_label, key in rows:
            vals = "".join(f"{s[key]:>{col_w}}" for s in stats)
            print(f"{row_label:<35}{vals}")

        print(f"{'---':<35}" + "".join(f"{'---':>{col_w}}" for _ in labels))
        vals = "".join(f"{s['testable']:>{col_w}}" for s in stats)
        print(f"{'Testable (clean + graph_break)':<35}{vals}")

        if has_any_explain:
            print()
            vals = "".join(f"{s['total_graph_breaks'] if s['total_graph_breaks'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Total graph breaks (explain)':<35}{vals}")
            vals = "".join(f"{s['explain_ok'] if s['explain_ok'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Explain OK':<35}{vals}")
            vals = "".join(f"{s['explain_error'] if s['explain_error'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Explain error':<35}{vals}")
            vals = "".join(f"{s['avg_breaks_per_broken'] if s['avg_breaks_per_broken'] is not None else 'N/A':>{col_w}}" for s in stats)
            print(f"{'Avg breaks per broken model':<35}{vals}")

        # Apples-to-apples comparison
        if has_any_explain:
            a2a = find_apples_to_apples(versions_data, mode, common_models)
            if a2a:
                print(f"\n{'─'*70}")
                print(f"APPLES-TO-APPLES: {len(a2a)} models broken in ALL versions ({mode})")
                print(f"{'─'*70}\n")

                a2a_header = f"{'Metric':<35}" + "".join(f"{l:>{col_w}}" for l in labels)
                print(a2a_header)
                print("-" * len(a2a_header))

                for vd in versions_data:
                    p2 = vd.get("explain_data")
                    if not p2:
                        continue

                a2a_breaks = []
                a2a_ok = []
                a2a_err = []
                for vd in versions_data:
                    p2 = vd.get("explain_data")
                    if not p2:
                        a2a_breaks.append(None)
                        a2a_ok.append(None)
                        a2a_err.append(None)
                        continue
                    tb = 0
                    ok = 0
                    err = 0
                    for n in a2a:
                        if n in p2 and mode in p2[n]:
                            r = p2[n][mode]
                            if r.get("status") == "ok":
                                ok += 1
                                tb += r.get("graph_break_count", 0)
                            elif r.get("status") == "explain_error":
                                err += 1
                    a2a_breaks.append(tb)
                    a2a_ok.append(ok)
                    a2a_err.append(err)

                vals = "".join(f"{b if b is not None else 'N/A':>{col_w}}" for b in a2a_breaks)
                print(f"{'Total graph breaks':<35}{vals}")
                vals = "".join(f"{o if o is not None else 'N/A':>{col_w}}" for o in a2a_ok)
                print(f"{'Explain OK':<35}{vals}")
                vals = "".join(f"{e if e is not None else 'N/A':>{col_w}}" for e in a2a_err)
                print(f"{'Explain error':<35}{vals}")
                if any(b is not None for b in a2a_breaks):
                    num_a2a = len(a2a)
                    vals = "".join(
                        f"{round(b/num_a2a, 1) if b is not None else 'N/A':>{col_w}}"
                        for b in a2a_breaks
                    )
                    print(f"{'Avg breaks per model':<35}{vals}")

        # Transitions
        transitions = find_transitions(versions_data, mode, common_models)
        if transitions:
            print(f"\n{'─'*70}")
            print(f"STATUS TRANSITIONS: {labels[0]} → {labels[-1]} ({mode})")
            print(f"{'─'*70}\n")

            for transition, models in sorted(transitions.items(), key=lambda x: -len(x[1])):
                print(f"  {transition}: {len(models)} models")
                if show_details:
                    for name, path in models:
                        path_str = " → ".join(path)
                        print(f"    {name}: {path_str}")

        # Fixed models detail (graph_break → clean)
        if show_details and has_any_explain:
            fixed = []
            for t, models in transitions.items():
                if "graph_break → clean" in t:
                    fixed.extend(models)
            if fixed:
                print(f"\n{'─'*70}")
                print(f"FIXED MODELS — break counts in {labels[0]} ({mode})")
                print(f"{'─'*70}\n")
                p2_first = versions_data[0].get("explain_data")
                if p2_first:
                    for name, path in sorted(fixed):
                        r = p2_first.get(name, {}).get(mode, {})
                        bc = r.get("graph_break_count", "?")
                        print(f"  {name}: {bc} breaks")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze graph break trends across PyTorch versions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("dirs", nargs="*", help="Version directories containing identify_results.json (auto-discovered if omitted)")
    parser.add_argument("--labels", nargs="*", help="Labels for each version directory")
    parser.add_argument("--train", action="store_true", help="Include train mode analysis")
    parser.add_argument("--details", action="store_true", help="Show per-model details for transitions")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    versions = find_version_dirs(args)
    if len(versions) < 2:
        print("ERROR: Need at least 2 versions to compare", file=sys.stderr)
        sys.exit(1)

    if not args.json:
        print(f"Comparing {len(versions)} versions: {', '.join(v['label'] for v in versions)}")
        for v in versions:
            print(f"  {v['label']}: identify={v['identify']}")
            if os.path.exists(v["explain"]):
                print(f"  {' ' * len(v['label'])}  explain={v['explain']}")

    # Load data
    for v in versions:
        v["identify_data"] = load_identify(v["identify"])
        v["explain_data"] = load_explain(v["explain"])

    # Find common models
    all_sets = [set(v["identify_data"].keys()) for v in versions]
    common_models = all_sets[0]
    for s in all_sets[1:]:
        common_models &= s

    modes = ["eval"]
    if args.train:
        modes.append("train")

    if args.json:
        output = {}
        for mode in modes:
            output[mode] = {
                "summary": analyze_mode(versions, mode, common_models),
                "transitions": {
                    k: [{"name": n, "path": p} for n, p in v]
                    for k, v in find_transitions(versions, mode, common_models).items()
                },
                "apples_to_apples_models": sorted(find_apples_to_apples(versions, mode, common_models)),
            }
        print(json.dumps(output, indent=2))
    else:
        print_report(versions, modes, common_models, show_details=args.details)


if __name__ == "__main__":
    main()
