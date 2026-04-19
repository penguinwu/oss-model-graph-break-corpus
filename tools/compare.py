#!/usr/bin/env python3
"""Compare two sweep results or corpus snapshots.

Usage:
    # Compare static vs dynamic=mark (from corpus)
    python tools/compare.py --corpus-dynamic

    # Compare two sweep result files
    python tools/compare.py sweep_results/static/identify_results.json sweep_results/dynamic_mark/identify_results.json

    # Compare with labels
    python tools/compare.py old.json new.json --labels "PyTorch 2.9" "PyTorch 2.10"
"""
import argparse
import json
import sys


def load_results(path):
    """Load results from JSON or JSONL file.

    Supports:
    - Standard JSON: array of results or {"results": [...]}
    - JSONL: one JSON object per line (sweep checkpoint format)
    """
    with open(path) as f:
        content = f.read()

    content = content.strip()
    if content.startswith('[') or (content.startswith('{') and '\n' not in content):
        # Standard JSON array, or single-line JSON object
        data = json.loads(content)
        results = data if isinstance(data, list) else data.get("results", [])
    else:
        # JSONL or multi-line JSON object — try JSON first, fall back to JSONL
        try:
            data = json.loads(content)
            results = data if isinstance(data, list) else data.get("results", [])
        except json.JSONDecodeError:
            results = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    by_key = {}
    for r in results:
        key = (r["name"], r.get("mode", "eval"))
        by_key[key] = r
    return by_key


def compare_two(results_a, results_b, label_a="A", label_b="B", mode="eval"):
    # Filter to requested mode
    a = {name: r for (name, m), r in results_a.items() if m == mode}
    b = {name: r for (name, m), r in results_b.items() if m == mode}

    all_names = sorted(set(a.keys()) | set(b.keys()))

    # Status distribution
    status_a, status_b = {}, {}
    for name in all_names:
        sa = a[name].get("status", "missing") if name in a else "missing"
        sb = b[name].get("status", "missing") if name in b else "missing"
        status_a[sa] = status_a.get(sa, 0) + 1
        status_b[sb] = status_b.get(sb, 0) + 1

    print(f"Comparison ({mode} mode): {label_a} vs {label_b}")
    print(f"{'Status':<20} {label_a:>10} {label_b:>10} {'Delta':>10}")
    print("-" * 55)
    all_statuses = sorted(set(list(status_a.keys()) + list(status_b.keys())))
    for s in all_statuses:
        ca = status_a.get(s, 0)
        cb = status_b.get(s, 0)
        d = cb - ca
        sign = "+" if d > 0 else ""
        print(f"{s:<20} {ca:>10} {cb:>10} {sign}{d:>9}")
    print()

    # Changed models
    changed = []
    for name in all_names:
        sa = a[name].get("status") if name in a else "missing"
        sb = b[name].get("status") if name in b else "missing"
        if sa != sb:
            changed.append((name, sa, sb))

    if changed:
        # Separate compile regressions from infra changes
        compile_regressions = []
        infra_changes = []
        other_changes = []
        for name, sa, sb in changed:
            if sa == "full_graph" and sb == "timeout":
                phase = b.get(name, {}).get("phase_at_timeout", "unknown")
                if phase == "compile":
                    compile_regressions.append((name, phase))
                else:
                    infra_changes.append((name, f"timeout at {phase}"))
            elif sa == "full_graph" and sb not in ("full_graph",):
                other_changes.append((name, sa, sb))
            else:
                other_changes.append((name, sa, sb))

        if compile_regressions:
            print(f"\n⚠ COMPILE REGRESSIONS ({len(compile_regressions)}):")
            for n, phase in sorted(compile_regressions):
                print(f"    {n}  (timeout during {phase})")

        print(f"\nChanged models ({len(changed)}):")
        from collections import defaultdict
        transitions = defaultdict(list)
        for name, sa, sb in changed:
            phase_info = ""
            if sb == "timeout" and name in b:
                phase = b[name].get("phase_at_timeout", "")
                if phase:
                    phase_info = f" [phase: {phase}]"
            transitions[(sa, sb)].append((name, phase_info))
        for (sa, sb), entries in sorted(transitions.items(), key=lambda x: -len(x[1])):
            print(f"\n  {sa} → {sb} ({len(entries)}):")
            for n, info in sorted(entries):
                print(f"    {n}{info}")
    else:
        print("No models changed status.")


def compare_corpus_dynamic():
    """Compare static vs dynamic=mark from the corpus file."""
    import os
    corpus_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "corpus", "corpus.json")
    with open(corpus_path) as f:
        corpus = json.load(f)

    for mode in ["eval", "train"]:
        static = {}
        dynamic = {}
        for m in corpus["models"]:
            if mode not in m:
                continue
            name = m["name"]
            static[(name, mode)] = {"name": name, "mode": mode, "status": m[mode].get("status")}
            dm = m[mode].get("dynamic_mark", {})
            if dm:
                dynamic[(name, mode)] = {"name": name, "mode": mode, "status": dm.get("status")}

        compare_two(static, dynamic, label_a="Static", label_b="Mark", mode=mode)
        print()


def main():
    parser = argparse.ArgumentParser(description="Compare sweep results")
    parser.add_argument("files", nargs="*", help="Two result JSON files to compare")
    parser.add_argument("--labels", nargs=2, default=["A", "B"], help="Labels for the two files")
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--corpus-dynamic", action="store_true",
                        help="Compare static vs dynamic=mark from corpus")
    args = parser.parse_args()

    if args.corpus_dynamic:
        compare_corpus_dynamic()
        return

    if len(args.files) != 2:
        parser.error("Provide exactly two result files to compare")

    a = load_results(args.files[0])
    b = load_results(args.files[1])
    compare_two(a, b, label_a=args.labels[0], label_b=args.labels[1], mode=args.mode)


if __name__ == "__main__":
    main()
