#!/usr/bin/env python3
"""Query the graph break corpus.

Usage:
    python tools/query.py                          # Summary
    python tools/query.py --status graph_break     # All graph break models
    python tools/query.py --error deepcopy         # Models with 'deepcopy' in error
    python tools/query.py --compare-dynamic        # Static vs dynamic=mark comparison
    python tools/query.py --json                   # Machine-readable output
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS_PATH = os.path.join(REPO_ROOT, "corpus", "corpus.json")


def load_corpus():
    with open(CORPUS_PATH) as f:
        return json.load(f)


def print_summary(corpus):
    models = corpus["models"]
    print(f"Corpus: {len(models)} models")
    print()

    for mode in ["eval", "train"]:
        by_status = {}
        for m in models:
            if mode in m:
                s = m[mode].get("status", "unknown")
                by_status[s] = by_status.get(s, 0) + 1
        print(f"{mode.upper()} mode:")
        for s, c in sorted(by_status.items(), key=lambda x: -x[1]):
            pct = c / len(models) * 100
            print(f"  {s:<20} {c:>4} ({pct:.0f}%)")
        print()

    # Dynamic mark summary if available
    has_dynamic = any("dynamic_mark" in m.get("eval", {}) for m in models)
    if has_dynamic:
        print("Dynamic=mark (eval):")
        dm_status = {}
        for m in models:
            dm = m.get("eval", {}).get("dynamic_mark", {})
            if dm:
                s = dm.get("status", "unknown")
                dm_status[s] = dm_status.get(s, 0) + 1
        for s, c in sorted(dm_status.items(), key=lambda x: -x[1]):
            pct = c / len(models) * 100
            print(f"  {s:<20} {c:>4} ({pct:.0f}%)")
        print()


def filter_models(corpus, status=None, error=None, mode="eval"):
    results = []
    for m in corpus["models"]:
        if mode not in m:
            continue
        rec = m[mode]
        if status and rec.get("status") != status:
            continue
        if error:
            err_text = rec.get("fullgraph_error", "") or rec.get("error", "")
            if error.lower() not in err_text.lower():
                continue
        results.append(m)
    return results


def compare_dynamic(corpus):
    print(f"{'Model':<35} {'Static':>12} {'Mark':>12} {'Change'}")
    print("-" * 75)
    changed = []
    for m in corpus["models"]:
        static = m.get("eval", {}).get("status")
        dm = m.get("eval", {}).get("dynamic_mark", {})
        mark = dm.get("status") if dm else None
        if static and mark and static != mark:
            changed.append((m["name"], static, mark))
            print(f"{m['name']:<35} {static:>12} {mark:>12}   {'NEW' if mark == 'graph_break' and static == 'clean' else ''}")
    print(f"\n{len(changed)} models changed status with dynamic=mark")


def main():
    parser = argparse.ArgumentParser(description="Query the graph break corpus")
    parser.add_argument("--status", help="Filter by status (clean, graph_break, eager_error, ...)")
    parser.add_argument("--error", help="Search error messages (e.g., 'deepcopy', 'Logger')")
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--compare-dynamic", action="store_true", help="Compare static vs dynamic=mark")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    corpus = load_corpus()

    if args.compare_dynamic:
        compare_dynamic(corpus)
        return

    if not args.status and not args.error:
        print_summary(corpus)
        return

    results = filter_models(corpus, status=args.status, error=args.error, mode=args.mode)

    if args.json:
        print(json.dumps([{"name": m["name"], "source": m["source"],
                           "status": m[args.mode]["status"],
                           "error": m[args.mode].get("fullgraph_error", "")[:200]}
                          for m in results], indent=2))
    else:
        print(f"Found {len(results)} models:")
        for m in results:
            rec = m[args.mode]
            err = (rec.get("fullgraph_error", "") or rec.get("error", ""))[:80]
            print(f"  {m['source']}/{m['name']:<30} {rec['status']:<15} {err}")


if __name__ == "__main__":
    main()
