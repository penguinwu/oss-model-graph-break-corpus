#!/usr/bin/env python3
"""Query the graph break corpus.

Usage:
    python tools/query.py                          # Summary
    python tools/query.py --status graph_break     # All graph break models
    python tools/query.py --error deepcopy         # Models with 'deepcopy' in error
    python tools/query.py --source hf              # Filter by source
    python tools/query.py --compare-dynamic        # Static vs dynamic=mark comparison
    python tools/query.py --top-errors             # Top error categories
    python tools/query.py --mode-diff              # Models that differ between eval/train
    python tools/query.py --json                   # Machine-readable output
    python tools/query.py --error deepcopy --output models.json  # Export for targeted sweep
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
CORPUS_PATH = os.path.join(REPO_ROOT, "corpus", "corpus.json")


def load_corpus():
    with open(CORPUS_PATH) as f:
        return json.load(f)


def _get_error(rec):
    """Get error text from a record, checking both field names."""
    return rec.get("fullgraph_error", "") or rec.get("error", "")


def _classify_error(error_text):
    """Classify an error string into a category."""
    e = error_text.lower()
    if "deepcopy" in e:
        return "copy.deepcopy"
    if "marked as skipped" in e:
        return "skipped function call"
    if "mark_static_address" in e:
        return "forbidden callable (mark_static_address)"
    if "data-dependent" in e or "data dependent" in e:
        if "guard" in e or "constraint" in e:
            return "data-dependent guard"
        return "data-dependent branching"
    if "logger" in e and "logging" not in e.split("hint")[0] if "hint" in e else "logger" in e:
        return "logging.Logger"
    if "as_proxy" in e or "proxy" in e:
        return "proxy conversion failure"
    if "requires_grad" in e and "setattr" in e:
        return "requires_grad mutation"
    if "callable" in e and "builtin" in e:
        return "builtin callable"
    if "context manager" in e or "ContextManag" in e.replace(" ", ""):
        return "unsupported context manager"
    if "fake tensor" in e or "faketensor" in e.replace(" ", ""):
        return "fake tensor error"
    if "non-tensor" in e or "non-Tensor" in error_text:
        return "non-Tensor return"
    if "observed exception" in e:
        return "observed exception"
    if "unbacked" in e or "symint" in e:
        return "unbacked SymInt"
    if not error_text.strip():
        return "(no error text)"
    return "other"


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


def filter_models(corpus, status=None, error=None, mode="eval", source=None):
    results = []
    for m in corpus["models"]:
        if source and m.get("source") != source:
            continue
        if mode not in m:
            continue
        rec = m[mode]
        if status and rec.get("status") != status:
            continue
        if error:
            err_text = _get_error(rec)
            if error.lower() not in err_text.lower():
                continue
        results.append(m)
    return results


def compare_dynamic(corpus, mode="eval"):
    print(f"{'Model':<35} {'Static':>12} {'Mark':>12} {'Change'}")
    print("-" * 75)
    changed = []
    for m in corpus["models"]:
        static = m.get(mode, {}).get("status")
        dm = m.get(mode, {}).get("dynamic_mark", {})
        mark = dm.get("status") if dm else None
        if static and mark and static != mark:
            changed.append((m["name"], static, mark))
            print(f"{m['name']:<35} {static:>12} {mark:>12}   {'NEW' if mark == 'graph_break' and static == 'full_graph' else ''}")
    print(f"\n{len(changed)} models changed status with dynamic=mark")


def top_errors(corpus, mode="eval", n=10):
    categories = Counter()
    examples = {}
    for m in corpus["models"]:
        rec = m.get(mode, {})
        if rec.get("status") != "graph_break":
            continue
        err = _get_error(rec)
        cat = _classify_error(err)
        categories[cat] += 1
        if cat not in examples:
            examples[cat] = m["name"]

    print(f"Top error categories ({mode} mode, {sum(categories.values())} graph break models):")
    print(f"{'#':<4} {'Category':<35} {'Count':>6} {'Example'}")
    print("-" * 80)
    for i, (cat, count) in enumerate(categories.most_common(n), 1):
        print(f"{i:<4} {cat:<35} {count:>6}   {examples[cat]}")


def mode_diff(corpus):
    eval_only = []
    train_only = []
    for m in corpus["models"]:
        e_status = m.get("eval", {}).get("status")
        t_status = m.get("train", {}).get("status")
        if e_status == "graph_break" and t_status != "graph_break":
            eval_only.append((m["name"], t_status))
        elif t_status == "graph_break" and e_status != "graph_break":
            train_only.append((m["name"], e_status))

    if eval_only:
        print(f"Graph break in EVAL only ({len(eval_only)} models):")
        for name, other in sorted(eval_only):
            print(f"  {name:<35} train={other}")
        print()

    if train_only:
        print(f"Graph break in TRAIN only ({len(train_only)} models):")
        for name, other in sorted(train_only):
            print(f"  {name:<35} eval={other}")
        print()

    both = sum(
        1 for m in corpus["models"]
        if m.get("eval", {}).get("status") == "graph_break"
        and m.get("train", {}).get("status") == "graph_break"
    )
    print(f"Summary: {len(eval_only)} eval-only, {len(train_only)} train-only, {both} both")


def _model_to_json(m, mode):
    """Build a full-field JSON record for a model."""
    rec = m.get(mode, {})
    result = {
        "name": m["name"],
        "source": m.get("source", ""),
        "status": rec.get("status", ""),
        "error": _get_error(rec)[:200],
        "error_category": _classify_error(_get_error(rec)),
    }
    for field in ["compile_time_s", "create_time_s", "eager_time_s",
                  "graph_break_count", "graph_count", "wall_time_s"]:
        if field in rec:
            result[field] = rec[field]
    dm = rec.get("dynamic_mark", {})
    if dm:
        result["dynamic_mark_status"] = dm.get("status", "")
    return result


def main():
    parser = argparse.ArgumentParser(description="Query the graph break corpus")
    parser.add_argument("--status", help="Filter by status (full_graph, graph_break, eager_error, ...)")
    parser.add_argument("--error", help="Search error messages (e.g., 'deepcopy', 'Logger')")
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--source", help="Filter by source (hf, diffusers)")
    parser.add_argument("--compare-dynamic", action="store_true", help="Compare static vs dynamic=mark")
    parser.add_argument("--top-errors", action="store_true", help="Show top error categories")
    parser.add_argument("--mode-diff", action="store_true", help="Show models that differ between eval and train")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--output", "-o", help="Write matching models to JSON file (for targeted sweeps)")
    args = parser.parse_args()

    corpus = load_corpus()

    if args.compare_dynamic:
        compare_dynamic(corpus, mode=args.mode)
        return

    if args.top_errors:
        top_errors(corpus, mode=args.mode)
        return

    if args.mode_diff:
        mode_diff(corpus)
        return

    # Summary mode (no filters) — respect --json
    if not args.status and not args.error and not args.source:
        if args.json:
            print(json.dumps([_model_to_json(m, args.mode) for m in corpus["models"]], indent=2))
        else:
            print_summary(corpus)
        return

    results = filter_models(corpus, status=args.status, error=args.error,
                            mode=args.mode, source=args.source)

    if args.output:
        # Export as sweep-compatible JSON (name + source, for run_sweep.py --models)
        export = [{"name": m["name"], "source": m.get("source", "hf")} for m in results]
        with open(args.output, "w") as f:
            json.dump(export, f, indent=2)
            f.write("\n")
        print(f"Wrote {len(export)} models to {args.output}")
        return

    if args.json:
        print(json.dumps([_model_to_json(m, args.mode) for m in results], indent=2))
    else:
        print(f"Found {len(results)} models:")
        for m in results:
            rec = m[args.mode]
            err = _get_error(rec)[:80]
            print(f"  {m['source']}/{m['name']:<30} {rec['status']:<15} {err}")


if __name__ == "__main__":
    main()
