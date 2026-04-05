#!/usr/bin/env python3
"""Compare old (explain) vs new (TORCH_LOGS + counting backend) graph break analysis.

Runs both approaches on the same model and reports differences.
This validates the migration before re-sweeping the full corpus.

Usage:
    # Compare a single model
    python3 tools/compare_explain_methods.py BartModel

    # Compare a list of models (recommended: mix of break counts)
    python3 tools/compare_explain_methods.py --auto

    # Compare specific models
    python3 tools/compare_explain_methods.py AriaModel MraModel FlaubertModel
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sweep"))

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def run_old_explain(model, inputs_dict, inputs_tuple, ctx):
    """Run the deprecated torch._dynamo.explain approach."""
    import torch._dynamo

    torch._dynamo.reset()
    start = time.perf_counter()

    with ctx:
        if inputs_tuple:
            explanation = torch._dynamo.explain(model)(*inputs_tuple)
        else:
            explanation = torch._dynamo.explain(model)(**inputs_dict)

    elapsed = round(time.perf_counter() - start, 3)

    reasons = []
    if hasattr(explanation, "break_reasons") and explanation.break_reasons:
        for br in explanation.break_reasons:
            reasons.append(str(getattr(br, "reason", str(br)))[:300])

    return {
        "graph_count": explanation.graph_count,
        "graph_break_count": explanation.graph_break_count,
        "break_reasons": reasons,
        "time_s": elapsed,
    }


def run_new_approach(model, inputs_dict, inputs_tuple, ctx):
    """Run the new TORCH_LOGS + counting backend approach."""
    import torch
    import torch._dynamo

    # Set up graph break log capture
    captured_messages = []

    class _BreakHandler(logging.Handler):
        def emit(self, record):
            captured_messages.append(record.getMessage())

    handler = _BreakHandler()
    dynamo_logger = logging.getLogger("torch._dynamo")
    dynamo_logger.addHandler(handler)

    try:
        import torch._logging
        torch._logging.set_logs(graph_breaks=True)
    except (ImportError, AttributeError):
        pass

    # Counting backend
    ops_per_graph = []

    def _counting_backend(gm, example_inputs):
        op_count = sum(1 for n in gm.graph.nodes
                       if n.op not in ('placeholder', 'output'))
        ops_per_graph.append(op_count)
        return gm.forward

    torch._dynamo.reset()
    start = time.perf_counter()

    compiled = torch.compile(model, backend=_counting_backend)
    with ctx:
        if inputs_tuple:
            compiled(*inputs_tuple)
        else:
            compiled(**inputs_dict)

    elapsed = round(time.perf_counter() - start, 3)

    # Cleanup
    dynamo_logger.removeHandler(handler)
    try:
        import torch._logging
        torch._logging.set_logs(graph_breaks=False)
    except (ImportError, AttributeError):
        pass

    return {
        "graph_count": len(ops_per_graph),
        "graph_break_count": max(0, len(ops_per_graph) - 1),
        "break_reasons": captured_messages,
        "ops_per_graph": ops_per_graph,
        "time_s": elapsed,
    }


def compare_results(model_name, old, new):
    """Compare old and new results, return a report dict."""
    report = {
        "model": model_name,
        "graph_count_match": old["graph_count"] == new["graph_count"],
        "break_count_match": old["graph_break_count"] == new["graph_break_count"],
        "old_graph_count": old["graph_count"],
        "new_graph_count": new["graph_count"],
        "old_break_count": old["graph_break_count"],
        "new_break_count": new["graph_break_count"],
        "old_reasons_count": len(old["break_reasons"]),
        "new_reasons_count": len(new["break_reasons"]),
        "old_time_s": old["time_s"],
        "new_time_s": new["time_s"],
    }

    # Check if break reasons overlap (substring matching since formats differ)
    if old["break_reasons"] and new["break_reasons"]:
        # For each old reason, check if any new reason contains similar keywords
        matched = 0
        for old_reason in old["break_reasons"]:
            # Extract key terms from old reason
            key_terms = [w.lower() for w in old_reason.split() if len(w) > 4][:5]
            for new_reason in new["break_reasons"]:
                new_lower = new_reason.lower()
                if any(term in new_lower for term in key_terms):
                    matched += 1
                    break
        report["reason_overlap_pct"] = round(100 * matched / len(old["break_reasons"]), 1)
    else:
        report["reason_overlap_pct"] = 100.0 if not old["break_reasons"] and not new["break_reasons"] else 0.0

    report["pass"] = report["graph_count_match"] and report["break_count_match"]
    return report


def get_auto_models():
    """Select ~10 models with varying break counts for comparison."""
    explain_path = os.path.join(REPO_ROOT, "sweep_results", "v2.10", "explain_results.json")
    with open(explain_path) as f:
        data = json.load(f)

    results = data["results"]
    break_models = [
        (r["name"], r.get("graph_break_count", 0))
        for r in results
        if r.get("graph_break_count", 0) > 0 and r.get("mode") == "eval"
    ]
    break_models.sort(key=lambda x: x[1])

    # Pick 3 low, 4 medium, 3 high
    n = len(break_models)
    selected = (
        break_models[:3] +
        break_models[n // 2 - 2: n // 2 + 2] +
        break_models[-3:]
    )
    return [m[0] for m in selected]


def main():
    parser = argparse.ArgumentParser(
        description="Compare old explain() vs new TORCH_LOGS approach"
    )
    parser.add_argument("models", nargs="*", help="Model names to compare")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select ~10 models with varying break counts")
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    if args.auto:
        model_names = get_auto_models()
        print(f"Auto-selected {len(model_names)} models for comparison")
    elif args.models:
        model_names = args.models
    else:
        parser.error("Provide model names or use --auto")

    # Enable TORCH_LOGS before importing torch
    os.environ["TORCH_LOGS"] = "+graph_breaks"

    import torch
    import torch._dynamo
    from worker import create_model

    # Load corpus for model lookup
    corpus_path = os.path.join(REPO_ROOT, "corpus", "corpus.json")
    with open(corpus_path) as f:
        corpus = json.load(f)
    corpus_models = {m["name"]: m for m in corpus["models"]}

    reports = []
    for model_name in model_names:
        if model_name not in corpus_models:
            print(f"SKIP: {model_name} not found in corpus")
            continue

        m = corpus_models[model_name]
        spec = {"name": m["name"], "source": m["source"]}
        print(f"\n{'='*70}")
        print(f"Model: {model_name} (mode={args.mode})")
        print(f"{'='*70}")

        try:
            model, inputs_dict, inputs_tuple = create_model(spec, args.device)
        except Exception as e:
            print(f"  CREATE ERROR: {e}")
            continue

        if args.mode == "train":
            model.train()
        else:
            model.eval()
        ctx = torch.no_grad() if args.mode == "eval" else torch.enable_grad()

        # Run old approach
        print("  Running old (explain)...", end="", flush=True)
        try:
            old = run_old_explain(model, inputs_dict, inputs_tuple, ctx)
            print(f" {old['graph_break_count']} breaks, {old['time_s']}s")
        except Exception as e:
            print(f" FAILED: {e}")
            old = None

        # Run new approach
        print("  Running new (TORCH_LOGS)...", end="", flush=True)
        try:
            new = run_new_approach(model, inputs_dict, inputs_tuple, ctx)
            print(f" {new['graph_break_count']} breaks, {new['time_s']}s")
        except Exception as e:
            print(f" FAILED: {e}")
            new = None

        if old and new:
            report = compare_results(model_name, old, new)
            reports.append(report)
            status = "PASS" if report["pass"] else "FAIL"
            print(f"  Result: {status}")
            print(f"    graph_count:  old={report['old_graph_count']}  new={report['new_graph_count']}  {'✓' if report['graph_count_match'] else '✗'}")
            print(f"    break_count:  old={report['old_break_count']}  new={report['new_break_count']}  {'✓' if report['break_count_match'] else '✗'}")
            print(f"    reasons:      old={report['old_reasons_count']}  new={report['new_reasons_count']}  overlap={report['reason_overlap_pct']}%")
            print(f"    time:         old={report['old_time_s']}s  new={report['new_time_s']}s")

        # Cleanup between models
        del model
        torch._dynamo.reset()
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # Summary
    if reports:
        passed = sum(1 for r in reports if r["pass"])
        total = len(reports)
        print(f"\n{'='*70}")
        print(f"SUMMARY: {passed}/{total} models passed comparison")
        print(f"{'='*70}")
        for r in reports:
            status = "PASS" if r["pass"] else "FAIL"
            print(f"  {status}  {r['model']:<35}  breaks: {r['old_break_count']}→{r['new_break_count']}  graphs: {r['old_graph_count']}→{r['new_graph_count']}")

        if args.json:
            print(f"\n--- JSON Report ---")
            print(json.dumps(reports, indent=2))

        if passed < total:
            print(f"\n⚠ {total - passed} model(s) have mismatches — investigate before full re-sweep")
            sys.exit(1)
        else:
            print(f"\nAll models match — safe to proceed with full re-sweep")


if __name__ == "__main__":
    main()
