#!/usr/bin/env python3
"""Reproduce a graph break for a single model.

Usage:
    python reproduce.py BartModel
    python reproduce.py BartModel --mode train
    python reproduce.py BartModel --mode train --device cpu
"""
import argparse
import json
import sys
import os

# Add sweep dir to path for model creation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sweep"))

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Reproduce a graph break for a model")
    parser.add_argument("model", help="Model name (e.g., BartModel)")
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--list", action="store_true",
                        help="List all graph-break models from the corpus")
    args = parser.parse_args()

    if args.list:
        corpus_path = os.path.join(REPO_ROOT, "corpus", "corpus.json")
        with open(corpus_path) as f:
            corpus = json.load(f)
        for m in corpus["models"]:
            if m["has_graph_break"]:
                ev = m["eval"]["status"]
                tr = m["train"]["status"]
                print(f"{m['source']}/{m['name']:30s}  eval={ev:12s}  train={tr}")
        return

    import torch
    import torch._dynamo
    from worker import create_model

    # Look up model in corpus to get source
    corpus_path = os.path.join(os.path.dirname(__file__), "corpus", "corpus.json")
    spec = None
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            corpus = json.load(f)
        for m in corpus["models"]:
            if m["name"] == args.model:
                spec = {"name": m["name"], "source": m["source"]}
                break

    if spec is None:
        # Try hf by default
        spec = {"name": args.model, "source": "hf"}
        print(f"Model not found in corpus, assuming source=hf")

    print(f"Model: {spec['name']} (source={spec['source']})")
    print(f"Mode:  {args.mode}")
    print(f"Device: {args.device}")
    print()

    # Step 1: Create model
    print("Creating model...")
    model, inputs_dict, inputs_tuple = create_model(spec, args.device)

    if args.mode == "train":
        model.train()
    else:
        model.eval()
    ctx = torch.no_grad() if args.mode == "eval" else torch.enable_grad()

    # Step 2: Eager forward
    print("Running eager forward...")
    torch._dynamo.reset()
    with ctx:
        if inputs_tuple:
            model(*inputs_tuple)
        else:
            model(**inputs_dict)
    print("  Eager: OK")

    # Step 3: Compile with fullgraph=True
    print("Running torch.compile(fullgraph=True, backend='eager')...")
    torch._dynamo.reset()
    compiled = torch.compile(model, fullgraph=True, backend="eager")
    try:
        with ctx:
            if inputs_tuple:
                compiled(*inputs_tuple)
            else:
                compiled(**inputs_dict)
        print("  Compile: CLEAN (no graph break)")
    except Exception as e:
        print(f"  Compile: GRAPH BREAK")
        print(f"  Error: {e}")

        # Step 4: Run explain() for details
        print()
        print("Running torch._dynamo.explain() for details...")
        torch._dynamo.reset()
        try:
            with ctx:
                if inputs_tuple:
                    explanation = torch._dynamo.explain(model)(*inputs_tuple)
                else:
                    explanation = torch._dynamo.explain(model)(**inputs_dict)
            print(f"  Graphs: {explanation.graph_count}")
            print(f"  Graph breaks: {explanation.graph_break_count}")
            if hasattr(explanation, "break_reasons") and explanation.break_reasons:
                print(f"  Break reasons:")
                for br in explanation.break_reasons:
                    reason = str(getattr(br, "reason", str(br)))
                    print(f"    - {reason[:200]}")
        except Exception as ex:
            print(f"  explain() failed: {ex}")


if __name__ == "__main__":
    main()
