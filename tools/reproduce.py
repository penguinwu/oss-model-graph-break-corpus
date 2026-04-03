#!/usr/bin/env python3
"""Reproduce a graph break for a single model.

Usage:
    python reproduce.py BartModel
    python reproduce.py BartModel --mode train
    python reproduce.py BartModel --mode train --device cpu
    python reproduce.py BartModel --explain          # show ALL graph breaks
    python reproduce.py BartModel --explain --verbose # include user stack traces
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
    parser.add_argument("model", nargs="?", help="Model name (e.g., BartModel)")
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--list", action="store_true",
                        help="List all graph-break models from the corpus")
    parser.add_argument("--explain", action="store_true",
                        help="Run torch._dynamo.explain() to show ALL graph breaks")
    parser.add_argument("--verbose", action="store_true",
                        help="Show user stack traces for each break (with --explain)")
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

    if not args.model:
        parser.error("model name is required (e.g., reproduce.py BartModel)")

    try:
        import torch
    except ImportError:
        print("Error: PyTorch not found.")
        print("Install the required packages:")
        print("  pip install torch==2.10.0 transformers==5.4.0 diffusers==0.37.1")
        print()
        print("Or create a virtual environment:")
        print("  python -m venv env && source env/bin/activate")
        print("  pip install torch==2.10.0 transformers==5.4.0 diffusers==0.37.1")
        sys.exit(1)

    import torch._dynamo
    from worker import create_model

    # Look up model in corpus to get source
    corpus_path = os.path.join(REPO_ROOT, "corpus", "corpus.json")
    corpus_models = {}
    if os.path.exists(corpus_path):
        with open(corpus_path) as f:
            corpus = json.load(f)
        for m in corpus["models"]:
            corpus_models[m["name"]] = m

    spec = None
    if args.model in corpus_models:
        m = corpus_models[args.model]
        spec = {"name": m["name"], "source": m["source"]}
    else:
        # Check for close matches
        close = [n for n in corpus_models if args.model.lower() in n.lower()]
        if close:
            print(f"Model '{args.model}' not found in corpus. Similar names:")
            for n in close[:10]:
                print(f"  {n}")
            if len(close) == 1:
                print(f"\nDid you mean '{close[0]}'?")
            sys.exit(1)
        else:
            print(f"Model '{args.model}' not found in corpus.")
            print("Use --list to see all available models.")
            sys.exit(1)

    # Check if diffusers model is supported
    if spec["source"] == "diffusers":
        print(f"Note: Diffusers model reproduction requires model-specific input configs.")
        print(f"Only 5 Diffusers families are currently supported.")
        print()

    print(f"Model: {spec['name']} (source={spec['source']})")
    print(f"Mode:  {args.mode}")
    print(f"Device: {args.device}")
    print()

    # Step 1: Create model
    print("Creating model...")
    try:
        model, inputs_dict, inputs_tuple = create_model(spec, args.device)
    except Exception as e:
        print(f"Error creating model: {e}")
        if spec["source"] == "diffusers":
            print("This Diffusers model may not have a supported constructor config.")
        sys.exit(1)

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

    if args.explain:
        # --explain mode: skip fullgraph compile, go straight to explain()
        # This shows ALL graph breaks, not just the first one
        print()
        print("Running torch._dynamo.explain()...")
        torch._dynamo.reset()
        try:
            with ctx:
                if inputs_tuple:
                    explanation = torch._dynamo.explain(model)(*inputs_tuple)
                else:
                    explanation = torch._dynamo.explain(model)(**inputs_dict)
        except Exception as ex:
            print(f"  explain() failed: {ex}")
            sys.exit(1)

        print(f"  Graphs:       {explanation.graph_count}")
        print(f"  Graph breaks: {explanation.graph_break_count}")

        if explanation.graph_break_count == 0:
            print()
            print("No graph breaks — model compiles cleanly.")
            return

        print()
        if hasattr(explanation, "break_reasons") and explanation.break_reasons:
            print(f"Break reasons ({len(explanation.break_reasons)} total):")
            print("-" * 70)
            for i, br in enumerate(explanation.break_reasons, 1):
                reason = str(getattr(br, "reason", str(br)))
                print(f"\n  [{i}] {reason}")

                if args.verbose and hasattr(br, "user_stack"):
                    stack = br.user_stack
                    if stack:
                        print("      User stack:")
                        for frame in stack:
                            if hasattr(frame, "filename"):
                                print(f"        {frame.filename}:{frame.lineno} in {frame.name}")
                            else:
                                print(f"        {frame}")
            print()
        else:
            print("  (no break_reasons attribute — older PyTorch version?)")

        # Summary
        if explanation.graph_break_count > 0:
            print(f"Summary: {explanation.graph_break_count} graph break(s) "
                  f"producing {explanation.graph_count} subgraph(s)")
            print(f"Tip: Use TORCH_TRACE=/tmp/trace to capture full compilation artifacts,")
            print(f"     then run: tlparse /tmp/trace")
        return

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
            print()
            print(f"  Tip: Run with --explain for full break details")
        except Exception as ex:
            print(f"  explain() failed: {ex}")


if __name__ == "__main__":
    main()
