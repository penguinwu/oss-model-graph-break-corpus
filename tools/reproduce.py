#!/usr/bin/env python3
"""Reproduce a graph break for a single model.

Usage:
    python reproduce.py BartModel
    python reproduce.py BartModel --mode train
    python reproduce.py BartModel --mode train --device cpu
    python reproduce.py BartModel --explain          # show ALL graph breaks
    python reproduce.py BartModel --explain --verbose # include stack traces via TORCH_TRACE
    python reproduce.py BartModel --dynamic mark     # dynamic shapes (realistic dims)
    python reproduce.py BartModel --dynamic true     # dynamic shapes (all dims)
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
                        help="Show ALL graph breaks with reasons and doc links")
    parser.add_argument("--verbose", action="store_true",
                        help="Also generate TORCH_TRACE report (with --explain)")
    parser.add_argument("--dynamic", choices=["mark", "true"],
                        help="Enable dynamic shapes: 'mark' = batch+seq_len, 'true' = all dims")
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

    # Enable graph break logging before importing torch (env var must be set
    # before torch configures its logging system at import time)
    if args.explain:
        existing = os.environ.get("TORCH_LOGS", "")
        os.environ["TORCH_LOGS"] = f"{existing},+graph_breaks" if existing else "+graph_breaks"

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
    from worker import create_model, _mark_dynamic_dims

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
    if args.dynamic:
        print(f"Dynamic: {args.dynamic}")
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
        # Compile with graph break logging to show ALL breaks with doc links
        # (replaces deprecated torch._dynamo.explain)
        print()
        print("Compiling with graph break logging enabled...")
        print("(Break reasons will appear below as they are detected)")
        print()
        torch._dynamo.reset()

        # Apply dynamic shapes if requested
        compile_dynamic = None
        if args.dynamic == "mark":
            _mark_dynamic_dims(inputs_dict, inputs_tuple, spec["source"],
                               spec.get("input_type", "auto"))
            print("  (dynamic=mark: batch + seq_len dims marked dynamic)")
        elif args.dynamic == "true":
            compile_dynamic = True
            print("  (dynamic=True: all dims symbolic)")

        # Counting backend to track subgraphs and ops
        graph_ops = []

        def _counting_backend(gm, example_inputs):
            op_count = sum(1 for n in gm.graph.nodes
                           if n.op not in ('placeholder', 'output'))
            graph_ops.append(op_count)
            return gm.forward

        # If --verbose, also capture TORCH_TRACE
        trace_dir = None
        if args.verbose:
            import tempfile
            trace_dir = tempfile.mkdtemp(prefix="graph_break_trace_")
            os.environ["TORCH_TRACE"] = trace_dir

        # Match fullgraph=True behavior: capture item() and dynamic output shape
        # ops instead of graph-breaking on them (see torch/_dynamo/variables/tensor.py).
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.capture_dynamic_output_shape_ops = True

        try:
            compiled = torch.compile(model, backend=_counting_backend,
                                     dynamic=compile_dynamic)
            with ctx:
                if inputs_tuple:
                    compiled(*inputs_tuple)
                else:
                    compiled(**inputs_dict)
        except Exception as ex:
            print(f"  Compilation failed: {ex}")
            sys.exit(1)

        graph_count = len(graph_ops)
        break_count = max(0, graph_count - 1)

        print()
        print(f"  Graphs:       {graph_count}")
        print(f"  Graph breaks: {break_count}")

        if break_count == 0:
            print()
            print("No graph breaks — model compiles cleanly.")
            return

        print()
        print(f"Summary: {break_count} graph break(s) producing {graph_count} subgraph(s)")

        if trace_dir:
            print()
            print(f"TORCH_TRACE captured to: {trace_dir}")
            print(f"Generate HTML report:  tlparse {trace_dir}")
        else:
            print()
            print("For richer analysis with stack traces and graph IR:")
            print(f"  python3 tools/reproduce.py {args.model} --mode {args.mode} --explain --verbose")
            print("Or manually:")
            print(f"  TORCH_TRACE=/tmp/trace python3 tools/reproduce.py {args.model} --mode {args.mode}")
            print("  tlparse /tmp/trace")
        return

    # Step 3: Compile with fullgraph=True
    compile_dynamic = None
    dynamic_label = ""
    if args.dynamic == "mark":
        _mark_dynamic_dims(inputs_dict, inputs_tuple, spec["source"],
                           spec.get("input_type", "auto"))
        dynamic_label = ", dynamic=mark"
    elif args.dynamic == "true":
        compile_dynamic = True
        dynamic_label = ", dynamic=True"

    print(f"Running torch.compile(fullgraph=True, backend='eager'{dynamic_label})...")
    torch._dynamo.reset()
    # Skip logging methods that are pure side effects
    import logging
    for method_name in (
        "debug", "info", "warning", "warn", "error", "critical",
        "fatal", "log", "exception", "warning_once",
    ):
        method = getattr(logging.Logger, method_name, None)
        if method is not None:
            torch._dynamo.config.ignore_logging_functions.add(method)
    compiled = torch.compile(model, fullgraph=True, backend="eager", dynamic=compile_dynamic)
    try:
        with ctx:
            if inputs_tuple:
                compiled(*inputs_tuple)
            else:
                compiled(**inputs_dict)
        print("  Compile: FULL_GRAPH (no graph break)")
    except Exception as e:
        print(f"  Compile: GRAPH BREAK")
        print(f"  Error: {e}")
        print()
        print("  For detailed break analysis, run with --explain:")
        dyn_flag = f" --dynamic {args.dynamic}" if args.dynamic else ""
        print(f"  python3 tools/reproduce.py {args.model} --mode {args.mode}{dyn_flag} --explain")


if __name__ == "__main__":
    main()
