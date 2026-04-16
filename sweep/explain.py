#!/usr/bin/env python3
"""Shared graph break analysis — single implementation for all suites.

Every suite (HF, diffusers, timm, custom) calls the same function to
analyze graph breaks. This ensures consistent methodology and output
format across the entire corpus.

Usage:
    from explain import run_graph_break_analysis

    result = run_graph_break_analysis(model, inputs, mode="eval")
    # result is a dict with: graph_count, graph_break_count, break_reasons,
    #   ops_per_graph, compile_times, explain_time_s, status
"""
import contextlib
import logging
import re
import time

import torch
import torch._dynamo


class _BreakCollector:
    """Logging handler that captures graph break messages from TORCH_LOGS."""

    def __init__(self):
        self.messages = []
        self._handler = None

    def install(self):
        class _Handler(logging.Handler):
            def __init__(self, collector):
                super().__init__()
                self.collector = collector

            def emit(self, record):
                self.collector.messages.append(record.getMessage())

        self._handler = _Handler(self)
        logger = logging.getLogger("torch._dynamo")
        logger.addHandler(self._handler)

        try:
            import torch._logging
            torch._logging.set_logs(graph_breaks=True)
        except (ImportError, AttributeError):
            pass

    def uninstall(self):
        if self._handler:
            logging.getLogger("torch._dynamo").removeHandler(self._handler)
        try:
            import torch._logging
            torch._logging.set_logs(graph_breaks=False)
        except (ImportError, AttributeError):
            pass

    def get_break_reasons(self):
        """Parse captured log messages into structured break_reasons."""
        _site_pkg_re = re.compile(r'/[^\s]*/site-packages/')
        reasons = []
        for msg in self.messages:
            cleaned = _site_pkg_re.sub('', msg)
            info = {"reason": cleaned[:2000]}

            # Classify break type
            if "nonzero" in cleaned:
                info["type"] = "aten.nonzero"
            elif "item()" in cleaned.lower() or "scalar" in cleaned.lower():
                info["type"] = "Tensor.item()"
            elif "generic_jump" in cleaned or "Data-dependent branching" in cleaned:
                info["type"] = "data-dependent-branch"
            else:
                info["type"] = "other"

            # Extract file:line location
            loc = re.search(r'(\S+\.py):(\d+)', msg)
            if loc:
                info["file"] = loc.group(1)
                info["line"] = int(loc.group(2))

            # Extract file/line/function from FrameSummary format
            frames = re.findall(r'file (\S+\.py), line (\d+) in (\w+)', msg)
            if frames:
                info["location"] = f"{frames[-1][0]}:{frames[-1][1]} in {frames[-1][2]}"

            reasons.append(info)
        return reasons


def run_graph_break_analysis(model, inputs, mode="eval"):
    """Analyze graph breaks using TORCH_LOGS + counting backend.

    This is the single, canonical way to analyze graph breaks across all
    suites in the corpus. Uses a lightweight counting backend that tracks
    subgraph count and ops per graph, plus TORCH_LOGS to capture structured
    break reasons.

    Args:
        model: The model (already on the right device, already in eval/train mode).
        inputs: dict of kwargs, tuple of args, or single tensor.
        mode: "eval" or "train" — controls torch.no_grad() context.

    Returns:
        dict with keys:
            status: "ok" or "explain_error"
            graph_count: number of compiled subgraphs
            graph_break_count: graph_count - 1 (0 means full graph)
            break_reasons: list of {type, reason, file?, line?, location?}
            ops_per_graph: list of op counts per subgraph
            compile_times: list of compile time per subgraph
            explain_time_s: total wall time
            error?: error message if status is "explain_error"
    """
    result = {}
    collector = _BreakCollector()
    collector.install()

    ops_per_graph = []
    compile_times = []

    def _counting_backend(gm, example_inputs):
        start_t = time.perf_counter()
        op_count = sum(1 for n in gm.graph.nodes
                       if n.op not in ('placeholder', 'output'))
        ops_per_graph.append(op_count)
        compile_times.append(round(time.perf_counter() - start_t, 3))
        return gm.forward

    torch._dynamo.reset()
    ctx = torch.no_grad() if mode == "eval" else contextlib.nullcontext()
    start = time.perf_counter()

    # Do NOT set capture_scalar_outputs or capture_dynamic_output_shape_ops here.
    # While fullgraph=True captures these implicitly, setting them in the explain
    # pass causes PendingUnbackedSymbolNotFound crashes on models with complex
    # data-dependent shapes (VL models, Funnel, etc.). The explain pass should
    # report scalar ops as graph breaks, not try to trace through them.

    try:
        compiled = torch.compile(model, backend=_counting_backend)
        with ctx:
            if isinstance(inputs, dict):
                compiled(**inputs)
            elif isinstance(inputs, tuple):
                compiled(*inputs)
            else:
                compiled(inputs)

        result["graph_count"] = len(ops_per_graph)
        result["graph_break_count"] = max(0, len(ops_per_graph) - 1)
        result["ops_per_graph"] = ops_per_graph
        result["compile_times"] = compile_times
        result["break_reasons"] = collector.get_break_reasons()
        result["status"] = "ok"

    except Exception as e:
        result["status"] = "explain_error"
        result["error"] = str(e)[:500]

    finally:
        collector.uninstall()

    result["explain_time_s"] = round(time.perf_counter() - start, 3)
    return result
