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


_RECOMPILE_REASONS_CAP = 20
_RECOMPILE_REASON_CHARS = 300
_METRICS_LIMIT = 10_000


def _metrics_api():
    """Return torch._dynamo.utils if it exposes the compilation-metrics API, else None."""
    try:
        import torch._dynamo.utils as dyn_utils
    except Exception:
        return None
    if not (hasattr(dyn_utils, "get_compilation_metrics")
            and hasattr(dyn_utils, "clear_compilation_metrics")):
        return None
    return dyn_utils


def clear_dynamo_state():
    """Reset Dynamo caches, compilation metrics and counters.

    Safe on any torch version: missing APIs are skipped. Called before every
    model run so earlier models (or calls) cannot affect recompile counts.
    """
    try:
        torch._dynamo.reset()
    except Exception:
        pass
    try:
        import torch._dynamo.utils as dyn_utils
    except Exception:
        return
    clear_metrics = getattr(dyn_utils, "clear_compilation_metrics", None)
    if clear_metrics is not None:
        try:
            clear_metrics()
        except Exception:
            pass
    counters = getattr(dyn_utils, "counters", None)
    if counters is not None:
        try:
            counters.clear()
        except Exception:
            pass
    set_limit = getattr(dyn_utils, "set_compilation_metrics_limit", None)
    if set_limit is not None:
        try:
            set_limit(_METRICS_LIMIT)
        except Exception:
            pass


def _read_metrics():
    """Return the list of CompilationMetrics, or None if unavailable."""
    api = _metrics_api()
    if api is None:
        return None
    try:
        return list(api.get_compilation_metrics())
    except Exception:
        return None


def _recompile_entries(metrics):
    return [m for m in metrics if getattr(m, "recompile_reason", None)]


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


def _record_recompiles(result, call):
    """Add recompile fields to `result`. `call` re-runs the compiled fn (call 2).

    Never raises: failures become None fields / recompile_steady_error.
    """
    try:
        result["recompile_limit"] = getattr(torch._dynamo.config, "recompile_limit", None)
    except Exception:
        result["recompile_limit"] = None

    metrics1 = _read_metrics()
    if metrics1 is None:
        result["recompile_metrics_source"] = "unavailable"
        result["frame_compile_count"] = None
        result["recompile_count"] = None
        result["recompile_count_steady"] = None
        result["recompile_reasons"] = None
        return
    result["recompile_metrics_source"] = "compilation_metrics"
    rc1 = _recompile_entries(metrics1)
    result["frame_compile_count"] = len(metrics1)
    result["recompile_count"] = len(rc1)

    rc2 = []
    try:
        call()
    except Exception as e:
        result["recompile_steady_error"] = f"{type(e).__name__}: {str(e)[:500]}"
        result["recompile_count_steady"] = None
    else:
        metrics2 = _read_metrics()
        if metrics2 is None:
            result["recompile_count_steady"] = None
        else:
            # The metrics store is append-only (capped deque; limit raised in
            # clear_dynamo_state), so entries past len(metrics1) are call 2's.
            rc2 = _recompile_entries(metrics2[len(metrics1):])
            result["recompile_count_steady"] = len(rc2)

    reasons, seen, truncated = [], set(), False
    for call_num, entries in ((1, rc1), (2, rc2)):
        for m in entries:
            co_name = getattr(m, "co_name", None)
            reason = str(m.recompile_reason)[:_RECOMPILE_REASON_CHARS]
            key = (co_name, reason, call_num)
            if key in seen:
                continue
            seen.add(key)
            if len(reasons) >= _RECOMPILE_REASONS_CAP:
                truncated = True
                break
            reasons.append({"co_name": co_name, "reason": reason, "call": call_num})
    result["recompile_reasons"] = reasons
    if truncated:
        result["recompile_reasons_truncated"] = True


def run_graph_break_analysis(model, inputs, mode="eval",
                              dynamic=False, compile_kwargs=None):
    """Analyze graph breaks using TORCH_LOGS + counting backend.

    This is the single, canonical way to analyze graph breaks across all
    suites in the corpus. Uses a lightweight counting backend that tracks
    subgraph count and ops per graph, plus TORCH_LOGS to capture structured
    break reasons.

    Args:
        model: The model (already on the right device, already in eval/train mode).
        inputs: dict of kwargs, tuple of args, or single tensor.
        mode: "eval" or "train" — controls torch.no_grad() context.
        dynamic: matches sweep/worker `--dynamic` semantics. True / "mark" / False.
        compile_kwargs: user-supplied torch.compile() kwargs (e.g. from
            `--compile-kwargs`). `fullgraph` and `backend` are forced — the
            counting backend can't run under fullgraph and must be _counting_backend.
            Other kwargs (notably `dynamic`) are honored.

    Returns:
        dict with keys:
            status: "ok" or "explain_error"
            graph_count: number of compiled subgraphs
            graph_break_count: graph_count - 1 (0 means full graph)
            break_reasons: list of {type, reason, file?, line?, location?}
            ops_per_graph: list of op counts per subgraph
            compile_times: list of compile time per subgraph
            explain_time_s: total wall time
            effective_compile_kwargs: the kwargs actually passed to torch.compile()
            frame_compile_count: Dynamo frame compiles during call 1 (None if
                the compilation-metrics API is unavailable)
            recompile_count: recompiles (guard failures) during call 1
            recompile_count_steady: recompiles during a 2nd same-input call
                (expected 0; >0 means unstable guards)
            recompile_reasons: deduped [{co_name, reason, call}] (cap 20)
            recompile_reasons_truncated?: True when the cap was hit
            recompile_limit: torch._dynamo.config.recompile_limit in effect
            recompile_metrics_source: "compilation_metrics" or "unavailable"
            recompile_steady_error?: error from call 2 (call-1 fields intact)
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

    clear_dynamo_state()
    ctx = torch.no_grad() if mode == "eval" else contextlib.nullcontext()
    start = time.perf_counter()

    # Do NOT set capture_scalar_outputs or capture_dynamic_output_shape_ops here.
    # While fullgraph=True captures these implicitly, setting them in the explain
    # pass causes PendingUnbackedSymbolNotFound crashes on models with complex
    # data-dependent shapes (VL models, Funnel, etc.). The explain pass should
    # report scalar ops as graph breaks, not try to trace through them.

    # Build effective compile_kwargs:
    # - Start from user-supplied (so dynamic=True etc. flow through)
    # - Force backend = counting backend (graph break count requires it)
    # - Force fullgraph = False (counting needs to continue past breaks)
    # - Default `dynamic` from the explicit `dynamic` arg if user didn't set it
    effective_kwargs = dict(compile_kwargs) if compile_kwargs else {}
    effective_kwargs["backend"] = _counting_backend
    effective_kwargs["fullgraph"] = False
    if "dynamic" not in effective_kwargs:
        # mirror worker's dynamic semantics: True = all dims symbolic, "mark" = realistic dims, False = static
        effective_kwargs["dynamic"] = True if dynamic is True else None
    # Record what we actually used (excluding the backend function — not JSON-friendly)
    result["effective_compile_kwargs"] = {
        k: v for k, v in effective_kwargs.items() if k != "backend"
    }

    try:
        compiled = torch.compile(model, **effective_kwargs)

        def _call():
            with ctx:
                if isinstance(inputs, dict):
                    compiled(**inputs)
                elif isinstance(inputs, tuple):
                    compiled(*inputs)
                else:
                    compiled(inputs)

        # Call 1 is inlined (not via _call) so graph-break stack traces keep
        # the same frame depth as before; break_reasons are capped at 2000
        # chars, so an extra harness frame would shift what gets truncated.
        with ctx:
            if isinstance(inputs, dict):
                compiled(**inputs)
            elif isinstance(inputs, tuple):
                compiled(*inputs)
            else:
                compiled(inputs)

        # Snapshot call-1 results BEFORE call 2: a recompile in call 2 would
        # append to ops_per_graph/compile_times and the break-log collector.
        result["graph_count"] = len(ops_per_graph)
        result["graph_break_count"] = max(0, len(ops_per_graph) - 1)
        result["ops_per_graph"] = list(ops_per_graph)
        result["compile_times"] = list(compile_times)
        result["break_reasons"] = collector.get_break_reasons()
        result["status"] = "ok"

        _record_recompiles(result, _call)

    except Exception as e:
        result["status"] = "explain_error"
        result["error"] = str(e)[:500]

    finally:
        collector.uninstall()

    result["explain_time_s"] = round(time.perf_counter() - start, 3)
    return result
