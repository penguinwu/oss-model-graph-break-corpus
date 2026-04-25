"""measure_perf — per-trial performance primitive for discovery agent + eval.

A *primitive*, not a sweep. Skill-eval, discovery harnesses, and one-off
evaluations (e.g. DeepSeek V4 Pro) call this on one model at a time and get
back eager / compiled latency, peak memory, compile time, and sanity numbers.

Designed to be cheap and self-contained. No corpus framework dependency.

Methodology aligns with `pytorch/pytorch/benchmarks/dynamo/common.py`
(latency_experiment shape) so numbers are comparable to upstream HF dashboards:

  - RNG determinism: patch_torch_manual_seed (always seed 1337) applied at
    process start. Per Animesh's note + common.py:540 — without this, HF models'
    internal randomness (dropout init, sampling) makes accuracy comparisons
    suspect and tier-1 perf show false variance.

  - Two timing layers: warmup (cold latency + peak mem capture) → steady-state
    timed loop with `--repeat 30` median (matches common.py default).

  - Cold vs warm peak memory: cold (default) captures one-time allocations;
    warm = post-warmup peak. Pass `warm_peak_mem=True` to capture both. Don't
    compare cold-vs-warm across runs — pick one and document.

  - Real compile time: uses `torch._dynamo.utils.compile_times()` for the
    canonical compile breakdown (vs the wall-time heuristic of the first call).
    Both are reported.

  - eager_self_check: runs eager twice, compares outputs. Detects models that
    are non-deterministic even with the seed patch. Equivalent to common.py's
    `non_deterministic_models` set, but per-trial.

Usage
-----
    from discovery.perf import measure_perf, patch_torch_manual_seed

    patch_torch_manual_seed()  # call ONCE at process start

    def model_fn():
        return MyModel(config).eval().cuda()

    def inputs_fn(model):
        return {"input_ids": torch.randint(0, 256, (1, 16), device="cuda")}

    result = measure_perf(model_fn, inputs_fn, n_warmup=10, n_repeat=30)
    # -> {"eager_ms": 12.4, "compiled_ms": 8.1, "speedup": 1.53,
    #     "compile_s": 7.84, "compile_times": {...}, ...}
"""
from __future__ import annotations

import functools
import gc
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch


@dataclass
class PerfResult:
    eager_ms: float
    compiled_ms: float
    speedup: float
    eager_peak_mem_mb: float
    compiled_peak_mem_mb: float
    n_warmup: int
    n_repeat: int
    device: str
    compile_s: float | None = None  # wall time of first compiled call
    compile_times: dict | None = None  # torch._dynamo.utils.compile_times() breakdown
    eager_warm_peak_mem_mb: float | None = None  # post-warmup peak (if warm_peak_mem=True)
    compiled_warm_peak_mem_mb: float | None = None
    eager_self_diff: float | None = None  # max_abs_diff between two eager runs (None if not checked)
    eager_deterministic: bool | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ----- RNG determinism (per pytorch/benchmarks/dynamo/common.py:540) -----

@functools.cache
def patch_torch_manual_seed(seed: int = 1337) -> None:
    """Monkey-patch torch.manual_seed to always seed deterministically.

    HF models call torch.manual_seed internally during forward (dropout init,
    sampling, layer-norm noise variants). Without this patch, accuracy
    comparisons are suspect and tier-1 perf shows false variance.

    Call ONCE at process start. Idempotent (functools.cache).

    Mirrors common.py:540 patch_torch_manual_seed exactly.
    """

    def deterministic_torch_manual_seed(*args, **kwargs):
        from torch._C import default_generator

        if torch.cuda.is_available():
            if not torch.cuda._is_in_bad_fork():
                torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            if not torch.xpu._is_in_bad_fork():
                torch.xpu.manual_seed_all(seed)
        return default_generator.manual_seed(seed)

    torch.manual_seed = deterministic_torch_manual_seed


# Compile time IS measured (compile_s) but kept SEPARATE from compiled_ms.
# - compiled_ms: steady-state median over n_repeat post-warmup iterations.
#   This is what the discovery rubric scores against (fix quality).
# - compile_s: wall time of the first compiled call (compile + 1 forward).
#   Corpus-level signal (e.g. "is compile getting slower across PyTorch
#   versions?") — NOT a fix-quality axis. Surfaced here so callers can use
#   it without the perf primitive needing to be re-instrumented.


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_peak_mem() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def _time_call(fn: Callable[[], Any], n_warmup: int, n_repeat: int) -> float:
    """Return median latency in milliseconds."""
    for _ in range(n_warmup):
        fn()
    _cuda_sync()

    timings_ms: list[float] = []
    for _ in range(n_repeat):
        _cuda_sync()
        t0 = time.perf_counter()
        fn()
        _cuda_sync()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    timings_ms.sort()
    return timings_ms[len(timings_ms) // 2]


def _eager_self_check(
    model_fn: Callable[[], torch.nn.Module],
    inputs_fn: Callable[[torch.nn.Module], dict[str, Any]],
) -> tuple[float, bool]:
    """Run model forward twice with identical inputs and weights; return (max_abs_diff, deterministic).

    Both runs use the SAME inputs (cloned) and the SAME weights — the only
    difference is wall-clock + any internal RNG calls inside the model. A
    diff > 0 means the model has internal non-determinism even with the seed
    patch (i.e. dropout-without-eval, stochastic routing, etc.).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    m = model_fn()
    inputs = inputs_fn(m)
    # Clone inputs so two forward calls see byte-identical data
    inputs_a = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    inputs_b = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    with torch.no_grad():
        out1 = m(**inputs_a)
        o1 = (out1.logits if hasattr(out1, "logits") else _to_tensor(out1)).detach().clone()
        del out1
        out2 = m(**inputs_b)
        o2 = (out2.logits if hasattr(out2, "logits") else _to_tensor(out2)).detach().clone()
        del out2
    del m, inputs, inputs_a, inputs_b
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    diff = (o1 - o2).abs().max().item()
    return diff, diff == 0.0


def _to_tensor(x: Any) -> torch.Tensor:
    """Pull a tensor out of a model output (handles HF output objects)."""
    if isinstance(x, torch.Tensor):
        return x
    for attr in ("logits", "last_hidden_state", "waveform", "pooler_output"):
        v = getattr(x, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    if hasattr(x, "to_tuple"):
        for v in x.to_tuple():
            if isinstance(v, torch.Tensor):
                return v
    raise TypeError(f"can't extract tensor from {type(x).__name__}")


def measure_perf(
    model_fn: Callable[[], torch.nn.Module],
    inputs_fn: Callable[[torch.nn.Module], dict[str, Any]],
    n_warmup: int = 10,
    n_repeat: int = 30,
    device: str = "cuda",
    compile_kwargs: dict[str, Any] | None = None,
    warm_peak_mem: bool = False,
    eager_self_check: bool = True,
) -> PerfResult:
    """Measure eager vs compiled performance for one model.

    Parameters
    ----------
    model_fn : returns a fresh model instance, already on target device, in eval mode.
        Called multiple times (eager_self_check + eager + compiled) so measurements
        don't share state.
    inputs_fn : given a model, returns a dict of kwargs to pass to forward.
    n_warmup : warmup iterations before timing.
    n_repeat : timed iterations; median is reported (default 30 matches upstream).
    device : "cuda" or "cpu".
    compile_kwargs : kwargs forwarded to torch.compile.
    warm_peak_mem : if True, ALSO capture post-warmup peak memory (in addition to cold).
    eager_self_check : if True, run eager twice and compare to detect non-determinism.

    Notes
    -----
    Caller MUST invoke patch_torch_manual_seed() once at process start before any
    model construction. Without it, HF models' internal RNG calls drift between
    runs and accuracy/perf numbers are suspect.
    """
    compile_kwargs = compile_kwargs or {}

    try:
        # ----- eager_self_check (optional, off by default in tight loops) -----
        eager_self_diff: float | None = None
        eager_deterministic: bool | None = None
        if eager_self_check:
            eager_self_diff, eager_deterministic = _eager_self_check(model_fn, inputs_fn)

        # ----- eager -----
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _reset_peak_mem()

        eager_model = model_fn()
        eager_inputs = inputs_fn(eager_model)

        with torch.no_grad():
            eager_call = lambda: eager_model(**eager_inputs)
            # Run one untimed warmup pass to capture cold peak (matches common.py warmup())
            eager_call()
            eager_peak_mem_mb_cold = _peak_mem_mb()
            # Then timed loop
            _reset_peak_mem()
            eager_ms = _time_call(eager_call, n_warmup, n_repeat)
            eager_warm_peak = _peak_mem_mb() if warm_peak_mem else None

        eager_peak_mem_mb = eager_peak_mem_mb_cold

        del eager_model, eager_inputs, eager_call
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----- compiled -----
        torch._dynamo.reset()
        _reset_peak_mem()

        compiled_model_raw = model_fn()
        compiled_inputs = inputs_fn(compiled_model_raw)
        compiled_model = torch.compile(compiled_model_raw, **compile_kwargs)

        with torch.no_grad():
            compiled_call = lambda: compiled_model(**compiled_inputs)
            # Time the first compiled call separately — this triggers compilation.
            _cuda_sync()
            t0 = time.perf_counter()
            compiled_call()
            _cuda_sync()
            compile_s = time.perf_counter() - t0
            compiled_peak_mem_mb_cold = _peak_mem_mb()

            # Capture real compile_times breakdown from dynamo
            # API: compile_times(repr='csv', aggregate=True) -> (headers, values)
            try:
                from torch._dynamo.utils import compile_times
                headers, values = compile_times(repr="csv", aggregate=True)
                compile_times_dict = {
                    k: float(v) for k, v in zip(headers, values) if v not in ("", None)
                }
            except Exception as ct_err:
                compile_times_dict = {"_error": str(ct_err)}

            # Now timed loop
            _reset_peak_mem()
            compiled_ms = _time_call(compiled_call, n_warmup, n_repeat)
            compiled_warm_peak = _peak_mem_mb() if warm_peak_mem else None

        compiled_peak_mem_mb = compiled_peak_mem_mb_cold

        speedup = eager_ms / compiled_ms if compiled_ms > 0 else float("nan")

        return PerfResult(
            eager_ms=eager_ms,
            compiled_ms=compiled_ms,
            speedup=speedup,
            eager_peak_mem_mb=eager_peak_mem_mb,
            compiled_peak_mem_mb=compiled_peak_mem_mb,
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            device=device,
            compile_s=compile_s,
            compile_times=compile_times_dict,
            eager_warm_peak_mem_mb=eager_warm_peak,
            compiled_warm_peak_mem_mb=compiled_warm_peak,
            eager_self_diff=eager_self_diff,
            eager_deterministic=eager_deterministic,
        )

    except Exception as e:
        return PerfResult(
            eager_ms=float("nan"),
            compiled_ms=float("nan"),
            speedup=float("nan"),
            eager_peak_mem_mb=float("nan"),
            compiled_peak_mem_mb=float("nan"),
            n_warmup=n_warmup,
            n_repeat=n_repeat,
            device=device,
            error=f"{type(e).__name__}: {e}",
        )


# ----- self-test -----
if __name__ == "__main__":
    """Smoke test: tiny MLP on a single GPU."""

    def make_mlp() -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(64, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 64),
        ).eval().cuda()

    def make_inputs(_model: torch.nn.Module) -> dict[str, Any]:
        return {"input": torch.randn(32, 64, device="cuda")}

    # Sequential.forward expects positional, not kwargs — wrap.
    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input):
            return self.m(input)

    def make_wrapped() -> torch.nn.Module:
        return Wrap(make_mlp()).eval().cuda()

    result = measure_perf(make_wrapped, make_inputs, n_warmup=5, n_repeat=20)
    print(json.dumps(result.to_dict(), indent=2))
