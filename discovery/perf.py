"""measure_perf — per-trial performance primitive for discovery agent.

A *primitive*, not a sweep. Skill-eval and discovery harnesses call this on one
model at a time and get back eager / compiled latency, peak memory, compile
time, and a few sanity numbers.

Designed to be cheap and self-contained. No corpus framework dependency.

Usage
-----
    from discovery.perf import measure_perf

    def model_fn():
        return MyModel(config).eval().cuda()

    def inputs_fn(model):
        return {"input_ids": torch.randint(0, 256, (1, 16), device="cuda")}

    result = measure_perf(model_fn, inputs_fn, n_warmup=10, n_repeat=30)
    # -> {"eager_ms": 12.4, "compiled_ms": 8.1, "speedup": 1.53,
    #     "compile_s": 7.84, ...}
    #
    # compile_s is the wall time of the first compiled call. It is reported
    # for corpus-level analysis but is NOT used in the discovery rubric —
    # compiled_ms (steady-state) is what scores fix quality.
"""
from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
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
    compile_s: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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


def measure_perf(
    model_fn: Callable[[], torch.nn.Module],
    inputs_fn: Callable[[torch.nn.Module], dict[str, Any]],
    n_warmup: int = 10,
    n_repeat: int = 30,
    device: str = "cuda",
    compile_kwargs: dict[str, Any] | None = None,
) -> PerfResult:
    """Measure eager vs compiled performance for one model.

    Parameters
    ----------
    model_fn : returns a fresh model instance, already on the target device, in eval mode.
        Called twice (once for eager, once for compiled) so that measurements
        don't share state.
    inputs_fn : given a model, returns a dict of kwargs to pass to forward.
        Called twice. Inputs should be on the target device.
    n_warmup : warmup iterations before timing.
    n_repeat : timed iterations; median is reported.
    device : "cuda" or "cpu".
    compile_kwargs : kwargs forwarded to torch.compile (e.g. {"mode": "reduce-overhead"}).
    """
    compile_kwargs = compile_kwargs or {}

    try:
        # ----- eager -----
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _reset_peak_mem()

        eager_model = model_fn()
        eager_inputs = inputs_fn(eager_model)

        with torch.no_grad():
            eager_call = lambda: eager_model(**eager_inputs)
            eager_ms = _time_call(eager_call, n_warmup, n_repeat)
        eager_peak_mem_mb = _peak_mem_mb()

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

        # Time the first compiled call separately — this triggers compilation.
        # Captured as compile_s for corpus-level analysis; NOT folded into
        # compiled_ms (which must be steady-state).
        with torch.no_grad():
            compiled_call = lambda: compiled_model(**compiled_inputs)
            _cuda_sync()
            t0 = time.perf_counter()
            compiled_call()
            _cuda_sync()
            compile_s = time.perf_counter() - t0

            # Now warmup + steady-state timing.
            compiled_ms = _time_call(compiled_call, n_warmup, n_repeat)
        compiled_peak_mem_mb = _peak_mem_mb()

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
