# Generalized Compiler Testing

**Status:** Design (not yet implemented)
**Date:** 2026-04-19

## Problem

The corpus infrastructure is designed around a single test: "does `fullgraph=True` succeed with `backend='eager'`?" This limits us to graph break detection. We want to measure broader compiler quality: backend errors, correctness, compile time, and runtime performance — across any backend and flag combination.

## Design

### Two independent axes

The current system conflates *what to compile with* and *what to measure* into a single hardcoded test. The generalization separates them into two orthogonal axes:

1. **Compile configuration** — backend, fullgraph, dynamo flags, any `torch.compile()` kwargs
2. **Measurements** — what data to capture from the run

These compose freely. Any compile configuration can be paired with any set of measurements.

### Compile configuration

Instead of hardcoding `torch.compile(model, backend="eager", fullgraph=True)`, the worker accepts arbitrary `torch.compile()` kwargs:

```python
# Current (hardcoded)
compiled = torch.compile(model, backend="eager", fullgraph=True)

# Generalized
compiled = torch.compile(model, **compile_kwargs)
```

`compile_kwargs` is a dict passed through from the experiment config. It handles any argument `torch.compile()` accepts today or adds in the future: `backend`, `fullgraph`, `mode`, `options`, `dynamic`, etc.

`dynamo_flags` remain separate — they configure `torch._dynamo.config.*` before compilation, not `torch.compile()` itself.

### Measurements

Each measurement type captures different data and has different cost:

| Measurement | What it captures | Cost | Infrastructure needed |
|-------------|-----------------|------|----------------------|
| `errors` | Success/error status + stack trace | Cheap (always on) | None — current worker already does this |
| `compile_timing` | Per-phase durations (Dynamo, AOT Autograd, backend) | Medium | Enable TORCH_LOGS, parse phase timing from logs |
| `correctness` | Eager vs compiled output comparison | High | Output capture, tolerance-based comparison |
| `runtime_perf` | Forward pass wall time, memory | High | Multiple warmup runs, statistical aggregation |

Measurements are additive. `["errors"]` is the cheapest sweep. `["errors", "compile_timing"]` adds log parsing overhead. The worker enables only the instrumentation needed for the requested measurements.

### Test profiles

A test profile combines compile configuration with measurements:

```json
{
  "compile_kwargs": {"backend": "aot_eager", "fullgraph": false},
  "dynamo_flags": {},
  "measurements": ["errors"]
}
```

Examples:

| Use case | compile_kwargs | measurements |
|----------|---------------|-------------|
| Graph break detection (Level 1, current) | `{backend: "eager", fullgraph: true}` | `["errors"]` + explain pass |
| Backend error detection (Level 2) | `{backend: "aot_eager", fullgraph: false}` | `["errors"]` |
| Inductor crash detection | `{backend: "inductor", fullgraph: false}` | `["errors"]` |
| Compile time tracking | `{backend: "inductor"}` | `["errors", "compile_timing"]` |
| Correctness validation (future) | `{backend: "aot_eager"}` | `["errors", "correctness"]` |
| Full quality profile (future) | `{backend: "inductor"}` | `["errors", "compile_timing", "correctness", "runtime_perf"]` |

### Relationship to existing graph break workflow

Graph break detection is a special case of error detection where:
- `compile_kwargs` includes `fullgraph: true`
- The error type is specifically `Unsupported` (Dynamo graph break), not a general crash
- An explain pass is available to extract break reasons via TORCH_LOGS

The explain pass only makes sense for graph breaks — it re-runs with Dynamo logging to get detailed break reasons. For general error detection, the error message from the initial run *is* the explanation. No second pass needed.

The two-pass workflow (identify → explain) becomes conditional:
- Graph break test type → two-pass
- Everything else → single-pass

### Success criteria by measurement

| Measurement | Success | Failure |
|-------------|---------|---------|
| `errors` | Compiled and ran without exception | Any exception during compile or forward pass |
| `compile_timing` | Always succeeds (captures timing data) | n/a — timing is data, not pass/fail |
| `correctness` | Compiled output matches eager within tolerance | Output divergence exceeds tolerance |
| `runtime_perf` | Always succeeds (captures perf data) | n/a — perf is data, not pass/fail |

### Status values

Generalized statuses that work across all test types:

| Status | Meaning |
|--------|---------|
| `success` | Test passed (compiled + ran without error) |
| `error` | Compilation or execution raised an exception |
| `timeout` | Exceeded per-model time limit |
| `create_error` | Model failed to instantiate |
| `eager_error` | Model failed during eager baseline |

For backward compatibility, the graph break test type continues to emit `full_graph` and `graph_break` as status values. These map to `success` and a specific subclass of `error` in the general framework.

## Worker changes

### New CLI flags

```
worker.py --model-json '...' \
  --compile-kwargs '{"backend": "aot_eager", "fullgraph": false}' \
  --measurements '["errors"]' \
  --mode eval --device cuda
```

When `--compile-kwargs` is not provided, defaults to current behavior: `{"backend": "eager", "fullgraph": true}`.

When `--measurements` is not provided, defaults to `["errors"]`.

### Compile step

The compile step (currently lines 3104-3154 in worker.py) changes:

```python
# Parse compile kwargs from args, with defaults for backward compat
compile_kwargs = json.loads(args.compile_kwargs) if args.compile_kwargs else {
    "backend": "eager", "fullgraph": True
}

# Dynamic shapes compose with compile_kwargs
if dynamic is not None:
    compile_kwargs["dynamic"] = dynamic

compiled = torch.compile(model, **compile_kwargs)
```

### Error classification

Currently the worker distinguishes graph breaks from infrastructure errors (lines 3131-3153). This classification is only relevant when `fullgraph=True`. For general error detection:

- If `fullgraph` is in `compile_kwargs` and is `True`: use current classification (graph break vs infra error)
- Otherwise: any exception = `error`, capture full stack trace

### Measurement: compile_timing

When `"compile_timing"` is in measurements:

1. Set `TORCH_LOGS` environment variable before compile to enable phase logging
2. Capture stderr during compilation
3. Parse phase durations from log output
4. Include in result: `{"dynamo_time_s": ..., "aot_time_s": ..., "backend_time_s": ...}`

This is the same log parsing approach used in the existing explain pass, but capturing timing instead of break reasons.

## Experiment config changes

The experiment config schema extends naturally:

```json
{
  "name": "inductor-error-sweep",
  "description": "Which models crash under inductor?",
  "models": {"source": "all"},
  "configs": [
    {
      "name": "inductor",
      "compile_kwargs": {"backend": "inductor", "fullgraph": false},
      "dynamo_flags": {},
      "measurements": ["errors"]
    }
  ],
  "settings": {"device": "cuda", "modes": ["eval"], "workers": 4, "timeout_s": 300}
}
```

Backward compatibility: configs without `compile_kwargs` default to `{"backend": "eager", "fullgraph": true}`. Configs without `measurements` default to `["errors"]`. Existing experiment configs continue to work unchanged.

## Results schema

Current `results.jsonl` line:
```json
{"model": "GPT2Model", "config": "baseline", "mode": "eval", "status": "full_graph", "wall_time_s": 12.3}
```

Extended:
```json
{"model": "GPT2Model", "config": "inductor", "mode": "eval", "status": "success", "wall_time_s": 12.3, "compile_kwargs": {"backend": "inductor", "fullgraph": false}}
```

With compile timing:
```json
{"model": "GPT2Model", "config": "inductor-timed", "mode": "eval", "status": "success", "wall_time_s": 12.3, "compile_kwargs": {"backend": "inductor"}, "compile_timing": {"dynamo_s": 1.2, "aot_s": 3.4, "backend_s": 5.6}}
```

## Validation findings

Tested 2026-04-19 on PyTorch 2.11.0+cu128, A100 80GB.

**aot_eager + fullgraph=False** (9 models, eval + train):
- All 9 succeeded, including 4 models with known graph breaks (BartModel, MBartModel, PLBartModel, BlenderbotModel)
- With fullgraph=False, Dynamo graph-breaks silently and the traced subgraphs run through aot_eager without error
- Compile times: 1-14s (within current 180s timeout)
- Memory: up to 4.1 GB (well within A100 80GB)

**aot_eager + fullgraph=True** (4 models):
- Same errors as eager backend — graph breaks happen at the Dynamo level before the backend sees them
- Confirms fullgraph is orthogonal to backend choice

**Implication:** aot_eager may not surface many *new* errors with fullgraph=False (Dynamo handles the hard parts, backend just runs the traced subgraphs). The more interesting Level 2 target is **inductor**, which does real code generation and can hit additional failure modes. The framework should make backend selection trivial so we can test inductor once the infra is ready.

## Implementation plan

### Phase 1: Generalize worker (Level 2 MVP)
1. Add `--compile-kwargs` and `--measurements` flags to worker.py
2. Parameterize the compile step
3. Make error classification conditional on fullgraph setting
4. Default to current behavior when new flags aren't provided
5. Smoke test: 5 models with aot_eager, verify results

### Phase 2: Update experiment system
1. Add `compile_kwargs` and `measurements` to experiment config schema
2. Update validation (flag typo detection works on compile_kwargs keys too)
3. Pass new fields through to worker invocations
4. Make explain pass conditional on fullgraph=True in compile_kwargs

### Phase 3: First backend sweep
1. Run full corpus with aot_eager backend
2. Run full corpus with inductor backend
3. Analyze error patterns — determine what classification/issue tooling is needed
4. Build Level 2 analysis tools based on actual data

### Future phases
- **Compile timing** — add TORCH_LOGS parsing to worker, build timing analysis tools
- **Correctness** — add output capture and comparison to worker
- **Performance** — add benchmarking harness with warmup and statistical aggregation
- **Corpus schema** — extend corpus.json to store multi-backend results (defer until we have data worth persisting)

## Open questions

1. **Inductor timeout:** Inductor compilation is significantly slower than eager/aot_eager. What's the right timeout? May need per-backend timeout defaults.
2. **Inductor on CPU vs GPU:** Inductor generates different code for CPU and GPU. Do we want both?
3. **Issue management generalization:** Current issue classification maps graph break patterns to issues. Level 2 errors will have different patterns (inductor codegen failures, lowering errors, etc.). Build new classifier rules or extend existing framework?
4. **Corpus schema evolution:** When do we add multi-backend results to corpus.json vs keeping them as experiment results? Probably after the first full inductor sweep shows the data is stable and worth tracking across versions.
