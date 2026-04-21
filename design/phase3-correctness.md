# Phase 3: Correctness Testing Infrastructure

**Status:** Draft, design questions resolved 2026-04-20 — pending Peng's final green-light to merge into design-doc.md as Section 8
**Owner:** Otter
**Target merge:** Section 8 of `design-doc.md` after approval

---

## 1. Goal

Surface compiler-introduced numerical errors by comparing eager vs compiled forward outputs across the corpus. Every divergence is a signal worth filing — not a pass/fail certification.

## 2. Scope

**MVP:** HF Transformers (~460 models), eval mode, fp32, `backend="eager"`.

**V2:** Train mode (gradient correctness), `backend="inductor"`, dtype matrix (fp16/bf16), Diffusers (~110 models, per-pipeline output extraction).

**Out of scope:** TIMM (V3), decoded output comparison (text/image/audio quality — separate generative-correctness phase).

## 3. Methodology

### 3.1 Capture

For each model in the corpus:

```python
set_seed(42)
inputs = generate_inputs(model)  # same as identify pass
out_eager = model(**inputs)         # eager forward
torch._dynamo.reset()               # full state reset
compiled = torch.compile(model, fullgraph=True, backend="eager")
set_seed(42)                        # re-seed before compiled fwd
out_compiled = compiled(**inputs)
```

Same input, same seed, same shape — only difference is the `torch.compile` wrapper.

### 3.2 Compare (HF-Style Recursive Walk)

Adopt HF Transformers' battle-tested approach from `tests/test_modeling_common.py`:

| Field type | Action |
|------------|--------|
| Floating-point tensor | `torch.testing.assert_close(rtol, atol)` |
| Integer/bool tensor | Skip (HF rationale: argmax-derived; tiny upstream diffs flip results) |
| 0-dim tensor (loss) | Skip in forward; compare in V2 backward-grad pass |
| `None` | Skip |
| `Cache` objects (`DynamicCache`, `HybridCache`) | Skip |
| `tuple`/`list`/`dict`/`ModelOutput` | Recurse |

NaN handling: skip if both `nan`; strip `nan`/`-inf` before max-diff.

### 3.3 Tolerance Table

Adopted verbatim from HF (`test_modeling_common.py:194-222`):

| dtype | atol | rtol |
|-------|------|------|
| fp32 (CUDA) | 1e-6 | 1e-4 |
| fp16 (CUDA) | 5e-3 | 5e-3 |
| bf16 (CUDA) | 1e-2 | 1e-2 (SDPA: 3e-2) |

MVP runs fp32 only. dtype matrix is V2.

### 3.4 Determinism Helpers

Adopt HF helpers verbatim (import paths verified against transformers 5.5.3):

- `from transformers import set_seed` → `set_seed(42)` before each forward
- `from transformers.testing_utils import set_config_for_less_flaky_test, set_model_for_less_flaky_test`
  - `set_config_for_less_flaky_test(config)` — disables dropout, fixes attention impl
  - `set_model_for_less_flaky_test(model)` — fixed init

### 3.5 Failure Classification (for triage, not grading)

| Class | Definition |
|-------|------------|
| `match` | Within `(atol, rtol)` |
| `divergence` | Past `(atol, rtol)`. Severity = continuous `max_diff / tolerance_threshold` ratio — sort triage queue by ratio, do **not** pre-bucket into "small" vs "large." |
| `nan_inf_introduced` | Compiled output has NaN/Inf, eager doesn't |
| `shape_mismatch` | Compiled output shape ≠ eager |
| `dtype_mismatch` | Compiled output dtype ≠ eager |

**Why one `divergence` class instead of two:** splitting into `numerical_drift` vs `divergence` at an arbitrary threshold (e.g., 10× vs 100× tolerance) silently re-introduces grading — implying "drift" is acceptable. Continuous severity preserves the full distribution and lets bimodality (if it exists) emerge from the data, not from our prior.

`compile_error` and `eager_error` cases are already covered in the identify pass and excluded from the correctness pass (correctness only runs on models where both eager and compile succeed).

**Every divergence is a signal worth filing.** Severity ratio decides which to file *first*, never which to *accept*. Pattern recognition emerges from looking at all of them.

### 3.6 Execution

- **When:** After identify pass (need known-clean candidate set)
- **Subset:** Models with `eval.fullgraph_ok=True` in corpus.json (~352 of 460 for HF v2.10)
- **Worker:** New `run_correctness(spec, device, mode)` in `worker.py`, dispatched via `--pass correctness`
- **Wall budget:** Same 180s default; large models inherit tier from issue #57
- **Storage:** New `correctness/` directory at repo root, schema mirrors `corpus.json`:

```json
{
  "name": "BertModel",
  "source": "hf",
  "eval": {
    "status": "match",
    "max_diff": 1.2e-5,
    "tolerance": {"atol": 1e-6, "rtol": 1e-4},
    "compared_fields": ["last_hidden_state", "pooler_output"],
    "skipped_fields": ["past_key_values"],
    "severity_ratio": 0.42,
    "wall_time_s": 12.4
  }
}
```

### 3.7 Why Not Pre-existing `run_validate`?

`worker.py:3248 run_validate` does a different test: dynamic-shape correctness (eager at shape A vs compiled-dynamic at shape B). It validates that `mark_dynamic` produces the same answer as eager when shape changes. Phase 3 is the simpler base case: same shape, eager vs compiled, does the answer match? Both are needed; this doc covers Phase 3 only.

## 4. Implementation Plan

1. **`run_correctness` in `worker.py`** — analogous to `run_validate`, simpler (one shape).
2. **HF-recursive comparator** — upgrade `_compare_outputs` (line 3210) with field-type taxonomy + skip rules.
3. **Import HF helpers** — `from transformers.testing_utils import set_seed, set_config_for_less_flaky_test, set_model_for_less_flaky_test`.
4. **CLI dispatcher** — add `--pass correctness` to `worker.py` arg parsing (line 3473).
5. **Smoke test on 5 known-clean models** — Bert, T5, GPT2, ViT, ResNet (TIMM via existing path).
6. **Run on full HF clean set** (~352 models) — surface every divergence.
7. **Triage divergences** — file each as a corpus issue with `correctness` label.

## 5. Resolved Decisions (from review 2026-04-20)

1. **Failure classes:** 5 classes (collapsed `numerical_drift`+`divergence` into one `divergence` with continuous severity ratio).
2. **Diff dump:** Default off; opt-in via `--dump-diffs` flag. Sizing rationale: ~1MB per tensor × ~50 divergent models × 2 tensors ≈ 100MB single-mode but accumulates across modes/backends/PT versions.
3. **Train mode:** V2 (post-MVP). MVP is eval-only.

## 6. Future Work (V2+)

- Train mode + gradient comparison (HF's `test_torch_compile_for_training` pattern)
- Backend matrix: `aot_eager`, `inductor`
- Dtype matrix: fp16, bf16
- Diffusers per-pipeline output capture
- TIMM (raw tensors — should be straightforward)
- Cross-version trend (does compile correctness improve PT 2.8 → 2.9 → 2.10 → 2.11?)
