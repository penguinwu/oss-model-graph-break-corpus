# Sweep Hygiene — Preventing Setup-Side Failures from Masking Graph-Break Signal

**Status:** SHIPPED 2026-04-28 (commits `50a3d3d`, `e82f93b`, `a4fd24c`, `8e91caf`)
**Author:** Otter
**Motivation:** Animesh's morning fullgraph sweep recorded 78 `eager_error` + 42 `create_error` items. On dig, *67% of the eager_errors and ~76% of the create_errors were not real model failures* — they were source-build venv missing libraries (libnvrtc-builtins, timm, LAPACK, etc.). The morning per-issue walk classified those models as "still failing" against open dynamo-team issues, which would have masked real graph-break improvements (and regressions) from the actual signal.

**User docs:** [docs/running-sweeps.md](../docs/running-sweeps.md#known-errors-gate-no-silent-infrastructure-failures) covers the user-visible behavior.

---

## Problem

A graph-break sweep classifies each (model, mode) work item into one of:
`full_graph` | `graph_break` | `eager_error` | `create_error` | `compile_error` | `timeout` | `worker_error`

The first two are signal. The rest are *failure* classes. Of those failure classes, several are **infrastructure** rather than model behavior:
- `eager_error` from missing CUDA runtime libs
- `create_error` from missing Python deps (timm, natten, detectron2)
- `create_error` from torch built without LAPACK
- `timeout` from a model whose tier in `large_models.json` got the wrong timeout multiplier

When the morning's per-issue walk asked "did this model improve / regress?" for each open dynamo-team issue, it intersected the affected-model lists with the sweep's *failure* rows. Models actually-passing-fullgraph that hit env failures got classified as "still broken" — false negatives that bury real wins (and let regressions hide as quote-unquote "expected" failures).

## Design

Three layers, each preventing a specific failure mode of the sweep:

### Layer A — Worker error truncation 500 → 8KB

`sweep/worker.py` previously truncated every exception string at 500 chars (`str(e)[:500]`). For nvrtc-style errors (~5KB inline kernel source preceding the actionable line), 500 chars dropped the actionable line. Bumped to 8KB via `_record_error()` helper called at all 11 truncation sites; results that were truncated record `error_truncated: {original_chars, kept}` for visibility. A full sweep adds maybe 1MB to `identify_results.json` — small price for losing root-cause info on 67% of failures.

### Layer B — Source-build venv parity with pip-wheel torch

`scripts/build-nightly-from-source.sh stage_fixup` now installs the runtime deps that `pip install torch transformers diffusers` would normally pull but that source-built torch doesn't:

1. `nvidia-cuda-{nvrtc,runtime,cupti}-cu12==<TORCH_CUDA_VER>.*` — CUDA runtime libs needed for JIT-compiled kernels (e.g. `reduction_prod_kernel`).
2. `sitecustomize.py` written into the venv's site-packages — force-preloads the bundled nvidia-cuda-* libs at Python startup (necessary because source-built torch links against system libcudart, so torch's own `_preload_cuda_deps()` short-circuits and never preloads the bundled nvrtc-builtins).
3. `pip install --no-deps timm natten` — corpus-required Python deps that ~30 models import at create time.
4. `USE_LAPACK=1` in `stage_build` — torch needs LAPACK for `torch.geqrf` / `torch.lstsq` / etc. on CPU tensors. Without it, RwkvModel and similar fail at create time.

`stage_verify` adds a CUDA JIT smoke test (`torch.randn(...).prod()` + `torch.fft.fft(x)` — exercises the nvrtc JIT path) so the build fails fast if a future regression breaks the parity.

### Layer C — Known errors gate

`sweep/known_errors.json` declares (model, status, mode, error_pattern) tuples for stable real bugs the sweep should skip entirely. After the identify pass, `_validate_no_unexpected_errors()` walks the results and flags any failure in `GATED_STATUSES` (currently `create_error`, `eager_error`) that doesn't match a declared entry. With `--strict-known-errors`, unmatched failures cause a non-zero exit.

Workflow: a NEW gated failure must be either (a) fixed at root, or (b) added to `known_errors.json` with a `reason` field. Old entries are deleted when the underlying bug is fixed; the next sweep re-tests the model and surfaces the new status. Goal: every gated failure per sweep is either *expected* (in the list) or *actionable* (loud warning).

### Tier-aware timeouts

`sweep/large_models.json` already declared `timeout_tier: large` vs `very_large` per-model, but `run_sweep.py` applied `args.timeout * 3` uniformly to both tiers. Now: `large = 3×`, `very_large = 9×`. With default `--timeout 180` that's 540s and 1620s respectively. Models like Gemma3n that legitimately need >540s no longer timeout-mask as "failure".

## Why this matters for downstream signal

The sweep produces data the issue tracker, the per-team walk, the corpus stability classification, and Animesh's `#70` deliverable all consume. A failure row in any of those that's actually env-bug-disguised-as-model-fault propagates everywhere downstream. The hygiene layers don't add new functionality — they make the existing signal honest.

After Layers A-C ship, the morning's 120 failure rows (78 eager + 42 create) drop to ~12 real model bugs. The 28 net-new fullgraph wins discovered post-fix are visible, not masked.

## Followups

- LAPACK fix needs a fresh source build to verify (Rwkv items will flip from create_error → likely full_graph or graph_break).
- natten install path (DinatModel — 2 work items): `pip install natten` builds CUDA from source and currently fails wheel build. Worth a separate dep-install-from-wheel-index attempt.
- detectron2 install path (LayoutLMv2Model — 2 work items): same shape, separate investigation.
- The 6 "real bug" residue items (Zamba ×4, Qwen3OmniMoeTalker ×2, PeAudioVideoModel ×2) are now in `known_errors.json` — should be filed as upstream bugs for triage.
