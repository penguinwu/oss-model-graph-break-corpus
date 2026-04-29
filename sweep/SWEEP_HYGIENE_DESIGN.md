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

### Layer C — Known errors gate (now with failure classification)

`sweep/known_errors.json` declares (model, status, mode, error_pattern) tuples for stable real bugs the sweep should skip entirely. After the identify pass, `_validate_no_unexpected_errors()` walks the results and processes any failure in `GATED_STATUSES` (currently `create_error`, `eager_error`).

Each unmatched gated failure is classified by `_classify_failure()` into one of three buckets via substring matching on the error text:

- **infra** — env/setup/dep issue (`libnvrtc`, `No module named`, `requires the natten library`, `CUDA out of memory`, `LAPACK`, etc.). Surfaced in an informational summary; NOT gate-blocking. Fix at the venv / build script level.
- **harness** — corpus tooling bug (`.forward() missing N required positional arguments`, `.__init__() missing`, `unsupported operand type(s) for ... NoneType`). Surfaced in a separate informational summary; NOT gate-blocking. Fix at `sweep/models.py` / `create_model`.
- **unknown** — neither — likely a real model bug. Loud warning + counts toward the strict-mode non-zero exit.

Rationale: the original gate flagged ALL unexpected gated failures as if they were equivalent — but env-bugs, harness-bugs, and real-model-bugs need different responses (re-build vs fix-tool vs file-issue). The classifier separates them so the gate's loud-warning path stays focused on what's actually a new model bug. Validated on PT 2.11 baseline data: 130 unexpected → 8 infra + 104 harness + 12 unknown — a 90% reduction in the loud-warning bucket.

Workflow for a NEW *unknown* gated failure: either (a) fix at root, or (b) add to `known_errors.json` with a `reason` field AND an `applies_to_versions` list (e.g. `["2.11"]`). Without `applies_to_versions`, the entry applies universally — that's a regression risk (a real bug fixed in PT 2.13 would still skip the model on the next sweep). Always scope new entries to the version(s) where the bug was actually observed; force re-verification on every other release.

The validator queries the active torch's `major.minor` from the target python at orchestrator startup (via subprocess), then filters entries whose `applies_to_versions` doesn't include it. Stderr summary reports `N/M entries (K filtered out by version X.Y)` so you can see what got dropped.

Goal: every gated failure per sweep is either *expected* (in the list, in scope for the active version), *infra* (env summary), *harness* (tooling summary), or *actionable* (loud warning).

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
