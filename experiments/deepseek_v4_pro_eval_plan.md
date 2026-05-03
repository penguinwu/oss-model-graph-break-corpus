---
plan: DeepSeek V4 — corpus enablement + evaluation
status: active
owner: Otter
created: 2026-04-25
last_check: 2026-05-02
revision: 2
forcing_function: tools/check_plan.py + daily brief at 7:30 AM ET
---

# DeepSeek V4 — Corpus Enablement + Evaluation Plan

> **Revision 2 — 2026-05-02:** Major rework after (a) the Apr 25 one-off `run_eval.py` was deleted in commit `c7987b5` per Peng's principle that adding a model = a new config, not a new runner; (b) PR #45643 (Add DeepSeek V4) merged into transformers main this morning at 11:41 UTC, eliminating the PR-branch dance; (c) GPU corrected from H100 → A100 80GB (devvm has 1× A100 80GB).

## Goal

Add DeepSeek V4 as a first-class corpus model and run the standard evaluation:

| Dimension | What we measure | Pass criterion |
|---|---|---|
| **Graph breaks** | `torch.compile(fullgraph=True)` graph break count + categorization | gb=0 ideal; >0 → file as corpus issue per category |
| **Correctness** | `torch.compile(default backend)` output vs eager output | `max_abs_diff` ≤ **1e-4** (default per upstream `pytorch/benchmarks/dynamo/common.py:1069`); flag bitwise non-equivalence separately |
| **Performance** | tier-1 (small / "fast") and tier-2 (realistic) speedup of compiled vs eager; compile time | speedup > 1.0x. (No prior-release regression check — Peng 2026-04-25: no comparable baseline; absolute numbers only.) |
| **Numerics** | `bitwise_equal` field + max_abs_diff distribution; bf16 vs fp32 stability | record per layer / per output head if model is composite |

## Why this model — and which variant

DeepSeek V4 series (released Apr 2026) is a fresh release not yet in the corpus. Standing eval question: when a high-profile new model lands, where do PyTorch's compile, correctness, and perf stories stand on it?

**The DSV4 series has two variants on HF Hub:**

| Variant | Total params | Activated | HF model_class | Context |
|---|---|---|---|---|
| `deepseek-ai/DeepSeek-V4-Pro` | 1.6T | 49B (MoE) | `DeepseekV4ForCausalLM` | 1M tokens |
| `deepseek-ai/DeepSeek-V4-Flash` | 284B | 13B (MoE) | `DeepseekV4ForCausalLM` | 1M tokens |

**Critical:** both variants use the **same HF model class** (`DeepseekV4ForCausalLM`, model_type `deepseek_v4`). The architectural feature set (Hybrid Attention via CSA+HCA, manifold-Constrained Hyper-Connections, hash layers, MoE routing, index_*) is identical between Pro and Flash. The variants differ only in size knobs (`hidden_size`, `num_hidden_layers`, `num_attention_heads`, `n_routed_experts`, `moe_intermediate_size`).

**Implication for corpus integration:** **one corpus entry covers both.** The corpus's `_reduce_model_size` shrinks `num_hidden_layers` to 2 anyway (graph-break behavior is determined by ops used, not depth). Whatever default `DeepseekV4Config()` ships with after the merge becomes the starting point; per-model overrides reduce remaining size knobs (notably `n_routed_experts`) so the model fits A100 80GB during instantiation.

**Use Flash's default dims as the starting reference point** (`hidden_size=4096`, `n_routed_experts=256` etc. — smaller than Pro's `7168` and `384`), but in practice the difference disappears after `_reduce_model_size` + per-model expert reduction. This is "use Flash" in spirit while not actually picking one over the other architecturally.

## A100 80GB constraint

- **Pro full weights at bf16:** 1.6T × 2B = 3.2 TB. Needs ~40+ A100s. Out of scope.
- **Flash full weights at bf16:** 284B × 2B = 568 GB. Needs ~7+ A100s. Also out of scope.
- **Reduced (corpus-style):** ~7B params at bf16 = ~13 GB. Fits comfortably (Apr 25 phase 1 ran at this scale).
- **Quantized variants on HF Hub:** Pro/Flash ship in FP4+FP8 mixed precision (their MoE expert weights are FP4). For corpus eval we want bf16 weights to test the true compute path; we construct the model from config + random init rather than loading published weights. This is the same approach Apr 25 used.

**Verdict:** corpus eval runs against a **scaled-down architecturally-complete config** instantiated from `DeepseekV4Config(...)` with reduced size knobs. Same approach as every other large MoE model in the corpus (Dbrx, Qwen MoE variants, etc.).

## What's done (Apr 25 Phase 1, pre-revision-2)

Phase 1 ran on Apr 25 via the deleted `run_eval.py` one-off. Tiny config: `vocab=4096, hidden=7168, layers=4, experts=16, moe_intermediate=3072, batch=1, seq=16` → 7.15B params, 13.3 GB at bf16. Loaded from `transformers PR #45643` (sha `a0a8482927a1...`), since DSV4 was not in mainline at the time.

**Findings:**
- ✅ **gb=0 (single fullgraph)** — model fully captures into one graph; ~1712 ops
- ❌ **Correctness FAIL: max_abs_diff = 1.07** (4 orders of magnitude above the 1e-4 tolerance)
- ✅ **Performance: 3.58× speedup** at tier-1 (eager 54.1ms vs compiled 15.1ms; compile_s 5.5)
- ✅ **eager_self_diff = 0.0** — divergence is REAL (Inductor/AOTAutograd), not RNG noise

Sub-issues filed: [#67 Phase 1 closed](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/67) ✓ ; [#108 correctness divergence](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/108) OPEN ; [#68 Phase 2](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/68) OPEN. Umbrella [#66](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/66) OPEN.

Historical record at `experiments/results/deepseek_v4_pro/phase1-tiny-20260425-183222/{results.json,README.md}`.

## What's changed since Apr 25 (this revision)

1. **PR #45643 merged into transformers main this morning** (2026-05-02 11:41 UTC, sha `08e4cf82819a`). DSV4 is now installable from `pip install -U transformers` (or `git+main` until the next PyPI tag drops). No more PR-branch + PYTHONPATH dance.
2. **The one-off `experiments/scripts/run_eval.py` is gone** (commit `c7987b5`). Eval work goes through `tools/run_experiment.py` like every other model in the corpus.
3. **GPU corrected:** devvm is 1× **A100 80GB**, not H100 as Revision 1 assumed. The compute envelope is the same in this case (both 80GB SKUs exist) but the doc should be accurate.

## Methodology

### Corpus integration shape

DSV4 is now a regular HF model. Integration steps:

1. **Upgrade transformers** in the corpus's nightly venv (`~/envs/torch-nightly-cu126`) to a version that contains the merged PR. Either:
   - `pip install -U transformers` (if a release post 11:41 UTC today is available on PyPI), OR
   - `pip install --upgrade git+https://github.com/huggingface/transformers.git@main` (always-latest from main)
2. **Verify `enumerate_hf` discovers DeepseekV4ForCausalLM + DeepseekV4Model** — should be automatic; `sweep/models.py:enumerate_hf()` walks `dir(transformers)` for ModelMixin subclasses.
3. **Verify `worker.py:create_model` succeeds with reduced size.** The default `DeepseekV4Config()` will probably allocate too many experts to fit A100 80GB during init. Two paths:
   - (a) Add a per-model override block in `sweep/worker.py:_fix_config` that reduces `n_routed_experts` (target: 4–8) and any other dims that scale aggressively with the default. Mirror the Bert/ViT/Jamba pattern (search `worker.py` for `name_lower == "..."` blocks).
   - (b) Add an entry to `sweep/large_models.json` if it ends up tier='large' (>120s wall) so timeout tier auto-adjusts.
4. **Run via the standard path:**
   ```bash
   ~/envs/torch-nightly-cu126/bin/python tools/run_experiment.py sweep --models DeepseekV4ForCausalLM --modes eval --workers 1
   ```
   This invokes the existing identify/explain/correctness pipeline. No new runner script.

### Per-dimension protocol

**Graph breaks.** Standard sweep `fullgraph=True` compile + `torch._dynamo.explain`. `tools/file_issues.py` categorizes any breaks via existing rules (`[dynamo] data-dependent branching`, `[dynamo] Tensor.item()`, etc.). Apr 25 found gb=0 — should reproduce.

**Correctness.** Standard `run_correctness` path (`sweep/worker.py:3880`+) with the corpus's existing 3-layer determinism handling: `set_seed(42, deterministic=True)` + CUBLAS deterministic + the less_flaky-retry methodology that flags `numeric_noise_floor_dominant`. Apr 25 found `max_abs_diff=1.07` (4 orders above 1e-4). If reproduced via the principled path, the divergence is real and stays as a Phase-2 blocker.

**Performance.** Standard sweep doesn't measure tier-1/tier-2 perf (that's only in `experiments/scripts/run_eval.py` which we deleted). For now: graph-break + correctness only. Perf measurement is Phase 2 territory and gates on the correctness story landing first.

## Output

**Source-of-truth artifact:** sweep results land at `sweep_results/experiments/deepseek-v4-corpus-add-<date>/` (the principled path's standard output convention). No need for a separate `experiments/results/deepseek_v4_pro/` tree going forward — that was tied to the run_eval.py shape.

**AutoDev Kanban board:** [github.com/users/penguinwu/projects/1](https://github.com/users/penguinwu/projects/1). Per Peng 2026-04-25 13:32 ET. Existing umbrella + sub-issues stay; status updates per phase.

## Resolved decisions (Peng 2026-04-25 + 2026-05-02)

| # | Question | Decision |
|---|---|---|
| 1 | "Kabana" instance | AutoDev Kanban board ([projects/1](https://github.com/users/penguinwu/projects/1)). |
| 2 | Tiny-config vs full-model | Two phases: tiny + correctness FIRST, then full+perf. Phase 2 deferred indefinitely until Phase 1 correctness divergence (#108) is understood — "full" is meaningless if the small case diverges. |
| 3 | Comparable baseline | None. Absolute numbers only. |
| 4 | Tolerance per dtype | **1e-4** (default per upstream `pytorch/benchmarks/dynamo/common.py:1069`). |
| 5 | Detailed writeup | Required for community release; Peng will guide the summary. |
| 6 | **Pro vs Flash variant** *(Rev 2)* | Same `DeepseekV4ForCausalLM` class; one corpus entry tests both. Default config dims used at construction time decide which "shape" we exercise — start with Flash's dims (smaller hidden_size + fewer experts) since Apr 25 already proved Pro-shape works at tiny scale. |
| 7 | **Custom model registry vs HF auto-discovery** *(Rev 2)* | DSV4 is in transformers main now → `enumerate_hf` auto-discovers it. **No custom-models registry entry.** Per-model size-reduction override goes in `sweep/worker.py:_fix_config` like every other large MoE. |
| 8 | **One-off runner vs general path** *(Rev 2)* | General path. `experiments/scripts/run_eval.py` deleted in `c7987b5` — DSV4 runs via `tools/run_experiment.py sweep --models DeepseekV4ForCausalLM` like any other corpus model. |

## Done means

**Phase 1 (corpus integration + correctness reproduction):**
- Transformers upgraded to a version containing PR #45643's merge (>= 5.7 or git+main)
- `python sweep/models.py --source hf | grep DeepseekV4` returns the class
- `tools/run_experiment.py sweep --models DeepseekV4ForCausalLM --modes eval --workers 1` runs to completion (no create_error)
- Results show `numeric_status` either `match` (Apr 25 wrong) or `divergence` with `numeric_max_diff > 1e-4` (Apr 25 reproduced)
- If correctness reproduces: file-update on #108 with the principled-path repro command + close #67 (already closed) + leave #66+#68 open
- If correctness does NOT reproduce: investigate (better seeding via the 3-layer methodology may have collapsed the noise; or the `n_routed_experts=4` reduction changed enough to mask the bug; or PR-branch sha `a0a8482927a1` had a bug since fixed in the merged sha `08e4cf82819a`)

**Phase 2 (perf at scale):** GATED on Phase 1 correctness verdict. If divergence persists at tiny scale, full-model perf is meaningless. Re-plan Phase 2 once #108 is resolved.

## Stop conditions

- *transformers upgrade fails* (BPF blocks, version conflict): document the install path, defer to Peng.
- *DSV4 still won't construct under reduced size:* iterate on `_fix_config` overrides; if 3+ rounds don't converge, defer + file as a corpus-tooling issue.
- *Eager itself fails:* file as a transformers issue; pause until upstream patches.
- *Compile crashes:* file as a corpus issue with the crash trace; reduce scope (drop correctness, keep gb-only) and continue.

## Out of scope

- Full-model evaluation (1.6T or 284B params won't fit single A100; would need sharded inference, separate workstream)
- Training-mode evaluation
- Quantized inference (FP4 + FP8 mixed) — that's the published variants' default, but corpus tests the bf16 compute path
- Latency under serving load (vLLM, TensorRT)

## Execution shape — tonight

1. ✅ Cleanup `run_eval.py` + Pro Phase 1 config — commit `c7987b5`
2. ✅ Investigate variants + PR status (this revision)
3. **Upgrade `~/envs/torch-nightly-cu126` transformers** to post-merge version. Verify `from transformers import DeepseekV4ForCausalLM` succeeds.
4. **Run `python sweep/models.py --source hf | grep -i deepseek`** to confirm enumerate_hf picks it up.
5. **Try `worker.py:create_model({"name": "DeepseekV4ForCausalLM", "source": "hf"}, "cuda")`** in isolation — see what fails. Iterate on `_fix_config` per-model override (set `n_routed_experts=4` etc.) until create succeeds at <16 GB.
6. **Run the full pipeline:** `tools/run_experiment.py sweep --models DeepseekV4ForCausalLM --modes eval --workers 1 --identify-only --output-dir sweep_results/experiments/dsv4-add-$(date +%Y%m%d)/` then explain + correctness as the second pass.
7. **Capture results** — file findings as a comment on #66 (umbrella) and #67 (close-out) and #108 (Phase 1 reproduction status).
8. **Commit** the per-model override + a `experiments/configs/dsv4-add.json` if needed for future re-runs.

## Revision log

- **Rev 1** *(2026-04-25)*: Initial plan. 5 open questions resolved in 3 rounds (13:15 → 13:18 → 13:22 → 13:32 ET). Two-phase execution adopted. Tolerance 1e-4. AutoDev board for tracking. *Phase 1 executed Apr 25 via one-off `run_eval.py` — gb=0 + correctness divergence + 3.58× speedup.*
- **Rev 2** *(2026-05-02)*: Major rework. (a) Deleted one-off `run_eval.py` per Peng's principle that custom models reuse general corpus path (`c7987b5`). (b) PR #45643 merged into transformers main 11:41 UTC today — DSV4 is now auto-discovered by `enumerate_hf`. (c) GPU corrected H100 → A100 80GB. (d) Pro vs Flash decision: same class, one corpus entry, use Flash dims as starting reference. (e) Phase 2 perf measurement removed from this plan's scope (gated on correctness; would need a `--measure-perf` extension to general path which is itself out-of-scope tonight).
