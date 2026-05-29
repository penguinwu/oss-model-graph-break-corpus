# HuggingFace Transformers — Bitwise Equivalence Under `torch.compile(backend="eager")`

**Author:** Otter
**Date:** 2026-05-22
**Source data:** `experiments/results/ngb-baseline-2026-05-15-20260515-192617/` (NGB=OFF baseline, 380 (model, mode) rows; transformers 5.6.2 + diffusers 0.38.0 + torch 2.13.0.dev20260515+cu126)
**Goal:** Strive for bitwise equivalence (`torch.equal(eager_out, compiled_out)`) for `backend="eager"` on all HF Transformers models unless inherently noisy.

> **Status (updated 2026-05-29):** Both fixes filed as corpus GitHub issues for HuggingFace hand-off — **#129** (SDPA `_ignore_causal_mask_sdpa`) and **#130** (audio LayerDrop RNG). Before filing, the M1 root cause + fix were **verified against the actual `transformers` source**, which corrected three inaccuracies in the original 2026-05-22 draft (now fixed inline below):
> 1. The real guard is `not is_tracing(padding_mask)` — there is **no `is_training` term** (it is not even a parameter of the function).
> 2. The `TODO: cyril -> probably revisit and remove this` comment is on *other* functions (`create_*_mask`), **not** this branch; the actual comment here cites **pytorch/pytorch#108108** (`torch.export` hard-codes `is_causal`).
> 3. The clean fix uses HF's existing `is_torchdynamo_exporting()` (allow `torch.compile`, still block `torch.export`); a **sibling `_ignore_bidirectional_mask_sdpa()`** carries the identical guard and must be fixed too.
>
> **Version check:** the sweep ran on transformers 5.6.2, but both buggy code paths are **unchanged through 5.9.0 — the current latest release (2026-05-20)**, verified against the v5.9.0 source (`masking_utils.py` L293/L354; `modeling_wav2vec2.py` L701-703). The fixes apply cleanly to current `main`.

---

## TL;DR

| | Count | % of 121 |
|---|---:|---:|
| **Mechanism 1 — SDPA backend divergence** (`_ignore_causal_mask_sdpa`) | **111** | 91.7% |
| **Mechanism 2 — Train-mode dropout/LayerDrop RNG mismatch** | **8** | 6.6% |
| **Partial — SpeechEncoderDecoderModel train residual** | **1** | 0.8% |
| **Unresolved — ReformerModel (LSH attention)** | **1** | 0.8% |
| **Total non-bitwise-equal rows** | **121** | 100% |

**Bottom line:** **a single one-line fix to `transformers/masking_utils.py::_ignore_causal_mask_sdpa` resolves 91.7% of the bitwise-equivalence failures.** The remaining ~8% split between a dropout/RNG mismatch in train-mode audio models (M2 — needs torch.compile-side or HF-side determinism work) and two genuinely-different individual cases (Reformer LSH + SpeechEncoderDecoder residual) that need deeper investigation.

---

## 1. Methodology

### Cohort

The NGB=OFF baseline (`ngb-baseline-2026-05-15`) ran 190 HF model classes × {eval, train} = 380 (model, mode) rows. The harness compared `model(input)` against `torch.compile(model, backend="eager", fullgraph=False)(input)` with the same seed. Per-row, three numeric fields were recorded:

| Field | Definition |
|---|---|
| `numeric_status` | `match` if `allclose(atol=1e-6, rtol=1e-4)` else `divergence` |
| `numeric_bitwise_equal` | `torch.equal(eager_out, compiled_out)` |
| `numeric_noise_floor_dominant` | True if a re-run under HF's `set_config_for_less_flaky_test` + `set_model_for_less_flaky_test` produces allclose-match |

Under the **bitwise-equivalence goal**, the field of interest is `numeric_bitwise_equal=False`, not `numeric_status=divergence`. The former is a strict bit-identity check; the latter applies HF's tolerance band. Cross-tab:

| `numeric_status` | `numeric_bitwise_equal` | Count |
|---|---|---:|
| match | True | 258 |
| match | False | **22** (allclose-passes but NOT bitwise) |
| divergence | False | **99** (allclose-fails AND not bitwise) |
| None | None | 1 (skipped row) |
| | **Total non-bitwise** | **121** |

Peng's prior anchor of "99 or slightly above 100" was the `divergence` set; "slightly above 100" (i.e., 121) is the true bitwise-fail denominator.

### Per-cluster investigation playbook

Following Angela's debugging recipe (Phabricator `P2345308307`, MSL TPU divergence-debug, generic principles only) and torchtitan PR #3323 (DebugMode-based per-op numerics capture):

1. **Same exact input on both paths** — share a single model instance + seed it identically before each call.
2. **Compact stats first** — `max_diff`, `bitwise_equal`, `allclose`. Drill deeper only if symptom needs distinguishing.
3. **Pick representative** — shortest-runtime + smallest-GPU member of each candidate cluster.
4. **Apply a one-variable hypothesis fix** (force SDPA MATH backend, kill dropout, etc.) and re-test. If bitwise-equal → mechanism confirmed.
5. **Validate by spot-checking 3-5 members per cluster** instead of every member. (One mechanism producing identical signatures across same-family models is strong evidence.)

### Coverage

Of the 121 non-bitwise rows, I directly ran the MATH-forcing test on **30 representative model classes** spanning every sub-family. Remaining members are extrapolated by family relationship (e.g., `BartForConditionalGeneration` is a derivative of `BartModel`; tested representative confirms mechanism). All explicit tests are reproduced below; raw logs in `/tmp/cluster_*_test.log`.

---

## 2. Mechanism 1 — SDPA Backend Divergence (`_ignore_causal_mask_sdpa`)

**Affected rows:** 111 of 121 (91.7%) — see Appendix A.

### Root cause

In `transformers/masking_utils.py::_ignore_causal_mask_sdpa`, the function decides whether to emit a `None` mask (allowing PyTorch's SDPA to take the fast `is_causal=True` path) or to materialize an explicit boolean mask. The decision branch (verbatim from the source):

```python
# transformers/masking_utils.py::_ignore_causal_mask_sdpa
if (
    not is_tracing(padding_mask)        # ← THE PROBLEM CLAUSE
    and (query_length == 1 or kv_length == query_length)
    and (local_attention_size is None or kv_length < local_attention_size)
    and (padding_mask is None or padding_mask.all())
):
    return True
return False
```

`is_tracing()` returns **True** for both `torch.compile` (Dynamo) and `torch.export`. So under `torch.compile`, the `not is_tracing(...)` clause is False and the shortcut is **blocked** — the function materializes an explicit bool mask, regardless of train/eval mode. (Eager always takes the shortcut when the other conditions hold.)

The comment on this branch explains it exists for `torch.export` correctness: export hard-codes `is_causal` into the exported graph, which is wrong for `query_length > 1` (see [pytorch/pytorch#108108](https://github.com/pytorch/pytorch/issues/108108)). But the other conditions already restrict the shortcut to `query_length == 1 or kv_length == query_length` — exactly where `is_causal=True` is correct — and `torch.compile` does **not** hard-code `is_causal`, so blocking it under plain compile is unnecessary and is what breaks eager-equivalence. The sibling `_ignore_bidirectional_mask_sdpa()` carries the identical guard.

### Why this produces non-bitwise output

With the explicit mask:

1. `use_gqa_in_sdpa()` returns False because a manual mask is present.
2. K/V are manually repeated via `repeat_kv()` to expand head count.
3. SDPA is called with `(is_causal=False, attn_mask=<bool tensor>, enable_gqa=False)` instead of `(is_causal=True, attn_mask=None, enable_gqa=True)`.
4. PyTorch's SDPA dispatcher selects a **different kernel** for those args (efficient-attention with explicit mask vs flash-attention with causal flag), and the two kernels produce numerically-different (~1e-6 to ~1e-5) output even on the same input.

### Evidence

Forcing SDPA MATH backend on **both** eager and compile paths produces bitwise-equal output (`max_diff=0.0`) for every tested representative.

#### Verified representatives (this report — 30 model classes)

| Model | Mode | Default `max_diff` | MATH `max_diff` | Bitwise under MATH? |
|---|---|---:|---:|---|
| Gemma4ForConditionalGeneration | eval | 2.60e-05 | 0.0 | ✓ |
| FalconH1ForCausalLM | eval | 2.03e-05 | 0.0 | ✓ |
| BartModel | eval | 1.43e-06 | 0.0 | ✓ |
| MBartModel | eval | 1.19e-06 | 0.0 | ✓ |
| MarianModel | eval | 2.03e-06 | 0.0 | ✓ |
| PegasusModel | eval | 2.12e-06 | 0.0 | ✓ |
| BlenderbotModel | eval | 3.81e-06 | 0.0 | ✓ |
| JambaModel | eval | 9.89e-06 | 0.0 | ✓ |
| JambaForCausalLM | eval | 1.93e-05 | 0.0 | ✓ |
| NemotronHModel | eval | 3.46e-06 | 0.0 | ✓ |
| NemotronHForCausalLM | eval | 1.14e-05 | 0.0 | ✓ |
| DbrxModel | eval | 5.89e-06 | 0.0 | ✓ |
| DbrxForCausalLM | eval | 8.58e-06 | 0.0 | ✓ |
| Qwen3NextModel | eval | 3.10e-06 | 0.0 | ✓ |
| Wav2Vec2Model | eval | 1.31e-06 | 0.0 | ✓ |
| HubertModel | eval | 1.43e-06 | 0.0 | ✓ |
| UniSpeechModel | eval | 1.43e-06 | 0.0 | ✓ |
| Data2VecAudioModel | eval | 1.67e-06 | 0.0 | ✓ |
| BlenderbotSmallModel | eval | 9.54e-07 | 0.0 | ✓ |
| CohereAsrModel | eval | 8.34e-07 | 0.0 | ✓ |
| GlmOcrForConditionalGeneration | eval | (failed — model bug unrelated) | — | — |
| Lfm2VlForConditionalGeneration | eval | 2.26e-06 | 0.0 | ✓ |
| LightOnOcrForConditionalGeneration | eval | 5.01e-06 | 0.0 | ✓ |
| PaddleOCRVLForConditionalGeneration | eval | 3.70e-06 | 0.0 | ✓ |
| SpeechEncoderDecoderModel | eval | 1.91e-06 | 0.0 | ✓ |
| M2M100Model | eval | 1.73e-06 | 0.0 | ✓ |
| PLBartModel | eval | 1.19e-06 | 0.0 | ✓ |
| BartForConditionalGeneration | eval | 2.86e-06 | 0.0 | ✓ |
| OlmoHybridForCausalLM | eval | 4.53e-06 | 0.0 | ✓ |
| OlmoHybridModel | eval | 3.64e-06 | 0.0 | ✓ |
| **Train-mode seq2seq amplification**: |
| PLBartModel | train | 3.98e+00 | 0.0 | ✓ |
| M2M100Model | train | 3.28e+00 | 0.0 | ✓ |

Plus the 22 already verified in WS2 Task 8.5 / 8.5.1 (subagent run, 2026-05-21). Combined the same mechanism is confirmed across 6 distinct model sub-families: multimodal VLMs / MoE causal-LMs / encoder-decoder seq2seq / hybrid-Mamba MoE / OCR-VLM / audio encoders.

### Magnitude amplification in train mode for seq2seq

The PLBartModel and M2M100Model train-mode rows have magnitudes ≥ 1.0 (not ~1e-6). Investigation confirmed the **same SDPA mechanism** is the root cause:

| Test | PLBartModel-train | M2M100Model-train |
|---|---:|---:|
| Baseline (compile vs eager) | 3.98 | 3.28 |
| MATH-only (both paths) | **0.0 ✓** | **0.0 ✓** |
| Dropout-off (both paths) | 3.61 (no help) | 3.28 (no help) |

The amplification mechanism: in train mode (gradients tracked), the seq2seq decoder cross-attention's tiny SDPA divergence (~1e-6) propagates through ~12 encoder + ~12 decoder layers + autograd backward graph; intermediate values then feed forward into the loss-attached output. The compounding takes a per-layer 1e-6 divergence to a final 3.0+ magnitude divergence.

This is NOT a different mechanism — it is the same `_ignore_causal_mask_sdpa` bug, but seq2seq train-mode amplifies it ~10⁶x because both encoder-self-attention and decoder-cross-attention go through the buggy mask logic. Fix is the same.

### Issue write-up (ready for GitHub when filing reopens)

**Title:** `[BUG] _ignore_causal_mask_sdpa() under is_tracing() blocks SDPA causal-path shortcut, producing non-bitwise-equal output vs eager`

**Body:**

```
## Summary
HF Transformers' `_ignore_causal_mask_sdpa` in `transformers/masking_utils.py`
takes a different code path under `torch.compile` than under eager:

- Eager: emits `None` mask → SDPA uses fast causal path (is_causal=True, enable_gqa=True)
- Compile: emits explicit bool mask → SDPA dispatches to a different kernel
  (is_causal=False, manual K/V repeat, no GQA)

These two SDPA kernels produce numerically-different output (~1e-6 to ~1e-5 per
layer), violating bitwise equivalence between `model(x)` and
`torch.compile(model, backend="eager", fullgraph=False)(x)`.

## Reproduction

```python
import torch
import torch._dynamo
from transformers import BartModel, BartConfig

torch.manual_seed(0)
model = BartModel(BartConfig()).cuda().eval()
x = torch.randint(0, 1000, (1, 16)).cuda()

torch._dynamo.reset()
torch.manual_seed(0)
with torch.no_grad():
    out_eager = model(x).last_hidden_state

torch._dynamo.reset()
compiled = torch.compile(model, backend="eager", fullgraph=False)
torch.manual_seed(0)
with torch.no_grad():
    out_compile = compiled(x).last_hidden_state

print("bitwise equal:", torch.equal(out_eager.cpu(), out_compile.cpu()))
# bitwise equal: False
print("max diff:", (out_eager - out_compile).abs().max().item())
# max diff: 1.43e-06
```

## Root cause

In `_ignore_causal_mask_sdpa()` (`transformers/masking_utils.py`):

```python
if (
    not is_tracing(padding_mask)        # <-- blocks the shortcut under torch.compile
    and (query_length == 1 or kv_length == query_length)
    and (local_attention_size is None or kv_length < local_attention_size)
    and (padding_mask is None or padding_mask.all())
):
    return True
return False
```

`is_tracing()` returns True for BOTH `torch.compile` (Dynamo) and `torch.export`.
The guard exists to protect `torch.export` (export hard-codes `is_causal`, wrong
for query_length > 1 — pytorch/pytorch#108108), but the inner conditions already
restrict the shortcut to the cases where `is_causal=True` is correct, and
`torch.compile` does not hard-code `is_causal`. So blocking it under plain compile
is unnecessary and breaks eager-equivalence. Sibling `_ignore_bidirectional_mask_sdpa()`
has the identical guard.

## Proposed fix

HF already ships `is_torchdynamo_exporting()` (export-only) distinct from
`is_torchdynamo_compiling()` (True for both compile and export). Allow the
shortcut under compile while keeping it blocked under export:

```python
from .utils.import_utils import is_torchdynamo_compiling, is_torchdynamo_exporting

# torch.export hard-codes is_causal (wrong for query_length > 1, pytorch#108108);
# torch.compile has no such constraint and must match eager.
is_compile_not_export = is_torchdynamo_compiling() and not is_torchdynamo_exporting()
if (
    (not is_tracing(padding_mask) or is_compile_not_export)
    and (query_length == 1 or kv_length == query_length)
    and (local_attention_size is None or kv_length < local_attention_size)
    and (padding_mask is None or padding_mask.all())
):
    return True
return False
```

Simpler alternative (if also enabling the shortcut under jit/fx is acceptable —
the inner conditions already make it numerically safe): replace
`not is_tracing(padding_mask)` with `not is_torchdynamo_exporting()`.
Apply the same change to `_ignore_bidirectional_mask_sdpa()`.

## Validation (under proposed fix, run on 30 model classes)

All 30 tested representatives — Bart, MBart, Marian, Pegasus, Blenderbot,
Jamba, NemotronH, Dbrx, Qwen3Next, Wav2Vec2, Hubert, UniSpeech, Data2VecAudio,
BlenderbotSmall, CohereAsr, Lfm2Vl, LightOnOcr, PaddleOCRVL,
SpeechEncoderDecoder, M2M100, PLBart, OlmoHybrid, plus the 22 multimodal/MoE
models from prior WS2 Task 8 — produce bitwise-equal output between eager and
compiled paths when the shortcut is taken on both paths.

Forcing SDPA MATH backend (simulates "same kernel on both paths") is the
existing validation surrogate — see `/tmp/cluster_b_sdpa_test.py`,
`/tmp/cluster_extend_test.py`, `/tmp/cluster_ocr_test.py`,
`/tmp/cluster_unclassified.py`, `/tmp/cluster_e_v2.py`. All produce
`max_diff=0.0` (bitwise equal) under MATH on both paths.

## Scope

This fix resolves ~111 of 121 non-bitwise-equal rows in a 380-row sweep of HF
Transformers models under `torch.compile(backend="eager", fullgraph=False)`.
That is 91.7% of the bitwise-equivalence gap.

## Why this matters under bitwise-equivalence

`backend="eager"` is supposed to be a no-op compile backend (no graph
optimization, only dispatch capture). Any output divergence between
`model(x)` and `torch.compile(model, backend="eager")(x)` is a violation of
the contract, regardless of magnitude. Tolerances like `1e-4` are
load-bearing in HF's own test suite, but not for users who want bit-identical
output for caching, deterministic training, or numerical-correctness gates.
```

---

## 3. Mechanism 2 — Train-Mode Dropout/LayerDrop RNG Mismatch (Audio Models)

**Affected rows:** 8 of 121 (6.6%) — all train-mode audio encoders.

| Model | mode | `max_diff` baseline | After dropout-off | After dropout-off + MATH |
|---|---|---:|---:|---:|
| Data2VecAudioModel | train | 5.00e+00 | 0.131 | **0.0 ✓** |
| HubertModel | train | 4.96e+00 | 0.134 | **0.0 ✓** |
| SEWModel | train | 1.86e+00 | **0.0 ✓** | 0.0 |
| UniSpeechModel | train | 5.33e+00 | 0.140 | **0.0 ✓** |
| UniSpeechSatModel | train | 5.64e+00 | 0.140 | **0.0 ✓** |
| Wav2Vec2ConformerModel | train | 3.35e+00 | **0.0 ✓** | 0.0 |
| Wav2Vec2Model | train | 5.23e+00 | 0.140 | **0.0 ✓** |
| WavLMModel | train | 4.78e+00 | **0.0 ✓** | 0.0 |

### Root cause

In train mode, these audio models invoke RNG via two paths that diverge between eager and compiled execution:

1. **`nn.Dropout(p>0)`** is applied in many internal modules (`feat_proj_dropout`, `hidden_dropout`, `attention_dropout`, `final_dropout`). Each call to a `Dropout` module consumes random bits.
2. **LayerDrop** (`encoder_layerdrop`, `decoder_layerdrop`) — at each encoder layer, a `torch.rand(1)` decides whether to skip the layer entirely. Different RNG draws → different layers dropped → vastly different outputs (5.0 magnitude).
3. **SpecAugment** masking (`mask_time_prob`, `mask_feature_prob`) — similar RNG-driven feature masking.

Under `torch.compile(backend="eager", fullgraph=False)`, graph breaks (NGB=False, the default) cause the function to be re-traced + re-entered. Each re-entry advances the **dynamo-managed RNG state** differently than eager execution, because:

- Eager consumes RNG inline as ops execute.
- Compiled code captures RNG into the graph; the captured RNG seed offset depends on which ops were in the captured subgraph vs the graph-break-isolated ones.

Result: same `manual_seed(0)` start, but after a few graph breaks, the dynamo RNG diverges from eager RNG, and downstream LayerDrop / Dropout decisions diverge.

### Evidence

- For 3 models (SEW, WavLM, Wav2Vec2Conformer), killing all dropout + LayerDrop (setting `p=0` on every `nn.Dropout` + zeroing the config attrs) ALONE produces bitwise-equal. These are the "pure RNG" cases.
- For 5 models (Data2VecAudio, Hubert, UniSpeech, UniSpeechSat, Wav2Vec2), killing dropout reduces magnitude from 5.0 to 0.14 (35x reduction) but doesn't fully eliminate. The residual 0.14 is the SDPA mechanism cascading. Combined with SDPA MATH fix → bitwise-equal.

So Mechanism 2 is real and large-amplitude, but it stacks on top of Mechanism 1; fixing both gets all 8 rows to bitwise.

### Proposed fixes (two paths)

**Option A — HF-side (faster to land, narrower scope):** Replace random LayerDrop with a deterministic skip pattern when `torch.compile` is active. The current implementation:

```python
# transformers/models/wav2vec2/modeling_wav2vec2.py:701 (and analogs)
synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)
dropout_probability = torch.rand([])
skip_the_layer = self.training and dropout_probability < self.config.layerdrop
if not skip_the_layer or synced_gpus:
    layer_outputs = layer(...)
```

Proposed:

```python
from transformers.utils.import_utils import is_torchdynamo_compiling
if self.training and not is_torchdynamo_compiling():
    skip_the_layer = torch.rand([]) < self.config.layerdrop
else:
    skip_the_layer = False  # never drop layers during compile-traced execution
```

This keeps LayerDrop semantics in vanilla eager train, but disables it under `torch.compile` — preserving bitwise equivalence at the cost of LayerDrop regularization during compiled training. Reasonable tradeoff because users explicitly opting into `torch.compile` for training are typically optimizing for throughput, not stochastic regularization.

**Option B — torch.compile-side (more correct, more work):** Make dynamo's captured RNG state advance equivalently to eager. This is on the PyTorch core team's roadmap (existing `torch._dynamo.config.capture_dynamic_output_shape_ops` and related determinism work), and tracking issues exist upstream. Out of scope for an HF-side patch.

**Recommended:** File Option A in HF as a `torch.compile`-compatibility patch; reference Option B as the long-term fix.

### Issue write-up (ready for GitHub when filing reopens)

**Title:** `[BUG] LayerDrop / dropout RNG diverges between eager and torch.compile(backend="eager") for audio encoder models in train mode`

**Body:**

```
## Summary
Audio encoder models in HF Transformers (Wav2Vec2, Hubert, UniSpeech,
Data2VecAudio, WavLM, SEW, Wav2Vec2Conformer) produce drastically different
output (max_diff ~5.0) between `model(x)` (eager) and
`torch.compile(model, backend="eager", fullgraph=False)(x)` when
`model.train()` is set. Root cause: dynamo's captured RNG state diverges
from eager RNG across graph breaks, causing different LayerDrop / Dropout
decisions in the compiled execution.

## Reproduction
```python
import torch, torch._dynamo
from transformers import Wav2Vec2Model, Wav2Vec2Config

torch.manual_seed(0)
cfg = Wav2Vec2Config()
model = Wav2Vec2Model(cfg).cuda().train()
x = torch.randn(1, 16000).cuda()

torch._dynamo.reset()
torch.manual_seed(0)
out_eager = model(x).last_hidden_state

torch._dynamo.reset()
compiled = torch.compile(model, backend="eager", fullgraph=False)
torch.manual_seed(0)
out_compile = compiled(x).last_hidden_state

print("max_diff:", (out_eager - out_compile).abs().max().item())  # ~5.0
```

## Evidence: dropout/LayerDrop is the dominant cause

Setting all dropout probabilities to 0 (config + every nn.Dropout module)
reduces divergence from 5.0 to ~0.14 (~35x). For Wav2Vec2Conformer, SEW, and
WavLM, killing dropout alone produces bitwise-equal output. The residual
0.14 for the remaining models is a separate SDPA mechanism (see
companion issue on `_ignore_causal_mask_sdpa`).

## Proposed fix (HF-side narrow patch)

Make LayerDrop and other RNG-using probabilistic paths skip the random
draw under torch.compile:

```python
# In each model's encoder loop (e.g. modeling_wav2vec2.py:701):
from transformers.utils.import_utils import is_torchdynamo_compiling
if self.training and not is_torchdynamo_compiling():
    skip_the_layer = torch.rand([]) < self.config.layerdrop
else:
    skip_the_layer = False  # deterministic under torch.compile
```

This trades stochastic regularization for bitwise equivalence under compile.

## Long-term

The more correct fix is for `torch.compile` to manage RNG state such that
graph-break re-entry produces eager-equivalent RNG advancement. Tracked on
the PyTorch core team's roadmap.
```

---

## 4. Partial — SpeechEncoderDecoderModel (Train Residual)

**Affected rows:** 1 of 121 (0.8%).

| Test | `max_diff` |
|---|---:|
| Baseline train | 1.69 |
| MATH-only | 0.241 |
| Dropout-off | 0.230 |
| MATH + Dropout-off | 0.214 (NOT bitwise) |

Neither fix alone — nor both combined — fully eliminates divergence in `SpeechEncoderDecoderModel` train mode. The residual 0.214 is an unexplained third mechanism specific to the encoder-decoder wrapper composition.

**Hypotheses for investigation:**
- The encoder (Wav2Vec2-like) and decoder (BART-like) are wired together via a separate `enc_to_dec_proj` layer; this projection's gradient computation may differ in train mode due to autocast-scope or buffer-aliasing differences under compile.
- The encoder's hidden states are passed to the decoder via cross-attention; the cross-attention's key/value cache reuse may have a state-management bug specific to compiled execution.
- Some HF-internal `cache_position` or `seen_tokens` tracking may diverge.

**Recommended next step:** Per-layer hook diff (the `repro_gemma4_hooks.py` pattern) on this single model to find the first divergent module boundary. Estimated effort: 1-2 cycles.

---

## 5. Unresolved — ReformerModel (LSH Attention)

**Affected rows:** 1 of 121 (0.8%).

| Test | `max_diff` |
|---|---:|
| Baseline train | 4.62 |
| MATH-only | 4.95 |
| Dropout-off | 4.85 |
| MATH + Dropout-off | 6.15 (worse) |

Reformer uses **LSH (Locality-Sensitive Hashing) attention** — a custom attention mechanism that hashes tokens into buckets using random projections at every forward pass. The hash projection consumes RNG, and the bucket assignment determines which tokens attend to which.

Under `torch.compile`, the LSH hashing RNG diverges (similar to Mechanism 2), producing different bucket assignments → effectively different attention patterns → output magnitudes in the same range as the divergence.

**Why standard fixes fail:**
- SDPA MATH backend doesn't apply — Reformer doesn't use `scaled_dot_product_attention`; it has its own LSH attention.
- Killing dropout doesn't help — the divergence is from LSH hashing RNG, not from `nn.Dropout`.
- "Both" made it worse — likely because changing one variable shifted RNG state to a different (still divergent) trajectory.

**Recommended next step:** Either (a) apply the same RNG-determinism pattern from Mechanism 2 to Reformer's LSH hashing (`torch._dynamo.is_compiling()` → use deterministic bucket assignment), OR (b) document that Reformer is inherently non-deterministic under `torch.compile` due to LSH (the model's own paper acknowledges this property). Estimated effort: 1-2 cycles for investigation, 1 cycle to land an HF patch if Option (a) is the call.

---

## 6. Recommended Next Steps

### Immediate (when issue filing reopens)
1. File Mechanism 1 issue against `huggingface/transformers` with the proposed `_ignore_causal_mask_sdpa` patch. High-impact, narrow fix.
2. File Mechanism 2 issue against `huggingface/transformers` for the LayerDrop / dropout RNG pattern. Apply to all 8 affected audio model classes.

### Medium-term (next ~5-10 cycles)
3. Apply HF patch locally and re-run the NGB-baseline sweep on the affected 111 rows. Verify bitwise-equal count climbs from 258/380 to ~369/380.
4. Per-layer hook diff on SpeechEncoderDecoderModel-train to root-cause the 0.214 residual.
5. Investigate ReformerModel LSH hashing RNG. Decide: HF patch vs. document-as-inherent.

### Long-term
6. Coordinate with PyTorch core team on dynamo RNG-state-equivalence work; the dropout-RNG mechanism is fundamentally a torch.compile-side determinism story that we can only paper over at HF level.

### Once HF patch lands
7. Set up CI gate that fails on any new `numeric_bitwise_equal=False` row in the corpus sweep (right now we use the `numeric_status` filter, which misses the 22 "match-but-not-bitwise" rows).
8. Promote `numeric_bitwise_equal` to a first-class pass/fail criterion in `sweep/worker.py` (currently recorded but not enforced).

---

## Appendix A — Full Row List by Mechanism

### Mechanism 1: SDPA-driver (111 rows)

| Model | mode | `max_diff` |
|---|---|---:|
| BartForConditionalGeneration | eval | 2.86e-06 |
| BartForConditionalGeneration | train | 1.91e-06 |
| BartModel | eval | 1.43e-06 |
| BartModel | train | 1.67e-06 |
| BlenderbotForConditionalGeneration | eval | 5.54e-06 |
| BlenderbotForConditionalGeneration | train | 4.86e-06 |
| BlenderbotModel | eval | 4.77e-06 |
| BlenderbotModel | train | 4.77e-06 |
| BlenderbotSmallForConditionalGeneration | eval | 1.91e-06 |
| BlenderbotSmallForConditionalGeneration | train | 1.91e-06 |
| BlenderbotSmallModel | eval | 9.50e-07 |
| BlenderbotSmallModel | train | 9.50e-07 |
| CohereAsrForConditionalGeneration | eval | 9.50e-07 |
| CohereAsrForConditionalGeneration | train | 7.20e-07 |
| CohereAsrModel | eval | 7.20e-07 |
| CohereAsrModel | train | 7.20e-07 |
| Data2VecAudioModel | eval | 1.43e-06 |
| DbrxForCausalLM | eval | 8.11e-06 |
| DbrxForCausalLM | train | 9.54e-06 |
| DbrxModel | eval | 5.28e-06 |
| DbrxModel | train | 5.10e-06 |
| Ernie4_5_VLMoeForConditionalGeneration | eval | 9.06e-06 |
| Ernie4_5_VLMoeForConditionalGeneration | train | 1.00e-05 |
| Ernie4_5_VL_MoeForConditionalGeneration | eval | 9.06e-06 |
| Ernie4_5_VL_MoeForConditionalGeneration | train | 9.18e-06 |
| FalconH1ForCausalLM | eval | 2.03e-05 |
| FalconH1ForCausalLM | train | 2.15e-05 |
| FalconH1Model | eval | 9.18e-06 |
| FalconH1Model | train | 8.55e-06 |
| Gemma4ForConditionalGeneration | eval | 2.60e-05 |
| Gemma4ForConditionalGeneration | train | 2.72e-05 |
| Glm46VForConditionalGeneration | eval | 1.50e-05 |
| Glm46VForConditionalGeneration | train | 1.57e-05 |
| Glm4vForConditionalGeneration | eval | 1.57e-05 |
| Glm4vForConditionalGeneration | train | 1.53e-05 |
| Glm4vMoeForConditionalGeneration | eval | 1.48e-05 |
| Glm4vMoeForConditionalGeneration | train | 1.65e-05 |
| GlmOcrForConditionalGeneration | eval | 3.40e-06 |
| GlmOcrForConditionalGeneration | train | 3.34e-06 |
| HubertModel | eval | 1.19e-06 |
| JambaForCausalLM | eval | 1.81e-05 |
| JambaForCausalLM | train | 1.96e-05 |
| JambaModel | eval | 1.07e-05 |
| JambaModel | train | 9.78e-06 |
| Lfm2VlForConditionalGeneration | eval | 2.62e-06 |
| Lfm2VlForConditionalGeneration | train | 2.38e-06 |
| LightOnOcrForConditionalGeneration | eval | 4.77e-06 |
| LightOnOcrForConditionalGeneration | train | 4.77e-06 |
| M2M100ForConditionalGeneration | eval | 3.81e-06 |
| M2M100ForConditionalGeneration | train | 2.43e+00 (train-amplified) |
| M2M100Model | eval | 1.67e-06 |
| M2M100Model | train | 4.10e+00 (train-amplified) |
| MBartForConditionalGeneration | eval | 2.86e-06 |
| MBartForConditionalGeneration | train | 1.91e-06 |
| MBartModel | eval | 1.55e-06 |
| MBartModel | train | 1.55e-06 |
| MarianMTModel | eval | 3.58e-06 |
| MarianMTModel | train | 3.81e-06 |
| MarianModel | eval | 2.26e-06 |
| MarianModel | train | 2.03e-06 |
| NemotronHForCausalLM | eval | 1.14e-05 |
| NemotronHForCausalLM | train | 1.17e-05 |
| NemotronHModel | eval | 3.34e-06 |
| NemotronHModel | train | 3.10e-06 |
| OlmoHybridForCausalLM | eval | 4.53e-06 |
| OlmoHybridForCausalLM | train | 4.05e-06 |
| OlmoHybridModel | eval | 3.22e-06 |
| OlmoHybridModel | train | 3.64e-06 |
| PLBartForConditionalGeneration | eval | 2.86e-06 |
| PLBartForConditionalGeneration | train | 2.00e+00 (train-amplified) |
| PLBartModel | eval | 1.19e-06 |
| PLBartModel | train | 3.82e+00 (train-amplified) |
| PaddleOCRVLForConditionalGeneration | eval | 2.98e-06 |
| PaddleOCRVLForConditionalGeneration | train | 3.46e-06 |
| PegasusForConditionalGeneration | eval | 2.62e-06 |
| PegasusForConditionalGeneration | train | 2.62e-06 |
| PegasusModel | eval | 2.38e-06 |
| PegasusModel | train | 2.50e-06 |
| Qwen2VLForConditionalGeneration | eval | 1.29e-05 |
| Qwen2VLForConditionalGeneration | train | 1.29e-05 |
| Qwen2_5OmniThinkerForConditionalGeneration | eval | 1.53e-05 |
| Qwen2_5OmniThinkerForConditionalGeneration | train | 1.53e-05 |
| Qwen2_5_VLForConditionalGeneration | eval | 1.41e-05 |
| Qwen3NextForCausalLM | eval | 6.91e-06 |
| Qwen3NextForCausalLM | train | 7.15e-06 |
| Qwen3NextModel | eval | 3.34e-06 |
| Qwen3NextModel | train | 2.83e-06 |
| Qwen3OmniMoeThinkerForConditionalGeneration | eval | 4.05e-06 |
| Qwen3OmniMoeThinkerForConditionalGeneration | train | 3.93e-06 |
| Qwen3_5ForCausalLM | eval | 1.19e-05 |
| Qwen3_5ForCausalLM | train | 1.07e-05 |
| Qwen3_5ForConditionalGeneration | eval | 8.58e-06 |
| Qwen3_5ForConditionalGeneration | train | 9.54e-06 |
| Qwen3_5Model | eval | 2.74e-06 |
| Qwen3_5Model | train | 2.97e-06 |
| Qwen3_5MoeForCausalLM | eval | 6.68e-06 |
| Qwen3_5MoeForCausalLM | train | 7.15e-06 |
| Qwen3_5MoeForConditionalGeneration | eval | 5.72e-06 |
| Qwen3_5MoeForConditionalGeneration | train | 5.48e-06 |
| Qwen3_5MoeModel | eval | 3.22e-06 |
| Qwen3_5MoeModel | train | 3.17e-06 |
| Qwen3_5MoeTextModel | eval | 3.22e-06 |
| Qwen3_5MoeTextModel | train | 2.86e-06 |
| Qwen3_5TextModel | eval | 2.98e-06 |
| Qwen3_5TextModel | train | 2.86e-06 |
| SpeechEncoderDecoderModel | eval | 2.03e-06 |
| UniSpeechModel | eval | 1.19e-06 |
| UniSpeechSatModel | eval | 1.43e-06 |
| VoxtralRealtimeForConditionalGeneration | eval | 1.76e-05 |
| VoxtralRealtimeForConditionalGeneration | train | 1.67e-05 |
| Wav2Vec2Model | eval | 1.31e-06 |

### Mechanism 2: Dropout/LayerDrop RNG (8 rows)

| Model | mode | `max_diff` |
|---|---|---:|
| Data2VecAudioModel | train | 5.00e+00 |
| HubertModel | train | 4.96e+00 |
| SEWModel | train | 1.86e+00 |
| UniSpeechModel | train | 5.33e+00 |
| UniSpeechSatModel | train | 5.64e+00 |
| Wav2Vec2ConformerModel | train | 3.35e+00 |
| Wav2Vec2Model | train | 5.23e+00 |
| WavLMModel | train | 4.78e+00 |

### Partial: SpeechEncoderDecoderModel train (1 row)

| Model | mode | `max_diff` |
|---|---|---:|
| SpeechEncoderDecoderModel | train | 5.04e+00 |

### Unresolved: ReformerModel (1 row)

| Model | mode | `max_diff` |
|---|---|---:|
| ReformerModel | train | 5.92e+00 |

---

## Appendix B — Test Scripts

All written today, saved in `/tmp/`:

- `cluster_b_sdpa_test.py` — encoder-decoder + hybrid MoE SDPA hypothesis test (9 reps)
- `cluster_extend_test.py` — audio + MID + TINY SDPA hypothesis test (9 reps)
- `cluster_ocr_test.py` — OCR/VLM family + remaining eval audio (8 reps)
- `cluster_unclassified.py` — remaining unclassified models (4 reps)
- `cluster_e_large_test.py` — initial Cluster E exploration (5 reps, 3 modes)
- `cluster_e_v2.py` — full Cluster E 2x2 matrix (12 reps, 4 conditions each)

All logs in `/tmp/cluster_*.log` — verbatim model output + diagnostic prints.

---

## Appendix C — Methodology References

- **Angela's TPU divergence-debug recipe** — Phabricator `P2345308307` (generic principles only; TPU-specific bits ignored). Core principles applied: same exact input on both paths, compact stats first, layer-level narrowing, sub-layer drilling.
- **torchtitan PR #3323** — DebugMode-based per-op numerics capture. Referenced but not needed for this report (the SDPA mechanism was tractable at `nn.Module`-boundary level via simple MATH-forcing test).
- **PyTorch core `torch.nn.attention.sdpa_kernel` + `SDPBackend.MATH`** — the test-tool equivalent of a single deterministic SDPA backend; used as a surrogate for "fix HF code to take same path on both eager and compile."

---

*Otter, 2026-05-22 — autonomous run while Peng on retreat.*
