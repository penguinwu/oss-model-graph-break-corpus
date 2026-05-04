# Sweep comparison report — 2026-05-03 vs 2026-04-26

**Baseline:** `sweep_results/nightly/2026-04-26`  (1514 work items)
**Current:**  `sweep_results/nightly/2026-05-03`  (1724 work items)
**Common:**   1494 work items
**New:**      230 work items (in current only)
**Removed:**  20 work items (in baseline only)
**Skipped (known_errors.json gated, in either nightly):** 4 work items

**Invariants:** ✓ all passed  (partition complete, explain coverage complete)

---

## 1. Improvements: error → compile-success (22 work items)

| Model | Mode | baseline | → | current | GB count |
|---|---|---|---|---|---|
| BltForCausalLM | eval | timeout | → | full_graph | 0 |
| BltForCausalLM | train | timeout | → | full_graph | 0 |
| BltModel | eval | timeout | → | full_graph | 0 |
| BltModel | train | timeout | → | full_graph | 0 |
| GPT-SoVITS-SynthesizerTrn | eval | worker_error | → | graph_break | 6 |
| GPT-SoVITS-SynthesizerTrn | train | worker_error | → | graph_break | 6 |
| GPT-SoVITS-SynthesizerTrn-forward | eval | worker_error | → | graph_break | 10 |
| GPT-SoVITS-SynthesizerTrn-forward | train | worker_error | → | graph_break | 10 |
| MllamaVisionModel | eval | eager_error | → | full_graph | 0 |
| MllamaVisionModel | train | eager_error | → | full_graph | 0 |
| Qwen2_5OmniToken2WavBigVGANModel | eval | eager_error | → | full_graph | 0 |
| Qwen2_5OmniToken2WavBigVGANModel | train | eager_error | → | full_graph | 0 |
| Qwen2_5OmniToken2WavDiTModel | eval | eager_error | → | full_graph | 0 |
| Qwen2_5OmniToken2WavDiTModel | train | eager_error | → | full_graph | 0 |
| Qwen3VLMoeVisionModel | eval | eager_error | → | graph_break | 14 |
| Qwen3VLMoeVisionModel | train | eager_error | → | graph_break | 14 |
| Qwen3VLVisionModel | eval | eager_error | → | graph_break | 14 |
| Qwen3VLVisionModel | train | eager_error | → | graph_break | 14 |
| Qwen3_5MoeVisionModel | eval | eager_error | → | graph_break | 14 |
| Qwen3_5MoeVisionModel | train | eager_error | → | graph_break | 14 |
| Qwen3_5VisionModel | eval | eager_error | → | graph_break | 14 |
| Qwen3_5VisionModel | train | eager_error | → | graph_break | 14 |

## 2. Regressions: compile-success → error (2 work items)

| Model | Mode | baseline | → | current | error message |
|---|---|---|---|---|---|
| MimiModel | eval | graph_break | → | eager_error | `_unsafe_index found unexpected index type Float` |
| MimiModel | train | graph_break | → | eager_error | `_unsafe_index found unexpected index type Float` |

## 3. Steady-state: compile-success in both (1430 work items)

**GB-count: improved 20 / regressed 10 / unchanged 1400.** Net GB delta across category: **-124**

### 3a. GB-count improved (fewer breaks)

| Model | Mode | baseline GB | current GB | Δ | shift class |
|---|---|---|---|---|---|
| SmolVLMForConditionalGeneration | train | 20 | 0 | -20 | REAL_NEW |
| SmolVLMForConditionalGeneration | eval | 20 | 0 | -20 | REAL_NEW |
| LwDetrModel | train | 12 | 0 | -12 | REAL_NEW |
| LwDetrModel | eval | 12 | 0 | -12 | REAL_NEW |
| PPDocLayoutV3Model | eval | 12 | 0 | -12 | REAL_NEW |
| RTDetrModel | eval | 9 | 0 | -9 | REAL_NEW |
| Siglip2Model | eval | 9 | 0 | -9 | REAL_NEW |
| PPDocLayoutV2Model | eval | 9 | 0 | -9 | REAL_NEW |
| Siglip2Model | train | 9 | 0 | -9 | REAL_NEW |
| PPDocLayoutV3Model | train | 17 | 9 | -8 | REAL_NEW |
| Siglip2VisionModel | eval | 7 | 0 | -7 | REAL_NEW |
| Siglip2VisionModel | train | 7 | 0 | -7 | REAL_NEW |
| PPDocLayoutV2Model | train | 14 | 9 | -5 | REAL_NEW |
| RTDetrModel | train | 14 | 9 | -5 | REAL_NEW |
| Qwen2AudioForConditionalGeneration | train | 11 | 10 | -1 | REAL_NEW |
| Qwen2AudioForConditionalGeneration | eval | 5 | 4 | -1 | REAL_NEW |
| GraniteMoeHybridModel | train | 11 | 10 | -1 | REAL_NEW |
| GraniteMoeHybridForCausalLM | train | 13 | 12 | -1 | REAL_NEW |
| GraniteMoeHybridModel | eval | 11 | 10 | -1 | REAL_NEW |
| GraniteMoeHybridForCausalLM | eval | 13 | 12 | -1 | REAL_NEW |

### 3b. GB-count regressed (more breaks)

| Model | Mode | baseline GB | current GB | Δ | shift class |
|---|---|---|---|---|---|
| InstructBlipVideoModel | train | 0 | 9 | +9 | REAL_NEW |
| InstructBlipModel | train | 0 | 9 | +9 | REAL_NEW |
| MMGroundingDinoModel | train | 11 | 12 | +1 | REAL_NEW |
| GroundingDinoModel | train | 11 | 12 | +1 | REAL_NEW |
| DFineModel | train | 15 | 16 | +1 | REAL_NEW |
| MMGroundingDinoModel | eval | 10 | 11 | +1 | REAL_NEW |
| GroundingDinoModel | eval | 10 | 11 | +1 | REAL_NEW |
| RTDetrV2Model | train | 15 | 16 | +1 | REAL_NEW |
| DFineModel | eval | 10 | 11 | +1 | REAL_NEW |
| RTDetrV2Model | eval | 10 | 11 | +1 | REAL_NEW |


## 4. New models in current (230 work items)

**Distinct new models:** 115

**Status distribution:**

| Status | Count |
|---|---|
| full_graph | 109 |
| eager_error | 65 |
| create_error | 23 |
| timeout | 18 |
| graph_break | 15 |

**Compile-clean (full_graph):** 109 of 230 work items
**Total graph breaks across new work items:** 143

### Top 10 break reasons in new models

| Count | Location | Op / Explanation |
|---|---|---|
| 57 | `unknown` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 16 | `unknown` | `aten.bincount.default` |
| 9 | `unknown` | `aten._local_scalar_dense.default` |
| 6 | `unknown` | `Dynamo does not know how to trace builtin operator `reversed` with argument types ['ModuleList'] (has_kwargs False)` |
| 5 | `transformers/utils/output_capturing.py:239` | `Dynamo has no bytecode reconstruction implemented for sourceless variable DictItemsIterator().` |
| 5 | `transformers/utils/output_capturing.py:224` | `Dynamo has no bytecode reconstruction implemented for sourceless variable DictIterator().` |
| 4 | `diffusers/models/transformers/transformer_hidream_image.py:392` | `aten.bincount.default` |
| 4 | `transformers/models/qwen3_5/modeling_qwen3_5.py:1312` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 4 | `diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py:209` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 4 | `diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py:166` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |

### NEW break reasons (not seen in any baseline model)

| Count | Location | Op / Explanation |
|---|---|---|
| 16 | `unknown` | `aten.bincount.default` |
| 4 | `diffusers/models/transformers/transformer_hidream_image.py:392` | `aten.bincount.default` |
| 4 | `transformers/models/qwen3_5/modeling_qwen3_5.py:1312` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 4 | `diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py:209` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 4 | `diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py:166` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 2 | `transformers/models/pp_formulanet/modeling_pp_formulanet.py:923` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 2 | `diffusers/models/transformers/transformer_hidream_image.py:396` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |
| 2 | `diffusers/models/transformers/transformer_hidream_image.py:699` | `Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False.` |
| 2 | `diffusers/models/transformers/transformer_lumina2.py:271` | `aten._local_scalar_dense.default` |
| 2 | `diffusers/models/autoencoders/autoencoder_kl_hunyuanimage_refiner.py:211` | `Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow` |

## 5. Removed models (in baseline only) — 20 work items

**Distinct removed models:** 10

DinatModel, LayoutLMv2Model, MusicFlamingoForConditionalGeneration, PeAudioVideoModel, Qwen2_5OmniThinkerTextModel, Qwen3OmniMoeTalkerCodePredictorModel, Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration, Qwen3OmniMoeTalkerModel, ZambaForCausalLM, ZambaModel

## 6. Stable failures: error in both (36 work items)

| Model | Mode | status (both) | error message |
|---|---|---|---|
| ConditionalDetrModel | eval | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| ConditionalDetrModel | train | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| DabDetrModel | eval | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| DabDetrModel | train | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| DeformableDetrModel | eval | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| DeformableDetrModel | train | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| DetrModel | eval | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| DetrModel | train | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| FLUX.1-DiT | eval | create_error | `No module named 'einops'` |
| FLUX.1-DiT | train | create_error | `No module named 'einops'` |
| FastVlmForConditionalGeneration | eval | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| FastVlmForConditionalGeneration | train | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| FastVlmModel | eval | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| FastVlmModel | train | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| Gemma3nForConditionalGeneration | eval | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| Gemma3nForConditionalGeneration | train | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| Gemma3nModel | eval | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| Gemma3nModel | train | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| HiggsAudioV2TokenizerModel | eval | eager_error | `Audio must be mono, but got 16000` |
| HiggsAudioV2TokenizerModel | train | eager_error | `Audio must be mono, but got 16000` |
| PI0Model | eval | eager_error | `'NoneType' object has no attribute 'get_seq_length'` |
| PI0Model | train | eager_error | `'NoneType' object has no attribute 'get_seq_length'` |
| PeVideoModel | eval | create_error | `TimmWrapperConfig requires the timm library but it was not found in your environment. You can install it with pip:` |
| PeVideoModel | train | create_error | `TimmWrapperConfig requires the timm library but it was not found in your environment. You can install it with pip:` |
| PerceptionLMForConditionalGeneration | eval | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| PerceptionLMForConditionalGeneration | train | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| PerceptionLMModel | eval | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| PerceptionLMModel | train | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| RwkvForCausalLM | eval | timeout | `` |
| RwkvForCausalLM | train | timeout | `` |
| RwkvModel | eval | timeout | `` |
| RwkvModel | train | timeout | `` |
| TableTransformerModel | eval | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| TableTransformerModel | train | create_error | `TimmBackbone requires the timm library but it was not found in your environment. You can install it with pip:` |
| TimmWrapperModel | eval | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |
| TimmWrapperModel | train | create_error | `TimmWrapperModel requires the timm library but it was not found in your environment. You can install it with pip:` |

---

_Generated by `tools/sweep_compare.py`. Reproduce with:_

```
python3 tools/sweep_compare.py \
    --baseline sweep_results/nightly/2026-04-26 \
    --current  sweep_results/nightly/2026-05-03
```
