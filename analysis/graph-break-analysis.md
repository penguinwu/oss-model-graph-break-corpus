# Graph Break Pattern Analysis for PT2 Team

**Date:** 2026-04-02 | **Corpus:** 468 models | **PyTorch:** 2.10.0+cu128 | **GPU:** A100 80GB

## 1. Executive Summary

118 of 468 open-source HuggingFace models have graph breaks in at least one of 6 tested configurations (static/mark\_dynamic/dynamic=true x eval/train). 85 break in all 6 configurations. All graph-break models are from HuggingFace Transformers; all 5 Diffusers models compile cleanly.

The breaks cluster into **12 root causes**. The top 3 actionable fixes (deepcopy, Logger, audio callables) would address **~48 models (45% of all breaks)**. The largest single category, data-dependent branching (49 models), is inherently hard to fix without `torch.cond()` or model restructuring.

**Key finding:** `mark_dynamic` is **stricter** than `dynamic=true` — 335 vs 339 clean (eval). This is counterintuitive: `mark_dynamic` enforces that marked dims must NOT be specialized, while `dynamic=true` allows the compiler to specialize freely.

## 2. Cross-Configuration Overview

### 2.1 Status Distribution

| Status | S/eval | S/train | M/eval | M/train | T/eval | T/train |
|--------|--------|---------|--------|---------|--------|---------|
| **Clean** | 352 (75%) | 337 (72%) | 335 (72%) | 318 (68%) | 339 (72%) | 321 (69%) |
| **Graph Break** | 92 (20%) | 106 (23%) | 97 (21%) | 113 (24%) | 90 (19%) | 107 (23%) |
| Eager Error | 13 | 14 | 18 | 19 | 18 | 19 |
| Create Error | 7 | 7 | 18 | 18 | 18 | 18 |
| Timeout | 4 | 4 | — | — | 3 | 3 |

**S** = static, **M** = mark\_dynamic (batch + seq\_len dims), **T** = dynamic=true (all dims symbolic).

### 2.2 Aggregate Counts

- **118 unique models** with graph\_break in any config/mode
- **85 models** break in all 6 configurations (stable breaks)
- **309 models** clean in all 6 configurations
- **9 models** break only in dynamic configs (not static)
- **17 models** break only in train mode (static)
- **3 models** break only in eval mode (static)

## 3. Root Cause Taxonomy

Root causes classified by primary error from `fullgraph_error` and `break_reasons` fields. Models with multiple break reasons are classified by their first/primary cause.

### 3.1 Summary Table

| # | Root Cause | Models | Occurrences | Owner | Fixable? |
|---|-----------|--------|-------------|-------|----------|
| 1 | Data-dependent branching | 49 | 176 | Model authors | Hard |
| 2 | `copy.deepcopy()` | 22 | 82 | HF Transformers | **Easy** |
| 3 | Data-dependent guard | 16 | 74 | Model/Dynamo | Hard |
| 4 | Skipped function call | 14 | 47 | PyTorch Dynamo | **Medium** |
| 5 | `logging.Logger` | 12 | 70 | PyTorch Dynamo | **Medium** |
| 6 | Proxy conversion failure | 11 | 47 | PyTorch Dynamo | **Medium** |
| 7 | Tensor mutation (`requires_grad`) | 11 | 33 | Model authors | Medium |
| 8 | Constraint violation | 9 | 16 | mark\_dynamic only | N/A |
| 9 | Fake tensor error | 4 | 13 | PyTorch Dynamo | Medium |
| 10 | Observed exception | 3 | 18 | Model authors | Medium |
| 11 | Non-Tensor return | 3 | 10 | PyTorch Dynamo | Medium |
| 12 | Other | 6 | ~20 | Various | Varies |

"Occurrences" counts model x mode x config instances. "Models" counts unique model names.

### 3.2 Detailed Root Causes

#### 3.2.1 Data-Dependent Branching (49 models)

The largest category. Models use `if tensor.sum() > 0:` or similar control flow that depends on tensor values at runtime. Dynamo cannot trace through these because the branch taken depends on actual data.

**Sub-patterns:**
- **MoE expert routing** (10): Qwen3\_5Model, Qwen3\_5MoeModel, Qwen3\_5MoeTextModel, Qwen3\_5TextModel, Qwen3NextModel, DbrxModel, NllbMoeModel, GraniteMoeHybridModel, SwitchTransformersModel, OlmoHybridModel
- **Encoder-decoder cross-attention masking** (12): BartModel, BigBirdPegasusModel, BlenderbotModel, BlenderbotSmallModel, M2M100Model, MBartModel, MarianModel, MvpModel, PegasusModel, PegasusXModel, PLBartModel, FSMTModel
- **Detection post-processing** (7): AutoformerModel, ConditionalDetrModel, DetrModel, DFineModel, RTDetrModel, RTDetrV2Model, TableTransformerModel
- **Gradient path conditionals** (15): BioGptModel, Blip2Model, CohereAsrModel, FlaubertModel, IBertModel, InformerModel, InstructBlipModel, InstructBlipVideoModel, Kosmos2Model, OPTModel, Speech2TextModel, TimeSeriesTransformerModel, WhisperModel, XGLMModel, Phi4MultimodalAudioModel

**Note:** 35 of these 49 models only exhibit data-dependent branching in **train mode**. In eval mode they break for different reasons (deepcopy, Logger) or compile cleanly.

**Fix path:** Requires `torch.cond()` or model restructuring. No quick fix.

#### 3.2.2 `copy.deepcopy()` (22 models)

Dynamo does not support `copy.deepcopy()`. These models clone decoder layers from encoder using deepcopy during forward.

**Affected models:** BartModel, BigBirdPegasusModel, BlenderbotModel, BlenderbotSmallModel, FSMTModel, M2M100Model, MBartModel, MT5Model, MarianModel, MoonshineModel, MoonshineStreamingModel, MvpModel, NllbMoeModel, PLBartModel, PegasusModel, PegasusXModel, ProphetNetModel, Speech2TextModel, T5Model, TimeSeriesTransformerModel, UMT5Model, WhisperModel

**Config stability:** Present in all 3 dynamic configs. Primarily eval-mode (21 models eval, 7 models train).

**Fix:** Replace `copy.deepcopy(layer)` with `layer.clone()` or explicit parameter copying. Single HF Transformers PR.

#### 3.2.3 Data-Dependent Guard (16 models)

Models trigger data-dependent guard failures where Dynamo cannot verify symbolic constraints at trace time.

**Sub-patterns:**
- **`aten._local_scalar_dense`** (8): Glm4vMoeVisionModel, Glm4vVisionModel, GlmImageVisionModel, GlmOcrVisionModel, PaddleOCRVisionModel, Siglip2Model, Siglip2VisionModel, Phi4MultimodalAudioModel
- **Unbacked symbol propagation** (3): FunnelModel, MimiModel, EncodecModel
- **Guard on symbolic expression** (5): AriaTextModel, DbrxModel, ViltModel, Wav2Vec2BertModel, UnivNetModel

**Fix path:** Requires `torch._check*()` annotations or model shape redesign.

#### 3.2.4 Skipped Function Call (14 models)

Dynamo developers have intentionally marked certain functions as "skipped" (e.g., `find_spec` from the import system). Audio models trigger this when calling feature extractor modules.

**Affected models:** Data2VecAudioModel, HubertModel, SEWModel, SpeechEncoderDecoderModel, SpeechT5Model, UniSpeechModel, UniSpeechSatModel, VitsModel, Wav2Vec2BertModel, Wav2Vec2ConformerModel, Wav2Vec2Model, WavLMModel, XcodecModel, RecurrentGemmaModel

**Config stability:** Primarily eval-mode (14 eval, 3 train). Stable across all dynamic configs.

**Fix:** Un-skip/whitelist audio feature extractor callables in Dynamo.

#### 3.2.5 `logging.Logger` (12 models)

`logging.Logger` method calls inside `forward()` cause graph breaks for non-export cases.

**Affected models:** AriaModel, BambaModel, FalconH1Model, Florence2Model, GotOcr2Model, InternVLModel, LEDModel, LongformerModel, PaliGemmaModel, RwkvModel, SeamlessM4TModel, SeamlessM4Tv2Model

**Config stability:** Both eval and train (11 models eval, 11 train). Consistent across all dynamic configs. AriaModel switches from "unsupported context manager" to Logger in dynamic configs (underlying cause is the same — Logger becomes the first-hit break after context manager support).

**Fix:** Support Logger methods in Dynamo (skip/inline). Medium effort.

#### 3.2.6 Proxy Conversion Failure (11 models)

Missing `as_proxy()` implementation for detection model output types.

**Affected models:** DFineModel, DeformableDetrModel, GroundingDinoModel, LwDetrModel, MMGroundingDinoModel, PPDocLayoutV2Model, PPDocLayoutV3Model, RTDetrModel, RTDetrV2Model, Siglip2Model, Siglip2VisionModel

**Config stability:** Primarily eval-mode (9 eval, 4 train). Siglip2 models switch from data-dependent guard to proxy conversion failure in dynamic configs.

**Fix:** Implement `as_proxy()` for detection output types in Dynamo.

#### 3.2.7 Tensor Mutation — `requires_grad` (11 models)

`setattr()` on `Tensor.requires_grad` not supported. All are audio/speech models in **train mode only**.

**Affected models:** Data2VecAudioModel, HubertModel, SEWDModel, SEWModel, SpeechEncoderDecoderModel, UniSpeechModel, UniSpeechSatModel, Wav2Vec2ConformerModel, Wav2Vec2Model, WavLMModel, XcodecModel

**Config stability:** Train-only. Stable across all 3 dynamic configs.

**Fix:** Refactor models to avoid mutating `requires_grad` during forward, or add Dynamo support.

#### 3.2.8 Constraint Violation (9 models, mark\_dynamic only)

Models specialize on dimensions that were marked as dynamic. **Only appears with `mark_dynamic`**, never with static or `dynamic=true`.

**Affected models:** BrosModel, CanineModel, CpmAntModel, DecisionTransformerModel, IdeficsModel, PixtralVisionModel, TapasModel, MusicgenMelodyModel, MusicgenModel

**Error pattern:** "You marked L['input_ids'].size()[1] as dynamic but your code specialized it to be a constant (32)."

**Why mark-only:** `mark_dynamic` tells Dynamo "this dim varies" but the model's code hardcodes the dimension. `dynamic=true` doesn't enforce this constraint.

#### 3.2.9 Other Categories

| Category | Models | Notes |
|----------|--------|-------|
| Fake tensor error | AutoformerModel, FalconMambaModel, LongcatFlashModel, MambaModel | FX node execution failure with symbolic shapes |
| Observed exception | LongT5Model, SwitchTransformersModel, UdopModel | Exception during tracing with no handler |
| Non-Tensor return | InformerModel, NllbMoeModel, PeAudioModel | `torch.*` op returns non-Tensor |
| Forbidden callable (`mark_static_address`) | FalconMambaModel, MambaModel | SSM cache initialization |
| Unsupported `requires_grad_()` | MraModel | Different from setattr pattern |
| Subgraph tracer error | YosoModel | `dynamic=true` train only |

## 4. Fix Impact Analysis

### 4.1 High-Impact Fixes (3 PRs, ~49 models)

| Fix | Models | Eval Fixed | Train Fixed | Fully Clean | Effort |
|-----|--------|-----------|-------------|-------------|--------|
| `deepcopy` → `clone()` | 22 | 21 | 7 | ~7* | Low |
| Un-skip audio callables | 14 | 14 | 3 | ~3* | Medium |
| Support `Logger` methods | 12 | 11 | 11 | ~11 | Medium |
| **Total unique models touched** | **~49** | | | | |

*\*Many models have different root causes in eval vs train. Fixing deepcopy (eval) still leaves data-dependent branching (train). "Fully clean" counts models where the fix resolves ALL modes.*

### 4.2 Medium-Impact Fixes

| Fix | Models | Notes |
|-----|--------|-------|
| `as_proxy()` for detection outputs | 11 | All DETR-family models |
| Tensor `requires_grad` mutation | 11 | All audio models, train-only |
| `mark_static_address` support | 2 | Mamba/FalconMamba, eval-only |

### 4.3 Hard-to-Fix Categories

| Category | Models | Why Hard |
|----------|--------|----------|
| Data-dependent branching | 49 | Requires `torch.cond()` or restructuring |
| Data-dependent guard | 16 | Requires `torch._check*()` or shape redesign |
| Constraint violation | 9 | mark\_dynamic incompatibility (model specializes on "dynamic" dims) |

## 5. Dynamic Shape Findings

### 5.1 Three-Mode Comparison

| Metric | Static | mark\_dynamic | dynamic=true |
|--------|--------|--------------|--------------|
| Clean (eval) | 352 | 335 | 339 |
| Clean (train) | 337 | 318 | 321 |
| Graph break (eval) | 92 | 97 | 90 |
| Graph break (train) | 106 | 113 | 107 |

### 5.2 Key Insight: mark\_dynamic Is Stricter

`mark_dynamic` produces **fewer** clean models than both static and `dynamic=true`. This is because:

1. **Constraint enforcement:** `mark_dynamic` declares "this dim varies at runtime." If the model's code specializes on that dim (e.g., `if seq_len == 32:`), Dynamo raises a constraint violation. `dynamic=true` doesn't enforce this — it lets Dynamo specialize.

2. **9 new breaks from mark\_dynamic:** 7 are constraint violations (BrosModel, CanineModel, CpmAntModel, DecisionTransformerModel, IdeficsModel, PixtralVisionModel, TapasModel), 1 is a fake tensor error (UnivNetModel), 1 is a subgraph tracer error (YosoModel — `dynamic=true` train only).

3. **Zero static breaks fixed by either dynamic config.** Dynamic shapes never help an already-broken model.

4. **Error type migration:** 11 model/mode pairs change error type in dynamic configs (e.g., AriaModel: unsupported context manager → Logger), but none change from graph\_break to clean.

### 5.3 Dynamic-Only Breaks (9 models)

| Model | mark | true | Root Cause |
|-------|------|------|------------|
| BrosModel | break | clean | constraint violation (seq\_len) |
| CanineModel | break | clean | constraint violation (seq\_len) |
| CpmAntModel | break | clean | constraint violation (seq\_len) |
| DecisionTransformerModel | break | clean | constraint violation (seq\_len) |
| IdeficsModel | break | clean | constraint violation (seq\_len) |
| PixtralVisionModel | break | clean | constraint violation (batch) |
| TapasModel | break | clean | constraint violation (batch) |
| UnivNetModel | break | break | fake tensor error (randn with symbolic shape) |
| YosoModel | clean | break (train) | subgraph tracer error |

7 of 9 are constraint violations exclusive to `mark_dynamic`. These models' code specializes on dimensions we marked as dynamic — they are fundamentally incompatible with `mark_dynamic` semantics.

## 6. Architecture Family Breakdown

| Family | Count | Primary Break Cause | Actionable? |
|--------|-------|---------------------|-------------|
| Encoder-decoder seq2seq | 23 | `copy.deepcopy()` (21/23) | **YES** — single HF PR |
| Audio/speech (Wav2Vec2 family) | 14 | Skipped callable (14/14) + requires\_grad (11/14 train) | **YES** (eval) |
| LLM/MoE/SSM | 13 | Data-dependent branching (10/13) | No |
| Detection/grounding (DETR family) | 11 | Proxy conversion failure (9/11) | **YES** — Dynamo fix |
| Vision encoder (Glm, Siglip, Paddle) | 10 | Data-dependent guard (8/10) | Partially |
| Multimodal (Aria, InternVL, etc.) | 8 | Logger (5/8), data-dependent (3/8) | Partially |
| Time-series/forecasting | 3 | Data-dependent branching, non-Tensor return | No |
| Other | 37 | Mixed | Varies |

**Source distribution:** All 119 graph-break models are from HuggingFace Transformers (`hf`). All 5 Diffusers models compile cleanly in all configurations.

## 7. Mode-Specific Patterns

### 7.1 Train-Only Breaks (17 models, static)

All but 2 are caused by **data-dependent branching** in gradient computation paths:

BioGptModel, Blip2Model, CohereAsrModel, ConditionalDetrModel, DetrModel, FlaubertModel, IBertModel, InstructBlipModel, InstructBlipVideoModel, Kosmos2Model, OPTModel, TableTransformerModel, XGLMModel (data-dependent branching); Phi4MultimodalAudioModel (data-dependent guard); SEWDModel (requires\_grad mutation); MusicgenMelodyModel, MusicgenModel (unclassified/step\_unsupported)

### 7.2 Eval-Only Breaks (3 models, static)

| Model | Eval Cause | Train Status |
|-------|-----------|-------------|
| FalconMambaModel | Forbidden `mark_static_address` | Clean |
| MambaModel | Forbidden `mark_static_address` | Clean |
| FastSpeech2ConformerModel | Data-dependent branching | Eager error |

### 7.3 Audio Model Pattern

The 14 audio/speech models exhibit a distinctive dual-break pattern:
- **Eval:** Skipped function call (feature extractor marked as skipped)
- **Train:** Tensor `requires_grad` mutation (in addition to skipped callable in some)

Fixing the skipped callable alone makes eval clean but not train.

## 8. Multi-Break-Point Models

Some models hit multiple distinct graph break points in a single forward pass. The `break_reasons` array captures these:

| Model | Break Points | Primary Categories |
|-------|-------------|-------------------|
| AriaModel | 27 | Dynamic shape op (4), Tensor.item() (8), context manager (1), ContextVar (2), step\_unsupported (2) |
| VideoLlama3VisionModel | ~12 | Dynamic shape op (2), data-dependent guard (8), step\_unsupported (2) |
| Glm vision models (4) | ~4 each | Data-dependent guard, dynamic shape op, step\_unsupported |

These models would need multiple fixes to become fully clean.

## 9. Recommendations

### For PyTorch (Dynamo) Team

1. **Logger support** — 12 models, medium effort, high ROI. These are popular models (Llama-based multimodal, Longformer, SeamlessM4T).
2. **Un-skip audio callables** — 14 models, medium effort. All Wav2Vec2-family.
3. **`as_proxy()` for detection types** — 11 models, medium effort. All DETR-family.
4. **`mark_static_address` support** — 2 models (Mamba, FalconMamba), low effort.

### For HuggingFace Transformers Team

1. **Replace `copy.deepcopy()` with `clone()`** — 22 models, single PR, mechanical change.
2. **Refactor `requires_grad` mutation** in audio models — 11 models, moderate refactor.
3. **Guard data-dependent branches** with `torch._check*()` where feasible.

### For Model Authors

1. **Avoid `if tensor.value():` control flow** — use `torch.cond()` or restructure.
2. **Avoid `copy.deepcopy()` in `forward()`** — use explicit cloning.
3. **Avoid `logging.Logger` calls in `forward()`** — move to `__init__` or use compile-safe alternatives.

## Appendix A: Full Model Status Table (118 models)

| Model | S/eval | S/train | M/eval | M/train | T/eval | T/train | Primary Cause |
|-------|--------|---------|--------|---------|--------|---------|--------------|
| AriaModel | GB | GB | GB | GB | GB | GB | context manager / Logger |
| AriaTextModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| AutoformerModel | GB | GB | GB | GB | GB | GB | fake tensor error |
| BambaModel | GB | GB | GB | GB | GB | GB | Logger |
| BartModel | GB | GB | GB | GB | GB | GB | deepcopy |
| BigBirdPegasusModel | GB | GB | GB | GB | GB | GB | deepcopy |
| BioGptModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| BlenderbotModel | GB | GB | GB | GB | GB | GB | deepcopy |
| BlenderbotSmallModel | GB | GB | GB | GB | GB | GB | deepcopy |
| Blip2Model | C | GB | C | GB | C | GB | data-dep branching (train) |
| BrosModel | C | C | GB | GB | C | C | constraint violation (mark) |
| CanineModel | C | C | GB | GB | C | C | constraint violation (mark) |
| CohereAsrModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| ConditionalDetrModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| CpmAntModel | C | C | GB | GB | C | C | constraint violation (mark) |
| DFineModel | GB | GB | GB | GB | GB | GB | proxy conversion |
| Data2VecAudioModel | GB | GB | GB | GB | GB | GB | skipped callable |
| DbrxModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| DecisionTransformerModel | C | C | GB | GB | C | C | constraint violation (mark) |
| DeformableDetrModel | GB | GB | GB | GB | GB | GB | proxy conversion |
| DetrModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| EncodecModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| FSMTModel | GB | GB | GB | GB | GB | GB | deepcopy |
| FalconH1Model | GB | GB | GB | GB | GB | GB | Logger |
| FalconMambaModel | GB | C | GB | GB | GB | GB | forbidden callable |
| FastSpeech2ConformerModel | GB | EE | GB | EE | GB | EE | data-dep branching |
| FlaubertModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| Florence2Model | GB | GB | GB | GB | GB | GB | Logger |
| FunnelModel | GB | GB | CE | CE | CE | CE | data-dependent guard |
| Glm4vMoeVisionModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| Glm4vVisionModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| GlmImageVisionModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| GlmOcrVisionModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| GotOcr2Model | GB | GB | GB | GB | GB | GB | Logger |
| GraniteMoeHybridModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| GroundingDinoModel | GB | GB | GB | GB | GB | GB | proxy conversion |
| HubertModel | GB | GB | GB | GB | GB | GB | skipped callable |
| IBertModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| IdeficsModel | C | C | GB | GB | C | C | constraint violation (mark) |
| InformerModel | GB | GB | GB | GB | GB | GB | non-Tensor return |
| InstructBlipModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| InstructBlipVideoModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| InternVLModel | GB | GB | GB | GB | GB | GB | Logger |
| JambaModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| Kosmos2Model | C | GB | C | GB | C | GB | data-dep branching (train) |
| LEDModel | GB | GB | GB | GB | GB | GB | Logger |
| LongT5Model | GB | GB | GB | GB | GB | GB | observed exception |
| LongcatFlashModel | GB | GB | GB | GB | GB | GB | fake tensor error |
| LongformerModel | GB | GB | GB | GB | GB | GB | Logger |
| LwDetrModel | GB | GB | GB | GB | GB | GB | proxy conversion |
| M2M100Model | GB | GB | GB | GB | GB | GB | deepcopy |
| MBartModel | GB | GB | GB | GB | GB | GB | deepcopy |
| MMGroundingDinoModel | GB | GB | GB | GB | GB | GB | proxy conversion |
| MT5Model | GB | GB | GB | GB | GB | GB | deepcopy |
| MambaModel | GB | C | GB | GB | GB | GB | forbidden callable |
| MarianModel | GB | GB | GB | GB | GB | GB | deepcopy |
| MimiModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| MoonshineModel | GB | GB | GB | GB | GB | GB | deepcopy |
| MoonshineStreamingModel | GB | GB | GB | GB | GB | GB | deepcopy |
| MraModel | GB | GB | GB | GB | GB | GB | requires\_grad\_() |
| MusicgenMelodyModel | C | GB | C | GB | C | GB | step\_unsupported (train) |
| MusicgenModel | C | GB | C | GB | C | GB | step\_unsupported (train) |
| MvpModel | GB | GB | GB | GB | GB | GB | deepcopy |
| NemotronHModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| NllbMoeModel | GB | GB | GB | GB | GB | GB | non-Tensor return |
| OPTModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| OlmoHybridModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| PLBartModel | GB | GB | GB | GB | GB | GB | deepcopy |
| PPDocLayoutV2Model | GB | GB | GB | GB | GB | GB | proxy conversion |
| PPDocLayoutV3Model | GB | GB | GB | GB | GB | GB | proxy conversion |
| PaddleOCRVisionModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| PaliGemmaModel | GB | GB | GB | GB | GB | GB | Logger |
| PeAudioModel | GB | GB | GB | GB | GB | GB | non-Tensor return |
| PegasusModel | GB | GB | GB | GB | GB | GB | deepcopy |
| PegasusXModel | GB | GB | GB | GB | GB | GB | deepcopy |
| Phi4MultimodalAudioModel | C | GB | C | GB | C | GB | data-dependent guard (train) |
| PixtralVisionModel | C | C | GB | GB | C | C | constraint violation (mark) |
| ProphetNetModel | GB | GB | CE | CE | CE | CE | deepcopy |
| Qwen3NextModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| Qwen3\_5Model | GB | GB | GB | GB | GB | GB | data-dep branching |
| Qwen3\_5MoeModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| Qwen3\_5MoeTextModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| Qwen3\_5TextModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| RTDetrModel | GB | GB | GB | GB | GB | GB | proxy conversion |
| RTDetrV2Model | GB | GB | GB | GB | GB | GB | proxy conversion |
| RecurrentGemmaModel | GB | GB | EE | EE | EE | EE | skipped callable |
| ReformerModel | GB | GB | GB | GB | GB | GB | data-dep branching |
| RwkvModel | GB | GB | GB | GB | GB | GB | Logger |
| SEWDModel | C | GB | C | GB | C | GB | requires\_grad (train) |
| SEWModel | GB | GB | GB | GB | GB | GB | skipped callable |
| SeamlessM4TModel | GB | GB | GB | GB | GB | GB | Logger |
| SeamlessM4Tv2Model | GB | GB | GB | GB | GB | GB | Logger |
| Siglip2Model | GB | GB | GB | GB | GB | GB | data-dependent guard |
| Siglip2VisionModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| Speech2TextModel | GB | GB | GB | GB | GB | GB | deepcopy |
| SpeechEncoderDecoderModel | GB | GB | GB | GB | GB | GB | skipped callable |
| SpeechT5Model | GB | GB | GB | GB | GB | GB | skipped callable |
| SwitchTransformersModel | GB | GB | GB | GB | GB | GB | observed exception |
| T5Model | GB | GB | GB | GB | GB | GB | deepcopy |
| TableTransformerModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| TapasModel | C | C | GB | GB | C | C | constraint violation (mark) |
| TimeSeriesTransformerModel | GB | GB | GB | GB | GB | GB | deepcopy |
| UMT5Model | GB | GB | GB | GB | GB | GB | deepcopy |
| UdopModel | GB | GB | GB | GB | GB | GB | observed exception |
| UniSpeechModel | GB | GB | GB | GB | GB | GB | skipped callable |
| UniSpeechSatModel | GB | GB | GB | GB | GB | GB | skipped callable |
| UnivNetModel | C | C | GB | GB | GB | GB | fake tensor error |
| VideoLlama3VisionModel | GB | GB | GB | GB | GB | GB | dynamic shape op |
| ViltModel | GB | GB | GB | GB | GB | GB | data-dependent guard |
| VitsModel | GB | GB | GB | GB | GB | GB | skipped callable |
| Wav2Vec2BertModel | GB | GB | GB | GB | GB | GB | skipped callable |
| Wav2Vec2ConformerModel | GB | GB | GB | GB | GB | GB | skipped callable |
| Wav2Vec2Model | GB | GB | GB | GB | GB | GB | skipped callable |
| WavLMModel | GB | GB | GB | GB | GB | GB | skipped callable |
| WhisperModel | GB | GB | GB | GB | GB | GB | deepcopy |
| XGLMModel | C | GB | C | GB | C | GB | data-dep branching (train) |
| XcodecModel | GB | GB | GB | GB | GB | GB | skipped callable |
| YosoModel | C | C | C | C | C | GB | subgraph tracer (T/train) |

**Legend:** GB = graph\_break, C = clean, EE = eager\_error, CE = create\_error

## Appendix B: Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |
| Python | 3.12.13 |
| CUDA | 12.8 |
| GPU | NVIDIA A100 80GB |
| Backend | `eager` (tests Dynamo tracing, not inductor codegen) |
| Batch size | 2 (avoids specialization on 0/1) |
