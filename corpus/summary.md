# Corpus Summary

**468 models** from HuggingFace Transformers + Diffusers, tested on PyTorch 2.10.0.

*Auto-generated from corpus.json — do not edit manually.*

## Status Distribution

| Status | Eval | Train |
|--------|------|-------|
| clean | 352 (75%) | 337 (72%) |
| graph_break | 92 (20%) | 106 (23%) |
| eager_error | 13 (3%) | 14 (3%) |
| create_error | 7 (1%) | 7 (1%) |
| timeout | 4 (1%) | 4 (1%) |

## Dynamic Shape Comparison (eval)

| Status | Static | Mark | True |
|--------|--------|------|------|
| clean | 352 | 335 | 339 |
| graph_break | 92 | 97 | 90 |
| eager_error | 13 | 18 | 18 |
| create_error | 7 | 18 | 18 |
| timeout | 4 | 0 | 3 |

## Key Findings

- **118 unique models** with graph breaks in any config/mode (85 break in all 6 configs)
- **3 targeted PRs** (deepcopy, Logger, audio callables) would fix ~48 models (45%)
- `mark_dynamic` is **stricter** than `dynamic=true` (335 vs 339 clean eval)
- Full analysis: [analysis/graph-break-analysis.md](../analysis/graph-break-analysis.md)

## Graph Break Models (static eval)

92 models:

- **AriaModel** — unsupported context manager
- **AriaTextModel** — data-dependent guard
- **AutoformerModel** — fake tensor error
- **BambaModel** — logging.Logger
- **BartModel** — copy.deepcopy
- **BigBirdPegasusModel** — copy.deepcopy
- **BlenderbotModel** — copy.deepcopy
- **BlenderbotSmallModel** — copy.deepcopy
- **DFineModel** — proxy conversion failure
- **Data2VecAudioModel** — skipped function call
- **DbrxModel** — data-dependent guard
- **DeformableDetrModel** — proxy conversion failure
- **EncodecModel** — data-dependent branching
- **FSMTModel** — copy.deepcopy
- **FalconH1Model** — logging.Logger
- **FalconMambaModel** — forbidden callable
- **FastSpeech2ConformerModel** — data-dependent branching
- **Florence2Model** — logging.Logger
- **FunnelModel** — data-dependent guard
- **Glm4vMoeVisionModel** — data-dependent guard
- **Glm4vVisionModel** — data-dependent guard
- **GlmImageVisionModel** — data-dependent guard
- **GlmOcrVisionModel** — data-dependent guard
- **GotOcr2Model** — logging.Logger
- **GraniteMoeHybridModel** — data-dependent branching
- **GroundingDinoModel** — proxy conversion failure
- **HubertModel** — skipped function call
- **InformerModel** — non-Tensor return
- **InternVLModel** — logging.Logger
- **JambaModel** — data-dependent branching
- **LEDModel** — logging.Logger
- **LongT5Model** — observed exception
- **LongcatFlashModel** — fake tensor error
- **LongformerModel** — logging.Logger
- **LwDetrModel** — proxy conversion failure
- **M2M100Model** — copy.deepcopy
- **MBartModel** — copy.deepcopy
- **MMGroundingDinoModel** — proxy conversion failure
- **MT5Model** — copy.deepcopy
- **MambaModel** — forbidden callable
- **MarianModel** — copy.deepcopy
- **MimiModel** — data-dependent guard
- **MoonshineModel** — copy.deepcopy
- **MoonshineStreamingModel** — copy.deepcopy
- **MraModel** — unsupported requires_grad_()
- **MvpModel** — copy.deepcopy
- **NemotronHModel** — data-dependent branching
- **NllbMoeModel** — non-Tensor return
- **OlmoHybridModel** — data-dependent branching
- **PLBartModel** — copy.deepcopy
- **PPDocLayoutV2Model** — proxy conversion failure
- **PPDocLayoutV3Model** — proxy conversion failure
- **PaddleOCRVisionModel** — data-dependent guard
- **PaliGemmaModel** — logging.Logger
- **PeAudioModel** — non-Tensor return
- **PegasusModel** — copy.deepcopy
- **PegasusXModel** — copy.deepcopy
- **ProphetNetModel** — copy.deepcopy
- **Qwen3NextModel** — data-dependent branching
- **Qwen3_5Model** — data-dependent branching
- **Qwen3_5MoeModel** — data-dependent branching
- **Qwen3_5MoeTextModel** — data-dependent branching
- **Qwen3_5TextModel** — data-dependent branching
- **RTDetrModel** — proxy conversion failure
- **RTDetrV2Model** — proxy conversion failure
- **RecurrentGemmaModel** — skipped function call
- **ReformerModel** — data-dependent branching
- **RwkvModel** — logging.Logger
- **SEWModel** — skipped function call
- **SeamlessM4TModel** — logging.Logger
- **SeamlessM4Tv2Model** — logging.Logger
- **Siglip2Model** — data-dependent guard
- **Siglip2VisionModel** — data-dependent guard
- **Speech2TextModel** — copy.deepcopy
- **SpeechEncoderDecoderModel** — skipped function call
- **SpeechT5Model** — skipped function call
- **SwitchTransformersModel** — observed exception
- **T5Model** — copy.deepcopy
- **TimeSeriesTransformerModel** — copy.deepcopy
- **UMT5Model** — copy.deepcopy
- **UdopModel** — observed exception
- **UniSpeechModel** — skipped function call
- **UniSpeechSatModel** — skipped function call
- **VideoLlama3VisionModel** — dynamic shape operator
- **ViltModel** — data-dependent guard
- **VitsModel** — skipped function call
- **Wav2Vec2BertModel** — skipped function call
- **Wav2Vec2ConformerModel** — skipped function call
- **Wav2Vec2Model** — skipped function call
- **WavLMModel** — skipped function call
- **WhisperModel** — copy.deepcopy
- **XcodecModel** — skipped function call
