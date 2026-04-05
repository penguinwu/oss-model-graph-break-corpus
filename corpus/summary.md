# Corpus Summary

**468 models** from HuggingFace Transformers + Diffusers, tested on PyTorch 2.10.0.

*Auto-generated from corpus.json — do not edit manually.*

## Status Distribution

| Status | Eval | Train |
|--------|------|-------|
| full_graph | 337 (72%) | 323 (69%) |
| graph_break | 90 (19%) | 105 (22%) |
| create_error | 18 (4%) | 18 (4%) |
| eager_error | 17 (4%) | 17 (4%) |
| worker_error | 6 (1%) | 5 (1%) |

## Dynamic Shape Comparison (eval)

| Status | Static | Mark | True |
|--------|--------|------|------|
| full_graph | 337 | 335 | 339 |
| graph_break | 90 | 97 | 90 |
| create_error | 18 | 18 | 18 |
| eager_error | 17 | 18 | 18 |
| timeout | 0 | 0 | 3 |

## Key Findings

- **116 unique models** with graph breaks in any config/mode (86 break in all 6 configs)
- `mark_dynamic` is **stricter** than `dynamic=true` (335 vs 339 full_graph eval)
- Full analysis: [analysis/graph-break-analysis.md](../analysis/graph-break-analysis.md)

## Graph Break Models (static eval)

90 models:

- **AriaModel** — builtin callable
- **AriaTextModel** — data-dependent branching
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
- **Florence2Model** — builtin callable
- **Glm4vMoeVisionModel** — fake tensor error
- **Glm4vVisionModel** — fake tensor error
- **GlmImageVisionModel** — fake tensor error
- **GlmOcrVisionModel** — fake tensor error
- **GotOcr2Model** — builtin callable
- **GraniteMoeHybridModel** — data-dependent branching
- **GroundingDinoModel** — proxy conversion failure
- **HubertModel** — skipped function call
- **InformerModel** — non-Tensor return
- **InternVLModel** — builtin callable
- **JambaModel** — data-dependent branching
- **LEDModel** — logging.Logger
- **LongT5Model** — observed exception
- **LongcatFlashModel** — data-dependent guard
- **LongformerModel** — logging.Logger
- **LwDetrModel** — proxy conversion failure
- **M2M100Model** — copy.deepcopy
- **MBartModel** — copy.deepcopy
- **MMGroundingDinoModel** — proxy conversion failure
- **MT5Model** — copy.deepcopy
- **MambaModel** — forbidden callable
- **MarianModel** — copy.deepcopy
- **MimiModel** — fake tensor error
- **MoonshineModel** — copy.deepcopy
- **MoonshineStreamingModel** — copy.deepcopy
- **MraModel** — unsupported requires_grad_()
- **MvpModel** — copy.deepcopy
- **NemotronHModel** — data-dependent branching
- **NllbMoeModel** — copy.deepcopy
- **OlmoHybridModel** — data-dependent branching
- **PLBartModel** — copy.deepcopy
- **PPDocLayoutV2Model** — proxy conversion failure
- **PPDocLayoutV3Model** — proxy conversion failure
- **PaddleOCRVisionModel** — data-dependent guard
- **PaliGemmaModel** — builtin callable
- **PeAudioModel** — non-Tensor return
- **PegasusModel** — copy.deepcopy
- **PegasusXModel** — copy.deepcopy
- **Qwen3NextModel** — data-dependent branching
- **Qwen3_5Model** — data-dependent branching
- **Qwen3_5MoeModel** — data-dependent branching
- **Qwen3_5MoeTextModel** — data-dependent branching
- **Qwen3_5TextModel** — data-dependent branching
- **RTDetrModel** — proxy conversion failure
- **RTDetrV2Model** — proxy conversion failure
- **RecurrentGemmaModel** — builtin callable
- **ReformerModel** — data-dependent branching
- **RwkvModel** — logging.Logger
- **SEWModel** — skipped function call
- **SeamlessM4TModel** — logging.Logger
- **SeamlessM4Tv2Model** — logging.Logger
- **Siglip2Model** — proxy conversion failure
- **Siglip2VisionModel** — proxy conversion failure
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
- **VideoLlama3VisionModel** — fake tensor error
- **ViltModel** — data-dependent branching
- **VitsModel** — skipped function call
- **Wav2Vec2BertModel** — skipped function call
- **Wav2Vec2ConformerModel** — skipped function call
- **Wav2Vec2Model** — skipped function call
- **WavLMModel** — skipped function call
- **WhisperModel** — copy.deepcopy
- **XcodecModel** — skipped function call
