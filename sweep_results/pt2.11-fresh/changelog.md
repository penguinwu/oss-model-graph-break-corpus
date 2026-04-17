# Corpus Update Changelog

**Sweep directory:** `sweep_results/pt2.11-fresh`
**Date:** 2026-04-17 10:14
**PyTorch:** 2.11.0+cu128
**Transformers:** 5.5.3
**Diffusers:** 0.37.1

## Corpus Summary
- Total models: 790
- Models with graph breaks: 240

## Changes (107 total)

### Fixes (4) — graph_break → full_graph
- **FalconMambaForCausalLM** (eval)
- **FalconMambaModel** (eval)
- **MambaForCausalLM** (eval)
- **MambaModel** (eval)

### Other Status Changes (25)
- **BltForCausalLM** (eval): full_graph → timeout
- **BltForCausalLM** (train): full_graph → timeout
- **BltModel** (eval): full_graph → timeout
- **BltModel** (train): full_graph → timeout
- **LlavaForConditionalGeneration** (eval): create_error → graph_break
- **OlmoHybridForCausalLM** (eval): eager_error → graph_break
- **OlmoHybridModel** (eval): eager_error → graph_break
- **Qwen3NextForCausalLM** (eval): eager_error → graph_break
- **Qwen3NextModel** (eval): eager_error → graph_break
- **Qwen3_5ForCausalLM** (eval): eager_error → graph_break
- **Qwen3_5ForConditionalGeneration** (eval): eager_error → graph_break
- **Qwen3_5Model** (eval): eager_error → graph_break
- **Qwen3_5MoeForCausalLM** (eval): eager_error → graph_break
- **Qwen3_5MoeForConditionalGeneration** (eval): eager_error → graph_break
- **Qwen3_5MoeModel** (eval): eager_error → graph_break
- **Qwen3_5MoeTextModel** (eval): eager_error → graph_break
- **Qwen3_5TextModel** (eval): eager_error → graph_break
- **RwkvForCausalLM** (eval): full_graph → timeout
- **RwkvModel** (eval): graph_break → timeout
- **Sam3Model** (train): worker_error → full_graph
- **Sam3TrackerModel** (train): worker_error → full_graph
- **Zamba2ForCausalLM** (eval): full_graph → eager_error
- **Zamba2Model** (eval): full_graph → eager_error
- **ZambaForCausalLM** (eval): graph_break → create_error
- **ZambaModel** (eval): graph_break → create_error

### New Model Families (71)
Entirely new architectures not previously in the corpus.

- **AllegroTransformer3DModel** (eval): eager_error
- **AsymmetricAutoencoderKL** (eval): eager_error
- **AudioLDM2ProjectionModel** (eval): create_error
- **AudioLDM2UNet2DConditionModel** (eval): eager_error
- **AuraFlowTransformer2DModel** (eval): eager_error
- **AutoencoderDC** (eval): eager_error
- **AutoencoderKLAllegro** (eval): eager_error
- **AutoencoderKLCogVideoX** (eval): eager_error
- **AutoencoderKLCosmos** (eval): eager_error
- **AutoencoderKLFlux2** (eval): eager_error
- **AutoencoderKLHunyuanImage** (eval): create_error
- **AutoencoderKLHunyuanImageRefiner** (eval): eager_error
- **AutoencoderKLHunyuanVideo** (eval): eager_error
- **AutoencoderKLHunyuanVideo15** (eval): eager_error
- **AutoencoderKLLTX2Audio** (eval): eager_error
- **AutoencoderKLLTX2Video** (eval): eager_error
- **AutoencoderKLLTXVideo** (eval): eager_error
- **AutoencoderKLMagvit** (eval): eager_error
- **AutoencoderKLMochi** (eval): eager_error
- **AutoencoderKLQwenImage** (eval): eager_error
- **AutoencoderKLTemporalDecoder** (eval): eager_error
- **AutoencoderKLWan** (eval): eager_error
- **AutoencoderOobleck** (eval): eager_error
- **AutoencoderRAE** (eval): eager_error
- **BriaFiboTransformer2DModel** (eval): timeout
- **BriaTransformer2DModel** (eval): timeout
- **CLIPImageProjection** (eval): eager_error
- **ChromaTransformer2DModel** (eval): eager_error
- **ChronoEditTransformer3DModel** (eval): create_error
- **CogVideoXTransformer3DModel** (eval): eager_error
- **CogView3PlusTransformer2DModel** (eval): eager_error
- **CogView4Transformer2DModel** (eval): eager_error
- **ConsisIDTransformer3DModel** (eval): eager_error
- **ConsistencyDecoderVAE** (eval): eager_error
- **ControlNetModel** (eval): eager_error
- **ControlNetUnionModel** (eval): create_error
- **ControlNetXSAdapter** (eval): eager_error
- **CosmosControlNetModel** (eval): eager_error
- **CosmosTransformer3DModel** (eval): eager_error
- **EasyAnimateTransformer3DModel** (eval): create_error
- **Flux2Transformer2DModel** (eval): timeout
- **FluxControlNetModel** (eval): timeout
- **FluxTransformer2DModel** (eval): timeout
- **Gemma4AudioModel** (eval): full_graph
- **GlmImageTransformer2DModel** (eval): eager_error
- **HeliosTransformer3DModel** (eval): timeout
- **HiDreamImageTransformer2DModel** (eval): create_error
- **HunyuanDiT2DControlNetModel** (eval): create_error
- **HunyuanDiT2DModel** (eval): create_error
- **HunyuanImageTransformer2DModel** (eval): timeout
- **HunyuanVideo15Transformer3DModel** (eval): timeout
- **HunyuanVideoFramepackTransformer3DModel** (eval): timeout
- **HunyuanVideoTransformer3DModel** (eval): timeout
- **I2VGenXLUNet** (eval): eager_error
- **Kandinsky3UNet** (eval): eager_error
- **Kandinsky5Transformer3DModel** (eval): eager_error
- **LTX2VideoTransformer3DModel** (eval): timeout
- **LTXVideoTransformer3DModel** (eval): eager_error
- **LatteTransformer3DModel** (eval): create_error
- **LongCatImageTransformer2DModel** (eval): timeout
- **Lumina2Transformer2DModel** (eval): eager_error
- **LuminaNextDiT2DModel** (eval): create_error
- **MochiTransformer3DModel** (eval): eager_error
- **MotionAdapter** (eval): eager_error
- **MusicFlamingoForConditionalGeneration** (eval): eager_error
- **NomicBertModel** (eval): full_graph
- **OmniGenTransformer2DModel** (eval): create_error
- **OvisImageTransformer2DModel** (eval): create_error
- **PRXTransformer2DModel** (eval): eager_error
- **PixArtTransformer2DModel** (eval): eager_error
- **PriorTransformer** (eval): eager_error

### New Configurations (5)
Task-head variants (ForCausalLM, ForConditionalGeneration, etc.) of model families already in the corpus.

- **Gemma4ForCausalLM** (eval): full_graph  ← Gemma4
- **Gemma4ForConditionalGeneration** (eval): compile_error  ← Gemma4
- **Gemma4Model** (eval): full_graph  ← Gemma4
- **Gemma4TextModel** (eval): full_graph  ← Gemma4
- **Gemma4VisionModel** (eval): full_graph  ← Gemma4

### Removed Models (2)
Models in the previous corpus but not in this sweep (--replace mode).

- **UNet2DConditionModel**
- **UNet2DModel**

## Health Warnings

### Persistent Failures (4)
Models with `create_error` in both the previous and current corpus.
These may have been removed upstream — consider removing from corpus.

- **DinatModel**
- **LayoutLMv2Model**
- **ZambaForCausalLM**
- **ZambaModel**

### Missing Explain Data (2 models)
Models with `graph_break` status but no `break_reasons`.
Run an explain pass to fill in break details.

- **AutoformerModel** (eval, train)
- **InformerModel** (eval, train)
