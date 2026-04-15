# Error Models — Why Certain Models Are Omitted

This document explains why certain HuggingFace Transformers models produce errors
in the sweep and cannot be included in the corpus. Updated as of PyTorch 2.10.0
sweep (v2.12).

## Summary

| Category | Count | Models |
|----------|-------|--------|
| Skipped pre-sweep (models.py) | 14 | See [Pre-Sweep Skip List](#pre-sweep-skip-list) |
| Eager errors | 3 | See [Eager Errors](#eager-errors) |
| Transient failures | 10 | See [Transient Failures](#transient-failures) |
| Missing dependencies | 3 | See [Missing Dependencies](#missing-dependencies) |
| Environment issues | 4 | See [Environment Issues](#environment-issues) |
| Train-only errors | 4 | See [Train-Only Errors](#train-only-errors) |

## Pre-Sweep Skip List

These models are excluded in `models.py:_SKIP_MODELS` before the sweep runs,
because they cannot be instantiated or invoked in a standard forward pass.

| Model | Reason |
|-------|--------|
| T5GemmaEncoderModel | Raises "only supports encoder-only" — use T5GemmaModel instead |
| BarkModel | Abstract — `forward()` is `_forward_unimplemented` |
| BartPretrainedModel | Abstract — `forward()` is `_forward_unimplemented` |
| PretrainedBartModel | Abstract — `forward()` is `_forward_unimplemented` |
| PretrainedFSMTModel | Abstract — `forward()` is `_forward_unimplemented` |
| Qwen3OmniMoeTalkerFCG | Internal sub-model: needs `inputs_embeds` from parent Thinker model |
| Qwen3OmniMoeCode2WavTransformerModel | Internal sub-model: does not accept `input_ids` |
| Qwen2_5OmniTalkerFCG | Internal sub-model: needs `inputs_embeds` from parent Thinker model |
| Qwen2_5OmniToken2WavModel | Internal sub-model: requires `batch_size=1` |
| Sam2VideoModel | Stateful: `forward()` requires `inference_session` with frame tracking state |
| Sam3VideoModel | Stateful: `forward()` requires `inference_session` with frame tracking state |
| Sam3TrackerVideoModel | Stateful: `forward()` requires `inference_session` with frame tracking state |
| EdgeTamVideoModel | Stateful: `forward()` requires `inference_session` with frame tracking state |
| PI0ForConditionalGeneration | `image_token_id` (257152) == `vocab_size`; model size reduction makes image token embedding out of range |

### Why video session models can't be tested

Sam2Video, Sam3Video, Sam3TrackerVideo, and EdgeTamVideo models use a stateful
inference session pattern. Their `forward()` requires an `inference_session` object
that tracks per-frame state across multiple calls:

```python
# Session maintains: processed_frames dict, frame_idx counter, memory banks
session = model.init_inference_session(...)
for frame_idx, frame in enumerate(video_frames):
    out = model(frames=frame, inference_session=session)
    # Session state mutates: session.processed_frames[frame_idx] = ...
```

This is fundamentally incompatible with `torch.compile(fullgraph=True)` because:
1. Dict key lookups on dynamic frame indices cause graph breaks
2. Session state mutation across calls prevents static graph tracing
3. The forward pass fails without prior frames populating the session

### Why PI0ForConditionalGeneration can't be tested

PI0 is a robotics policy model (VLM + DiT architecture) that requires:
- `pixel_values` (B, num_cameras, C, H, W) — multi-camera image inputs
- `pixel_attention_mask` (B, num_cameras) — camera validity mask
- `input_ids` with embedded `image_token_id` placeholders (token ID 257152)
- `state`, `noise`, `timestep` tensors for action denoising

The `image_token_id` equals `vocab_size` (257152). The sweep's `_reduce_model_size()`
shrinks vocab to fit in GPU memory, making the image token ID out of range for the
embedding layer. Without reduction, the model's embedding alone exceeds GPU memory.

## Eager Errors

Models that create successfully but fail during eager (non-compiled) forward pass.
These represent genuine model issues, not compilation problems.

| Model | Error | Root Cause |
|-------|-------|------------|
| ClvpModelForConditionalGeneration | `You have to specify either input_ids or inputs_embeds` | CLVP FCG needs specialized inputs (text + speech + conditioning). Generic input handler doesn't provide the right combination. Fixable with a custom handler. |
| PI0Model | `'NoneType' object has no attribute 'get_seq_length'` | PI0 is a VLM (PaliGemma + DiT) that requires `pixel_values` for vision backbone. Without images, `past_key_values` is None. Fixable with custom handler but model is very large (257K vocab). |
| PeAudioVideoModel | `KeyError: None` in ModernBERT | Internal ModernBERT component generates a `None` key during `forward()`. Bug in the HF implementation — not related to our input generation. |

## Transient Failures

Models that pass when run individually but fail intermittently in multi-worker
sweep (4 workers). These pass on retry and are classified as "flaky" by the
auto-retry system.

**Root cause**: Unknown. NOT GPU OOM (verified with 25% GPU memory fraction —
model uses 7.6GB, well within 10GB limit). All models pass in train sweep and
when run serially. The error is always `list index out of range` in eval mode
with 4 workers.

| Model | Modes Affected |
|-------|----------------|
| Qwen3_5Model | eval |
| Qwen3_5TextModel | eval |
| Qwen3_5ForCausalLM | eval |
| Qwen3_5ForConditionalGeneration | eval |
| Qwen3_5MoeModel | eval |
| Qwen3_5MoeTextModel | eval |
| Qwen3_5MoeForCausalLM | eval |
| Qwen3_5MoeForConditionalGeneration | eval |
| Qwen3NextModel | eval |
| Qwen3NextForCausalLM | eval |
| OlmoHybridModel | eval |
| OlmoHybridForCausalLM | eval |

The auto-retry feature in `run_sweep.py` re-runs these models serially after
the main sweep pass. Models that pass on retry are marked "flaky" and their
retry results are used in the corpus.

## Missing Dependencies

Models that require external libraries not installed in the sweep environment.

| Model | Dependency | Notes |
|-------|------------|-------|
| DinatModel | `natten` | Neighborhood attention library (shi-labs.com/natten). Specialized CUDA kernels. |
| LayoutLMv2Model | `detectron2` | Facebook's object detection library. Large dependency with CUDA build requirements. |

These are not installed because they are specialized libraries with complex build
requirements. They could be added in the future if needed.

## Environment Issues

Models that fail due to environment configuration, not model logic.

| Model | Error | Notes |
|-------|-------|-------|
| LlavaForConditionalGeneration | `dlopen: libtorch_cuda_linalg.so not found` | Missing CUDA linalg shared library. Environment-specific — works on other setups. Only affects eval mode. |
| EdgeTamModel | Can't load `timm/repvit_m1.dist_in1k` config | Requires HuggingFace Hub network access to download timm backbone config. Blocked on devvm. |
| EdgeTamVisionModel | Can't load `timm/repvit_m1.dist_in1k` config | Same as EdgeTamModel — shares the same timm backbone. |

## Train-Only Errors

Models that pass in eval mode but fail in train mode.

| Model | Error | Root Cause |
|-------|-------|------------|
| Zamba2Model | `eager_error: 1` | Likely an assertion failure in the Mamba2/attention hybrid architecture during backward pass. Only fails in train mode. |
| Zamba2ForCausalLM | `eager_error: 1` | Same as Zamba2Model — shares the same backbone. |
| ZambaModel | `create_error: tie_weights_keys` issue | Weight tying regex pattern fails during model creation in train mode. Bug in HF model definition. |
| ZambaForCausalLM | `create_error: tie_weights_keys` issue | Same as ZambaModel — shares the same backbone. |
| Sam3Model | `worker_error` during compile | Times out or crashes during compilation in train mode. Passes in eval. |
| Sam3TrackerModel | `worker_error` during eager | Crashes during eager forward in train mode. Passes in eval. |
