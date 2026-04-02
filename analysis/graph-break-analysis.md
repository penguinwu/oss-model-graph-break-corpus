# Graph Break Analysis — R8 Corpus (110 Models)

**Date:** 2026-04-01
**Corpus:** 468 HF Transformers + Diffusers models, PyTorch 2.10.0, A100 GPU
**Coverage:** 352 clean (75%), 93 graph_break eval (20%), 13 eager_error (3%)

## Executive Summary

110 of 468 HF models have graph breaks in at least one mode (eval or train). The breaks cluster into 10 root causes, with **3 actionable categories covering 45% of all broken models** (49/110) that can be fixed via PyTorch core or HF model patches.

## Fix Impact Matrix

| Fix | Owner | Models Fixed | % of Breaks | Effort |
|-----|-------|-------------|-------------|--------|
| Replace `copy.deepcopy()` with `clone()` | HF Transformers | 21 | 19% | Low — single PR |
| Support `Logger` methods in Dynamo | PyTorch Dynamo | 12 | 11% | Medium — skip/inline |
| Un-skip audio feature extractor callables | PyTorch Dynamo | 12 | 11% | Medium — whitelist |
| Implement `as_proxy()` for detection outputs | PyTorch Dynamo | 11 | 10% | Medium |
| Support `mark_static_address` in Dynamo | PyTorch Dynamo | 2 | 2% | Low |
| **Subtotal (actionable)** | | **49** | **45%** | |
| Data-dependent branching | Model authors | 28 | 25% | Hard — torch.cond() |
| Unbacked symbols | Model authors | 10 | 9% | Hard — shape redesign |
| Other (try/except, requires_grad, etc.) | Various | 12 | 11% | Medium |
| **Subtotal (inherently hard)** | | **50** | **45%** | |
| **Unclassified / other** | | **4** | **4%** | |

## Breakdown by Architecture Family

| Family | Count | Primary Break Cause | Actionable? |
|--------|-------|-------------------|-------------|
| Encoder-decoder (Bart, T5, Whisper, etc.) | 23 | copy.deepcopy() (21/23) | YES — single HF PR |
| Vision/multimodal (Glm, Siglip, BLIP, etc.) | 20 | Mixed: unbacked symbols (5), data-dependent (5), as_proxy (2), logger (5) | Partially |
| LLM/MoE/SSM (Qwen3, Falcon, Mamba, etc.) | 16 | Data-dependent branching (10/16) | NO — inherent |
| Audio/speech (Wav2Vec2, HuBERT, etc.) | 13 | Skipped callable (12/13) | YES — Dynamo whitelist |
| Detection/grounding (DETR, RT-DETR, etc.) | 11 | as_proxy() missing (9/11) | YES — Dynamo fix |
| NLP/seq2seq (LED, Longformer, etc.) | 5 | Logger (4/5) | YES — Dynamo fix |
| Other | 22 | Mixed | Partially |

## Key Findings for PT2 Team

### 1. Three PRs Would Fix 45 Models

- **HF PR: Replace deepcopy with clone()** in encoder-decoder models (21 models). All use `copy.deepcopy()` to clone decoder layers from encoder. A mechanical replacement.
- **Dynamo: Un-skip audio feature extractor functions** (12 models). All Wav2Vec2-family models call conv-based feature extractors marked as "skipped" by Dynamo.
- **Dynamo: Support Logger methods** (12 models). `logging.Logger` method calls inside `forward()` cause breaks for non-export cases.

### 2. Data-Dependent Branching Is the Largest Hard Category

28 models (25%) use control flow that depends on tensor values at runtime. This includes:
- MoE expert routing (Qwen3_5, Qwen3_5Moe, Dbrx, Jamba, GraniteMoeHybrid)
- Encoder-decoder cross-attention masking (Blip2, InstructBlip, Kosmos2)
- Positional encoding conditionals (OPT, BioGpt, Reformer)
- Detection NMS/filtering (ConditionalDetr, Detr, TableTransformer)

These require `torch.cond()` or model restructuring — no quick fix.

### 3. Unbacked Symbols Emerged as a New Category

R8 exposed 10 models with unbacked symbol issues (was 2 in R7). These are primarily:
- **Glm vision models** (4): Position indices with data-dependent shapes from `grid_thw`
- **VideoLlama3Vision**: Same pattern — pre-patchified inputs with dynamic grid
- **Dbrx, PaddleOCR, Phi4MultimodalAudio**: Various data-dependent shape computations

### 4. Mode Distribution

- 90 models break in both eval AND train
- 3 models break in eval only (FalconMamba, FastSpeech2Conformer, Mamba)
- 17 models break in train only (Blip2, CohereAsr, InstructBlip, etc.)

Train-only breaks are typically data-dependent branching in gradient computation paths.

### 5. Coverage After R8

| Metric | R1 | R5 | R7 | R8 |
|--------|----|----|----|----|
| Clean (eval) | 220 | 261 | 296 | **352** |
| Graph break (eval) | 56 | 78 | 69 | **93** |
| Eager error | 72 | 105 | 95 | **13** |
| Models with any break | ~60 | ~90 | 80 | **110** |

The graph break count increased because 24 formerly-masked models (hidden behind input/config failures) now test successfully and reveal real graph breaks. This is positive signal — the corpus is more complete.
