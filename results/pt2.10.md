# PyTorch 2.10 Sweep Results

**Date:** 2026-04-15 (expanded corpus)
**Models:** 2031 (468 original + 1563 expanded)

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |

## Summary (2031 models, all sources)

|  | eval | train |
|---|---|---|
| **full\_graph** | 1513 (74%) | 1478 (73%) |
| **graph\_break** | 166 (8%) | 196 (10%) |
| **eager\_error** | 41 (2%) | 42 (2%) |
| **create\_error** | 26 (1%) | 24 (1%) |
| **timeout** | 269 (13%) | 269 (13%) |
| **worker\_error** | 11 | 13 |
| **zombie** | 5 | 5 |

- **235 models** have graph breaks in at least one mode
- **1478 models** compile fully (full\_graph) in both modes
- Timeout rate is elevated due to large generative models (ForCausalLM variants)

## Original 468 Models (version trend comparison)

|  | eval | train |
|---|---|---|
| **full\_graph** | 337 (72%) | 323 (69%) |
| **graph\_break** | 90 | 105 |
| **error** | 41 | 40 |

## Changes from 2.9 (original 468 models)

- **12 graph break fixes** (graph\_break -> full\_graph), **0 regressions**

### Fixes (12 model x mode pairs)

| Model | Mode | 2.9 | 2.10 |
|-------|------|-----|------|
| BltModel | eval, train | graph\_break | full\_graph |
| FlaubertModel | eval | graph\_break | full\_graph |
| HiggsAudioV2Model | eval, train | graph\_break | full\_graph |
| Idefics2Model | eval, train | graph\_break | full\_graph |
| Phi4MultimodalAudioModel | eval | graph\_break | full\_graph |
| Phi4MultimodalVisionModel | eval, train | graph\_break | full\_graph |
| PixtralVisionModel | eval, train | graph\_break | full\_graph |
| Sam3Model | eval, train | graph\_break | full\_graph |
| TapasModel | eval, train | graph\_break | full\_graph |
| VJEPA2Model | eval, train | graph\_break | full\_graph |
| XLMModel | eval, train | graph\_break | full\_graph |
| XmodModel | eval, train | graph\_break | full\_graph |

## Corpus Expansion

The expanded corpus adds ForCausalLM and ForConditionalGeneration variants of existing families, testing the full model stack users actually compile. This revealed additional graph breaks in vision-text merge paths invisible at the base model level.
