# PyTorch 2.11 Sweep Results

**Date:** 2026-04-12
**Models:** 473 (468 existing + 5 new)

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.5.3 |
| Diffusers | 0.37.1 |

## Summary

|  | eval | train |
|---|---|---|
| **full\_graph** | 350 (74%) | 333 (70%) |
| **graph\_break** | 87 | 104 |
| **create\_error** | 19 | 19 |
| **eager\_error** | 16 | 16 |
| **timeout** | 1 | 1 |

- **104 models** (22%) have graph breaks in at least one mode
- **87 models** break in both eval and train
- All Diffusers models (5/5) compile clean
- All graph breaks come from HF Transformers models

## Changes from 2.10

- **2 graph break fixes**, **0 regressions**

### Fixes

| Model | Mode | 2.10 | 2.11 |
|-------|------|------|------|
| FalconMambaModel | eval | graph\_break | full\_graph |
| MambaModel | eval | graph\_break | full\_graph |

Both are Mamba/SSM architecture models — likely a targeted Dynamo fix for selective scan operations.

### New Models (5)

All from Transformers 5.5.3 (Gemma4 family + NomicBert):

| Model | eval | train |
|-------|------|-------|
| Gemma4AudioModel | full\_graph | full\_graph |
| Gemma4Model | full\_graph | full\_graph |
| Gemma4TextModel | full\_graph | full\_graph |
| Gemma4VisionModel | full\_graph | full\_graph |
| NomicBertModel | full\_graph | full\_graph |

All 5 new models compile clean in both modes.

### Other Status Changes

| Model | eval | train | Notes |
|-------|------|-------|-------|
| ChameleonModel | worker\_error → full\_graph | worker\_error → full\_graph | Worker fix (stale data) |
| ChineseCLIPModel | worker\_error → full\_graph | worker\_error → full\_graph | Worker fix |
| ChineseCLIPVisionModel | worker\_error → full\_graph | worker\_error → full\_graph | Worker fix |
| Cohere2Model | worker\_error → full\_graph | worker\_error → full\_graph | Worker fix |
| Cohere2VisionModel | worker\_error → full\_graph | worker\_error → full\_graph | Worker fix |
| CohereModel | worker\_error → full\_graph | — | Worker fix |
| RwkvModel | graph\_break → timeout | graph\_break → timeout | Model size issue |
| ZambaModel | eager\_error → create\_error | eager\_error → create\_error | Transformers compat |

## Version Trend

| Version | eval full\_graph | train full\_graph | Fixes | Regressions |
|---------|-----------------|-------------------|-------|-------------|
| 2.8 | 298 (64%) | 288 (62%) | — | — |
| 2.9 | 324 (69%) | 314 (67%) | 0 | 0 |
| 2.10 | 337 (72%) | 323 (69%) | 12 | 0 |
| **2.11** | **350 (74%)** | **333 (70%)** | **2** | **0** |

Steady improvement across four releases. Zero regressions in any release.
