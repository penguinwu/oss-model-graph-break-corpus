# PyTorch 2.11 Sweep Results

**Date:** 2026-04-17 (expanded corpus)
**Models:** 790 (714 from pt2.10 expanded corpus + 76 new)

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.5.3 |
| Diffusers | 0.37.1 |

## Summary (790 models)

|  | eval | train |
|---|---|---|
| **full\_graph** | 531 (67%) | 489 (62%) |
| **graph\_break** | 177 (22%) | 219 (28%) |
| **eager\_error** | 49 (6%) | 49 (6%) |
| **create\_error** | 16 (2%) | 16 (2%) |
| **timeout** | 16 (2%) | 15 (2%) |
| **compile\_error** | 1 | 1 |
| **zombie** | — | 1 |

- **240 models** have graph breaks in at least one mode
- **489 models** compile fully (full\_graph) in both modes
- All graph breaks come from HF Transformers models

## Original 468 Models (version trend comparison)

|  | eval | train |
|---|---|---|
| **full\_graph** | 345 (74%) | 328 (70%) |
| **graph\_break** | 87 | 104 |
| **error** | 36 | 36 |

## Changes from 2.10 (714 common models)

- **4 graph break fixes** (graph\_break -> full\_graph), **0 regressions**
- **148 other status changes** (mostly infra improvements: errors/timeouts resolving)

### Fixes

| Model | Mode | 2.10 | 2.11 |
|-------|------|------|------|
| FalconMambaForCausalLM | eval | graph\_break | full\_graph |
| FalconMambaModel | eval | graph\_break | full\_graph |
| MambaForCausalLM | eval | graph\_break | full\_graph |
| MambaModel | eval | graph\_break | full\_graph |

All Mamba/SSM architecture — targeted Dynamo fix for selective scan operations.

### Infra Improvements (selected)

| Category | Count | Examples |
|----------|-------|---------|
| eager\_error -> full\_graph | 38 | BayesianDetectorModel, SpeechT5Model |
| create\_error -> full\_graph | 26 | ClapModel, Aimv2TextModel |
| timeout -> full\_graph | 17 | GPTBigCodeModel, LlamaModel |
| timeout -> graph\_break | 18 | CpmBeeModel, MegaModel |
| eager\_error -> graph\_break | 16 | AudioFlamingo3, MusicFlamingo |

These reflect Transformers 5.4.0 -> 5.5.3 compatibility improvements, not PyTorch compiler changes.

### New Models (76)

76 models new in the pt2.11 corpus (not present in pt2.10 expanded sweep). Mostly Diffusers models added via library upgrade and new HF Transformers models (Gemma4, NomicBert).

Notable new models compiling clean:
- Gemma4AudioModel, Gemma4Model, Gemma4TextModel, Gemma4VisionModel, Gemma4ForCausalLM
- NomicBertModel

## Version Trend (original 468 models)

| Version | eval full\_graph | train full\_graph | Fixes | Regressions |
|---------|-----------------|-------------------|-------|-------------|
| 2.8 | 298 (64%) | 288 (62%) | -- | -- |
| 2.9 | 324 (69%) | 314 (67%) | 0 | 0 |
| 2.10 | 337 (72%) | 323 (69%) | 12 | 0 |
| **2.11** | **345 (74%)** | **328 (70%)** | **2** | **0** |

Steady improvement across four releases. Zero regressions in any release.
