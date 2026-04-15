# PyTorch 2.10 Sweep Results

**Date:** 2026-04-15 (expanded corpus)
**Models:** 732 (468 original + 264 new)
**Model families:** 424 (30 new families, 234 new configurations)

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |

## Summary

|  | eval | train |
|---|---|---|
| **full\_graph** | 533 (73%) | 490 (67%) |
| **graph\_break** | 171 (23%) | 219 (30%) |
| **error** | 28 (4%) | 21 (3%) |

- **235 models** (32%) have graph breaks in at least one mode
- **696 models** (95.1%) work in both eval and train modes
- **486 models** compile fully (full\_graph) in both modes

## Corpus Expansion (v2.12 sweep iteration)

264 new models added to the corpus:
- **30 new model families** — entirely new architectures (Llama4, Gemma3n, Dots1, etc.)
- **234 new configurations** — ForCausalLM and ForConditionalGeneration variants of existing families

28 previously-broken models fixed via config patches, dimension fixes, and input shape corrections.

See `sweep_results/v2.12/changelog.md` for the full change list.

## Changes from 2.9 (original 468 models)

- **12 graph break fixes** (graph\_break → full\_graph), **0 regressions**

### Fixes (12 model×mode pairs)

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

## Notes

The largest batch of genuine graph break fixes in the corpus. These reflect real PyTorch Dynamo improvements between 2.9 and 2.10.

The corpus expansion (v2.12 sweep) added ForCausalLM and ForConditionalGeneration variants, testing the full model stack users actually compile — not just base backbones. This revealed additional graph breaks in vision-text merge paths that were invisible at the base model level.
