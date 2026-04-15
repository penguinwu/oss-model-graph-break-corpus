# Corpus Summary

**732 models** (424 families) from HuggingFace Transformers + Diffusers, tested on PyTorch 2.10.0.

*Auto-generated from corpus.json — do not edit manually.*

## Status Distribution

| Status | Eval | Train |
|--------|------|-------|
| full_graph | 533 (73%) | 490 (67%) |
| graph_break | 171 (23%) | 219 (30%) |
| create_error | 10 (1%) | 9 (1%) |
| eager_error | 15 (2%) | 11 (2%) |
| worker_error | 3 (<1%) | 3 (<1%) |

## Coverage

- **696 models** (95.1%) work in both eval and train modes
- **486 models** (66%) compile fully (full_graph) in both modes
- **235 models** (32%) have graph breaks in at least one mode
- **34 models** have errors in at least one mode
- **14 models** skipped pre-sweep (abstract, stateful, or unfixable)

## Model Variants

| Variant | Count | Full Graph (eval) | Graph Break (eval) |
|---------|-------|-------------------|--------------------|
| Base Model | ~424 | ~350 | ~60 |
| ForCausalLM | ~190 | ~170 | ~15 |
| ForConditionalGeneration | ~115 | ~55 | ~50 |

ForConditionalGeneration variants have the highest graph break rate (~43%) because vision-text merge paths introduce dynamic shape ops and control flow invisible at the base model level.

## Key Findings

- **235 unique models** with graph breaks in any mode
- ForConditionalGeneration is the compile quality frontier — real-world VLM usage
- All graph breaks come from HF Transformers; Diffusers models compile clean
- Train mode has more graph breaks than eval (+48 models)
- Full analysis: [analysis/graph-break-analysis.md](../analysis/graph-break-analysis.md)
- Error model documentation: [docs/error-models.md](../docs/error-models.md)
