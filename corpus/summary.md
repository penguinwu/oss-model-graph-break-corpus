# Corpus Summary

**716 models** from HuggingFace Transformers + Diffusers, tested on PyTorch 2.10.0.

*Auto-generated from corpus.json — do not edit manually.*

## Status Distribution

| Status | Eval | Train |
|--------|------|-------|
| full_graph | 521 (73%) | 478 (67%) |
| graph_break | 169 (24%) | 217 (30%) |
| create_error | 6 (1%) | 7 (1%) |
| eager_error | 20 (3%) | 10 (1%) |
| timeout | 0 | 2 (<1%) |
| worker_error | 0 | 2 (<1%) |

## Coverage

- **716 models** tested across eval and train modes
- **521 models** (73%) compile fully (full_graph) in eval mode
- **478 models** (67%) compile fully (full_graph) in train mode
- **169 models** (24%) have graph breaks in eval mode
- **18 models** skipped pre-sweep (abstract, stateful, or unfixable)

## Key Findings

- ForConditionalGeneration variants have the highest graph break rate — vision-text merge paths introduce dynamic shape ops
- All graph breaks come from HF Transformers; Diffusers models compile clean
- Train mode has more graph breaks than eval (+48 models)
- Full analysis: [analysis/graph-break-analysis.md](../analysis/graph-break-analysis.md)
- Error model documentation: [docs/error-models.md](../docs/error-models.md)
