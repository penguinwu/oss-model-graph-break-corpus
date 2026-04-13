# PyTorch 2.9 Sweep Results

**Date:** 2026-04-02
**Models:** 468

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.9.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |

## Summary

|  | eval | train |
|---|---|---|
| **full\_graph** | 324 (69%) | 314 (67%) |
| **graph\_break** | 101 | 110 |
| **create\_error** | 27 | 27 |
| **eager\_error** | 16 | 17 |

- **112 models** (24%) have graph breaks in at least one mode

## Changes from 2.8

- **0 graph break fixes** (graph\_break → full\_graph)
- **0 regressions** (full\_graph → graph\_break)
- **+26 eval full\_graph, +26 train full\_graph** — gains came from models moving out of eager\_error (47→16), not from fixing graph breaks
- The large eager\_error reduction (47→16) reflects worker improvements, not PyTorch changes
