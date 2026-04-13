# PyTorch 2.8 Sweep Results

**Date:** 2026-04-02
**Models:** 468

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.4.0 |
| Diffusers | 0.37.1 |

## Summary

|  | eval | train |
|---|---|---|
| **full\_graph** | 298 (64%) | 288 (62%) |
| **graph\_break** | 96 | 106 |
| **create\_error** | 27 | 27 |
| **eager\_error** | 47 | 47 |

- **108 models** (23%) have graph breaks in at least one mode
- Baseline release for version trend tracking

## Notes

This is the earliest version in the corpus. All subsequent versions are compared against this baseline.
