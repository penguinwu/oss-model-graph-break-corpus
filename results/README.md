# Sweep Results

Per-version summaries and nightly tracking for `torch.compile` quality across open-source models.

## Version Trend

| Version | eval full\_graph | train full\_graph | eval break | train break | Models | Fixes | Regressions |
|---------|-----------------|-------------------|------------|-------------|--------|-------|-------------|
| [2.8](pt2.8.md)   | 298 (64%) | 288 (62%) | 96  | 106 | 468 | —  | — |
| [2.9](pt2.9.md)   | 324 (69%) | 314 (67%) | 101 | 110 | 468 | 0  | 0 |
| [2.10](pt2.10.md) | 337 (72%) | 323 (69%) | 90  | 105 | 468 | 12 | 0 |
| [2.11](pt2.11.md) | 350 (74%) | 333 (70%) | 87  | 104 | 473 | 2  | 0 |

**Zero full\_graph→graph\_break regressions across all releases.**

## Release Summaries

- [pt2.11](pt2.11.md) — Latest release. 473 models, 2 fixes (FalconMamba, Mamba), 5 new models (Gemma4 + NomicBert)
- [pt2.10](pt2.10.md) — 12 graph break fixes, largest batch of genuine fixes
- [pt2.9](pt2.9.md) — Baseline improvements from worker fixes (eager\_error reduction)
- [pt2.8](pt2.8.md) — First sweep, baseline for all comparisons

## Nightly Tracking

Weekly nightly sweeps track fixes landing between releases.

- [2026-04-12](nightly/2026-04-12.md) — torch 2.12.0.dev20260407. 1 fix (MraModel), 0 regressions. Gemma4 nightly compat issue.

## Methodology

Each sweep runs every model through `torch.compile(fullgraph=True)` in both eval and train modes. Models are instantiated with randomized configs (reduced size for GPU fit), run through eager mode to verify correctness, then compiled.

Status definitions:
- **full\_graph** — compiles successfully with no graph breaks
- **graph\_break** — compiles but with one or more graph breaks
- **create\_error** — model fails to instantiate
- **eager\_error** — model fails in eager mode (before compile)
- **timeout** — exceeded time limit (180s standard, 600s for large models)

Fixes are counted as graph\_break→full\_graph transitions between consecutive versions. Regressions are full\_graph→graph\_break transitions.
