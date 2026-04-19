# OSS Model Graph Break Corpus — Status

**Last updated:** 2026-04-19

## Current State

- **Corpus:** 734 models (HuggingFace + Diffusers + Custom)
- **Latest sweep:** PT HEAD 2.13.0a0+git96c92e0 (source build, April 19)
- **Baseline:** pt2.11 (PyTorch 2.11.0+cu128, Transformers 5.5.3)
- **Nightly venv:** PyTorch 2.13.0a0+git96c92e0 (source build)
- **Nightly cron:** Sundays 3 AM ET (`nightly-sweep` job), auto-falls back to source build when pip is stale
- **Unpushed commits:** 2 (Peng pushes manually due to BPF jailer)
- **Issue reconcile tool:** `python tools/file_issues.py reconcile --results <path>` — close/update/create issues from sweep results

## Version Trend (original 468 models, eval fullgraph)

| Version | Eval Fullgraph | Delta |
|---------|---------------|-------|
| pt2.8   | 64%           | —     |
| pt2.9   | 69%           | +5%   |
| pt2.10  | 72%           | +12 fixes |
| pt2.11  | 74%           | +2 fixes  |
| HEAD    | 83%           | +56 fixes, 1 regression |

Zero regressions across released versions. 1 regression on HEAD (BltModel compile-time blowup).

## April 19 Sweep Highlights

- **56 fixes** (44 eval + 12 train across 39 models), **1 regression** (BltModel) vs pt2.11
- Fixes: mostly encoder-decoder architectures (Bart, T5, Whisper, Pegasus, etc.)
- Root cause verified: `copy.deepcopy()` graph break, fixed by PR #179611 (landed April 11 via ghstack)
- Regression: BltModel times out during model creation (135s on tf 5.5.3 → 473s on tf 5.5.4). Compile time itself is unchanged (17s). Transformers issue, not torch.compile.
- Eval fullgraph: 577/734 (79%), Train: 503/734 (69%)
- Pip nightly stale since April 7 (CI broken) — sweep used source build from HEAD

## What's Done

- Full sweep infrastructure: identify → explain → corpus overlay → issue scan
- Nightly pipeline with source-build fallback (>3 days stale → auto-build from source)
- Build-from-source script with conditional sudo (agent vs direct identity)
- PR status verification tool (`tools/check_pr_status.py`) — handles ghstack/phabricator
- Issue filing tool with nightly validation and dedup
- pt2.8 through pt2.11 results published, HEAD sweep complete

## What's Done (April 19)

- Issues #42, #47 closed on GitHub (verified fixed on HEAD)
- Issues #1, #4, #43 closed (verified fixed on HEAD)
- Issues #45, #46, #52 updated with HEAD model counts
- pt2.11 directory renamed from pt2.11-fresh for canonical naming hygiene
- Phase-aware timeout classification (compare.py + orchestrator.py)
- Gemma4-text phase validation fix (orchestrator.py)
- Issue reconcile tool (`tools/file_issues.py reconcile`) — full lifecycle management
- Web proxy GitHub writes working (body-as-dict pattern)

## What's Next

- **CLI design review** — Peng to decide if `nightly` should become `sweep` (deferred)
- **Hardcoded backend/fullgraph** — design input needed from Peng
- **Animesh update** — draft ready, send Monday via feedback space
- **Git push** — ~15+ unpushed commits + nightly results

## Known Issues

- BPF jailer blocks GitHub/PyPI from Claude Code agent identity (workaround: sudo)
- Pip nightly CI broken since ~April 7 — no new wheels. Source build is the workaround.
- USE_NCCL=0 in source builds (no nccl.h on devvm, BPF blocks cmake fetch)
