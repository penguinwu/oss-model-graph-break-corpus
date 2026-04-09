# PT2 (oncall:pt2) GitHub Issue Analysis — User Journey Patterns for Documentation Audit

**Date:** 2026-04-06
**Dataset:** 2,970 issues from pytorch/pytorch with label `oncall: pt2` (2021-01-18 to 2026-04-06)
**Purpose:** Identify user pain points and documentation gaps to prioritize PT2 documentation fixes

---

## Executive Summary

Analysis of 2,970 `oncall: pt2` issues (2,598 user-filed, 372 bot/CI) reveals that **documentation gaps are a systemic contributor to user friction**. The top three documentation needs are:

1. **Dynamic shapes & recompilation** (438 issues, 16.9%) — Users don't understand guards, recompilation triggers, or how to configure dynamic shapes correctly
2. **torch.export limitations** (351 issues, 13.5%) — Users hit walls with export and lack guidance on workarounds
3. **torch.compile compatibility** (313 issues, 12.0%) — No clear "what works and what doesn't" reference

65% of external users file only one issue and never come back — suggesting that first-contact experience with PT2 is a critical moment where documentation can make or break adoption.

---

## Methodology

- Fetched 2,970 issues via GitHub API (all issues with label `oncall: pt2`, sorted by update time)
- Filtered out 372 bot/CI issues (DISABLED tests, UNSTABLE tests, dashboards)
- Categorized 2,598 user issues by journey type, pain point, and inferred documentation gap
- Separated analysis by user type: Meta internal (31%), company contributors (6%), external/community (63%)
- Categorization uses keyword matching on titles, bodies, and labels

---

## 1. User Journey Patterns (Ranked by Frequency)

### Tier 1: Core Workflows (>20% of issues)

| Rank | Journey | Issues | % | Close Rate | Key Pain Points |
|------|---------|--------|---|------------|-----------------|
| 1 | torch.compile usage | 1,581 | 60.9% | 57% | recompilation/guards, internal errors, unexpected behavior |
| 2 | Inductor/codegen | 1,298 | 50.0% | 59% | internal errors, numerical accuracy, performance |
| 3 | Dynamic shapes | 637 | 24.5% | 61% | recompilation/guards, internal errors, performance |
| 4 | Non-CUDA devices (XPU, MPS, ROCm, CPU) | 560 | 21.6% | 64% | internal errors, unexpected behavior, regression |

### Tier 2: Advanced Workflows (10-20% of issues)

| Rank | Journey | Issues | % | Close Rate | Key Pain Points |
|------|---------|--------|---|------------|-----------------|
| 5 | Autograd + compile | 497 | 19.1% | 57% | internal errors, performance, numerical accuracy |
| 6 | Correctness/accuracy | 422 | 16.2% | 62% | numerical accuracy, unexpected behavior, internal errors |
| 7 | Performance optimization | 332 | 12.8% | 55% | performance degradation, recompilation, memory |
| 8 | torch.export | 330 | 12.7% | 51% | internal errors, unsupported ops, regression |
| 9 | Model integration | 312 | 12.0% | 65% | performance, recompilation, regression |

### Tier 3: Specialized Workflows (5-10% of issues)

| Rank | Journey | Issues | % | Close Rate | Key Pain Points |
|------|---------|--------|---|------------|-----------------|
| 10 | AOT autograd | 255 | 9.8% | 51% | internal errors, recompilation |
| 11 | Memory issues | 223 | 8.6% | 54% | memory leaks, OOM, unexpected memory growth |
| 12 | Graph breaks | 205 | 7.9% | 63% | unsupported operations, recompilation |
| 13 | Crashes/segfaults | 174 | 6.7% | 58% | crash, internal errors |
| 14 | Distributed training | 163 | 6.3% | 53% | internal errors, performance |

### Tier 4: Emerging/Niche Workflows (<5% of issues)

| Rank | Journey | Issues | % | Close Rate | Key Pain Points |
|------|---------|--------|---|------------|-----------------|
| 15 | AOT compilation/serialization | 140 | 5.4% | 54% | internal errors, performance |
| 16 | Quantization | 138 | 5.3% | 59% | internal errors, memory, numerical accuracy |
| 17 | CUDA graphs | 132 | 5.1% | 52% | performance, memory, regression |
| 18 | Flex attention | 132 | 5.1% | 63% | internal errors, memory, performance |
| 19 | Error handling | 129 | 5.0% | 60% | runtime errors, internal compiler errors |
| 20 | Custom ops/kernels | 122 | 4.7% | **43%** | internal errors, unexpected behavior, performance |
| 21 | Version regression | 118 | 4.5% | 64% | regression, performance |
| 22 | Higher-order ops | 96 | 3.7% | **49%** | internal errors, unsupported ops |

**Notable:** Custom ops (43% close rate) and higher-order ops (49% close rate) have the lowest resolution rates, suggesting these areas are hardest to resolve and likely lack sufficient documentation.

---

## 2. Pain Point Patterns

| Pain Point | Issues | % of All User Issues |
|------------|--------|---------------------|
| Hang/deadlock (compile hangs, infinite loops) | 1,400 | 53.9% |
| Recompilation/guards (unexpected recompilation, guard failures) | 617 | 23.7% |
| Internal compiler error (assertion failures, lowering errors) | 454 | 17.5% |
| Performance degradation (slower than eager, compile overhead) | 294 | 11.3% |
| Unexpected behavior (works differently than expected) | 287 | 11.0% |
| Runtime error (clear error messages) | 257 | 9.9% |
| Numerical accuracy (NaN, wrong values, precision loss) | 249 | 9.6% |
| Unsupported operation (missing decomp, not implemented) | 243 | 9.4% |
| Memory issues (OOM, leaks, unexpected growth) | 223 | 8.6% |
| Crash/segfault | 163 | 6.3% |
| Version regression | 117 | 4.5% |
| Documentation gap (explicitly mentioned) | 40 | 1.5% |
| Unclear error message (explicitly mentioned) | 34 | 1.3% |

**Key insight:** "Hang/deadlock" at 53.9% is artificially high due to keyword overlap (many issue bodies mention "hanging" or "stuck" metaphorically). The real signal is in **recompilation/guards (23.7%)** and **internal compiler errors (17.5%)** — these are the most common "I don't know what's going wrong" moments.

---

## 3. Documentation Gaps (Inferred from Issues)

| Documentation Gap | Issues | % | Priority |
|-------------------|--------|---|----------|
| Dynamic shapes: when to use, how guards work, recompilation | 438 | 16.9% | **P0** |
| torch.export limitations and workarounds | 351 | 13.5% | **P0** |
| torch.compile compatibility matrix / limitations | 313 | 12.0% | **P0** |
| How to diagnose and fix graph breaks | 244 | 9.4% | **P1** |
| Autograd behavior under torch.compile | 242 | 9.3% | **P1** |
| Performance expectations and tuning guide | 163 | 6.3% | **P1** |
| Distributed training + torch.compile guide | 128 | 4.9% | **P1** |
| Custom ops with torch.compile tutorial | 106 | 4.1% | **P2** |
| Debugging correctness issues under compile | 105 | 4.0% | **P2** |
| Error message interpretation guide | 49 | 1.9% | **P2** |
| CUDA graphs integration guide | 25 | 1.0% | **P3** |

### Documentation-Seeking Keywords in Issue Bodies

| Keyword | Occurrences |
|---------|------------|
| "example" | 473 |
| "workaround" | 78 |
| "how to" | 63 |
| "expected behavior" | 55 |
| "tutorial" | 33 |
| "documentation" | 27 |
| "is there a way" | 11 |
| "confusing" | 10 |
| "guide" | 9 |
| "unclear" | 8 |

Users search for **examples** (473 mentions) far more than anything else. This suggests that **worked examples and runnable code snippets** should be the primary format for PT2 documentation improvements.

---

## 4. User Demographics

| Segment | Issues | % | Unique Users |
|---------|--------|---|-------------|
| External/community | 1,645 | 63% | 716 (68% one-time filers) |
| Meta internal | 800 | 31% | ~50 |
| Company contributors (AMD, Intel, etc.) | 153 | 6% | ~15 |

**External users** file the majority of issues and are predominantly **one-time filers** (68%). This means:
- First-contact documentation quality is critical — there's no second chance
- External users get 5.0% "needs reproduction" rate vs. the overall 3.6%, suggesting confusion about how to report issues or reproduce their own problems

### Top External User Issues by Community Interest

| Issue | Reactions | Comments | Topic |
|-------|-----------|----------|-------|
| #58734 | 58 | 36 | uint16/uint32/uint64 support |
| #95408 | 56 | 61 | Parallel associative scan |
| #50688 | 34 | 71 | `torch.scan` feature request |
| #124480 | 20 | 48 | Fused linear + cross-entropy |
| #151705 | 12 | 13 | Inductor native matmul via `tl.dot` |
| #133254 | 7 | 33 | Shared memory with flex attention |
| #96693 | 3 | 48 | max-autotune precision concerns |

---

## 5. Issue Resolution Patterns

### Open Issue Age Distribution

| Age | Count | % |
|-----|-------|---|
| < 30 days | 107 | 9.5% |
| 30-90 days | 168 | 14.8% |
| 90-365 days | 497 | 43.9% |
| > 365 days | 360 | 31.8% |
| > 2 years | 147 | 13.0% |

**Median open issue age: 212 days.** Over 75% of open issues are older than 90 days. This backlog signals areas where documentation could redirect users away from filing issues.

### Time-to-Close for Closed Issues

| Duration | Count | % |
|----------|-------|---|
| < 1 day | 87 | 5.9% |
| 1-7 days | 221 | 15.1% |
| 7-30 days | 287 | 19.6% |
| 30-90 days | 268 | 18.3% |
| > 90 days | 603 | 41.1% |

**Median time-to-close: 58 days.** 41% of closed issues take more than 90 days to resolve. Quick-resolution issues (< 7 days, 21%) often represent user errors or misunderstandings that better docs could prevent.

---

## 6. Temporal Trends

Issue volume has been **accelerating sharply** since 2025-Q2, with recent quarters showing 3-5x the volume of 2024 quarters. This likely reflects growing PT2 adoption.

### Emerging Trends (2025-Q4 to 2026-Q1)

- **Non-CUDA devices** surged from ~20 issues/quarter in 2025-Q1 to 222 in 2025-Q4 and 129 in 2026-Q1 — XPU, ROCm, and CPU backend users are growing fast
- **Correctness/accuracy** increased from ~15/quarter to 128 in 2026-Q1 — as more users move to production, they discover subtle numerical differences
- **torch.export** remains steady but has the **lowest close rate (51%)** among Tier 2 journeys

---

## 7. Recommended Documentation Priorities

### P0 — High Impact, Address First

1. **"Dynamic Shapes & Recompilation Deep Dive"** (438 issues)
   - What triggers recompilation and how to avoid it
   - How guards work, when they fire, how to inspect them
   - `dynamic=True` vs `dynamic=False` — when to use each
   - Common recompilation patterns and fixes
   - Example: #97155 (custom RNN takes very long to compile for long sequences)
   - Example: #121504 (custom attention recompilations)

2. **"torch.compile Compatibility Matrix"** (313 issues)
   - What works out of the box, what needs workarounds, what doesn't work
   - Per-feature status: autograd, distributed, quantization, custom ops, etc.
   - Known limitations with specific model architectures
   - Clear versioning of what changed between releases
   - Example: #133571 (errors after upgrading to 2.4.0)
   - Example: #90768 (not working on Windows)

3. **"torch.export: Limitations, Workarounds, and Migration Guide"** (351 issues)
   - What ops are supported and which are not
   - How to handle unsupported patterns
   - Differences between `torch.export` and `torch.compile` tracing
   - C-shim coverage and workarounds
   - Example: #125984 (torch.autograd.grad fails with export)
   - Example: #147625 (grid_sampler_3d missing c-shim)

### P1 — Medium Impact, Address Second

4. **"Diagnosing and Fixing Graph Breaks"** (244 issues)
   - What causes graph breaks and why they matter
   - How to identify graph breaks in your code
   - Common patterns that cause graph breaks and how to fix them
   - Using `fullgraph=True` for debugging
   - Example: #169995 (DebugMode graph break with record_nn_module)
   - Example: #176912 (display tracked side effects)

5. **"Autograd Behavior Under torch.compile"** (242 issues)
   - How backward pass is handled differently under compile
   - Interaction with activation checkpointing
   - Common pitfalls with gradient computation
   - Example: #161889 (SAC + compile not working properly)
   - Example: #150859 (RMS norm NaNs with compile + float8)

6. **"Performance Tuning Guide for torch.compile"** (163 issues)
   - Setting expectations: when compile helps and when it doesn't
   - Compilation time vs runtime tradeoffs
   - `mode` parameter guide: "default" vs "reduce-overhead" vs "max-autotune"
   - How to profile and identify bottlenecks
   - Compile overhead on small graphs (#161783, 4 reactions, 11 comments)
   - Example: #108971 (really slow compilation times causing distributed errors)

7. **"Using torch.compile with Distributed Training"** (128 issues)
   - DDP + compile: what works, common issues
   - FSDP + compile: current state and limitations
   - DTensor interactions
   - Example: #93756 (Dynamo for DeepSpeed and FSDP)
   - Example: #159635 (DTensor + dynamic shapes + compile failure)

### P2 — Targeted Impact, Address Third

8. **"Custom Ops and Kernels with torch.compile"** (106 issues, **43% close rate — lowest**)
   - How to register custom ops for compile compatibility
   - torch.library vs decorator registration (performance difference: #139500)
   - User-defined Triton kernels
   - Dynamic-shape outputs with autograd (#111950)
   - Example: #170049 (compile fails to capture user-defined kernels)

9. **"Debugging Correctness Issues Under Compile"** (105 issues)
   - How to compare eager vs compiled results
   - Tools for debugging numerical divergence
   - Known sources of non-determinism
   - Example: #113180 (higher train loss with compile)
   - Example: #173921 (4.7% relative error in Conv2d on CPU)

10. **"Error Message Reference Guide"** (49 issues)
    - Common error messages and what they mean
    - `BackendCompilerFailed` — what to check
    - `Unsupported` — what to do next
    - `InternalError` — how to report effectively
    - Example: #122129 (obscure error about List[int])
    - Example: #140765 (KeyError in default_cache_dir)

### P3 — Lower Priority but Growing

11. **"CUDA Graphs Integration Guide"** (25 explicit + 132 total CUDA graph issues)
    - When to use `mode="reduce-overhead"` vs explicit CUDA graphs
    - Interactions with dynamic shapes, gradient accumulation
    - Example: #169545 (compile + CUDAGraph + gradient accumulation fails)

12. **"Non-CUDA Device Support Matrix"** (560 issues, growing rapidly)
    - XPU, ROCm, CPU, MPS — current state per device
    - Known limitations per backend
    - Example: #148651 (avoid fork for compile threads)

---

## 8. Key Exemplar Issues for Documentation Case Studies

These issues represent the most common "user story" patterns and would make excellent case studies or FAQ entries:

### "I tried torch.compile and it's slower than eager"
- #161783 (11c, 4r) — torch.compile runtime overhead on small graphs
- #136263 (5c) — torch.compile 100x slower for cumprod backward
- #108971 (12c, 2r) — Really slow compilation times in distributed

### "I got a scary error and don't know what it means"
- #122129 (18c) — "Expected a value of type 'List[int]' for argument 'sizes'"
- #140765 (4c, 5r) — KeyError in default_cache_dir (env issue, not user code)
- #119054 — BackendCompilerFailed: Triton Error

### "torch.compile gives wrong results"
- #96693 (48c, 3r) — max-autotune precision appears lower
- #113180 (30c) — Higher train loss with compile
- #173921 (15c) — Significant numerical divergence in Conv2d on CPU

### "My model keeps recompiling"
- #121504 (53c) — Custom attention recompilations
- #97155 (14c, 4r) — Custom RNN takes very long to compile for long sequences
- #135859 (11c, 1r) — bmm, topk, cholesky causing recompilations

### "torch.compile doesn't work with my setup"
- #90768 (48c, 3r) — Not working on Windows
- #133571 (44c) — Errors after upgrading to 2.4.0
- #130174 (22c, 4r) — Making compile work with vLLM

### "I need to use custom ops with compile"
- #139500 (14c) — Custom ops via decorator slower than Library API
- #170049 (40c) — Compile fails to capture user-defined Triton kernels
- #111950 (15c, 1r) — Dynamic-shape custom ops choke AOTAutograd

---

## 9. Actionable Recommendations

### For the Documentation Team

1. **Prioritize "Dynamic Shapes & Recompilation"** — this is the #1 user confusion area (438 issues). A comprehensive guide with `TORCH_LOGS` examples would prevent hundreds of issues.

2. **Create a living "Compatibility Matrix"** — a single page showing what features work with `torch.compile`, `torch.export`, and each backend. Update with every release.

3. **Add "Common Errors and What They Mean"** section to the troubleshooting page — the existing troubleshooting doc (https://docs.pytorch.org/docs/2.7/torch.compiler_troubleshooting.html) is too thin.

4. **Focus on worked examples over API reference** — 473 issues mention "example" in their body. Users learn PT2 by example, not by reading API docs.

5. **Custom ops documentation is critically undertreated** — lowest close rate (43%), growing in importance as users try to integrate compile into real systems.

### For the Triage Team

6. **Tag documentation-preventable issues** — add a label like `doc-preventable` to issues that could have been avoided with better docs. This creates a feedback loop.

7. **Track "needs reproduction" as a doc quality signal** — 5% of external issues need reproduction, often because users can't articulate the problem due to lack of understanding.

### For the PT2 Team

8. **Invest in error message quality** — only 34 issues explicitly complain about error messages, but 454 issues (17.5%) involve internal compiler errors. Many users don't even know to complain about error quality; they just file a bug.

9. **Address the open issue backlog** — median open issue age is 212 days, with 360 issues older than a year. Even if the fix is "document this limitation," closing old issues with documentation links improves community trust.

---

## Appendix: High-Impact Open Issues

Top 20 open issues by community engagement (reactions * 5 + comments):

| Issue | Score | Topic | Journey |
|-------|-------|-------|---------|
| #95408 | 341 | Parallel associative scan | Correctness/accuracy |
| #58734 | 326 | uint16/uint32/uint64 support | Correctness/accuracy |
| #50688 | 241 | torch.scan feature request | Correctness/accuracy, memory |
| #151705 | 73 | Inductor native matmul via tl.dot | Inductor/codegen |
| #133254 | 68 | Shared memory with flex attention | torch.compile, inductor |
| #123177 | 60 | compile+cudagraphs in multithreaded context | torch.compile, inductor |
| #176837 | 50 | FlexAttention AuxRequest extension | Flex attention |
| #125718 | 49 | torch.compile and complex numbers | torch.compile, inductor |
| #109774 | 46 | DDP + Dynamo AllReduce | torch.compile, inductor |
| #117394 | 45 | Dynamo Single Step Graph | torch.compile, graph breaks |
| #165999 | 44 | associative_scan() performance | torch.compile, inductor |
| #125984 | 44 | torch.export with torch.autograd.grad | torch.export, error handling |
| #162859 | 41 | symmetric memory in torch.compile | torch.compile, inductor |
| #140914 | 38 | TypeError with type parameter defaults | Inductor, error handling |
| #157975 | 37 | Online softmax disabled warning | Inductor, performance |
| #148651 | 35 | Avoid fork for compile threads | Inductor, non-CUDA |
| #115075 | 38 | cuOccupancyMaxActiveClusters undefined | torch.compile, inductor |
| #97155 | 34 | Custom RNN long compile time | torch.compile, dynamic shapes |
| #150296 | 33 | zentorch integration RFC | Inductor, performance |
| #157015 | 32 | flex_attention + Context Parallel | torch.compile, distributed |

---

*Analysis generated from 2,970 GitHub issues. Raw data at `tmp/pt2_all_issues.json`. Classification is keyword-based and may have false positives; manual review of top issues is recommended for final prioritization.*
