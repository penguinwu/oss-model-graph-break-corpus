# Discovery Report — `dbrx_moe_data_dep`

**Case:** `dbrx_moe_data_dep` (BS-103)
**Model:** `DbrxForCausalLM` (transformers `models/dbrx/modeling_dbrx.py`)
**Run:** `20260424-173949` (Phase 1 v2, V0×3 + V1×3, tier-2 enriched 2026-04-24 ~18:25 UTC)
**Author:** Otter

---

## TL;DR

The discovery agent produced **three distinct fix strategies** for the data-dependent expert-dispatch graph break in `DbrxExperts.forward`:

| Rank | Strategy | gb | T1 vs base | T2 vs base | LoC | Notes |
|---|---|---|---|---|---|---|
| 1 | masked-dense | 0 | 4.62–4.80x | **4.45–4.87x** | +12/-11 | clean across V0×3 |
| 2 | bmm | 0 | 4.25x | **3.87x** | +36/-17 | preserves sparsity, more code |
| 3 | config-flag + half-fix | 8 | 1.17x | **1.03–1.28x** | +6/-7 | CONTAMINATED — agent edited test file |

(Speedups vs tier-2 baseline = compiled-with-graph-breaks at realistic input. Baseline itself is 1.21x over eager — even the broken-compile path gets some win at 2048 tokens.)

**Headline:** Masked-dense wins on perf at realistic input AND on code complexity, beating bmm by ~16%. This is opposite of what FLOP-cost reasoning predicts.

---

## The Break

`transformers/models/dbrx/modeling_dbrx.py:313` — data-dependent for-loop in MoE expert dispatch:

```python
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()  # data-dependent
for expert_idx in expert_hit:                                            # iterates over .item()s
    idx, token_idx = torch.where(expert_mask[expert_idx])                # also data-dependent
    states = self.mlp(hidden_states[token_idx], ...)                     # dynamic shape
    next_states.index_add_(0, token_idx, states)
```

`nonzero()` + iteration over the result + per-expert `where(...)` produce shapes Dynamo cannot specialize on. Each loop iteration creates a graph break, and the `index_add_` accumulator pattern compounds it. Baseline reports `graph_break_count=9` under `torch.compile`.

---

## Methodology

- **6 trials** across 2 constraint variants (V0 = bare, V1 = sparsity-preserved), 3 trials each.
- **Two perf tiers** measured per trial:
  - **Tier-1 (fast):** `(1, 16) = 16 tokens` — discovery turnaround budget, ~3–8s eager.
  - **Tier-2 (realistic):** `(4, 512) = 2048 tokens` — strategy ranking signal, ~6–10s eager.
- **Dual-file restore between trials** — both `modeling_dbrx.py` AND `baseline_dbrx.py` reset to `.original`. Post-trial diff check raises `file-mutated:<file>` flag if agent edits beyond what its captured diff shows.
- **Subprocess-isolated perf measurement** — fresh Python per measurement. Avoids `sys.modules` contamination after agent overwrites model source.
- **Tier-1/Tier-2 direction-mismatch detection** — flags trials where compile speedup crosses the 1.0 line between tiers.

Per-case baselines captured against unmodified `modeling_dbrx.py`:
- Tier-1: eager 5.59ms, compiled 6.00ms, speedup **0.93x** (compile loses at tiny input)
- Tier-2: eager 11.65ms, compiled 9.63ms, speedup **1.21x** (compile wins modestly even with breaks)

---

## Strategy 1: masked-dense (V0_1, V0_2, V0_3)

**Idea:** Run *all* tokens through *all* experts in a static loop; multiply by routing weights (zero for non-selected experts) and accumulate. The data-dependent `nonzero()` and per-expert `where()` disappear entirely — every shape is statically known.

**Diff shape (V0_1):** +12/-11 lines in `modeling_dbrx.py` only.

```python
# Before: data-dependent expert hit list + per-expert masked indexing
expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
for expert_idx in expert_hit:
    idx, token_idx = torch.where(expert_mask[expert_idx])
    states = self.mlp(hidden_states[token_idx], w1, v1, w2)
    states = states * top_k_weights[token_idx, idx, None]
    next_states.index_add_(0, token_idx, states)

# After: static range loop, all-tokens × all-experts, weight-mask the result
expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
token_expert_weights = (expert_mask * top_k_weights.unsqueeze(-1)).sum(dim=1)  # [N, E]
for expert_idx in range(self.num_experts):
    states = self.mlp(hidden_states, w1[expert_idx], v1[expert_idx], w2[expert_idx])
    next_states += states * token_expert_weights[:, expert_idx:expert_idx+1]
```

**Trade-off:** Pays `num_experts × num_tokens` FLOPs (4× the original, since top_k=2 of 4 experts). But the matmul is dense and contiguous — exactly the shape GPU + Inductor are best at.

**Result:** All 3 V0 trials converged on this exact strategy. gb=0. Tier-2 speedup vs broken-compile baseline: **4.45x, 4.87x, 4.53x**. Eager *also* sped up (1.21–1.94x) because the original had a hidden GPU sync per loop iteration.

---

## Strategy 2: bmm (V1_2)

**Idea:** Flatten (token, top_k) pairs into a single batch dimension; gather the right expert weights per pair and run `bmm` for the full GLU expert forward. Preserves sparsity (only `N × top_k` expert evaluations, not `N × num_experts`).

**Diff shape:** +36/-17 lines in `modeling_dbrx.py` only.

```python
flat_expert_idx = top_k_index.reshape(-1)               # [N * top_k]
flat_weights = top_k_weights.reshape(-1, 1)
token_idx = arange(N).unsqueeze(1).expand(-1, top_k).reshape(-1)
flat_hidden = hidden_states[token_idx]                  # [N * top_k, D]
w1_sel = w1[flat_expert_idx]                            # [N * top_k, F, D]
gate_proj = torch.bmm(flat_hidden.unsqueeze(1), w1_sel).squeeze(1)
# ... bmm for v1, then GLU activation, then bmm for w2 ...
weighted = down_proj * flat_weights
next_states.index_add_(0, token_idx, weighted)
```

**Trade-off:** Same FLOP count as the original sparse compute (`N × top_k`), but bookkeeping (gather, reshape, scatter-add) takes real GPU time. More code, more conceptual surface area.

**Result:** gb=0. Tier-2 speedup vs baseline: **3.87x**. Slower than masked-dense at this input size; predicted to widen the lead at larger top_k or num_experts (untested here — single trial).

---

## Strategy 3: config-flag + half-fix (V1_1, V1_3) — CONTAMINATED

**Idea:** Set `torch._dynamo.config.capture_dynamic_output_shape_ops = True` to let Dynamo proceed past the data-dependent ops; convert the `for expert_idx in expert_hit` loop to `for expert_id in range(num_experts)` (so the loop iterator is no longer data-dependent), but leave the `torch.where(expert_mask[expert_id])` inside.

**Diff shape:** +6/-7 lines split between `modeling_dbrx.py` AND `baseline_dbrx.py`.

```python
# baseline_dbrx.py — agent edited the TEST FILE, not the model:
+ torch._dynamo.config.capture_dynamic_output_shape_ops = True
```

**Result:** gb=**8** (still has 8 graph breaks, the flag just lets compile keep going). Tier-2 speedup: **1.03–1.28x** — essentially baseline behavior. V1_1 also fired the `tier1-tier2-direction-mismatch` flag (compile loses at tier-1, barely wins at tier-2).

**Why flagged contaminated:** The harness's dual-file restore + post-trial diff check caught both trials editing `baseline_dbrx.py`. Per design doc v0.2 §4.3, mutations to the test file invalidate the trial's strategy fingerprint — the agent shouldn't be allowed to "fix" a graph break by neutering the test that detects it.

This is the same contamination shape Pilot 4 surfaced. Recurring in V1 suggests the V1 prompt ("must not collapse sparse compute to dense") implicitly steers some agents toward escape-hatch flag-flipping when they can't see a clean sparsity-preserving fix.

---

## Findings

### 1. Masked-dense > bmm at realistic input (counter to prediction)

I predicted bmm would pull ahead at tier-2 because masked-dense pays 2× FLOPs. Wrong — at 2048 tokens the dense matmul stays compute-bound and amortizes the FLOPs cleanly; bmm's gather/reshape/scatter overhead dominates. **Don't reason from FLOP counts alone — measure tier-2.**

This is exactly the design intent of the multi-tier perf strategy. Tier-1 alone would have shown both at ~4x and the gap would have looked like noise. Tier-2 gave a stable, reproducible signal.

### 2. Tier-1/Tier-2 direction-mismatch flag has real bite

V1_1 fired the flag: T1 compile/eager = 0.87x (compile loses), T2 = 1.07x (compile wins). For a corpus consumer, that's a "don't trust tier-1 alone" signal. Worth keeping in the schema.

### 3. Eager-mode side-effect of the fix

Masked-dense made *eager* faster too (1.21–1.94x at tier-2 vs original eager). The original used implicit `nonzero().item()` iteration, which forces a CPU-GPU sync per expert. Removing that sync helps eager even before compile gets involved. Worth surfacing in the trade-off matrix.

### 4. Compile-time

All clean strategies compile in ~9–12s at tier-2 (vs baseline 17.1s). Faster than baseline because the simpler graph (no break recovery) needs fewer guards. Compile-time is captured as a corpus-level signal, not a fix-quality scoring axis.

### 5. Strategy distribution skewed by variant

V0 (bare) → 3/3 masked-dense.
V1 (sparsity-preserved) → 1 bmm + 2 contaminated config-flag fixes.

The bare prompt is a strong attractor for masked-dense. The sparsity-preservation constraint successfully pushes off masked-dense, but it splits the remaining strategy space between bmm (clean) and escape-hatch flagging (contaminated). Suggests adding **V6** (no config flags) from the design doc would be high-value — it would force V1's escape-hatch trials toward bmm.

---

## Open Questions / Next Probes

1. **V6 trial.** Does forbidding `torch._dynamo.config` mutations push the V1 contamination cohort toward bmm or toward some new strategy?
2. **bmm at extreme top_k.** Does bmm overtake masked-dense if we test at top_k=8 of 32 experts (a more realistic MoE config)? The current test config (top_k=2 of 4) is artificial.
3. **Numerical equivalence.** ~~`validate.py` reports `output_match=None`...~~ **Correction (2026-04-24 evening):** validate.py already emits `max_diff_compiled_vs_eager_now` and `max_diff_vs_eager_baseline` — the accuracy axis has signal (V0/V1 clean trials all at ~1e-7). The morning report misread the schema. Open follow-on: surface these into the trade-off table explicitly so the assessor can score the accuracy axis.
4. **Constraint generalization.** The "agent edited the test file" failure mode appeared in 2/3 V1 trials. Does this pattern repeat across other cases, or is it Dbrx-specific (because the test-file-edit is a one-line config flag)?

---

## Trial-by-trial appendix

| Trial | Variant | Exit | Elapsed | gb | Strategy | T1 base | T2 base | T2 compile_s | Flags |
|---|---|---|---|---|---|---|---|---|---|
| V0_1 | bare | 0 | 222s | 0 | masked-dense | 4.62x | 4.45x | 11.5s | — |
| V0_2 | bare | 0 | 247s | 0 | masked-dense | 4.66x | 4.87x | 9.2s | — |
| V0_3 | bare | 0 | 249s | 0 | masked-dense | 4.80x | 4.53x | 10.6s | — |
| V1_1 | sparsity | 0 | 273s | **8** | config-flag + half-fix | 1.17x | 1.28x | 7.5s | direction-mismatch, file-mutated:baseline |
| V1_2 | sparsity | 0 | 459s | 0 | bmm | 4.25x | 3.87x | 10.1s | — |
| V1_3 | sparsity | 0 | 342s | **8** | config-flag + half-fix | 1.11x | 1.03x | 8.7s | file-mutated:baseline |

Per-trial diffs and logs: `discovery/runs/dbrx_moe_data_dep/20260424-173949/<trial>/` (archived from `/tmp/` 2026-04-24 evening).

---

## Addendum — V6 trial (2026-04-24 evening)

**Run:** `20260424-232954` (V6×3, tier-2 enriched).
**Constraint:** "do not modify torch._dynamo.config; fix in source code only."
**Question this answers (Open Probe #1):** does forbidding flag mutations push the V1 contamination cohort toward bmm, or toward something new?

### Result: convergence to masked-dense

All 3 V6 trials produced **masked-dense**, the same strategy class as V0. The escape-hatch family disappears when the flag is forbidden; agents do not invent a new strategy class — they fall back to the V0 attractor.

| Trial | gb | T1 compile_vs_base | T2 compile_vs_base | T2 compiled_ms | T2 mem (compiled) |
|---|---|---|---|---|---|
| V6_1 | 0 | 4.32x | **4.54x** | 2.12ms | 56MB |
| V6_2 | 0 | 4.72x | **4.60x** | 2.10ms | 19MB |
| V6_3 | 0 | 4.93x | **4.64x** | 2.07ms | 56MB |

V6 sits squarely in the V0 masked-dense band (V0×3 was 4.45–4.87x at tier-2). max_diff = 1.19–1.27e-7 across all trials. Memory peak compiled (~56MB) is much lower than eager (~158MB) — dense path's compile lowers it further.

### Updated strategy distribution

| Variant | n | Strategies produced |
|---|---|---|
| V0 (bare) | 3 | masked-dense × 3 |
| V1 (sparsity-preserved) | 3 | bmm × 1, config-flag + half-fix × 2 (CONTAMINATED) |
| V6 (no config flags) | 3 | masked-dense × 3 |

### What this confirms

1. *V0 is a strong attractor for masked-dense.* V6's bareness-minus-flags converges to the same place — flag-flipping was a *deflection from V0*, not a separate stable strategy.
2. *bmm only emerges under the V1 sparsity constraint.* Without that prompt steering, agents pick masked-dense regardless of whether flags are allowed.
3. *Contamination class is V1-specific, not a general escape-hatch tendency.* The agent reaches for the flag *to satisfy V1's sparsity demand* — not because flags are the path of least resistance in general.

### Dbrx Phase 1 — closed

Three variants × three trials = 9 trials, three distinct strategies surfaced (masked-dense, bmm, config-flag + half-fix), with the V1-sparsity-vs-V6-no-flags pair isolating *why* the contamination happens. This is the first complete discovery-agent case in the corpus.

Per-trial diffs and logs: `discovery/runs/dbrx_moe_data_dep/20260424-232954/<trial>/`.
