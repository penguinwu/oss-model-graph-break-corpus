# PT2 Nightly Sweep Brief — 2026-05-09 vs 2026-05-03 (Dynamo scope)

**Window:** 6 days. torch nightly `2.13.0.dev20260502+cu126` → `2.13.0.dev20260507+cu126`.
**Scope:** Dynamo-relevant changes only — apple-to-apple on common (name, mode) HF transformers pairs that ran in BOTH baseline and current. Diffusers + custom suites + timm-dependent models excluded. Eager-side timeouts and harness gaps tracked separately.

---

## 1. Headline

Net **+4 graph breaks** on the apple-to-apple set (1432 common pairs), entirely from 4 GraniteMoeHybrid pairs already tracked at issue **[#55](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/55)** (`_local_scalar_dense` from `.tolist()`); most likely a transformers `5.8.0 → 5.6.2` downgrade artifact (see §2). Zero models flipped graph_break → full_graph. Zero real upstream compile regressions.

## 2. Setup

⚠️ **Methodology note:** baseline 2026-05-03 has NO recorded `transformers` / `diffusers` versions in `sweep_state.json` (the sweep predates the fail-loud guardrail shipped 2026-05-10). However, last week's published brief reports baseline transformers as `5.8.0` (pip-installed venv). This week ran on `5.6.2` (modellibs tree) — i.e., a TRANSFORMERS DOWNGRADE between the two sweeps. This is the load-bearing methodology issue this week:
- the cohort delta in §5.1 (9 model classes "absent") is the downgrade — those classes exist in transformers 5.7+ but not 5.6.2;
- the +4 GraniteMoeHybrid GB delta in §1 is most likely transformers source drift in the GraniteMoeHybrid forward path between `5.8.0` and `5.6.2` (downgrade direction); already covered by #55.

| Setup         | Last week (2026-05-03)        | This week (2026-05-09)       |
|---|---|---|
| torch nightly | `2.13.0.dev20260502+cu126`    | `2.13.0.dev20260507+cu126`   |
| transformers  | `5.8.0` (pip-installed)       | `5.6.2` (modellibs)          |
| diffusers     | _(not recorded)_              | `0.38.0`                     |

The modellibs upgrade to `5.8.0+` is queued for next week to restore version parity (the apparent "cohort shrinkage" in §5.1 then reverses).

## 3. Apple-to-apple Topline

(1432 (model × mode) HF pairs present in BOTH sweeps)

| Metric                                          | Last week     | This week     | Δ                  |
|---|---|---|---|
| Compiles fullgraph                              | 1098          | 1098          | 0                  |
| Compiles with graph breaks                      | 326           | 328           | +2                 |
| Pairs with errors                               | 10            | 8             | −2                 |
| Total graph breaks (1424 reliable pairs; 10 with explain-coverage gaps excluded) | 3850 | 3854 | **+4** (entirely from 4 GraniteMoeHybrid pairs covered by #55) |

Status counts on the common-pair set are by definition byte-identical UNLESS Dynamo behavior changed. Total-GB delta is the load-bearing apple-to-apple metric.

## 4. Dynamo focus — top-5 lists + actionable this week

The "if you read nothing else, read this" surface for the Dynamo team. Three lenses on the same question.

### 4.1 Top 5 Dynamo issues by blast radius (model count × break count)

| # | Symptom | Scope |
|---|---|---|
| [117](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/117) | `aten.nonzero.default` data-dependent output shape | 59 models, 573 breaks |
| [96](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/96)   | `DictIterator/DictItemsIterator` gb0092 in transformers' `output_capturing` decorator | 103 models |
| [55](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/55)   | `_local_scalar_dense` from `.tolist()` (incl. this week's +4 GraniteMoeHybrid GB delta) | ~64 models, 100+ breaks |
| [115](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/115) | `Tensor.item()` with `capture_scalar_outputs=False` | 27 models, 246 breaks |
| [78](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/78)   | Data-dep branching on `torch.all(mask==1)` mamba/linear-attn fast-path | 20 models, 204 breaks |

(Umbrella [#122](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/122) — 129 models, 857 breaks — is even larger but its top-9 named sites collapse into #77 + #78. See §7 for the scope-reduction edit applied this week.)

### 4.2 Top 5 easy fixes (narrow scope, specific symptom, MRE-anchored)

| # | Symptom | Scope | Why easy |
|---|---|---|---|
| [125](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/125) | FX `setitem` on FakeTensor for UdopEncoderModel | 1 model, 2 modes | ~25-line MRE verified live; specific failing call path |
| [126](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/126) | `PendingUnbackedSymbolNotFound` in `nn.functional.pad` (MimiModel `_pad1d`) | 1 model, 2 modes | ~30-line MRE verified live; specific symbol-tracking gap |
| [98](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/98)   | `parametrize.weight_norm` trips `recompile_limit` via `___check_type_id` churn | 1 model, generic mechanism | Narrow guard-class churn; MRE in body |
| [99](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/99)   | `SymInt / SymInt` division not supported | 3 sliding-window models, 100 breaks | Narrow algorithmic gap; MRE in body |
| [116](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/116) | `range(int, Tensor, int)` builtin not traced | 2 models, 12 breaks | Very specific builtin gap |

### 4.3 Actionable this week

1. **[#125](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/125)** — `[dynamo]` FX `setitem` on FakeTensor fails for UdopEncoderModel. Fake-tensor `setitem` on a `(2, 196)` bool tensor cannot be traced when the index is a tuple of 0-d int64 tensors (produced by `zip(*ind)` over an `(N, 2)` index tensor). 2 model×mode pairs currently zero-coverage on the corpus. ~25-line standalone MRE in body, verified live. Leverage: if FX fake-tensor setitem path handles tuples of 0-d tensor indices, both pairs become Dynamo-testable.
2. **[#126](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/126)** — `[dynamo]` `PendingUnbackedSymbolNotFound` in `nn.functional.pad` with int-tensor padding (MimiModel `_pad1d`). Newly surfaced after `pytorch/pytorch#182339` eager fix landed (which previously masked this Dynamo gap). ~30-line standalone MRE in body, verified live. Leverage: 2 MimiModel pairs currently graph_break instead of full_graph; likely affects any audio-model family that does dynamic-shape padding via int-tensor.
3. **[#55](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/55)** — `_local_scalar_dense` from `.tolist()` (already tracked). This week's +4 GB delta on GraniteMoeHybrid is the entire net-GB regression. No new ask; tracking issue is already actionable.

## 5. Cohort + newly compile-testable models

### 5.1 Cohort changes (set arithmetic)

|                                                  | Count           |
|---|---|
| Models added (in this week, NOT in last week)    | 0               |
| Models removed (in last week, NOT in this week)  | 18 work items / 9 distinct |

Removed breakdown:
- **1 entry REMOVED from `known_errors.json`**: `MimiModel` — eager-side bug [`pytorch/pytorch#182339`](https://github.com/pytorch/pytorch/issues/182339) closed upstream 2026-05-06 and verified locally; both modes now flip `eager_error → graph_break`. Net `known_errors` count: 8 → 7.
- **9 model classes** absent from this week's modellibs transformers `5.6.2` but present in last week's pip-installed `5.8.0` — i.e., the transformers DOWNGRADE between sweeps removed them from the cohort: DeepseekV4ForCausalLM, DeepseekV4Model, Deimv2Model, GraniteSpeechPlusForConditionalGeneration, LagunaForCausalLM, LagunaModel, MiniCPMV4_6ForConditionalGeneration, MiniCPMV4_6Model, PPFormulaNetForConditionalGeneration. (Modellibs upgrade to `5.8.0+` queued for next week to restore version parity.)

### 5.2 Newly compile-testable models (state flips: error → success)

- Total work items: **2**
- Distinct model count: **1** (MimiModel)
- % full_graph out of newly-compile-testable: **0%** (both pairs flipped `eager_error → graph_break`, not `full_graph`)
- Decomposition: 2 pairs that were `eager_error` in baseline are now compile-testable (MimiModel|eval, MimiModel|train) after [`pytorch/pytorch#182339`](https://github.com/pytorch/pytorch/issues/182339) closed upstream — both surface a new Dynamo gap (filed as #126).

## 6. Model state changes

### 6.1 Pure Dynamo wins — 0 this week

No flips graph_break → full_graph on the apple-to-apple HF set.

### 6.2 Reduced GBs — 0 pairs this week

Zero pairs reduced GB count. Four GraniteMoeHybrid pairs increased by +1 GB each (still compile-success):

| Model | Mode | baseline GB | current GB | Δ |
|---|---|---|---|---|
| GraniteMoeHybridForCausalLM | train | 12 | 13 | +1 |
| GraniteMoeHybridModel       | train | 10 | 11 | +1 |
| GraniteMoeHybridForCausalLM | eval  | 12 | 13 | +1 |
| GraniteMoeHybridModel       | eval  | 10 | 11 | +1 |

All four already tracked at #55.

### 6.3 Compile regressions — 0 real

Zero pairs flipped from compile-success → error.

## 7. Issues — actions taken

**Closed:** 1 (upstream)
- [`pytorch/pytorch#182339`](https://github.com/pytorch/pytorch/issues/182339) (`scaled_dot_product_attention` query-pad eager bug) closed upstream 2026-05-06. We removed `MimiModel` from `sweep/known_errors.json` and re-ran both modes: both flipped `eager_error → graph_break` (now compiles eagerly; surfaces a separate Dynamo gap, see #126 below).

**New issues filed:** 4 (filed 2026-05-11)

| #   | Title | Pairs | Models |
|---|---|---|---|
| [124](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/124) | `[corpus-tooling]` QianfanOCR fixture bug — image tokens/features mismatch (4 pairs) | 4 | QianfanOCRForConditionalGeneration ×2, QianfanOCRModel ×2 |
| [125](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/125) | `[dynamo]` FX setitem on FakeTensor fails for UdopEncoderModel (eval + train) | 2 | UdopEncoderModel ×2 |
| [126](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/126) | `[dynamo]` PendingUnbackedSymbolNotFound in nn.functional.pad with int tensor padding (MimiModel _pad1d) | 2 | MimiModel ×2 (surfaced after #182339 close) |
| [127](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/127) | `[corpus-tooling]` Explain pass async-CUDA failure not reproducible in isolation (Blip2\|eval, sweep 2026-05-09) | 1 | Blip2ForConditionalGeneration\|eval |

**Comments added to existing issues:** 1
- [`#55`](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/55) (`_local_scalar_dense` from `.tolist()` at modeling_granitemoehybrid.py:924) — added comment listing the 4 GraniteMoeHybrid pairs that newly hit this pattern this week.

**Edited:** 1 (umbrella scope reduction)
- [`#122`](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/122) (data-dep control-flow umbrella) — title + body edited to reflect that 9 of its 9 top-named source sites collapse into existing #77 (LayerDrop family, 56 breaks across 6 sites) and #78 (`torch.all(mask==1)` family, 28 breaks across 3 sites). New title: `[dynamo] Data-dep control-flow — residual ~773 breaks (LayerDrop @ #77, mask==1 @ #78 already split out)`. Residual scope is now ~520 "unknown source" breaks (blocked on a corpus-side `worker.py` upgrade for richer break-reason capture) plus a long-tail of named sites with ≤5 breaks each.

**Net effect on tracked issues (corpus repo):** new issues +4 / closed 0 → net +4 open. (One upstream pytorch/pytorch close drove a known_errors removal but didn't close any corpus-side issue. #122 scope-reduction is an edit, not a close.)

## 8. NEW break-reason types surfaced (not seen in any baseline model)

Zero new break-reason types on the apple-to-apple set.

- The MimiModel `_pad1d` `PendingUnbackedSymbolNotFound` is a new Dynamo break-reason but does NOT appear in the apple-to-apple set because the pair was `eager_error` in baseline; it surfaces only after the `pytorch/pytorch#182339` eager fix lands and the model becomes Dynamo-testable. **Newly filed this sweep as [#126](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/126).**
- The +4 GraniteMoeHybrid GB delta (§1) is the existing `_local_scalar_dense` from `.tolist()` pattern. **Already tracked at [#55](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/55)** (comment posted this week linking the 4 pairs). NOT a new break-reason type.

No new operators (e.g. no `aten.bincount`-class additions) need an ops-coverage decision this week.

## 9. Major rewrite of Dynamo issues this week (one-off)

The corpus has been investing heavily in raising the contribution-quality bar of Dynamo issues. Two waves of rewrite work landed in the last 7 days:

**Last week (2026-05-04 → 2026-05-09):**
- **#77** rewritten end-to-end via the new `file-issue` subagent (Mode A adversary review → Mode B body assembly → live `--edit`). Discovery-lineage corrected, criterion-4 fix-suggestions removed.
- **MRE-subagent dogfooding on ~19 dynamo issues** (#10, #12, #14, #16, #17, #19, #21, #23, #24, #25, #26, #28, #76, #78, #92, #96, #97, #98, #99): standalone reproducers constructed from existing sweep evidence and added to each issue body. Most issues went from "see corpus harness" to "here is a ~20-line MRE you can run."
- **Title click-decision rewrites** on #92, #98, #99 — titles now encode subsystem tag + scope + just-enough-mechanism so a maintainer can decide-to-click from the issue list without opening.

**This week (2026-05-11):**
- **#124-#127** (4 new issues filed) all rewritten same-day via the file-issue subagent. The recurring defect caught in all 4: a defensive "What this issue is NOT" section that paradoxically enumerated fix directions (criterion-4 anti-pattern). Bodies rewritten, MREs added for #125 + #126 (verified live to reproduce verbatim sweep error), titles trimmed to drop overclaimed causal language.
- **#122** umbrella scope reduction (see §7) — title + body edited to acknowledge that the top-9 named source sites are already covered by #77 and #78.

**Total rewrite footprint: ~25 dynamo issues touched in 7 days.** The recurring "What this issue is NOT" defensive-disclaimer pattern has been encoded as a hard-block in the file-issue persona for all future filings.

---

_Generated 2026-05-11 from `sweep_results/nightly/2026-05-09/` augmented with amendment `2026-05-11T01-49Z-mimi-eager-fix-pt182339`. Reproduce: `tools/sweep_compare.py --baseline sweep_results/nightly/2026-05-03 --current sweep_results/nightly/2026-05-09 --source hf --json /tmp/cmp.json --ignore-invariants`._
