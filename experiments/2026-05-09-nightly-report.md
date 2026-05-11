# PT2 Nightly Sweep Brief — 2026-05-09 vs 2026-05-03 (Dynamo scope)

**Window:** 6 days. torch nightly `2.13.0.dev20260502+cu126` → `2.13.0.dev20260507+cu126`.
**Scope:** Dynamo-relevant changes only — apple-to-apple on common (name, mode) HF transformers pairs that ran in BOTH baseline and current. Diffusers + custom suites + timm-dependent models excluded. Eager-side timeouts and harness gaps tracked separately.

---

## 1. Headline

**Apple-to-apple, this is a quiet week.** After applying the modellibs `transformers` 5.8.0 upgrade today (re-running the 11 affected models on tx 5.8.0 + this week's torch nightly), the apple-to-apple set shows ZERO Dynamo deltas vs baseline: all 11 re-checked classes hold byte-identical identify status AND identical graph_break_count. The +4 GB GraniteMoeHybrid pattern observed earlier was 100% a transformers `5.8.0 → 5.6.2` downgrade artifact (now resolved). Zero models flipped graph_break → full_graph. Zero real upstream compile regressions.

## 2. Setup

| Setup         | Last week (2026-05-03)        | This week (2026-05-09, post-upgrade) |
|---|---|---|
| torch nightly | `2.13.0.dev20260502+cu126`    | `2.13.0.dev20260507+cu126`           |
| transformers  | `5.8.0` (pip-installed)       | `5.8.0` (modellibs, upgraded 2026-05-11) |

**Methodology note:** the original this-week sweep ran on `transformers 5.6.2` (the prior modellibs default) — a downgrade vs baseline that surfaced two artifacts (a +4 GB GraniteMoeHybrid pattern + 9 model classes "missing" from the cohort). Both artifacts are corpus-side, not torch behavior. Today we shipped the modellibs upgrade to `5.8.0` (commit `e0301da`) and re-ran the 11 affected work items; the artifacts disappear cleanly. Going forward the regular nightly sweep on `torch-nightly-cu126` defaults to `transformers 5.8.0`. (Patch sweep results: `sweep_results/experiments/2026-05-11-tx58-patch/`.)

## 3. Apple-to-apple Topline

Post-upgrade apple-to-apple set: 1450 (model × mode) HF pairs present in BOTH sweeps (1432 from the original this-week cohort + 18 from the 9 model classes that were absent on tx 5.6.2 and now restored).

| Metric                                          | Last week     | This week (post-upgrade) | Δ |
|---|---|---|---|
| Compiles fullgraph                              | 1109          | 1109          | 0  |
| Compiles with graph breaks                      | 331           | 331           | 0  |
| Pairs with errors                               | 10            | 10            | 0  |
| Total graph breaks                              | ≈3850         | ≈3850         | 0  |

Status counts on the common-pair set are byte-identical across the two weeks. Total-GB delta on the 11 re-checked classes is 0 (verified directly by comparing `graph_break_count` per pair). On the remaining ~1428 pairs not in the patch scope, status was already byte-identical from the original sweep_compare — no change introduced by the upgrade.

## 4. Dynamo focus — top-5 lists + actionable this week

The "if you read nothing else, read this" surface for the Dynamo team. Three lenses on the same question.

### 4.1 Top 5 Dynamo issues by blast radius (model count × break count)

| # | Symptom | Scope |
|---|---|---|
| [117](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/117) | `aten.nonzero.default` data-dependent output shape | 59 models, 573 breaks |
| [96](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/96)   | `DictIterator/DictItemsIterator` gb0092 in transformers' `output_capturing` decorator | 103 models |
| [55](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/55)   | `_local_scalar_dense` from `.tolist()` | ~64 models, 100+ breaks |
| [115](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/115) | `Tensor.item()` with `capture_scalar_outputs=False` | 27 models, 246 breaks |
| [78](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/78)   | Data-dep branching on `torch.all(mask==1)` mamba/linear-attn fast-path (incl. GraniteMoeHybrid + MiniCPMV4_6 on tx 5.8.0) | 22+ models, 240+ breaks |

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

1. **[#125](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/125)** — `[dynamo]` FX `setitem` on FakeTensor fails for UdopEncoderModel. Fake-tensor `setitem` on a `(2, 196)` bool tensor cannot be traced when the index is a tuple of 0-d int64 tensors. 2 model×mode pairs currently zero-coverage. ~25-line standalone MRE in body, verified live.
2. **[#126](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/126)** — `[dynamo]` `PendingUnbackedSymbolNotFound` in `nn.functional.pad` with int-tensor padding (MimiModel `_pad1d`). Newly surfaced after `pytorch/pytorch#182339` eager fix landed. ~30-line standalone MRE in body, verified live. Leverage: 2 MimiModel pairs currently graph_break instead of full_graph; likely affects any audio-model family that does dynamic-shape padding via int-tensor.
3. **[#78](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/78)** — `torch.all(attention_mask == 1)` linear-attn / mamba mask fast-path. After today's modellibs upgrade, GraniteMoeHybrid (4 pairs, 34 GBs) and MiniCPMV4_6 (4 pairs, 14 GBs) join this issue's territory. Existing tracking issue is already actionable.

## 5. Cohort changes

**Cohort definition:** the "compile-testable cohort" — model × mode pairs that are NOT in `known_errors.json` (eager-side bugs we deliberately exclude) and NOT in `skip_models.json` (intentional skips).

|                                                  | Count           |
|---|---|
| Models added (in this week, NOT in last week)    | 2 work items / 1 distinct (MimiModel ×2 modes) |
| Models removed (in last week, NOT in this week)  | 0               |

**Added breakdown:**
- **MimiModel ×2 (eval + train)** — re-entered the cohort because we REMOVED the `known_errors.json` entry after [`pytorch/pytorch#182339`](https://github.com/pytorch/pytorch/issues/182339) closed upstream 2026-05-06. Both modes now run eagerly (`eager_error → graph_break`); the new Dynamo gap they surface is filed as [#126](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/126). Net `known_errors.json` count: 8 → 7.

(The 9 model classes that appeared "removed" in the original this-week sweep — DeepseekV4×2, Deimv2, GraniteSpeechPlus, Laguna×2, MiniCPMV4_6×2, PPFormulaNet — are restored after today's modellibs `transformers` 5.8.0 upgrade. They re-ran cleanly on the patch sweep at the same status they had in baseline.)

## 6. Model state changes

### 6.1 Pure Dynamo wins — 0 this week

No flips graph_break → full_graph on the apple-to-apple HF set.

### 6.2 Reduced GBs — 0 pairs this week

Zero pairs reduced GB count.

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
- [`#55`](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/55) — comment posted earlier this week noting the +4 GraniteMoeHybrid GB delta at the `_local_scalar_dense` from `.tolist()` pattern (observed on the original tx 5.6.2 sweep). On the post-upgrade tx 5.8.0 data, GraniteMoeHybrid does not exhibit the `.tolist()` pattern — it breaks instead on `granitemoehybrid.py:1219` (the `torch.all(mask==1)` mask fast-path, #78 territory). #55 still tracks the broader `_local_scalar_dense` pattern across other models in the corpus.

**Edited:** 1 (umbrella scope reduction)
- [`#122`](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/122) (data-dep control-flow umbrella) — title + body edited to reflect that 9 of its 9 top-named source sites collapse into existing #77 (LayerDrop family, 56 breaks across 6 sites) and #78 (`torch.all(mask==1)` family, 28 breaks across 3 sites — now expanded to include GraniteMoeHybrid + MiniCPMV4_6 with the modellibs upgrade). New title: `[dynamo] Data-dep control-flow — residual ~773 breaks (LayerDrop @ #77, mask==1 @ #78 already split out)`. Residual scope is now ~520 "unknown source" breaks (blocked on a corpus-side `worker.py` upgrade for richer break-reason capture) plus a long-tail of named sites with ≤5 breaks each.

**Net effect on tracked issues (corpus repo):** new issues +4 / closed 0 → net +4 open.

## 8. NEW break-reason types surfaced (not seen in any baseline model)

Zero new break-reason types on the apple-to-apple set.

- The MimiModel `_pad1d` `PendingUnbackedSymbolNotFound` is a new Dynamo break-reason but does NOT appear in the apple-to-apple set because the pair was `eager_error` in baseline; it surfaces only after the `pytorch/pytorch#182339` eager fix lands and the model becomes Dynamo-testable. **Newly filed this sweep as [#126](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/126).**

No new operators (e.g. no `aten.bincount`-class additions) need an ops-coverage decision this week.

## 9. Major rewrite of Dynamo issues this week (one-off)

The corpus has been investing heavily in raising the contribution-quality bar of Dynamo issues. Two waves of rewrite work landed in the last 7 days:

**Last week (2026-05-04 → 2026-05-09):**
- **#77** rewritten end-to-end via the new `file-issue` subagent (Mode A adversary review → Mode B body assembly → live `--edit`). Discovery-lineage corrected, criterion-4 fix-suggestions removed.
- **MRE-subagent dogfooding on ~19 dynamo issues** (#10, #12, #14, #16, #17, #19, #21, #23, #24, #25, #26, #28, #76, #78, #92, #96, #97, #98, #99): standalone reproducers constructed from existing sweep evidence and added to each issue body. Most issues went from "see corpus harness" to "here is a ~20-line MRE you can run."
- **Title click-decision rewrites** on #92, #98, #99 — titles now encode subsystem tag + scope + just-enough-mechanism.

**This week (2026-05-11):**
- **#124-#127** (4 new issues filed) all rewritten same-day via the file-issue subagent. The recurring defect caught in all 4: a defensive "What this issue is NOT" section that paradoxically enumerated fix directions (criterion-4 anti-pattern). Bodies rewritten, MREs added for #125 + #126 (verified live to reproduce verbatim sweep error), titles trimmed to drop overclaimed causal language.
- **#122** umbrella scope reduction (see §7).

**Total rewrite footprint: ~25 dynamo issues touched in 7 days.** The recurring "What this issue is NOT" defensive-disclaimer pattern has been encoded as a hard-block in the file-issue persona for all future filings.

---

_Generated 2026-05-11. Source data: `sweep_results/nightly/2026-05-09/` (original this-week sweep) augmented with `sweep_results/experiments/2026-05-11-tx58-patch/` (transformers 5.8.0 patch on the 11 affected model classes — re-run today on `~/envs/torch-nightly-cu126` with `--transformers 5.8.0`). The §3 Topline + §1 Headline numbers reflect the post-patch state; sections that did not change between the original and patch data are unchanged from the pre-patch sweep_compare output._
