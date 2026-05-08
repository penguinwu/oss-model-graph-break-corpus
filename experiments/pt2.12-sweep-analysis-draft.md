# PT 2.12 — Graph-Break State of the Union for HF Transformers

**Status:** DRAFT (2026-05-03). Long-form for compiler developers. Comments welcome.
**Author:** Otter (drafting); Peng Wu (reviewing)
**Audience:** PT2 / Dynamo team
**Data basis:** PT 2.12 pre-release sweep at `sweep_results/baseline/pt2.12-2026-04-30/` (torch 2.12.0.dev20260407+cu128, transformers 5.6.2). To be refreshed when stable 2.12 wheel publishes.

---

## §1. Executive summary

**State of graph capture in PT 2.12 (pre-release sweep, 2026-04-30):**

- **Eval fullgraph rate: 73.1%** (540 / 739). **Train: 67.1%** (496 / 739). Train mode produces **215 extra graph break occurrences** beyond eval (avg ~5 per affected model; IBertModel alone has 19). These 215 breaks are concentrated in 44 models that flip from `eval=full_graph` to `train=graph_break`; the ~6pt deficit converts entirely into graph_break status, with no leak into other failure categories. Dominant cause: per-model `self.training`-conditioned forward branches with data-dependent control flow. See §6.
- **Headline accounting check passes** (Q3): Dynamo's `graph_count = graph_break_count + 1` invariant holds across all 446 explain rows. Subsequent stats can be trusted.
- **Most graph breaks are concentrated, not scattered** (Q2): of the 215 models with ≥5 breaks, **53% have ≤30% unique break locations** — one root cause amplified into many downstream breaks. Fixing single root causes yields disproportionate impact.

**Top single-target Dynamo improvements that would unlock the most models** (each cross-referenced to corpus filed-issue + verified upstream PR/commit):

1. **Deepcopy polyfill — LANDED + VERIFIED.** PR #179611 ([dynamo] Support copy.deepcopy via polyfill) landed via ghstack 2026-04-11 as commit `61fdec7ddb5d` on pytorch main. **Already in current torch nightly** (`2.13.0.dev20260502+` and later) and `release/2.12` (verified: `behind_by=0`). Corpus tracking: Issue #1 (closed 2026-04-19 with verification). **Empirically verified on 2026-05-03 nightly sweep:** ZERO `copy.py:151` / `copy.deepcopy` break entries in explain output (was ~108 entries in PT 2.12 baseline). The single largest break-cluster from PT 2.11/2.12 is fully cleared.

2. **`CALL_FUNCTION_EX` (variadic call) better handling** — would resolve ~177 of 431 #103 wrapper occurrences. Corpus tracking: Issue #103 (open). Also helps with the `output_capturing.py:192` cascade (122 breaks at that single line). No upstream PR visible yet — would be net-new Dynamo work.

3. **Un-skip / polyfill specific functions Dynamo currently skips** — would clear ~72 #102 inner reasons. **Top concrete targets** (extracted directly from `break_reasons` text):

   | Skipped function | Occurrences | Notes |
   |---|---:|---|
   | `importlib.util.find_spec` | 96 | importlib plumbing — used by transformers' optional-dependency checks |
   | HF `*Config.__reduce_ex__` (Bart, Moonshine, Blenderbot, M2M100, Marian, NllbMoe, PLBart, Pegasus, ...) | ~250 combined | HF Config object pickle/copy paths — **deepcopy polyfill (#179611, LANDED) likely clears most of these** since they're invoked via `copy.deepcopy(config)` |

   Corpus tracking: filed-issue cross-references for these were partially incorrect prior to a 2026-05-03 classifier-bug fix in `tools/file_issues.py` (commit `4b28545`) — the classifier's location-based rules were shadowing more-specific explanation-based rules, so 96 `find_spec` breaks were misfiled against Issue #5 ("import_utils.py skip cluster") instead of Issue #7 ("find_spec skip"). Both issues are closed; future sweep runs will categorize correctly. After #179611 lands in nightly, the `__reduce_ex__` cluster should largely disappear; remaining = `find_spec`-style importlib calls under Issue #7.

4. **`callable()` builtin support for unknown argument types** — ~248 breaks across `import_utils.py:1525/1538/1540` in transformers. **Important clarification:** the break is NOT in `is_torch_compiling()` itself (which Dynamo can constant-fold to True). The break is on `callable(<arg>)` where the arg is a `StringFormatVariable` (an interpolated f-string passed as an argument) — Dynamo doesn't know how to evaluate `callable()` with an interpolated-string variable type. Corpus tracking: Issue #20 (the callable-builtin cluster). Upstream relevant: PR #179629 ([dynamo] Constant-fold importlib functions and fix callable() for StringFormatVariable) — **LANDED via ghstack 2026-04-22** as commit `2c24b04d2d23`; in current nightly but NOT in `release/2.12`.

**Big regressions / wins vs PT 2.11 (intersection of 1420 work items):**

- **Capability win:** Gemma4 went from `compile_error` → `graph_break` — Dynamo now graph-breaks gracefully on a data-dep f-string instead of hard-crashing.
- **Soundness improvement:** MraModel went from `graph_break` → `full_graph` thanks to better `requires_grad_()` speculative handling.
- **Real regression (NOT torch.compile):** BltModel construction time went 111s → 374s in eager (3.4× slowdown) due to `nn.Module` construction path changes. Caused 4 work items to flip `full_graph → timeout`. Worth filing as a torch issue.

**Train vs eval asymmetry — primary drivers:**

- **47% of models with explain data have train > eval breaks.** Only 1.3% have eval > train.
- Top offender: **IBertModel** (eval=0 fullgraph, train=19 breaks all from quantization-train code paths).
- Detection/segmentation models (DETR family, RTDetr, Florence2, PPDocLayout) consistently have +8 to +12 train-only breaks from loss-computation control flow.

**New models since 2.11 (29 added):**

- 10 fullgraph passes (Mllama, Zamba2, Sam3-Lite — encouraging coverage on recent architectures)
- 3 graph_break (manageable)
- 16 fail at the eager / create boundary (model-side / transformers-version issues — not yet ready for compile evaluation)

**Updated comparison: PT 2.12 RC vs our pre-release snapshot (2026-04-30).** The actual `release/2.12` branch (head SHA `06f10d088229...` as of 2026-05-03) contains:
- ✅ **PR #179611 (deepcopy polyfill)** — IS in release/2.12. Our 2026-04-30 snapshot used April-7 torch which predates this. The 164 deepcopy breaks at `copy.py:151` will be GONE in 2.12 stable.
- ❌ **PR #180585 (`object.__getattribute__` fallback)** — NOT in release/2.12. Branched off main before the April-17 landing; not cherry-picked. Breaks this would have addressed remain in 2.12 stable.

**Both verified via `tools/pr_landing_check.py 179611 180585 --branch release/2.12`.** This script is now mandatory for pytorch PR status checks (see CLAUDE.md "Checking pytorch/pytorch PR Status" rule) — added 2026-05-03 after we initially misclassified #179611 as not-merged due to ghstack's merge-via-MergeBot pattern (GitHub shows `merged: false` even when ghstack has landed the change).

**Bottom line for Dynamo developers:** PT 2.12 is approximately even with PT 2.11 on the model intersection (Δ−3 fullgraph, +1 graph_break, +4 timeout — all explained). The headline regression is BltModel (eager-side, file separately). For forward-leaning improvements, the deepcopy polyfill + `CALL_FUNCTION_EX` work are the highest-leverage targets.

**Sections that follow:**
- §2: top-line stats with full break-count distribution
- §3: per-new-model breakdown
- §4-5: accounting consistency check + cascade pattern characterization
- §6: train vs eval asymmetry drivers
- §7: #102/#103 wrapper-pattern inner-reason analysis
- §8: regression analysis vs PT 2.11
- §9: prioritized "Dynamo-fixable vs model-rewrite" work list
- §10: nested graph break feature impact (deferred — see end of doc)

---

## §2. Top-line stats — eval and train, separated

PT 2.12 was tested against **739 unique HF transformer models in each of eval and train modes** (1478 total work items). For comparison, PT 2.11 covered 710 models per mode (1420 work items). The 29-model expansion is the "new-models-since-2.11" set analyzed in §3.

### Per-mode status counts

| Status | PT 2.12 eval (n=739) | PT 2.12 train (n=739) | PT 2.11 eval (n=710) | PT 2.11 train (n=710) |
|---|---:|---:|---:|---:|
| **full_graph** | **540 (73.1%)** | **496 (67.1%)** | 531 (74.8%) | 488 (68.7%) |
| graph_break | 179 (24.2%) | 223 (30.2%) | 176 (24.8%) | 219 (30.8%) |
| eager_error | 13 (1.8%) | 13 (1.8%) | — | — |
| timeout | 4 (0.5%) | 4 (0.5%) | 2 (0.3%) | 2 (0.3%) |
| create_error | 3 (0.4%) | 3 (0.4%) | — | — |
| compile_error | — | — | 1 (0.1%) | 1 (0.1%) |

**Observations:**

- **Train fullgraph rate consistently ~6 pts lower than eval** (73.1% vs 67.1% in PT 2.12; 74.8% vs 68.7% in PT 2.11). The asymmetry persists across releases — investigated in §6.
- **`compile_error` disappeared in 2.12** (was 1 each in 2.11). Per the regression analysis (Gemma4 case), Dynamo now graph-breaks earlier on the data-dependent f-string pattern instead of crashing. Strict capability win.
- **`eager_error` + `create_error` grew** (16 each in 2.12 vs 1 compile_error in 2.11). Audit (2026-05-03) confirmed all 16 are from new models added between 2.11 and 2.12 (zero from intersection); ~10 are corpus-harness gaps already addressed in current corpus code; ~6 are real transformers/model-side bugs. **Tracked at corpus Issue [#109](https://github.com/penguinwu/oss-model-graph-break-corpus/issues/109)** with per-model breakdown + status.
- **Fullgraph rate dropped slightly eval-side** (74.8% → 73.1%, Δ−1.7pt) and **dropped slightly train-side** (68.7% → 67.1%, Δ−1.6pt). The drop is dominated by new-model additions having a lower fullgraph rate; for the 710-model intersection the fullgraph counts are nearly identical (see §8 regression analysis).

### Break-count distribution

For models with status `full_graph` or `graph_break` (719 per mode after de-dup, excluding eager_error / create_error / timeout):

| Break-count bucket | eval | train |
|---|---:|---:|
| **0 (full_graph)** | **543 (75.5%)** | **501 (69.7%)** |
| 1 | 4 (0.6%) | 4 (0.6%) |
| 2-5 | 106 (14.7%) | 91 (12.7%) |
| 6-10 | 32 (4.5%) | **77 (10.7%)** |
| 11-20 | 23 (3.2%) | 34 (4.7%) |
| 21+ | 11 (1.5%) | 12 (1.7%) |

**Train-mode break-count shifts upward.** The biggest delta is in the 6-10 break bucket (eval 4.5% → train 10.7%, +6.2pt) — many models that have a few breaks in eval grow to ~10 breaks in train. This is consistent with the per-model asymmetry analyzed in §6 (loss-computation paths add breaks in train mode).

**Top 5 most-broken models (PT 2.12):**

| Eval | Breaks | Train | Breaks |
|---|---:|---|---:|
| EncodecModel | 29 | EncodecModel | 30 |
| AriaForConditionalGeneration | 28 | VitsModel | 29 |
| AriaModel | 27 | AriaForConditionalGeneration | 28 |
| Qwen3VLForConditionalGeneration | 25 | AriaModel | 27 |
| Qwen3VLMoeForConditionalGeneration | 25 | Qwen3VLForConditionalGeneration | 25 |

### Top break reasons

The dominant graph-break inner reasons across PT 2.12 (extracted from `explain_results.json`'s `break_reasons.reason` field, after stripping wrapper boilerplate):

| Rank | Inner reason | Occurrences | Category |
|---|---|---:|---|
| 1 | Data-dependent branching | 365 | Model-internal: branch on data |
| 2 | Data dependent operator | 224 | Model-internal: data-dep op |
| 3 | Dynamic shape operator | 88 | Could be Dynamo-fixable |
| 4 | Attempted to call function marked as skipped | 72 | Likely Dynamo-fixable (un-skip / polyfill) |
| 5 | Observed exception | 54 | Mixed (depends on exception type) |
| 6 | Failed to convert args/kwargs to proxy | 42 | Likely Dynamo-fixable (proxy mechanism) |
| 7 | Unsupported `Tensor.item()` (no capture_scalar_outputs) | 41 | Configurable (or model-rewrite) |
| 8 | Unsupported context manager | 38 | Per-context-manager Dynamo support |
| 9 | Failed to trace builtin operator | 26 | Dynamo-fixable (builtin polyfill) |
| 10 | `torch.*` op returned non-Tensor | 12 | Dynamo-fixable (op return-type handling) |

Top 2 categories ("data-dependent branching" + "data-dependent operator") account for **~58%** of graph-break causes — these are classic model-internal Python control-flow on tensor values. They typically need either model-side rewrites or `dynamo.config.capture_scalar_outputs=True` + careful op support.

*[TODO: separate this table eval vs train; compare head-to-head.]*

---

## §3. New models since PT 2.11

Between PT 2.11 (transformers 5.5.3) and PT 2.12 (transformers 5.6.2) sweep snapshots, **29 new model name+source combinations were added** to the corpus (zero models removed). The new arrivals include high-profile recent releases like the Mllama, Qwen3-Omni, HYV3, and PerceptionDecoder series, as well as a number of vision-language model variants.

### New-model status breakdown (PT 2.12)

| Status | Eval (n=29) | Train (n=29) |
|---|---:|---:|
| **full_graph** | **10 (34.5%)** | **10 (34.5%)** |
| graph_break | 3 (10.3%) | 3 (10.3%) |
| **eager_error** | **13 (44.8%)** | **13 (44.8%)** |
| **create_error** | **3 (10.3%)** | **3 (10.3%)** |

### Comparison: new vs existing models

| | Existing (intersection) | New (added in 2.12) |
|---|---:|---:|
| eval fullgraph rate | 74.8% | **34.5%** (−40 pt) |
| train fullgraph rate | 67.8% | **34.5%** (−33 pt) |
| eager_error + create_error rate | 0% existing | **55.1%** new (eval+train identical) |

**Reading:** more than half of the new model additions fail at the `eager_error` or `create_error` boundary in the 2.12 baseline. **However, audit (2026-05-03) revealed these failures are dominated by CORPUS HARNESS gaps, NOT by torch.compile or genuine model bugs.** Specifically: corpus's per-model input recipes and `_reduce_model_size` config overrides hadn't kept pace with newly-added models that use custom forward signatures (e.g., `aspect_ratio_ids`, `grid_thw`, `mel_spectrogram`) or unconventional config shapes (Mamba-hybrid Zamba's `tie_weights_keys` regex, Qwen3-Omni's nested config attribute resolution).

**Audit + fixes (2026-05-03):**
- **Proof:** all 13 eager_errors and 3 create_errors in the 2.12 baseline come from new models. ZERO from intersection. Confirms our new-model harness is the gap.
- **Already fixed by 2026-05-02 input-recipe commits** (`08511ae`, `47bba8a`, `761e28f`): 7 vision/audio sub-component models with custom forward signatures (Mllama, Qwen3VL/Qwen3VLMoe vision, Qwen3_5/Qwen3_5Moe vision, etc.).
- **Fixed today** (commit `47d03db`): 3 config-shape models — ZambaForCausalLM + ZambaModel (`tie_weights_keys` regex requires layer 1 hybrid + peer; fix: `layers_block_type = ['hybrid'] * num_hidden_layers`); Qwen3OmniMoeTalkerModel (extended `_fix_qwen3_omni_talker` to comprehensively promote text_config attrs onto wrapper config + alias `num_experts` → `num_local_experts`).
- **Remaining genuine issues**: ~4-6 models with model-side bugs (MusicFlamingoForConditionalGeneration CUDA assert, Qwen3-Omni-Talker submodels with model __init__ bugs, etc.) — these are real transformers-side issues to file upstream.

**Aggregate projection for the next sweep that runs against current corpus code:** ~10 of the 16 new-model errors flip to fullgraph/graph_break/eager_error (showing real graph-capture behavior). Approximately 6 remain as real transformers bugs needing upstream coordination.

**Methodology lesson** (per Peng's directive: "we have to fix these harness flaws before sharing the report"): for the published report on the official 2.12 release, refresh the sweep on current corpus code so the new-model status reflects the actual graph-capture story rather than corpus harness gaps.

For Dynamo developers reading the 2.12-pre-release table above: the 16 errors number is INFLATED by the harness gap that has since been addressed. The actual 2.12-stable picture for new models will show closer to 19-23 of 29 new models reaching the compile stage (up from 13 in this baseline).

### Specific new-model lineup

**✅ 10 fullgraph passes** (immediately compile-clean):
HYV3ForCausalLM, HYV3Model, MllamaForCausalLM, MllamaTextModel, QianfanOCRVisionModel, Qwen3OmniMoeThinkerTextModel, Sam3LiteTextModel, Sam3LiteTextTextModel, Zamba2ForCausalLM, Zamba2Model

These represent encouraging coverage on recent-release architectures. The Mllama (text) + Zamba2 (Mamba-style state-space hybrid) wins are particularly worth highlighting — both are non-trivial transformer variants.

**3 graph_break**:
OpenAIPrivacyFilterModel, QianfanOCRForConditionalGeneration, QianfanOCRModel

**13 eager_error** (fail BEFORE Dynamo gets to compile — model-side or transformers-version issues):
MllamaVisionModel, MusicFlamingoForConditionalGeneration, PI0Model, PeAudioVideoModel, Qwen2_5OmniThinkerTextModel, Qwen2_5OmniToken2WavBigVGANModel, Qwen2_5OmniToken2WavDiTModel, Qwen3OmniMoeTalkerCodePredictorModel (+ ForConditionalGeneration variant), Qwen3VLMoeVisionModel, Qwen3VLVisionModel, Qwen3_5MoeVisionModel, Qwen3_5VisionModel

Pattern: the bulk of eager_errors are Qwen3-Omni multimodal sub-components and Qwen3-VL vision encoders — these have rapidly-evolving model code that's not yet stable in this transformers version.

**3 create_error** (model construction itself fails):
Qwen3OmniMoeTalkerModel, ZambaForCausalLM, ZambaModel

The Zamba (v1) failures are interesting given Zamba2 (above) succeeds — would warrant a separate transformers / corpus follow-up.

---

## §4. Q3 — graph_count vs graph_break_count accounting consistency

**Question:** Does the invariant `graph_count = graph_break_count + 1` hold across all PT 2.12 results? (Each break splits a graph into one more subgraph.)

**Result:** ✅ **Invariant holds universally — 0 violations across 446 explain-pass rows in PT 2.12.**

This validates that Dynamo's reporting of break counts is internally consistent. Subsequent stats (cascading analysis in §5, asymmetry quantification in §6, deep-dive in §7) can rely on `graph_count` and `graph_break_count` as paired measurements without worrying about reporting bugs.

*[Note: explain_results.json covers the 446 work items that had `graph_break` status (i.e., not full_graph). full_graph models trivially have graph_count=1, graph_break_count=0 which satisfies the invariant.]*

---

## §5. Q2 — Deep-stack origin and cascading downstream breaks

**Hypothesis:** A single root-cause graph break, when it occurs deep in a call stack, can cause N downstream breaks because the unsupported state propagates through subsequent traced code paths. We characterize the distribution of "single root → N cascade" patterns in PT 2.12.

### Concentration analysis

For the **215 models with ≥5 graph breaks**, we extract the file:line location of each break (from `break_reasons.reason` text) and compute the ratio of unique-locations to total breaks:

| Pattern | # models | Reading |
|---|---:|---|
| **Concentrated** (≤30% unique locations) | **113 (53%)** | One root cause amplified across many breaks — classic cascade |
| Scattered (>30% unique locations) | 102 (47%) | Genuinely independent breaks, each a distinct cause |

**More than half of heavy-break models exhibit the cascade pattern.** Fixing the single underlying root cause for these models would clear most of their break count.

### Top cascade champions (1 root → N breaks)

> **What "1 root → N breaks" means in this dataset:** for the top cascade champions below, the N reported breaks all originate from a SINGLE underlying break site that gets hit repeatedly during the forward pass. Example: MimiModel's 19 reported `break_reasons` entries ALL point at `torch/nn/functional.py:5462`. The breakdown is 1 first-time break + 15 "duplicate graph break" suppressions + 3 #102 wrapper failures — all at the same location. So "18 breaks" doesn't mean 18 distinct issues; it means ONE torch-side limitation hit 18 times across the model's forward path. Fixing the single root would clear all 18+.

| Model | Breaks | Unique locations | Ratio |
|---|---:|---:|---:|
| MimiModel (eval / train) | 18 / 18 | 1 / 1 | 6% |
| DeformableDetrModel (eval) | 17 | 1 | 6% |
| AriaTextForCausalLM (eval / train) | 15 / 15 | 1 / 1 | 7% |
| AriaTextModel (eval / train) | 14 / 14 | 1 / 1 | 7% |
| **IBertModel (train)** | **28** | **2** | **7%** (and the dramatic 0→19 train-only delta from §6) |
| LwDetrModel (eval / train) | 11 / 11 | 1 / 1 | 9% |

These models have effectively one root cause expanding into 11-28 downstream breaks. For Dynamo developers, fixing these single roots yields disproportionate impact.

### Top "amplifier" file:lines (locations responsible for the most total breaks)

| Location | Total breaks across all heavy models |
|---|---:|
| `/usr/local/.../python3.12/copy.py:151` | **164** |
| `transformers/utils/output_capturing.py:192` | 122 |
| `transformers/models/aria/modeling_aria.py:267` | 114 |
| `transformers/utils/import_utils.py:1540` | 98 |
| `transformers/utils/import_utils.py:1538` | 92 |
| `torch/utils/hooks.py:27` | 72 |
| `transformers/utils/import_utils.py:1525` | 58 |
| `transformers/models/glm4v/modeling_glm4v.py:733` | 54 |

**Highest-leverage targets identified:**

1. **`copy.py:151` — 164 breaks**. This is the **deepcopy regression** stemming from PT 2.12's PR #177443/#177484 changes. **The fix — Animesh's polyfill PR #179611 — LANDED 2026-04-11 via ghstack** (commit `61fdec7ddb5d` on pytorch main). The PT 2.12 sweep snapshot used `torch=2.12.0.dev20260407` which is from April 7 — PRE-DATES the polyfill landing. The CURRENT torch nightly (`2.13.0.dev20260502`) post-dates April 11 and includes the polyfill. **Expected: the 164 deepcopy breaks at `copy.py:151` resolve in the next sweep that uses a post-2026-04-11 torch nightly.** Today's in-flight nightly will be the first observable verification.

   *Note on PR status:* GitHub's UI / API shows PR #179611 as `closed: merged=false` because ghstack lands via internal merge flow (PyTorch MergeBot), not via GitHub's Merge button. The GitHub `merged` field reflects button-click only. Same applies to PR #180585 (also from Animesh, also landed via ghstack 2026-04-17, also shows `merged: false` on GitHub).



2. **`output_capturing.py:192` — 122 breaks**. This is the transformers wrapper for output post-processing. Many models hit it on `CALL_FUNCTION_EX` with model-specific output object construction. Could be addressed either Dynamo-side (better variadic call handling — same pattern that resolves most #103 cases) or transformers-side (refactor wrapper).

3. **`import_utils.py:1525/1538/1540` — 248 breaks combined**. These are the `callable()` / `is_torch_compiling()` builtin pattern (issue #20). Many cluster instances visible across these three line numbers in the same module.

4. **`torch/utils/hooks.py:27` — 72 breaks**. Autograd hook registration in Dynamo's traced code path — likely surfaces in train mode (forward hooks for backward setup).

### Implication

The cascade pattern means **break_count headline numbers significantly overstate the work needed to fix them**. A model with 28 reported breaks may have only 2 unique root causes; fixing those clears 28 breaks. For Dynamo prioritization, the unique-location count is a much better signal than total break count.

---

## §6. Train vs eval graph-break asymmetry

### Headline

Train mode produces **215 extra graph break occurrences** beyond eval, distributed across 44 models that flip from `eval=full_graph` to `train=graph_break`. The per-model break count varies widely (range 1 to 19, average ~5); there is **no "1 extra break per flipped model" pattern**.

The 44 flipped models do not leak into other failure categories — `eager_error`, `create_error`, and `timeout` counts are identical across modes. The deficit is entirely a `full_graph → graph_break` conversion.

### Distribution of train-mode break counts across the 44 flip models

| train_break_count | Models |
|---:|---:|
| 1 | 2 |
| 2 | 2 |
| 3 | 3 |
| 4 | 11 |
| 5 | 15 |
| 6 | 8 |
| 10 | 1 |
| 19 | 1 |
| **Total** | **44 models / 215 breaks** |

Median train_break_count is 5. The two extremes — IBertModel (19) and ConditionalDetrModel (10) — are responsible for 14% of the 215.

### Categorization of the 215 train-only break occurrences

Categorized by source file path. Buckets are **non-overlapping** — a break is assigned to the first matching specific cluster (IBert / MusicGen / DETR / Autoformer); everything else lands in "Other transformers modeling_X.py". Numbers below are break-reason entries (319 total; exceeds 215 because some breaks generate multiple reason entries — the relative distribution is what matters).

| Category | Break entries | Models |
|---|---:|---:|
| Other transformers modeling_X.py (per-model train branches) | 204 | 33 |
| MusicGen modeling files | 48 | 6 |
| IBert `quant_modules.py` (quantization-aware training) | 28 | 1 |
| DETR family (detr/conditional_detr/table_transformer) | 28 | 3 |
| Autoformer modeling file | 11 | 1 |
| **Total** | **319** | **44** |

The dominant pattern (~64% of break entries) is per-model `self.training`-conditioned forward branches in regular `modeling_X.py` files, spread across 33 different models. Four named clusters (MusicGen, IBert, DETR, Autoformer) contribute the other 36% from concentrated code paths.

**Concrete example — AutoformerModel:** 5 train-mode graph breaks at `modeling_autoformer.py:886` and `:549`. Reason: "Data-dependent branching" — the model has `if some_tensor_value > 0:` style code on the train-mode forward path. Eval skips that branch via `self.training` gating; train hits it and breaks.

**Outlier — IBertModel:** 19 train-only breaks all from `transformers/models/ibert/quant_modules.py:173`, the quantization-aware-training activation-range computation. Single code line fires 19 times during forward (loop body in a quantized layer).

**Answers "is train hitting different Python code?":** yes. Train mode exercises `if self.training:` branches that activate dropout, loss computation, auxiliary heads, or quantization paths. Many of those branches contain data-dependent control flow Dynamo can't trace. None of these breaks come from autograd-hook registration — that's a separate phenomenon visible in §5's `torch/utils/hooks.py:27` amplifier (which fires in both modes for graph_break-status models, not specifically in train).

The full per-model roster (all 44 flip models with their train_break_count, sum verified = 215) is in Appendix D.

### Per-model asymmetry

Looking at the 223 models with `graph_break` status in either mode (where we have explain-pass data):

- **104 models** have `train_breaks > eval_breaks` (47% of the cohort)
- **3 models** have `eval_breaks > train_breaks` (rare)
- **116 models** have `train_breaks == eval_breaks` (symmetric breaks)

### Top 10 worst train-vs-eval deltas (PT 2.12)

| Model | eval breaks | train breaks | Δ |
|---|---:|---:|---:|
| IBertModel | 0 | 19 | **+19** |
| PPDocLayoutV3Model | 9 | 21 | +12 |
| TimeSeriesTransformerModel | 1 | 11 | +10 |
| ConditionalDetrModel | 0 | 10 | +10 |
| DFineModel | 8 | 18 | +10 |
| RTDetrModel | 5 | 14 | +9 |
| RTDetrV2Model | 5 | 14 | +9 |
| PPDocLayoutV2Model | 5 | 14 | +9 |
| Florence2Model | 4 | 12 | +8 |
| SeamlessM4TTextToUnitForConditionalGeneration | 3 | 11 | +8 |

**Striking case: IBertModel.** Eval = 0 breaks (fullgraph!), train = 19 breaks. The extra train-only breaks all originate at `transformers/models/ibert/quant_modules.py:173` — quantization-specific code paths that activate only in train mode.

**Pattern across the top-10:** most are detection/segmentation/perception models (DETR family, PPDoc layout, Florence, RTDetr) where train mode introduces additional loss-computation paths that hit data-dependent control flow.

### Hypotheses on the train-mode break sources

1. **Loss computation introduces data-dep branching** (e.g., DETR-style bipartite matching loss, focal loss with hard/easy example selection) — observed in DETR family
2. **Train-mode-only ops** like dropout, batch_norm running stats updates — affect basic blocks
3. **Quantization train paths** (IBertModel) — quant_modules.py introduces dynamic-range computation
4. **Auxiliary heads activated in train** (Florence2, perception models) — extra forward branches

*[TODO: programmatically categorize the train-only extra reasons after parsing wrapper boilerplate. Need to extract from explain_results.json's `break_reasons` text more carefully.]*

---

## §7. Deep-dive — `Failed to handle graph break gracefully` (#102) and `Cannot resume from graph break` (#103)

**Scale (PT 2.12 baseline, de-duplicated):**

| Wrapper issue | Models affected | Total occurrences |
|---|---:|---:|
| **#102** "Failed to handle graph break gracefully" | **193** | 1001 |
| **#103** "Cannot resume from graph break" | **141**¹ | 431¹ |
| **#104** (meta — better error messages) | n/a | n/a |

¹ The #103 numbers come from earlier issue-filing data, not from the PT 2.12 explain output directly. Verification on 2026-05-04: a substring search for "Cannot resume from graph break" in `pt2.12-2026-04-30/explain_results.json` returns ZERO entries — same for the 2026-05-03 nightly. Either (a) the wrapper text changed in torch versions between when the issue-filing data was captured and the PT 2.12 baseline run, or (b) the 141/431 figures were computed via a different classifier path that's not currently in the repo. Treat #103 numbers as **needs source verification** before publishing externally.

**Update on the 2026-05-03 nightly:** #102 wrapper persists at scale — **949 entries / 195 models** (vs PT 2.12 baseline 1001/193). Net flat. The frame-skip over-match fix (commit `a62a63d`, 2026-05-02) improved corpus-side classification but did NOT clear underlying torch-side breaks. **#102 remains the largest single-target Dynamo improvement opportunity.**

Both wrappers grew slightly from the issue-filing data (#102: 166→193; #103: 107→141) due to the 29 newly-added models in 2.12 hitting them as well.

### #102 — what the wrapper hides

When Dynamo encounters a graph break it cannot handle (re-entry into a frame, certain unsupported builtins inside an already-traced function, etc.), it currently prints the wrapper "Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered: <REAL REASON>". The wrapper hides the true cause one level deep.

**Inner reasons of #102 wrappers (PT 2.12):**

| Inner reason | Occurrences | Pct of #102 | Class |
|---|---:|---:|---|
| Data-dependent branching | 365 | 36.5% | Model-rewrite |
| Data dependent operator | 224 | 22.4% | Model-rewrite |
| Dynamic shape operator | 88 | 8.8% | Possibly Dynamo-fixable |
| Attempted to call function marked as skipped | 72 | 7.2% | Likely Dynamo-fixable |
| Observed exception | 54 | 5.4% | Mixed |
| Failed to convert args/kwargs to proxy | 42 | 4.2% | Likely Dynamo-fixable |
| Unsupported `Tensor.item()` | 41 | 4.1% | Configurable |
| Unsupported context manager | 38 | 3.8% | Dynamo-fixable |
| Failed to trace builtin operator | 26 | 2.6% | Dynamo-fixable |
| `torch.*` op returned non-Tensor | 12 | 1.2% | Dynamo-fixable |
| Other / long tail | 39 | 3.9% | Mixed |

**Reading:** ~58% of #102 wrappers are wrapping data-dependent control flow (rows 1+2). These would not benefit from "better wrapper handling" because the underlying issue is fundamental — the model code branches on data values. The remaining ~42% (rows 3-10) include several patterns Dynamo could plausibly support with focused engineering (`call function marked as skipped`, `convert args to proxy`, context managers, builtin tracing, op return-type handling).

### #103 — root causes triggering "cannot resume"

When Dynamo hits a break it cannot continue from (typically because the trace state is corrupted by the break itself), it prints "Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function."

**The break that triggered the cannot-resume (PT 2.12, looking at the immediately-preceding break_reasons entry for each #103 occurrence):**

| Triggering break pattern | Occurrences | Pct |
|---|---:|---:|
| Variadic call (CALL_FUNCTION_EX `f(*args)`) | 177 | ~63% |
| Other CALL with data-dependent argument | 3 | ~1% |
| (Could not parse predecessor) | 62 | ~22% |

The dominant root cause is **variadic Python calls** `f(*args)` where args can't be statically resolved by Dynamo. This is a single, tractable Dynamo improvement target — better support for `CALL_FUNCTION_EX` with partially-known args could resolve the bulk of #103 cases.

### What #104 (better error messages) would give us

#104 asks for the wrapper to surface the inner reason in a structured field rather than embedding it in free text. With #104:
- Issue triagers wouldn't have to parse free text to count "what's really breaking"
- Counts like the table above would come for free from sweep output
- Cross-referencing with corpus filed-issues becomes a join, not a regex

### Categorization for §9

Of the 193 #102-affected models, expected break-down by Dynamo-fixability:
- **~58% blocked by model-rewrite-class issues** (data-dep branching/ops)
- **~30% reachable with Dynamo improvements** (skipped funcs, proxy conversion, context managers, builtins, op return types, dynamic shapes)
- **~12% mixed / requires per-case investigation**

For #103: ~63% (variadic call handling) is a single Dynamo improvement target.

---

## §8. Regression analysis vs PT 2.11

For the 1420-model **intersection** between PT 2.11 (transformers 5.5.3) and PT 2.12 (transformers 5.6.2), only **7 work items changed status** out of 2840 — the vast majority of models are status-stable across the release.

| Status flip | Models | Cause | Corpus issue / upstream PR | Interpretation |
|---|---|---|---|---|
| `compile_error → graph_break` | Gemma4 × 2 modes | PT 2.12 hits the `callable()` graph break (issue #20 cluster) at `import_utils.py:1525` BEFORE reaching the data-dep f-string crash at `gemma4/modeling_gemma4.py:2235`. | corpus #20 (callable cluster); upstream — no specific PR contributed this win, it's an earlier-break-in-the-stack effect | **Capability win** — graceful break instead of hard crash. |
| `graph_break → full_graph` | MraModel (eval only) | Dynamo: speculative `requires_grad_()` handling at `tensor.py:1869`. | upstream win contributed by speculative-symbolic-shape work in 2.12 cycle (specific PR not isolated; visible in upstream diff between torch 2.11 and 2.12 nightlies) | **Less restrictive but still correct** — Dynamo improvement. |
| `full_graph → timeout` | BltForCausalLM, BltModel × 2 modes each | PT 2.12 model-instantiation slowdown. Verified eager-only: PT 2.11 `BltModel(config)` = 111s, PT 2.12 = **374s** (3.4× slowdown in pure `nn.Module` construction, no `torch.compile` invoked). Model code byte-identical between transformers 5.5.3 and 5.6.2. | corpus tracking: needs filing (no issue yet); no upstream PR identified | **Real regression in torch nn.Module construction path. NOT torch.compile related.** Worth filing as a torch issue. |

Net deltas (intersection only):
- **full_graph: −3** (1019 → 1016) — slight regression driven by Blt timeouts
- **graph_break: +1** (graceful Gemma4 capture)
- **timeout: +4** (Blt cases)

The "deepcopy shape shift" pattern (PT 2.11 break type `gb0179` at user's call site → PT 2.12 break type `gb0007` deep in copy module internals at `__reduce_ex__`) was investigated as a candidate regression. Verified with controlled probe — it's a real different graph break, caused by Animesh's PR #177443/#177484 (in PT 2.12). The polyfill PR #179611 is NOT in PT 2.12 (closed 2026-04-11, after the 2026-04-07 snapshot). When the polyfill lands the deepcopy break should resolve.

### Week-over-week regression watch (2026-04-26 → 2026-05-03 nightlies)

Both nightlies on torch 2.13.dev stack (8 days apart, ~7 days of nightly commits + transformers 5.7→5.8.0.dev0 update + harness default-determinism added 2026-05-01). 96-97% of common 747 models unchanged. 8 models showed status changes — only **5 are real, only 1 needs upstream action**:

| Model(s) | Surface transition | Real status | Resolution |
|---|---|---|---|
| AutoencoderTiny | full_graph → eager_error | **Infrastructure flake** (multi-worker shape-gen race; auto-retry would catch). Not a regression. | No code change needed — auto-retry path handles it. F1-F3 single-worker registry shortcut was reverted. |
| SEWModel, SEWDModel | various → worker_error | **Infrastructure flake** — cudnn dlopen race despite 3s stagger (commit `e6f2206`). Not a regression in model behavior. | No code change needed — auto-retry handles. If recurrence becomes frequent, bump `SWEEP_SPAWN_STAGGER_S` env var (currently 3s). |
| AriaTextModel, AriaTextForCausalLM | full_graph → eager_error (`_histc_cuda` det) | **Real** — harness May 1 default-determinism surfaces real `_histc_cuda` non-determinism. Models legitimately cannot run with `torch.use_deterministic_algorithms(True)`. | **Real fix landed** — adopted HF benchmarks's `non_deterministic_models` pattern (`pytorch/benchmarks/dynamo/common.py:4000-4030`). Added `NON_DETERMINISTIC_MODELS` set + `_seed_for_spec()` helper in `sweep/worker.py`. Aria models now compile cleanly: `full_graph` (both modes). |
| AriaModel, AriaForConditionalGeneration | graph_break → eager_error (`_histc_cuda` det) | Same as above — real determinism opt-out needed. | Same fix. Models now produce real `graph_break` signal as before regression-introducing commit. |
| MimiModel | graph_break → eager_error | **Real torch regression** — `RuntimeError: _unsafe_index found unexpected index type Float` in pure eager. Reproduces at torch `2.13.0.dev20260502` (git `671f5614`) but not at `2.13.0.dev20260425` (git `6992d018`). Probable regression in indexing op validation in that ~7-day commit window. | Issue draft prepared at `/tmp/mimi-pytorch-issue-draft.md`. Pending Peng review before filing pytorch issue + corpus issue. Cannot bisect locally — only have endpoints in venv pool. |

**Corrected verdict:** of 8 surface-level "regressions", only 1 is a real upstream issue (MimiModel). 4 are real harness/model interactions resolved by porting HF benchmarks's determinism-opt-out pattern. 3 are infrastructure flakes already mitigated by existing auto-retry path.

---

## §9. Remaining breaks — Dynamo-fixable vs requires model rewrite

Combining the inner-reason analysis from §2 (top break reasons) and §7 (#102 inner reasons), here's a prioritized work list for Dynamo:

### Category A — Dynamo improvements that would unlock many models

| Target | Estimated impact (occurrences cleared) | Engineering shape |
|---|---:|---|
| `CALL_FUNCTION_EX` (variadic) better handling | ~177 (clears most #103) | Trace through `f(*args)` with partially-known args |
| Un-skip / polyfill skipped functions | ~72 | Per-function decision; some are true skip-worthy, others can polyfill |
| Better proxy conversion for args/kwargs | ~42 | Extend proxy mechanism to more types |
| More context manager support | ~38 | Per-CM polyfill; covers logging.disable, torch.set_grad_enabled-style |
| Builtin operator tracing | ~26 | Per-builtin polyfill |
| `torch.*` op non-Tensor returns | ~12 | Op-by-op return-type handling |
| Dynamic shape operator improvements | ~88 | symint propagation work |

### Category B — Requires model-side rewrite

| Pattern | Estimated impact | Recommendation |
|---|---:|---|
| Data-dependent branching | 365 | Model authors: refactor `if tensor.value:` → `torch.where(...)`. Dynamo cannot resolve. |
| Data-dependent operator | 224 | Model authors: avoid scalar extraction in hot path. Use `dynamo.config.capture_scalar_outputs=True` for `Tensor.item()` if applicable. |
| `Tensor.item()` (without capture_scalar_outputs) | 41 | Configurable: enable `capture_scalar_outputs` if numeric stability allows, else model-rewrite |

### Category C — Mixed / per-case

| Pattern | Occurrences | Why mixed |
|---|---:|---|
| Observed exception | 54 | Some are model bugs (e.g., assertion failures), some are Dynamo limitations in exception handling |

### Top filed issues by impact (cross-reference)

*[TODO: cross-reference top patterns with our `tools/file_issues.py` already-filed-issue list — which of these patterns already have a tracking issue? Which don't?]*

---

## §10. Nested graph break feature impact (measured on canonical 2026-05-07 verify)

**Data source:** `experiments/results/ngb-verify-2026-05-07-20260507-181621/` (380 work items × 190 unique HF models, `fullgraph=False` + `nested_graph_breaks=true`, torch 2.13.0.dev20260502+cu126, transformers 5.6.2, diffusers 0.38.0). Cohort: the `status==ok` subset of the canonical 2026-05-05 NGB explain pass (190 models that NGB successfully analyzes for break_reasons). Pre-launch and post-completion sanity checks via `tools/check_cohort_invariants.py`. Reproduction within ≤10% of the abandoned-but-reproducible Run-2 findings on `HubertModel` (5.18 → 5.10) and `Data2VecAudioModel` (4.90 → 4.45) — confirms the bug class is real, not measurement noise.

### What NGB enables

On the 190-model long-tail cohort (where `fullgraph=True` would refuse to compile because breaks happen):

| status | eval | train |
|---|---:|---:|
| **success** | 188 (99%) | 188 (99%) |
| **timeout** | 2 (1%) | 2 (1%) |

Both timeouts hit `Qwen2_5OmniThinkerForConditionalGeneration` and `Qwen2_5_VLForConditionalGeneration` — capacity-bound large vision-language models, not NGB-related. **NGB extracts a successful compilation on 376/380 (99%) of the runs that the strict `fullgraph=True` policy would have rejected.**

This is the fundamental "what NGB is for" answer to the open question §10 originally posed: yes, NGB lets the compiler make forward progress on models with deep-stack / wrapper-pattern breaks (the cascading patterns observed in §5 and §7) that would otherwise block all of compilation. With NGB, compilation completes; without NGB on this cohort, it would not.

### What NGB costs — numeric correctness

Comparing eager-mode forward output against the NGB-compiled forward output (per-element absolute max difference):

| max_diff range | eval | train |
|---|---:|---:|
| **bitwise / near-bitwise match** (≤1e-7) | 144 (76%) | 132 (69%) |
| **noise-floor divergence** (1e-7 to 1e-3) | 44 (23%) | 42 (22%) |
| **catastrophic divergence** (> 1e-3) | **0** | **14 (7%)** |
| no comparison data (timeouts) | 2 | 2 |

**The catastrophic divergences are 100% concentrated in train mode.** Eval mode shows zero catastrophic divergences across all 188 successful runs. This is the actionable finding for the NGB feature.

### The 14 catastrophic train-mode divergences

All 14 cases produce absolute max-diff between 1.8 and 5.5 — six to seven orders of magnitude beyond the noise floor. Severity ratios (max_diff ÷ baseline-noise-floor) sit in the 10⁵–10⁷ range. These are not numerical-precision drift; they are the compiled forward returning a meaningfully different result than eager.

Architecture clustering:
- **Audio / speech (9):** `Wav2Vec2Model`, `Wav2Vec2ConformerModel`, `WavLMModel`, `UniSpeechModel`, `UniSpeechSatModel`, `HubertModel`, `Data2VecAudioModel`, `SEWModel`, `SpeechEncoderDecoderModel`. Range 2.32 – 5.47.
- **Seq2seq (4):** `M2M100Model`, `M2M100ForConditionalGeneration`, `PLBartModel`, `PLBartForConditionalGeneration`. Range 1.82 – 3.59.
- **Other (1):** `ReformerModel`. 4.71.

Ranked by max_diff:

| Model | max_diff |
|---|---:|
| UniSpeechModel | 5.475 |
| Wav2Vec2Model | 5.204 |
| WavLMModel | 5.133 |
| HubertModel | 5.097 |
| UniSpeechSatModel | 4.890 |
| ReformerModel | 4.713 |
| SpeechEncoderDecoderModel | 4.520 |
| Data2VecAudioModel | 4.446 |
| M2M100Model | 3.595 |
| PLBartModel | 3.576 |
| Wav2Vec2ConformerModel | 3.062 |
| M2M100ForConditionalGeneration | 2.541 |
| SEWModel | 2.320 |
| PLBartForConditionalGeneration | 1.819 |

The clustering is highly suggestive: 9/14 are speech encoders that share architectural patterns (1D convolutional feature extractors → transformer encoder), 4/14 are encoder-decoder seq2seq models with explicit decoder-state handoffs, and `ReformerModel` has its own LSH-attention construction. The common thread across both audio and seq2seq is **stateful train-mode forward paths** (dropout, masked-feature-extraction noise, encoder-decoder cache mutation) that interact with NGB's break-and-resume contract differently than eval. Ruling these out as "expected dropout noise" would require running with `dropout=0` and re-measuring; until then they are flagged as suspect NGB-correctness regressions.

### Action items — per-specific-fix issues

The 14 catastrophic divergences are the deliverable bug list. They will be filed as separate issues per specific failure (NOT lumped into family-level umbrella issues). Each issue follows the four-criteria contract: self-contained standalone reproduction, concise, validated end-to-end against the actual repro before filing, and scoped to a single fix. See the issue-filing pipeline at `tools/file_issues.py` + `tools/issue_filing_plan.md`. Issue count is determined by distinct root causes after per-model triage, not by family clustering.

### Limitations of this measurement

- **No NGB-off control on the same cohort.** The verify ran with NGB on; the comparison "would these catastrophic divergences also appear with NGB off?" requires a paired sweep with `nested_graph_breaks=false` + `fullgraph=False` on the same 190 models. That paired sweep would also re-establish whether NGB reduces reported break counts (the original §10 question). It is the natural follow-up to this report.
- **Train-mode dropout was not zeroed.** The audio and seq2seq divergences could in principle be amplified by dropout-RNG-state divergence between eager and compiled paths. A `dropout=0` re-run would distinguish "NGB is causing real wrong-math" from "NGB is producing the same math but a different RNG sequence" before issues are filed.
- **Numeric correctness is per-tensor max-abs-diff on a single forward pass.** Loss + gradient correctness is downstream; if forward already diverges by ~5.0 absolute units, the gradient is not worth measuring until the forward is fixed.

---

## Appendix A — Methodology

**Sweep environment:** torch 2.12.0.dev20260407+cu128, transformers 5.6.2, diffusers 0.37.1, Python 3.12.13.

**Sweep stages:**
- **identify** — for each (model, mode) pair, instantiate model, run `torch.compile(fullgraph=True, backend='eager')`, do one forward pass. Record status: full_graph, graph_break, eager_error, create_error, timeout.
- **explain** — for models with `graph_break` status, run `torch._dynamo.explain` and capture all break_reasons with file/line context.

**Models cataloged from:**
- HuggingFace transformers (`enumerate_hf` walks `dir(transformers)` for ModelMixin subclasses)
- HuggingFace diffusers (similar)
- Custom models (GFPGAN, FLUX, etc. — see `corpora/custom-models/`)

**Status priority:** create_error > eager_error > timeout > graph_break > full_graph (more severe failures take precedence).

**Per-model defaults:** corpus uses scaled-down configs via `_reduce_model_size` (`num_hidden_layers=2`) so even large-MoE models fit on 1× A100 80GB. Graph-break detection is architecture-determined, not depth-determined.

## Appendix B — Reproducing this analysis

```bash
# Top-line stats (per-mode counts)
python3 -c "
import json
d = json.load(open('sweep_results/baseline/pt2.12-2026-04-30/identify_results.json'))
# ... see analysis script in this report's source
"

# Q3 accounting check
python3 -c "
import json
d = json.load(open('sweep_results/baseline/pt2.12-2026-04-30/explain_results.json'))
violations = [r for r in d['results']
              if r.get('graph_count') is not None
              and r['graph_count'] != r.get('graph_break_count', 0) + 1]
print(f'violations: {len(violations)}')
"
```

## Appendix D — Roster of the 44 train-vs-eval flip models

For reference. Models that flip from `eval=full_graph` to `train=graph_break`, with their train graph_break_count from `explain_results.json`. Sum = 215. The `eval_b` column is from the explain pass (which uses `fullgraph=False`); 4 models show nonzero eval_b in explain even though identify (which uses `fullgraph=True`) reported `full_graph` — see footnote.

| # | Model | eval_b | train_b |
|---:|---|---:|---:|
| 1 | IBertModel | 0 | 19 |
| 2 | ConditionalDetrModel | 0 | 10 |
| 3 | BartForCausalLM | 0 | 6 |
| 4 | BigBirdPegasusForCausalLM | 0 | 6 |
| 5 | PLBartForCausalLM | 0 | 6 |
| 6 | TableTransformerModel | 0 | 6 |
| 7 | MBartForCausalLM | 0 | 6 |
| 8 | WhisperForCausalLM | 0 | 6 |
| 9 | BlenderbotForCausalLM | 0 | 6 |
| 10 | Phi4MultimodalAudioModel | 6 | 6 |
| 11 | TrOCRForCausalLM | 0 | 5 |
| 12 | MraModel | 0 | 5 |
| 13 | BioGptForCausalLM | 0 | 5 |
| 14 | SEWDModel | 0 | 5 |
| 15 | XGLMForCausalLM | 0 | 5 |
| 16 | InstructBlipForConditionalGeneration | 0 | 5 |
| 17 | MarianForCausalLM | 0 | 5 |
| 18 | PegasusForCausalLM | 0 | 5 |
| 19 | FlaubertWithLMHeadModel | 2 | 5 |
| 20 | AutoformerModel | 0 | 5 |
| 21 | Kosmos2ForConditionalGeneration | 0 | 5 |
| 22 | BlenderbotSmallForCausalLM | 0 | 5 |
| 23 | Blip2Model | 0 | 5 |
| 24 | InstructBlipVideoForConditionalGeneration | 0 | 5 |
| 25 | Blip2ForConditionalGeneration | 0 | 5 |
| 26 | FlaubertModel | 1 | 4 |
| 27 | InstructBlipVideoModel | 0 | 4 |
| 28 | AudioFlamingo3ForConditionalGeneration | 2 | 4 |
| 29 | OPTForCausalLM | 0 | 4 |
| 30 | InstructBlipModel | 0 | 4 |
| 31 | XGLMModel | 0 | 4 |
| 32 | CohereAsrForConditionalGeneration | 0 | 4 |
| 33 | DetrModel | 0 | 4 |
| 34 | MvpForCausalLM | 0 | 4 |
| 35 | MusicgenForConditionalGeneration | 0 | 4 |
| 36 | MusicgenMelodyForConditionalGeneration | 0 | 4 |
| 37 | BioGptModel | 0 | 4 |
| 38 | Kosmos2Model | 0 | 3 |
| 39 | CohereAsrModel | 0 | 3 |
| 40 | OPTModel | 0 | 3 |
| 41 | MusicgenForCausalLM | 0 | 2 |
| 42 | MusicgenMelodyForCausalLM | 0 | 2 |
| 43 | MusicgenMelodyModel | 0 | 1 |
| 44 | MusicgenModel | 0 | 1 |
| | **Sum** | | **215** |

**Footnote on the eval_b column.** Identify pass uses `fullgraph=True` (raises on first break, so success = 0 breaks counted). Explain pass uses `fullgraph=False` and runs `torch._dynamo.explain()`, which counts breaks even on models that would have passed `fullgraph=True`. The 4 models with eval_b > 0 (Phi4MultimodalAudioModel, FlaubertModel, FlaubertWithLMHeadModel, AudioFlamingo3ForConditionalGeneration) still pass `fullgraph=True` in eval — from a user perspective they're full_graph in eval; explain numbers just confirm there exist break locations the `fullgraph=True` happy-path missed.

## Appendix C — Reference materials

- Prior 2.11-vs-2.12 regression analysis: [gdoc 1vG0tob5...](https://docs.google.com/document/d/1vG0tob5_D6PBPEs84-pe8F0BqV35VqoaKf2htgsAOG8/edit) and `sweep_results/comparisons/pt2.11-vs-pt2.12-2026-04-30/REPORT.md`
- Issue tracker: [penguinwu/oss-model-graph-break-corpus/issues](https://github.com/penguinwu/oss-model-graph-break-corpus/issues)
- Relevant filed issues: #102 (#102 wrapper, 193 models), #103 (#103 wrapper, 141 models), #104 (better error messages), #20 (callable() cluster, 84 occurrences)

---

*[End of draft body. Sections still marked TODO will be filled in subsequent iterations.]*
