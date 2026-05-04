# 2026-05-03 Nightly Sweep Report

**Baseline:** 2026-04-26 nightly (Sunday) — `torch=2.13.0.dev20260425+cu126`, `transformers=5.6.2`, `diffusers=0.37.1`
**Current:** 2026-05-03 nightly (Sunday) — `torch=2.13.0.dev20260502+cu126`, `transformers=5.8.0.dev0`, `diffusers=0.38.0`
**Window:** 7 days of torch nightly head + transformers 5.6→5.8 + diffusers 0.37→0.38

**Amendment applied:** `2026-05-04T14-59Z-aria-sew-autotiny-fix` (commit `588b659`) — folds in 2026-05-04 morning's harness fixes for AutoencoderTiny + SEW family + Aria family. Original sweep `results[]` is byte-identical to last night's data; amendment data is in `identify_results.json`'s new `amendments[]` array. All consumer tools route through `sweep/results_loader.py` which merges amendments transparently.

---

## Part 1 — Apple-to-apple: Common models only (1494 work items)

Comparison restricted to (name, mode) pairs present in BOTH nightlies. New models added in 2026-05-03 are excluded here and analyzed separately in Part 2.

### Status counts on common models

| Status | 04-26 eval | 05-03 eval | Δ eval | 04-26 train | 05-03 train | Δ train |
|---|---|---|---|---|---|---|
| full_graph | 581 | 595 | **+14** | 509 | 520 | **+11** |
| graph_break | 137 | 134 | −3 | 205 | 207 | +2 |
| eager_error | 13 | 14 | +1 | 13 | 14 | +1 |
| worker_error | 2 | 0 | **−2** | 2 | 0 | **−2** |
| timeout | 4 | 5 | +1 | 4 | 5 | +1 |
| create_error | 20 | 20 | 0 | 20 | 20 | 0 |
| skipped | 0 | 0 | 0 | 4 | 4 | 0 |

**Net fullgraph delta on common models: +25 work items.** All seven worker_errors from baseline are gone (one was a real cudnn race that the harness fix addressed; the rest matured into clean eager paths).

### Improvements (transitions to better state)

37 work items moved to better state on common models:

- **graph_break → full_graph: 15 work items** — compiler now traces these without breaks
  - Siglip2VisionModel (eval), Siglip2Model (train)
  - PPDocLayoutV2Model (eval), SmolVLMForConditionalGeneration (eval)
  - LwDetrModel (eval), RTDetrV2Model (eval), GroundingDinoModel (eval), DFineModel (eval)
  - ... and 7 more
- **eager_error → graph_break: 8 work items** — Qwen3VL/Qwen3.5 vision encoders now load + run eager
  - Qwen3_5VisionModel, Qwen3_5MoeVisionModel, Qwen3VLVisionModel, Qwen3VLMoeVisionModel (both modes each)
- **eager_error → full_graph: 6 work items**
  - Qwen2_5OmniToken2WavDiTModel, MllamaVisionModel, Qwen2_5OmniToken2WavBigVGANModel (both modes each)
- **worker_error → graph_break: 4 work items** — GPT-SoVITS variants no longer crash workers
- **timeout → full_graph: 4 work items** — Blt models now finish + compile clean
  - BltModel, BltForCausalLM (both modes each)

### GB shape changes (graph_break in BOTH nightlies, count differs)

14 models had `graph_break` in both nightlies but with different break counts. **Net delta: −24 graph breaks** (29 fewer minus 5 more).

**Reductions (improvements):**

| Model | Mode | 04-26 GBs | 05-03 GBs | Δ |
|---|---|---|---|---|
| PPDocLayoutV3Model | train | 21 | 11 | **−10** |
| RTDetrModel | train | 14 | 8 | **−6** |
| PPDocLayoutV2Model | train | 14 | 8 | **−6** |
| TimeSeriesTransformerModel | train | 11 | 9 | −2 |
| GraniteMoeHybridForCausalLM | eval | 10 | 9 | −1 |
| GraniteMoeHybridForCausalLM | train | 10 | 9 | −1 |
| GraniteMoeHybridModel | eval | 9 | 8 | −1 |
| GraniteMoeHybridModel | train | 9 | 8 | −1 |
| LongcatFlashForCausalLM | eval | 13 | 12 | −1 |

PPDocLayoutV3 train −10 is the standout — likely a single hot break-pattern got fixed upstream. Worth investigating which `break_reason` no longer appears.

**Increases:**

| Model | Mode | 04-26 GBs | 05-03 GBs | Δ |
|---|---|---|---|---|
| LongcatFlashModel | eval | 11 | 12 | +1 |
| M2M100ForConditionalGeneration | train | 7 | 8 | +1 |
| SeamlessM4TModel | train | 5 | 6 | +1 |
| LongcatFlashModel | train | 11 | 12 | +1 |
| LongcatFlashForCausalLM | train | 12 | 13 | +1 |

All +1, no large new break clusters.

### Regressions

**Zero real upstream torch regressions** in this nightly post-amendment.

The 2 surface-level MimiModel regressions (eval + train, `graph_break → eager_error` with `_unsafe_index found unexpected index type Float`) were initially diagnosed as a torch nightly regression in the 04-25 → 05-02 window, but verification on 2026-05-04 across torch 2.8 / 2.9 / 2.10 / 2.11 / 2.12-dev / 2.13-dev showed the bug **reproduces unchanged on every version**. It is a long-standing PyTorch decomp bug in `_replication_pad` (filed as [pytorch/pytorch#182339](https://github.com/pytorch/pytorch/issues/182339)) exposed only because we enabled `torch.use_deterministic_algorithms(True)` by default in commit `816a203` on 2026-05-01. MimiModel is the only model in the corpus that hits the trigger condition (`nn.functional.pad(mode='replicate')` with a tensor padding amount); EncodecModel uses identical `_pad1d` code but `mode='reflect'` and is unaffected. Added to `sweep/known_errors.json` (eager_error, all observed versions) — will be skipped on future sweeps until pytorch/pytorch#182339 lands.

| Model | Mode | 04-26 | 05-03 | Status |
|---|---|---|---|---|
| MimiModel | eval | graph_break | eager_error | known_errors (PT decomp bug, pytorch#182339) |
| MimiModel | train | graph_break | eager_error | known_errors (PT decomp bug, pytorch#182339) |

The other 8 surface-level regressions detected in the raw sweep (AutoencoderTiny, SEW family, Aria family) were resolved in code by 2026-05-04 commits and folded in via the amendment. Their `result_source` field reads `amended:2026-05-04T14-59Z-aria-sew-autotiny-fix`.

---

## Part 2 — Detailed analysis: NEW models (in 05-03 only)

230 new work items across **115 new models** added between sweeps — driven by `transformers 5.6.2 → 5.8.0.dev0` and `diffusers 0.37.1 → 0.38.0`.

### Status distribution on new models

| Status | eval | train |
|---|---|---|
| full_graph | 55 | 54 |
| graph_break | 7 | 8 |
| eager_error | 33 | 32 |
| timeout | 9 | 9 |
| create_error | 11 | 12 |

**109 of 230 work items (47%) compile cleanly out of the box** — strong landing rate for diffusers / transformers additions.

### A. Full_graph new models (109 work items, 55 distinct models)

Grouped by family — these are "free wins" the corpus picked up:

**Transformer{N}D family (19 models):** AllegroTransformer3DModel, AuraFlowTransformer2DModel, BriaTransformer2DModel, ChromaTransformer2DModel, CogVideoXTransformer3DModel, CogView3PlusTransformer2DModel, CogView4Transformer2DModel, ConsisIDTransformer3DModel, and 11 more

**AutoencoderKL family (15 models):** AsymmetricAutoencoderKL, AutoencoderKLAllegro, AutoencoderKLCogVideoX, AutoencoderKLCosmos, AutoencoderKLFlux2, AutoencoderKLHunyuanVideo, AutoencoderKLKVAE, AutoencoderKLKVAEVideo, and 7 more

**Other Autoencoders (3):** AutoencoderOobleck, AutoencoderRAE, AutoencoderVidTok

**DeepseekV4 (2):** DeepseekV4ForCausalLM, DeepseekV4Model

**UNet (2):** UNetControlNetXSModel, UNetMotionModel

**Single-model standouts:** AudioLDM2UNet2DConditionModel, GraniteSpeechPlusForConditionalGeneration, LagunaForCausalLM, LagunaModel, LongCatAudioDiTVae, LuminaNextDiT2DModel, MotionAdapter, PriorTransformer, UNet1DModel, UNet3DConditionModel, VQModel, etc.

The diffusers transformer/autoencoder ecosystem dominates the wins.

### B. Graph breaks in new models

**15 work items in `graph_break` status, 164 total breaks. Mean: 10.9 GBs per work item.**

Top new models by GB count:

| Model | Mode | GBs |
|---|---|---|
| AutoencoderKLHunyuanImageRefiner | eval | 26 |
| AutoencoderKLHunyuanImageRefiner | train | 26 |
| HiDreamImageTransformer2DModel | eval | 19 |
| HiDreamImageTransformer2DModel | train | 19 |
| ConsistencyDecoderVAE | train | 13 |
| AutoencoderDC | eval | 12 |
| AutoencoderDC | train | 12 |
| ConsistencyDecoderVAE | eval | 10 |
| PPFormulaNetForConditionalGeneration | train | 7 |
| MiniCPMV4_6ForConditionalGeneration | eval | 4 |

The two AutoencoderKLHunyuanImageRefiner work items contribute 52/164 (32%) of all new-model GBs alone — a single fix on this model could move the needle substantially.

### C. Top GB reasons across new models

```
   79× Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`)
   20× Operator `aten.bincount.default`'s output shape depends on input Tensor data.   [NEW]
   17× Operator `aten._local_scalar_dense.default` has a non-Tensor output...
    8× Dynamo does not know how to trace builtin operator `reversed`...
    5× Dynamo has no bytecode reconstruction implemented for sourceless DictItemsIterator
    5× Dynamo has no bytecode reconstruction implemented for sourceless DictIterator
    3× Operator `aten.nonzero.default`'s output shape depends on input Tensor data
    2× Dynamo does not support tracing `Tensor.item()` with config.capture_scalar_outputs=False
    2× Dynamo does not know how to iterate over `UserDefinedObjectVariable(reversed)`
```

Most patterns are well-known Dynamo limitations (data-dep branching → #54/#77/#78; `_local_scalar_dense` → #55; `Tensor.item()` → #56; DictIterator → #96; reversed → no existing issue). New models are exhibiting the same break-pattern catalogue as the baseline corpus, not introducing fundamentally new failure modes.

### D. New GB reasons surfaced (not seen in baseline)

**Exactly ONE explanation cluster appears in new models that wasn't in the 04-26 baseline:**

```
   20× Operator `aten.bincount.default`'s output shape depends on input Tensor data.
```

This is a data-dependent output shape op (same family as `aten.nonzero` / `aten.repeat_interleave` already covered by issue #18). The `bincount` op is structurally identical: output shape depends on input data. **Not a new class of break — fits cleanly under issue #18.** Worth adding `bincount` to that issue's covered ops list, or filing a follow-up if there's a path to fix it independently.

All other 8 distinct explanations seen in new models also appear in baseline — the new models are exercising existing break patterns, not introducing surprises.

---

## Amendment metadata (provenance)

```
amendment_id:       2026-05-04T14-59Z-aria-sew-autotiny-fix
applied_at:         2026-05-04T15:03:40+00:00
fix_commit:         588b659
fix_description:    HF non_deterministic_models pattern + cuDNN warm-with-retry + Option 1 first-pool gating
trigger:            post-nightly regression triage 2026-05-04 morning
env_constraints:    torch=2.13.0.dev20260502+cu126, transformers=5.8.0.dev0, diffusers=0.38.0
rows:               14 (all 7 Aria/SEW/AutoencoderTiny models × 2 modes)
```

Original `results[]` array (1724 rows) is byte-identical to what the sweep wrote at 2026-05-03T22:01:59. Amendment was authored 2026-05-04 morning per `amend_sweep.py` workflow (commit `588b659`, dedup guard added in `8955184`).

---

## Open follow-ups

- **MimiModel pytorch issue:** filed as [pytorch/pytorch#182339](https://github.com/pytorch/pytorch/issues/182339) on 2026-05-04. Added to `sweep/known_errors.json`. Re-verify on torch 2.14+.
- **Issue #110:** Switch identify_results.json to JSONL for true append-only amendments — scheduled for tomorrow
- **PPDocLayoutV3Model train −10 GB reduction:** worth investigating which break_reason got fixed (could close out a corpus issue)
- **Issue #18 update:** add `aten.bincount.default` to covered ops (20 occurrences in new models)

## Corpus dynamo issues — status check

Cross-checked named-model issues against the 04-26 → 05-03 transition. **No named models in any open issue changed status.** Specifically checked: #14 (OpenVoice), #16 (MiniCPMV), #17 (Gemma3/4), #76 (Gemma3n), #98 (EncodecModel) — all still in the same status as when the issues were filed.

Wrapper-pattern issues (#102, #103) and pattern-aggregate issues (#77, #78) need verification — issue #79 was queued exactly for this purpose. Following up on #79 with a comment containing the 05-03 baseline numbers + regex-extraction notes for #78.
