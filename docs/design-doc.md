# Design Doc: OSS Model Graph Break Corpus

**Revision:** 28
**Owner:** Peng Wu
**Date:** 2026-04-01
**Status:** Design Review
**Google Drive:** [OSS Model Graph Break Corpus](https://drive.google.com/drive/folders/1r74REnQBKK6ssoF6dS9mcbBrIZ8hrtBd)

---

## 1. Goal

Build a curated corpus of open-source PyTorch models that exhibit **graph break problems** under `torch.compile(fullgraph=True)`, to enable:
1. **Graph break skill analysis** — categorize and prioritize the root causes
2. **Compiler hardening** — regression testing as Dynamo improves
3. **Progress tracking** — measure graph break elimination across PyTorch versions

## 2. Scope

### Model Sources

| Source | Models | Graph Break Rate | Status |
|--------|--------|-----------------|--------|
| **HF Transformers** | 463 | 20% (93 eval) | Primary focus — richest signal |
| **HF Diffusers** | 5 (configured) | 0% | Expansion planned (~110 total) |
| **TIMM** | 1,284 | 0.2% (3 models, all RNN) | Excluded from default sweep |

**Why HF Transformers is the focus:**
1. Highest graph break density (20% vs 0.2% for TIMM)
2. Covers transformers, MoE, SSMs, RNNs, CNNs, hybrids, multimodal, audio, vision, detection
3. Industry standard — graph breaks here directly impact real users
4. Uniform `PreTrainedModel` + `Config()` API makes automated sweeps straightforward

TIMM is excluded from the default sweep: 73% of sweep time but only 3 graph breaks (all RNN wrapping, one root cause). Available via `--source all` for comprehensive runs.

### PyTorch Version

All results use **PyTorch 2.10.0 stable** from pytorch.org on an A100 GPU (80GB, sm_80).

## 3. Methodology

### 3.1 Two-Pass Compile Test

Each model is tested under `torch.compile(fullgraph=True, backend="eager")` in both eval and train modes. The sweep uses a two-pass approach:

**Pass 1 — Fast Identification (all models):**
```
1. Eager test:    output = model(inputs)                                          → must pass (else skip)
2. Fullgraph:     torch.compile(model, fullgraph=True, backend="eager")(inputs)   → pass/fail
```

`fullgraph=True` fails immediately on the first graph break — broken models fail in <0.1s, clean models take 3-10s. This makes Pass 1 fast even on the full corpus.

**Pass 2 — Detailed Analysis (broken models only):**
```
1. Explain:       explanation = torch._dynamo.explain(model)(*inputs)   → collect ALL graph breaks
2. TORCH_TRACE:   structured compilation artifacts for root cause analysis
```

`explain()` reports every break location and reason (not just the first). Combined with TORCH_TRACE + tlparse, this gives full tracebacks, FX output graphs, and guard conditions.

**Classification:**
- **Clean** = fullgraph compile succeeds
- **Graph break** = Dynamo raises `Unsupported` or similar
- **Eager error** = model fails in eager mode (input/config issue, not a compile problem)
- **Create error** = model instantiation fails
- **Timeout** = exceeds time limit

### 3.2 Input Generation

Models are instantiated locally using `Config()` objects — no network access or pretrained weights required. Model size is reduced to 2 layers and hidden_size ≤ 1024 via `_reduce_model_size()`, since graph break behavior depends on architecture (ops in forward()), not depth.

Input generation detects model modality (text, vision, audio, seq2seq, multimodal, video, time_series) and generates matching tensors. Batch size = 2 (PyTorch specializes on 0 and 1, so ≥2 is required for valid dynamic shape testing).

### 3.3 Test Modes

Each model is tested in 2 modes:

| Mode | Setup | Why |
|------|-------|-----|
| **Eval** | `model.eval()` | Inference path — baseline |
| **Train** | `model.train()` | Training path — can expose different graph breaks via dropout, batch norm |

### 3.4 Backend Choice

`backend="eager"` tests Dynamo tracing only. Benchmarked 1.3–2.4x faster than `aot_eager` with identical graph break detection. `inductor` is 5.5x slower with no change in graph break detection. v2 may add `aot_eager` for AOTAutograd-specific failures.

### 3.5 Execution

Single A100 GPU, 4 parallel worker subprocesses, 180s timeout per model. Each model runs in complete isolation (separate subprocess). The orchestrator uses checkpoint+resume for crash resilience.

```bash
# Default sweep: HF + Diffusers (~468 models, ~1 hour)
python run_sweep.py --source hf+diffusers --device cuda --workers 4 \
  --python ~/envs/graph-break-corpus/bin/python

# Full sweep including TIMM (~1,752 models, ~6 hours)
python run_sweep.py --source all --device cuda --workers 4 \
  --python ~/envs/graph-break-corpus/bin/python
```

### 3.6 Dynamic Shape Testing

Two modes test Dynamo's handling of symbolic shapes:

| Mode | How | What it catches |
|------|-----|----------------|
| `--dynamic true` | All tensor dims symbolic | Everything — most aggressive |
| `--dynamic mark` | Only batch (dim 0) + seq_len (dim 1 for NLP) symbolic | What users actually encounter |

Static graph breaks persist under dynamic shapes. Dynamic testing reveals **constraint violation** errors where models specialize on dimensions we marked as dynamic — a distinct failure mode from static graph breaks.

## 4. Results (R8 — 468 HF+Diffusers Models, PyTorch 2.10.0)

### 4.1 Corpus Summary

| Status | Eval | Train |
|--------|------|-------|
| Clean | **352 (75%)** | **337 (72%)** |
| Graph Break | **93 (20%)** | **107 (23%)** |
| Eager Error | 13 (3%) | 14 (3%) |
| Create Error | 6 (1%) | 6 (1%) |
| Timeout | 4 (1%) | 4 (1%) |

**110 models** have graph breaks in at least one mode. 93 break in eval; 17 additional models break only in train (BioGpt, Blip2, CohereAsr, ConditionalDetr, Detr, Flaubert, IBert, InstructBlip, InstructBlipVideo, Kosmos2, MusicgenMelody, Musicgen, OPT, Phi4MultimodalAudio, SEWD, TableTransformer, XGLM).

### 4.2 Graph Break Taxonomy (110 Models, 10 Root Causes)

| Root Cause | Count | % | Priority | Fix Location |
|------------|-------|---|----------|-------------|
| **Data-dependent branching** | 28 | 25% | LOW | Model code: requires torch.cond() or restructuring |
| **copy.deepcopy()** | 21 | 19% | HIGH | HF model code: replace deepcopy with clone() |
| **Skipped/forbidden callable** | 16 | 15% | HIGH | PyTorch core: support these callables in Dynamo |
| **logging.Logger** | 12 | 11% | MEDIUM | PyTorch core: skip/inline logger calls in Dynamo |
| **as_proxy() missing** | 11 | 10% | HIGH | PyTorch core: implement as_proxy() for failing types |
| **Unbacked symbols** | 10 | 9% | MEDIUM | Model code: shapes generated dynamically from data |
| **Observed exception (try/except)** | 3 | 3% | MEDIUM | Dynamo exception handler support |
| **Non-Tensor return** | 3 | 3% | MEDIUM | Dynamo op return type support |
| **requires_grad()** | 2 | 2% | MEDIUM | Dynamo mutation support |
| **Other** | 4 | 4% | VARIES | Per-model investigation needed |

### 4.3 Fix Impact Analysis

The top 3 actionable categories cover **45% of all broken models** (49/110):

| Fix | Owner | Models Fixed | Effort |
|-----|-------|-------------|--------|
| Replace `copy.deepcopy()` with `clone()` in encoder-decoder models | HF Transformers | 21 | Low — single PR |
| Un-skip audio feature extractor callables | PyTorch Dynamo | 12 | Medium |
| Support `Logger` methods in Dynamo | PyTorch Dynamo | 12 | Medium |
| Implement `as_proxy()` for detection/vision output types | PyTorch Dynamo | 11 | Medium |

**Inherently hard (35%):** Data-dependent branching (28 models) and unbacked symbols (10 models) require `torch.cond()`, model restructuring, or shape redesign. No quick fix.

**Detailed breakdown by category:**

**1. Data-dependent branching — 28 models (25%)**
AriaTextModel, BioGptModel, Blip2Model†, CohereAsrModel†, ConditionalDetrModel, DetrModel, EncodecModel, FastSpeech2ConformerModel, FlaubertModel, GraniteMoeHybridModel, IBertModel, InstructBlipModel†, InstructBlipVideoModel†, JambaModel, Kosmos2Model†, LongcatFlashModel, NemotronHModel, OPTModel, OlmoHybridModel, Qwen3NextModel, Qwen3_5Model, Qwen3_5MoeModel, Qwen3_5MoeTextModel, Qwen3_5TextModel, ReformerModel, TableTransformerModel, ViltModel, XGLMModel. (†train-only)

**2. copy.deepcopy() — 21 models (19%)**
BartModel, BigBirdPegasusModel, BlenderbotModel, BlenderbotSmallModel, FSMTModel, M2M100Model, MBartModel, MT5Model, MarianModel, MoonshineModel, MoonshineStreamingModel, MvpModel, PLBartModel, PegasusModel, PegasusXModel, ProphetNetModel, Speech2TextModel, T5Model, TimeSeriesTransformerModel, UMT5Model, WhisperModel. All encoder-decoder models using `copy.deepcopy()` to clone decoder layers.

**3. Skipped/forbidden callable — 16 models (15%)**
Audio feature extractors (12): Data2VecAudioModel, HubertModel, SEWModel, SpeechT5Model, UniSpeechModel, UniSpeechSatModel, VitsModel, Wav2Vec2BertModel, Wav2Vec2ConformerModel, Wav2Vec2Model, WavLMModel, XcodecModel. SSM (2): FalconMambaModel, MambaModel. Other (2): RecurrentGemmaModel, SpeechEncoderDecoderModel.

**4. logging.Logger — 12 models (11%)**
BambaModel, FalconH1Model, Florence2Model, Gemma3nModel, GotOcr2Model, InternVLModel, LEDModel, LongformerModel, PaliGemmaModel, RwkvModel, SeamlessM4TModel, SeamlessM4Tv2Model.

**5. as_proxy() missing — 11 models (10%)**
DFineModel, DeformableDetrModel, GroundingDinoModel, LwDetrModel, MMGroundingDinoModel, PPDocLayoutV2Model, PPDocLayoutV3Model, RTDetrModel, RTDetrV2Model, Siglip2Model, Siglip2VisionModel.

**6. Unbacked symbols — 10 models (9%)**
DbrxModel, FunnelModel, Glm4vMoeVisionModel, Glm4vVisionModel, GlmImageVisionModel, GlmOcrVisionModel, MimiModel, PaddleOCRVisionModel, Phi4MultimodalAudioModel†, VideoLlama3VisionModel. (†train-only)

**7–10. Other categories (12 models, 11%)**
Observed exception: LongT5Model, SwitchTransformersModel, UdopModel. Non-Tensor return: InformerModel, NllbMoeModel, PeAudioModel. requires_grad(): MraModel, SEWDModel. Other: AriaModel, AutoformerModel, MusicgenModel†, MusicgenMelodyModel†. (†train-only)

### 4.4 Remaining Gaps

| Gap | Count | Description |
|-----|-------|-------------|
| **Eager errors** | 13 | Structurally untestable (see below) |
| **Create errors** | 6 | Missing deps (natten, detectron2, flash_attn) or unresolvable config |
| **Timeouts** | 4 | EdgeTam (×2), ZambaModel, xLSTMModel |
| **TIMM** | excluded | 1,284 models, 99% clean, 3 graph breaks — available via `--source all` |

**13 Structurally Untestable Models:**

| Model | Root Cause |
|-------|-----------|
| BarkModel | `forward()` is a generation pipeline, not a standard forward pass |
| BayesianDetectorModel | `forward()` signature incompatible with expected types |
| DeepseekV2Model | MoE routing creates non-contiguous tensors (stride alignment) |
| DiaModel | CUDA device-side assert in DAC codec |
| EdgeTamVideoModel | Requires `EdgeTamInferenceSession` for temporal state |
| PI0Model | Multi-stage vision-language-action model with custom input format |
| PeAudioVideoModel | Custom audio-video fusion requiring synchronized inputs |
| Pix2StructVisionModel | CUDA assert in patch embedding (needs preprocessor) |
| Qwen2_5OmniToken2WavModel | Hardcoded `batch_size=1` in forward() |
| RagModel | Requires DPR retriever with FAISS index |
| Sam2VideoModel | Requires `Sam2InferenceSession` for temporal propagation |
| Sam3TrackerVideoModel | Requires tracking session for multi-object video |
| Sam3VideoModel | Requires inference session for video segmentation |

### 4.5 Dynamic Shape Results

#### Three-mode comparison (eval)

| Status | Static | Mark | True |
|--------|--------|------|------|
| Clean | **352 (75%)** | **329 (70%)** | **339 (72%)** |
| Graph Break | **93 (20%)** | **97 (21%)** | **90 (19%)** |
| Eager Error | 13 (3%) | 18 (4%) | 18 (4%) |
| Create Error | 6 (1%) | 18 (4%) | 18 (4%) |
| Timeout | 4 (1%) | 0 | 3 (1%) |
| Worker Error | 0 | 6 (1%) | 0 |

**Key finding: `dynamic=mark` is stricter than `dynamic=true`.** Mark produces 329 clean vs true's 339 — counterintuitively, making all dims symbolic is more permissive than marking specific dims. This is because `mark_dynamic` enforces that marked dims must NOT be specialized, while `dynamic=true` allows the compiler to specialize freely and just replays with symbolic shapes.

#### dynamic=mark details

Using `mark_dynamic()` on batch (dim 0) and seq_len (dim 1 for NLP models), **8 models that compile cleanly with static shapes break due to constraint violations:**

| Model | Violated Dimension |
|-------|-------------------|
| BrosModel | attention_mask dim 1 vs input_ids dim 1 |
| CanineModel | attention_mask dim 1 vs input_ids dim 1 |
| CpmAntModel | input_ids dim 1 specialized to constant |
| DecisionTransformerModel | attention_mask dim 1 specialized to constant |
| IdeficsModel | input_ids dim 1 specialized to constant |
| PixtralVisionModel | pixel_values dim 0 specialized to constant |
| TapasModel | input_ids dim 0 specialized to constant |
| UnivNetModel | randn with symbolic shapes |

These models internally assume fixed sequence length or batch size — a real issue users would hit with `mark_dynamic`. All 8 compile cleanly under `dynamic=true`, confirming the constraint violations are mark-specific.

Additional mark degradation: 12 new create errors (Swin variants), 5 new eager errors (Gemma3n/Zamba2), 6 worker errors (Gemma family — OOM under dynamic overhead).

#### dynamic=true details

Only **1 new graph break** vs static: UnivNetModel (randn with symbolic shapes — same failure as mark). 3 models (Sam2HieraDetModel, Sam2Model, Sam2VisionModel) that were clean statically now timeout under symbolic shape overhead.

No static graph breaks were eliminated by either dynamic mode.

## 5. Repository Guide

The GitHub repo at `~/projects/oss-model-graph-break-corpus/` serves two audiences:

### 5.1 For Corpus Consumers (Studying Compile Behavior)

The corpus is a JSON file (`corpus/corpus.json`) with per-model compile results. Each model record contains:

```json
{
  "name": "T5Model",
  "source": "hf",
  "has_graph_break": true,
  "eval": {"status": "graph_break", "fullgraph_ok": false, "compile_time_s": 2.1, ...},
  "train": {"status": "graph_break", "fullgraph_ok": false, "compile_time_s": 3.4, ...}
}
```

**To reproduce a single model's graph break:**

```bash
cd sweep/
python worker.py --model hf/T5Model --device cuda
```

**To study a specific category (e.g., all copy.deepcopy models):**

```bash
python -c "
import json
with open('corpus/corpus.json') as f:
    corpus = json.load(f)
for m in corpus['models']:
    if m.get('has_graph_break'):
        print(f\"{m['name']}: {m['eval'].get('fullgraph_error', '')[:80]}\")
"
```

**Beyond graph breaks:** The corpus can also be used to study:
- **Compile time** — `compile_time_s` per model for performance analysis
- **Dynamic shape behavior** — run with `--dynamic true` or `--dynamic mark`
- **Recompilation** — compile with one shape, run with another
- **Backend comparison** — change `backend="eager"` to `"inductor"` or `"aot_eager"`

### 5.2 For Corpus Builders (Running and Updating Sweeps)

**Running a full sweep:**

```bash
python sweep/run_sweep.py --source hf+diffusers --device cuda --workers 4 \
  --python /path/to/python
```

**Key flags:**
- `--source hf+diffusers|all` — which model zoos to sweep
- `--workers N` — parallel worker count (4 recommended for A100)
- `--dynamic true|mark` — enable dynamic shape testing
- `--timeout N` — per-model timeout in seconds
- `--models file.json` — sweep a specific subset of models

**Updating the corpus after fixes:**

1. Fix the model in `sweep/worker.py` (input generation, config patches, etc.)
2. Re-sweep the affected models: `--models /path/to/retry_models.json`
3. Merge results into `corpus/corpus.json`

**Where to add fixes:**
- **Input generation:** `_generate_inputs()` in `worker.py` — model-specific input overrides
- **Config patches:** `_fix_config()` in `worker.py` — fix missing/invalid config values
- **Config creation:** `_create_config()` in `worker.py` — composite models needing factory methods
- **Size reduction:** `_reduce_model_size()` in `worker.py` — cap layers, hidden dims, MoE experts

## 6. Future Work

| Item | Description | Priority |
|------|-------------|----------|
| **Diffusers expansion** | ~110 models with per-family constructor args and inputs | High |
| **Multi-version comparison** | Run corpus across PyTorch 2.8, 2.9, 2.10 to track Dynamo progress | Medium |
| **aot_eager sweep** | Catch AOTAutograd-specific training failures | Medium |
| **Gradient checkpointing mode** | Test `model.gradient_checkpointing_enable()` | Medium |
| **Autocast mode** | Test `torch.autocast('cuda', dtype=torch.float16)` | Medium |
| **torchaudio, ultralytics** | Additional model sources (~25 models) | Low |
| **OSS long-tail discovery** | Crawl GitHub for wild PyTorch modules (separate project) | Deferred |

## 7. Open Questions

1. **Diffusers constructor args:** Can we auto-discover minimal constructor args, or need per-model configs?
2. **TorchBench overlap:** Does the team already have fullgraph pass/fail data from CI?
3. **Multimodal models:** 44 HF composite models skipped — some (Llava, CLIP) are high-usage. Worth adding?

## 8. Prior Art

- **Repo:** https://github.com/jansel/pytorch-jit-paritybench
- **Dashboard:** [pytorch/pytorch#93667](https://github.com/pytorch/pytorch/issues/93667)
- **Umbrella task:** [pytorch/pytorch#92670](https://github.com/pytorch/pytorch/issues/92670)
- **Core IR Opset Analysis:** [Google Doc](https://docs.google.com/document/d/1XR73gknq3gAh6nHuG-jDUjS1_vwjhY6zatn-Tx4zz2c)
- **Op frequency spreadsheet:** [Google Sheet](https://docs.google.com/spreadsheets/d/1sEt0HD-0YAF5lfdOUPPZd2xIvwPL0emE7GaiqgMaTSM)

## 9. Artifacts

| File | Location |
|------|----------|
| Sweep orchestrator | `sweep/run_sweep.py` |
| Sweep worker | `sweep/worker.py` |
| Model enumeration | `sweep/models.py` |
| Corpus (JSON) | `corpus/corpus.json` |
| Graph break analysis | `docs/graph-break-analysis.md` |
| Design doc | `docs/design-doc.md` |
| Google Drive folder | [Link](https://drive.google.com/drive/folders/1r74REnQBKK6ssoF6dS9mcbBrIZ8hrtBd) |

---

## Appendix A: Prototype Sweep (PyTorch 2.8.0a0)

Early sweep on prototype methodology, included here for historical context.

### TIMM Sweep (1,121 vision models)

| Result | Count | % |
|--------|-------|---|
| Pass both configs | 946 | 84.4% |
| Default compile fails | 172 | 15.3% |
| **Graph breaks** | **3** | **0.3%** |

Graph break models: `sequencer2d_{s,m,l}` — all fail because Dynamo can't trace RNN/GRU/LSTM.

### HF Transformers Sweep (277 model classes)

| Result | Count | % |
|--------|-------|---|
| Pass both configs | 118 | 42.6% |
| Default compile fails | 91 | 32.9% |
| Create failures | 44 | 15.9% |
| **Graph breaks** | **24** | **8.7%** |

### Prototype Graph Break Taxonomy (27 models)

| Root Cause | Count | Models |
|------------|-------|--------|
| Logger in compile mode | 8 | BigBird, BigBirdPegasus, GraniteMoe, GraniteMoeShared, Longformer, MT5, T5, UMT5 |
| Dynamic shape operators | 5 | Dbrx, Mixtral, Olmoe, Qwen2Moe, Qwen3Moe |
| Data-dependent branching | 4 | Cohere2, Gemma2, Gemma3Text, Reformer |
| RNN/GRU/LSTM wrapping | 3 | sequencer2d_{s,m,l} |
| Tensor.item() | 2 | Flaubert, XLM |
| Other | 5 | Phimoe, MRA, FlavaText, GPTSanJapanese, NllbMoe |

### Prototype Limitations

- TIMM: 137 models failed due to hardcoded 224×224 inputs (fixed: use `model.default_cfg['input_size']`)
- HF: 67 models failed due to wrong modality inputs (fixed: modality detection)
- These setup errors inflated the "compile fails" category — resolved in production methodology

## Appendix B: Sweep History (R1–R8)

Eight rounds of sweeps, each improving input generation and error recovery:

| Round | Config | Clean | Graph Break | Eager Err | Create Err | Timeout | Key Changes |
|-------|--------|-------|------------|-----------|------------|---------|-------------|
| R1 | bs=1, all sources (1,752) | 1,433 (82%) | 40 (2%) | 104 (6%) | 31 (2%) | 143 (8%) | Initial sweep |
| R2 | bs=1, dynamic=True (1,433) | 1,305 (91%) | 0 | 5 (<1%) | — | 123 (9%) | Selection bias: only clean models |
| R3 | bs=2, HF+diffusers (468) | 220 (47%) | 56 (12%) | 72 (15%) | 36 (8%) | 84 (18%) | Fixed batch specialization |
| R4 | R3 + fix validation (468) | 236 (50%) | 61 (13%) | 58 (12%) | 33 (7%) | 80 (17%) | 19 error models recovered |
| R5 | R4 + retry sweep (468) | 261 (56%) | 78 (17%) | 105 (22%) | 22 (5%) | 2 (<1%) | Model size reduction, 30+ input fixes |
| R6 | R5 + config creation (468) | 296 (63%) | 69 (15%) | 95 (20%) | 6 (1%) | 2 (<1%) | `_create_config()`, MoE guards |
| R7 | R6 + retry confirm (468) | 296 (63%) | 69 (15%) | 95 (20%) | 6 (1%) | 2 (<1%) | Confirmed R6 via retry_v3 |
| **R8** | **R7 + mass fixes (468)** | **352 (75%)** | **93 (20%)** | **13 (3%)** | **6 (1%)** | **4 (1%)** | **82 eager_error models fixed** |

### R3→R4: Fix Validation (19 models recovered)

Retested 108 error models with improved input generation: +16 clean, +3 graph breaks (new signal), −1 create_error shift.

### R4→R5: Model Size Reduction + Input Fixes (net +25 clean, +17 graph break)

- **Model size reduction:** Generic `_reduce_model_size()` caps all models at 2 layers, hidden_size ≤ 1024, MoE experts ≤ 4. Graph break behavior depends on architecture (ops), not depth.
- **Config fixes:** `_fix_config()` patches 20+ models' broken default configs (None values, missing fields).
- **Input fixes:** 30+ model-specific input fixes (time series, multimodal image tokens, audio, vision).
- **Image token retry:** Catches "tokens: X, features: Y" errors and retries with correct count.

### R5→R6: Config Creation + Size Guards (net +35 clean, −9 graph break)

- **`_create_config()`:** Composite models (EncoderDecoder, Rag, Musicgen, etc.) whose config constructors fail before patching.
- **Head divisibility guard:** Ensures `hidden_size % num_attention_heads == 0` after reduction.
- **MoE topk guard:** Ensures `num_experts_per_tok ≤ num_local_experts` after reduction.
- 9 graph breaks eliminated — were caused by misconfigured internals, not real graph break patterns.

### R7→R8: Mass Error Model Fixes (82 of 95 eager_error models fixed)

Six iterative fix-and-sweep passes:
1. **mrope rescaling removed** — generic rescaler incorrectly double-scaled Qwen2VL sections
2. **VL text-only expansion** — 20+ vision-language models now use text-only inputs to avoid None image iteration
3. **Vision input fixes** — Siglip2 pre-patchified, Glm 2D patches, Blip2/InstructBlip image tokens, Kosmos2 position mask, PaddleOCR, VideoLlama3, etc.
4. **Audio/multimodal fixes** — CohereAsr mel spectrogram, Phi4MultimodalAudio, KyutaiSpeechToText 3D input_ids
5. **Config/architecture fixes** — Glm4v post-reduction mrope, DepthPro sub-config sync, Llama4Vision pixel_shuffle_ratio, Aria patch dict, Idefics perceiver embeddings, LwDetr, UVDoc, TvpModel, TimesFm

## Appendix C: Sweep Configuration Rationale

**Why bs=2 (not bs=1 or bs=3):** PyTorch specializes on dimension values 0 and 1, treating them as constants. Any batch ≥ 2 avoids this. bs=2 gives same graph break detection as bs=3 with 33% less GPU memory.

**Why `backend="eager"` (not `aot_eager` or `inductor`):** `eager` tests Dynamo tracing only. `aot_eager` is 1.3–2.4x slower with identical graph break detection (catches additional AOTAutograd failures, deferred to v2). `inductor` is 5.5x slower with no change in graph break detection.

**Why eval-only Pass 1:** Graph breaks in train mode are almost always a superset of eval. Running eval-only in Pass 1 identifies the broken set with minimal compute; Pass 2 tests both modes.

**Why 4 workers (not 16):** Default is 16 but 4 workers on A100 with reduced models (~few hundred MB each) provides safe headroom. Orchestrator checks nvidia-smi before spawning; if >80% memory used, halves worker count.

**Dynamic shapes add one new failure mode:** "Constraints violated" errors where models specialize on dimensions marked as dynamic. This is distinct from static graph break root causes. The `dynamic=mark` sweep found 8 models that compile cleanly with static shapes but break when batch/seq_len dims are dynamic (see Section 4.5). All other static graph break root causes (deepcopy, data-dependent, skipped callable, etc.) persist unchanged under dynamic shapes.

## Appendix D: Backend Benchmark

Benchmarked on PyTorch 2.8.0a0 (CPU, first-compile cost):

| Model | eager eval | aot_eager eval | Slowdown |
|-------|-----------|---------------|----------|
| resnet50 | 3.31s | 8.10s | 2.4x |
| vit_base_patch16_224 | 3.19s | 4.63s | 1.5x |
| efficientnet_b0 | 3.60s | 5.73s | 1.6x |

`inductor` on resnet50 eval: 38.2s (5.5x slower than eager).

## Appendix E: Revision Log

| Rev | Date | Changes |
|-----|------|---------|
| 1–15 | 2026-03-26–27 | Initial design, methodology iterations, GPU baseline, two-pass approach |
| 16 | 2026-03-29 | Production sweep (1,752 models, PyTorch 2.10.0). 1,433 clean, 40 graph breaks |
| 17 | 2026-03-29 | Per-phase observability (create/eager/compile timing) |
| 18 | 2026-03-29 | Pass 2 results + explain_error category |
| 19 | 2026-03-29 | Dynamic shape methodology (three-tier approach) |
| 20 | 2026-03-29 | Batch=1 specialization fix + two-shape validation |
| 21 | 2026-03-30 | TIMM exclusion, batch size analysis, input fixes |
| 22 | 2026-03-30 | Eval+train merge, GitHub repo creation |
| 23 | 2026-03-31 | R5: model size reduction, config fixes, image token retry |
| 24 | 2026-03-31 | R6/R7: config creation, size reduction guards |
| 25 | 2026-04-01 | R8: 82 eager_error models fixed, taxonomy updated to 110 models |
| 26 | 2026-04-01 | **Major restructure.** Methodology before results. Prototype/TIMM/round details → appendix. Project B parked. Added repository guide for two audiences (builders vs consumers). Addressed Peng's review comments. |
| 27 | 2026-04-01 | Dynamic=mark sweep results. 329 clean (70%, down from 75% static). 8 new constraint-violation graph breaks. Updated Section 4.5, Appendix C. |
| 28 | 2026-04-02 | Dynamic=true sweep results. Key finding: mark is stricter than true (329 vs 339 clean). Three-mode comparison table. Both dynamic sweeps complete. |
