# Design Doc: OSS Model Compiler Quality Corpus

**Revision:** 35
**Owner:** Peng Wu
**Date:** 2026-04-20
**Status:** Design Review
**Google Drive:** [OSS Model Compiler Quality Corpus](https://drive.google.com/drive/folders/1r74REnQBKK6ssoF6dS9mcbBrIZ8hrtBd)

---

## 1. Goal

Build a reusable corpus of open-source PyTorch models for measuring and improving `torch.compile` quality. The first application tracks `fullgraph=True` success rates; the infrastructure is designed to extend to other compiler quality studies.

Use cases:
1. **Compiler quality measurement** — track graph break rates, compilation success, and version-over-version trends
2. **Fix validation** — validate graph break fix skills/tools against known breaks (e.g., Arsh's GraphBreak skill)
3. **Compiler hardening** — regression testing as Dynamo improves
4. **Extended diagnostics** — dynamic shape behavior, recompilation patterns, performance analysis

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
1. Explain:       TORCH_LOGS=+graph_breaks + counting backend   → collect ALL graph breaks
2. TORCH_TRACE:   structured compilation artifacts for root cause analysis
```

The explain pass uses `TORCH_LOGS=+graph_breaks` (set before importing torch) combined with a custom counting backend that records subgraph counts and ops per graph. A `_BreakCollector` logging handler captures graph break log messages from `torch._dynamo`. This approach replaced the deprecated `torch._dynamo.explain()` API (removed in PyTorch 2.10). Combined with TORCH_TRACE + tlparse, this gives full tracebacks, FX output graphs, and guard conditions.

**Classification:**
- **Clean** = fullgraph compile succeeds
- **Graph break** = Dynamo raises `Unsupported` or similar
- **Explain error** = model passes eager but `explain()` crashes during tracing (compile-time error — valuable signal for PT2 developers)
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
  --python /path/to/your/venv/bin/python

# Full sweep including TIMM (~1,752 models, ~6 hours)
python run_sweep.py --source all --device cuda --workers 4 \
  --python /path/to/your/venv/bin/python
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

*Note: Section 4 reflects the R8 analysis pass. The corpus has since been updated — see README.md and corpus/summary.md for current numbers (116 models with breaks across 6 configurations).*

| Status | Eval | Train |
|--------|------|-------|
| Clean | **352 (75%)** | **337 (72%)** |
| Graph Break | **92 (20%)** | **107 (23%)** |
| Eager Error | 13 (3%) | 14 (3%) |
| Create Error | 7 (1%) | 6 (1%) |
| Timeout | 4 (1%) | 4 (1%) |

**107 models** have graph breaks in at least one mode. 90 break in eval; 17 additional models break only in train (BioGpt, Blip2, CohereAsr, ConditionalDetr, Detr, Flaubert, IBert, InstructBlip, InstructBlipVideo, Kosmos2, MusicgenMelody, Musicgen, OPT, Phi4MultimodalAudio, SEWD, TableTransformer, XGLM).

### 4.2 Graph Break Taxonomy (109 Models, 10 Root Causes)

| Root Cause | Count | % | Priority | Fix Location |
|------------|-------|---|----------|-------------|
| **Data-dependent branching** | 28 | 25% | LOW | Model code: requires torch.cond() or restructuring |
| **copy.deepcopy()** | 21 | 19% | HIGH | HF model code: replace deepcopy with clone() |
| **Skipped/forbidden callable** | 16 | 15% | HIGH | PyTorch core: support these callables in Dynamo |
| **logging.Logger** | 11 | 10% | MEDIUM | PyTorch core: skip/inline logger calls in Dynamo |
| **as_proxy() missing** | 11 | 10% | HIGH | PyTorch core: implement as_proxy() for failing types |
| **Unbacked symbols** | 10 | 9% | MEDIUM | Model code: shapes generated dynamically from data |
| **Observed exception (try/except)** | 3 | 3% | MEDIUM | Dynamo exception handler support |
| **Non-Tensor return** | 3 | 3% | MEDIUM | Dynamo op return type support |
| **requires_grad()** | 2 | 2% | MEDIUM | Dynamo mutation support |
| **Other** | 4 | 4% | VARIES | Per-model investigation needed |

#### All-Breaks Deep Dive (Pass 2 — explain)

Pass 1 counts one break per model (first encountered). Pass 2 uses `TORCH_LOGS=+graph_breaks` with a counting backend to enumerate **all** graph breaks across 107 broken models (214 model×mode pairs). Total: **1,254 graph breaks**.

4 models (4 model×mode pairs in eval) produce **explain_error** — eager passes but explain crashes during Dynamo tracing. 2 models (AutoformerModel, InformerModel) succeed in train mode with the new methodology despite failing in eval:

| Model | Mode | Error |
|-------|------|-------|
| AutoformerModel | eval | FakeTensor symbolic shape bug in FFT autocorrelation (`irfft` + `view` shape divergence) |
| InformerModel | eval | Conv downsampling (`distil=True`) shrinks hidden_states but mask size stays fixed |
| MambaModel | eval | `mark_static_address()` forbidden callable during cache initialization |
| FalconMambaModel | eval | `mark_static_address()` forbidden callable during cache initialization |

Additionally, 2 model×mode pairs (RwkvModel eval + train) timeout during explain.

| Root Cause | Breaks | % | Models | Description |
|------------|--------|---|--------|-------------|
| **Data-dependent branching** | 237 | 36.5% | 56 | Control flow depends on tensor values (`if tensor.sum() > 0`, `_local_scalar_dense`) |
| **as_proxy() missing** | 82 | 12.6% | 10 | Dynamo can't convert arg types to proxy (DETR models pass ValueError/bool) |
| **Dynamic shape operator** | 77 | 11.9% | 16 | Op output shape depends on data (`aten.nonzero`, `repeat_interleave`) |
| **copy.deepcopy()** | 76 | 11.7% | 25 | Encoder-decoder models clone layers with `copy.deepcopy()` |
| **Tensor.item()** | 48 | 7.4% | 7 | Tensor→scalar conversion breaks tracing |
| **Skipped function call** | 48 | 7.4% | 15 | Dynamo-marked not-traceable (audio `importlib.util.find_spec`) |
| **Unsupported op/step** | 30 | 4.6% | 11 | Bytecode pattern Dynamo hasn't implemented (Aria/Glm4v vision) |
| **Tensor requires_grad mutation** | 18 | 2.8% | 12 | In-place `requires_grad_()` mutation |
| **Unsupported method/builtin** | 15 | 2.3% | 4 | `ContiguousFormat.get()`, RNG `.seed()`, context manager `lock` |
| **logging.Logger** | 15 | 2.3% | 6 | Logger calls during tracing |
| **Non-Tensor return** | 2 | 0.3% | 1 | torch ops returning ints/tuples Dynamo can't trace |

**Key insight:** Data-dependent branching dominates at the break level (36%) even more than at the model level (25%). A single data-dependent model (e.g., EncodecModel with 29 eval breaks) generates many breaks because each branch point creates a separate graph break. By contrast, `copy.deepcopy()` affects more models (25) but generates fewer total breaks (77) because each model typically has just 2–4 deepcopy calls.

### 4.3 Fix Impact Analysis

The top 3 actionable categories cover **45% of all broken models** (49/109):

| Fix | Owner | Models Fixed | Effort |
|-----|-------|-------------|--------|
| Replace `copy.deepcopy()` with `clone()` in encoder-decoder models | HF Transformers | 21 | Low — single PR |
| Un-skip audio feature extractor callables | PyTorch Dynamo | 12 | Medium |
| Support `Logger` methods in Dynamo | PyTorch Dynamo | 11 | Medium |
| Implement `as_proxy()` for detection/vision output types | PyTorch Dynamo | 11 | Medium |

**Inherently hard (35%):** Data-dependent branching (28 models) and unbacked symbols (10 models) require `torch.cond()`, model restructuring, or shape redesign. No quick fix.

**Detailed breakdown by category:**

**1. Data-dependent branching — 28 models (26%)**
AriaTextModel, BioGptModel, Blip2Model†, CohereAsrModel†, ConditionalDetrModel, DetrModel, EncodecModel, FastSpeech2ConformerModel, FlaubertModel, GraniteMoeHybridModel, IBertModel, InstructBlipModel†, InstructBlipVideoModel†, JambaModel, Kosmos2Model†, LongcatFlashModel, NemotronHModel, OPTModel, OlmoHybridModel, Qwen3NextModel, Qwen3_5Model, Qwen3_5MoeModel, Qwen3_5MoeTextModel, Qwen3_5TextModel, ReformerModel, TableTransformerModel, ViltModel, XGLMModel. (†train-only)

**2. copy.deepcopy() — 21 models (19%)**
BartModel, BigBirdPegasusModel, BlenderbotModel, BlenderbotSmallModel, FSMTModel, M2M100Model, MBartModel, MT5Model, MarianModel, MoonshineModel, MoonshineStreamingModel, MvpModel, PLBartModel, PegasusModel, PegasusXModel, ProphetNetModel, Speech2TextModel, T5Model, TimeSeriesTransformerModel, UMT5Model, WhisperModel. All encoder-decoder models using `copy.deepcopy()` to clone decoder layers.

**3. Skipped/forbidden callable — 16 models (15%)**
Audio feature extractors (12): Data2VecAudioModel, HubertModel, SEWModel, SpeechT5Model, UniSpeechModel, UniSpeechSatModel, VitsModel, Wav2Vec2BertModel, Wav2Vec2ConformerModel, Wav2Vec2Model, WavLMModel, XcodecModel. SSM (2): FalconMambaModel, MambaModel. Other (2): RecurrentGemmaModel, SpeechEncoderDecoderModel.

**4. logging.Logger — 11 models (10%)**
BambaModel, FalconH1Model, Florence2Model, GotOcr2Model, InternVLModel, LEDModel, LongformerModel, PaliGemmaModel, RwkvModel, SeamlessM4TModel, SeamlessM4Tv2Model.

**5. as_proxy() missing — 11 models (10%)**
DFineModel, DeformableDetrModel, GroundingDinoModel, LwDetrModel, MMGroundingDinoModel, PPDocLayoutV2Model, PPDocLayoutV3Model, RTDetrModel, RTDetrV2Model, Siglip2Model, Siglip2VisionModel.

**6. Unbacked symbols — 10 models (9%)**
DbrxModel, FunnelModel, Glm4vMoeVisionModel, Glm4vVisionModel, GlmImageVisionModel, GlmOcrVisionModel, MimiModel, PaddleOCRVisionModel, Phi4MultimodalAudioModel†, VideoLlama3VisionModel. (†train-only)

**7–10. Other categories (12 models, 11%)**
Observed exception: LongT5Model, SwitchTransformersModel, UdopModel. Non-Tensor return: NllbMoeModel, PeAudioModel. requires_grad(): MraModel, SEWDModel. Explain error: AutoformerModel, InformerModel (eval), MambaModel (eval), FalconMambaModel (eval). Other: AriaModel, MusicgenModel†, MusicgenMelodyModel†. (†train-only)

### 4.4 Remaining Gaps

| Gap | Count | Description |
|-----|-------|-------------|
| **Breaks without reasons** | 78 | 6.2% of 1,254 breaks have no diagnostic reason from TORCH_LOGS (coverage gap) |
| **Truncated sub-causes** | 702 | CALL/CALL_FUNCTION_EX entries truncated at 300 chars — sub-cause unclassifiable |
| **Explain errors** | 4 | AutoformerModel, InformerModel, MambaModel, FalconMambaModel crash during Dynamo tracing (eval only, all 3 versions) |
| **Explain timeout** | 1 | RwkvModel (new in v2.10) |
| **Eager errors** | 13 | Structurally untestable (see below) |
| **Create errors** | 7 | Missing deps (natten, detectron2, flash_attn), unresolvable config, or instantiation failure |
| **Timeouts** | 4 | EdgeTam (×2), ZambaModel, xLSTMModel |
| **tlparse traces** | — | Trace hosting not yet set up; issues lack tlparse links for interactive debugging |
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
| Clean | **352 (75%)** | **335 (72%)** | **339 (72%)** |
| Graph Break | **92 (20%)** | **97 (21%)** | **90 (19%)** |
| Eager Error | 13 (3%) | 18 (4%) | 18 (4%) |
| Create Error | 7 (1%) | 18 (4%) | 18 (4%) |
| Timeout | 4 (1%) | 0 | 3 (1%) |

**Key finding: `dynamic=mark` is stricter than `dynamic=true`.** Mark produces 335 clean vs true's 339 — counterintuitively, making all dims symbolic is more permissive than marking specific dims. This is because `mark_dynamic` enforces that marked dims must NOT be specialized, while `dynamic=true` allows the compiler to specialize freely and just replays with symbolic shapes.

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

Additional mark degradation: 12 new create errors (Swin variants), 5 new eager errors (Gemma3n/Zamba2).

#### dynamic=true details

Only **1 new graph break** vs static: UnivNetModel (randn with symbolic shapes — same failure as mark). 3 models (Sam2HieraDetModel, Sam2Model, Sam2VisionModel) that were clean statically now timeout under symbolic shape overhead.

No static graph breaks were eliminated by either dynamic mode.

### 4.6 Version Trend (PyTorch 2.8 → 2.9 → 2.10)

Same 468 models tested with identical sweep code, transformers 5.4.0, diffusers 0.37.1, Python 3.12.13. Only variable: PyTorch version.

#### Eval Mode

| Status | v2.8 | v2.9 | v2.10 |
|--------|------|------|-------|
| **Clean** | **298 (63.7%)** | **324 (69.2%)** | **337 (72.0%)** |
| **Graph Break** | **96 (20.5%)** | **101 (21.6%)** | **90 (19.2%)** |
| Eager Error | 47 (10.0%) | 16 (3.4%) | 17 (3.6%) |
| Create Error | 27 (5.8%) | 27 (5.8%) | 18 (3.8%) |
| Worker Error | 0 | 0 | 6 (1.3%) |

#### Train Mode

| Status | v2.8 | v2.9 | v2.10 |
|--------|------|------|-------|
| **Clean** | **288 (61.5%)** | **314 (67.1%)** | **323 (69.0%)** |
| **Graph Break** | **106 (22.6%)** | **110 (23.5%)** | **105 (22.4%)** |
| Eager Error | 47 (10.0%) | 17 (3.6%) | 17 (3.6%) |
| Create Error | 27 (5.8%) | 27 (5.8%) | 18 (3.8%) |
| Worker Error | 0 | 0 | 5 (1.1%) |

**Note:** v2.10 has 6 eval / 5 train worker_error models due to cuDNN library loading failures (infra issue, not compiler regression). All 3 versions use identical sweep code and library versions (transformers 5.4.0, diffusers 0.37.1, Python 3.12.13).

#### Key Findings

1. **12 graph breaks fixed in v2.10** — BltModel, FlaubertModel, HiggsAudioV2Model, Idefics2Model, Phi4MultimodalAudioModel, Phi4MultimodalVisionModel, PixtralVisionModel, Sam3Model, TapasModel, VJEPA2Model, XLMModel, XmodModel. All were graph_break in both v2.8 and v2.9, fixed only in v2.10.
2. **0 new graph breaks introduced** — no clean→graph_break regressions across two major PyTorch releases.
3. **Graph break count increased in v2.9 (96→101)** — not regressions. 5 models moved from eager_error→graph_break (became testable, revealing pre-existing breaks) and 1 from create_error→graph_break. No clean→graph_break transitions in any version.
4. **Massive eager error cleanup** — 47→16→17 (eval). Models that were untestable in v2.8 due to PyTorch/library compatibility issues became testable as PyTorch improved. 26 models moved from eager_error→clean between v2.8 and v2.10.
5. **Create error reduction** — 27→27→18. Config/constructor compatibility improved significantly in v2.10.
6. **Worker errors in v2.10** — 6 eval models hit cuDNN library loading failures (infra issue, not compiler regression). These models were clean in v2.8/v2.9.

#### Net Movement (Eval, v2.8 → v2.10)

| Transition | Count |
|-----------|-------|
| eager_error → clean | 26 |
| graph_break → clean | 12 |
| create_error → clean | 7 |
| clean → worker_error | 6 |
| eager_error → graph_break | 5 |
| create_error → graph_break | 1 |
| create_error → eager_error | 1 |

#### Total Graph Break Counts (Pass 2 — explain)

Beyond counting *models* with graph breaks, we track the *total number of graph breaks* across all models using `TORCH_LOGS=+graph_breaks` with a counting backend. This matters because a version might have the same number of broken models but fewer breaks per model (or vice versa).

| Metric | v2.8 | v2.9 | v2.10 |
|--------|------|------|-------|
| Testable models | 394 | 425 | 427 |
| Total graph breaks (eval) | **487** | **506** | **467** |
| Explain OK | 103 | 108 | 102 |
| Explain error | 5 | 4 | 4 |
| Explain timeout | 0 | 0 | 2 |
| Avg breaks per broken model | 4.7 | 4.7 | 4.6 |

| Metric | v2.8 | v2.9 | v2.10 |
|--------|------|------|-------|
| Testable models | 394 | 424 | 428 |
| Total graph breaks (train) | **790** | **786** | **787** |
| Explain OK | 106 | 110 | 106 |
| Explain error | 2 | 2 | 0 |
| Explain timeout | 0 | 0 | 0 |
| Avg breaks per broken model | 7.5 | 7.1 | 7.4 |

**Normalization note:** The headline reduction (487→467 eval breaks) is partly confounded by changing denominators — more models became testable in later versions (394→427), while some broken models moved to clean. To isolate compiler improvement from denominator changes, we use an apples-to-apples comparison.

#### Apples-to-Apples Comparison

84 models were broken in ALL 3 versions (eval mode). Comparing total break counts on this stable set:

| Metric (84 models, eval) | v2.8 | v2.9 | v2.10 |
|--------------------------|------|------|-------|
| Total graph breaks | **419** | **427** | **438** |
| Avg breaks per model | 5.0 | 5.1 | 5.2 |

*Note: 91 models have `ok` explain status across all 3 versions; the 84-model set from the prior analysis round is retained here for continuity. See `tools/analyze_trend.py` for updated normalized comparisons.*

94 models were broken in ALL 3 versions (train mode):

| Metric (94 models, train) | v2.8 | v2.9 | v2.10 |
|--------------------------|------|------|-------|
| Total graph breaks | **713** | **702** | **716** |
| Avg breaks per model | 7.6 | 7.5 | 7.6 |

**Key insight:** For persistently-broken models, break counts are slightly *increasing* (eval: 419→438, +4.5%), driven by stricter soundness checks in v2.10 (e.g., `aten._local_scalar_dense` data-dependent checks in MoE routing). This is more correct behavior — the compiler is catching real issues it previously missed — but it means the headline total break reduction (487→467) understates the improvement from fixing whole models. The 12 models fixed in v2.10 eliminated ~68 breaks, but stricter checks added ~19 breaks to persistently-broken models, for a net reduction of ~20.

**Reproducible:** `python tools/analyze_trend.py` auto-discovers version directories and generates normalized comparisons. Use `--train` for train mode, `--json` for machine-readable output, `--details` for per-model transition tracking.

**Bottom line:** PyTorch Dynamo is steadily improving. Clean compile rate grew from 64% to 72% across two releases, driven primarily by 12 graph break fixes and improved model compatibility. Zero regressions. Break counts within persistently-broken models are slightly increasing due to stricter soundness checks — this is more correct, not worse. Improvement comes from fixing whole models, not reducing individual breaks.

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

## 6. GitHub Issue Tracking

GitHub Issues on this repo serve as the interface between corpus findings and actionable work items for the Dynamo team, HuggingFace maintainers, and corpus contributors.

### 6.1 Label Taxonomy

Issues use `for:*` audience labels so teams can quickly filter for their items:

| Label | Audience | Description |
|-------|----------|-------------|
| `for:dynamo-team` | PyTorch Dynamo team | Compiler-side fixes (frame skip, FakeTensor, tracing) |
| `for:hf-transformers` | HuggingFace team | Model library fixes (deepcopy, config, annotations) |
| `for:corpus-tooling` | Corpus maintainers | Pipeline/sweep infrastructure improvements |

Additional labels for type (`bug`, `feature-request`, `enhancement`, `question`), priority (`high-impact`, `good first issue`), and domain (`dynamo`, `data-quality`).

**Filtering:** `label:for:dynamo-team` shows only Dynamo-actionable issues.

### 6.2 Initial Issues (v2.10 corpus)

| # | Title | Audience | Breaks |
|---|-------|----------|--------|
| 1 | copy.deepcopy() → clone() | `for:hf-transformers` | 90 |
| 2 | Frame skip in generic.py + output_capturing.py | `for:dynamo-team` | 226 |
| 3 | 300-char truncation in break reasons | `for:corpus-tooling` | 97 unclassifiable |
| 4 | Dynamo compile crashes on 4 models | `for:dynamo-team` | 4 models blocked |
| 5 | import_utils.py + configuration_utils.py hotspot | `for:hf-transformers` | 851 (68% of all) |

All issues currently lack tlparse trace links (hosting not yet configured).

### 6.3 Issue Monitor

`tools/github_issue_monitor.py` polls the GitHub API for new issues and comments, alerting via GChat. Runs as a cron job every 30 minutes during business hours (weekdays 9am–7pm ET).

This enables a feedback loop: Dynamo engineers or users can file issues against the corpus, and the corpus maintainer is notified automatically.

## 7. Future Work

| Item | Description | Priority |
|------|-------------|----------|
| **Diffusers expansion** | ~110 models with per-family constructor args and inputs | High |
| **~~Multi-version comparison~~** | ~~Run corpus across PyTorch 2.8, 2.9, 2.10~~ | ✅ Done (Section 4.6) |
| **aot_eager sweep** | Catch AOTAutograd-specific training failures | Medium |
| **Gradient checkpointing mode** | Test `model.gradient_checkpointing_enable()` | Medium |
| **Autocast mode** | Test `torch.autocast('cuda', dtype=torch.float16)` | Medium |
| **torchaudio, ultralytics** | Additional model sources (~25 models) | Low |
| **OSS long-tail discovery** | Crawl GitHub for wild PyTorch modules (separate project) | Deferred |

### 7.1 Graph Break Fix Study (V2 Use Case)

**Goal:** Enable engineers to study graph breaks in pip-installed HuggingFace models, develop workarounds in user code, and validate fixes — without modifying the library source.

**Challenge:** Graph breaks occur inside library code (e.g., `copy.deepcopy()` in transformers encoder-decoder models). Fixing the library is one path, but users also want to know: *can I work around this break by changing how I call the model?*

**Workflow:**
1. **Select a broken model** from the corpus (e.g., `T5Model` — breaks on `copy.deepcopy()`)
2. **Reproduce the break** using `worker.py` with TORCH_TRACE enabled
3. **Study the break** using tlparse output — identify the exact op and location
4. **Develop a workaround** — e.g., monkey-patch the forward method, wrap with `torch._dynamo.allow_in_graph`, or restructure the calling code
5. **Validate** — re-run the model with the workaround applied, confirm graph break is eliminated

**What needs to be built (V2):**
- Workaround templates for common break categories (deepcopy, data-dependent, skipped callable)
- A `--workaround` flag in worker.py that applies known patches before compile
- Documentation of which workarounds are user-side vs require library PRs

**Priority:** After initial corpus + version trend data is complete.

## 8. Phase 3: Correctness Testing

**Goal.** Surface compiler-introduced numerical errors by comparing eager vs compiled forward outputs across the corpus. Every divergence is a signal worth filing — not a pass/fail certification.

**Scope (MVP).** HF Transformers (~352 fullgraph_ok models from PT 2.10), eval mode, fp32, `backend="eager"`. Diffusers, train mode, `backend="inductor"`, dtype matrix → V2.

### 8.1 Methodology

For each model: run eager forward, reset Dynamo state, compile + run with the same input and seed. Compare outputs with HF's recursive walk over `ModelOutput`/dict/tuple/list, applying field-type rules:

| Field type | Action |
|------------|--------|
| Floating-point tensor | `torch.testing.assert_close(rtol, atol)` |
| Integer/bool tensor | Skip (argmax-derived; tiny upstream diffs flip results) |
| 0-dim tensor (loss) | Skip in forward; compare in V2 backward-grad pass |
| `None` | Skip |
| `Cache` objects (`DynamicCache`, `HybridCache`) | Skip |
| `tuple`/`list`/`dict`/`ModelOutput` | Recurse |

NaN handling: skip if both `nan`; strip `nan`/`-inf` before max-diff.

**Tolerance table** (adopted verbatim from HF `test_modeling_common.py:194-222`):

| dtype | atol | rtol |
|-------|------|------|
| fp32 (CUDA) | 1e-6 | 1e-4 |
| fp16 (CUDA) | 5e-3 | 5e-3 |
| bf16 (CUDA) | 1e-2 | 1e-2 (SDPA: 3e-2) |

MVP runs fp32 only; dtype matrix is V2.

**Determinism helpers** (from `transformers`): `set_seed(42)`, `set_config_for_less_flaky_test(config)`, `set_model_for_less_flaky_test(model)`.

### 8.2 Failure Classification (for triage, not grading)

| Class | Definition |
|-------|------------|
| `match` | Within `(atol, rtol)` |
| `divergence` | Past `(atol, rtol)`. Severity = continuous `max_diff / atol` ratio — sort triage queue by ratio, do **not** pre-bucket into "small" vs "large." |
| `nan_inf_introduced` | Compiled output has NaN/Inf, eager doesn't |
| `shape_mismatch` | Compiled output shape ≠ eager |
| `dtype_mismatch` | Compiled output dtype ≠ eager |

**One `divergence` class, not two.** Splitting into `numerical_drift` vs `divergence` at an arbitrary threshold (e.g., 10× vs 100× tolerance) silently re-introduces grading — implying "drift" is acceptable. Continuous severity preserves the full distribution and lets bimodality (if it exists) emerge from the data, not from our prior. Every divergence is a signal worth filing; severity decides which to file first, never which to accept.

### 8.3 Execution

- **When:** After identify pass (need known-clean candidate set)
- **Subset:** Models with `eval.fullgraph_ok=True` in corpus.json (~352 of 460 for HF v2.10)
- **Worker:** `run_correctness(spec, device, mode)` in `worker.py`, dispatched via `--pass-num 4`
- **Wall budget:** Same 180s default; large-model tier inherits from issue #57
- **Storage:** New `correctness/` directory at repo root, schema mirrors `corpus.json`
- **Diff dump:** Default off; opt-in via `--dump-diffs` flag (sizing: ~100 MB single-mode, accumulates across modes/backends/PT versions)

### 8.4 Why Not Pre-existing `run_validate`?

`run_validate` does a different test: dynamic-shape correctness (eager at shape A vs compiled-dynamic at shape B). Phase 3 is the simpler base case: same shape, eager vs compiled, does the answer match? Both are needed.

### 8.5 Smoke Validation (2026-04-20)

5/5 known-clean models matched on PT 2.11.0:

| Model | Status | max_diff | Compared fields | Skipped |
|-------|--------|---------:|-----------------|---------|
| BertModel | match | 0.0 | last_hidden_state, pooler_output | — |
| GPT2Model | match | 0.0 | last_hidden_state | Cache |
| ViTModel | match | 0.0 | pooler_output, last_hidden_state | — |
| ResNetModel | match | 0.0 | last_hidden_state, pooler_output | — |
| T5GemmaModel | match | 1.43e-6 | encoder_last_hidden_state, last_hidden_state, decoder_hidden_states[0] | Cache |

Full design lives in `design/phase3-correctness.md`.

## 9. Open Questions

1. **Diffusers constructor args:** Can we auto-discover minimal constructor args, or need per-model configs?
2. **TorchBench overlap:** Does the team already have fullgraph pass/fail data from CI?
3. **Multimodal models:** 44 HF composite models skipped — some (Llava, CLIP) are high-usage. Worth adding?

## 10. Prior Art

- **Repo:** https://github.com/jansel/pytorch-jit-paritybench
- **Dashboard:** [pytorch/pytorch#93667](https://github.com/pytorch/pytorch/issues/93667)
- **Umbrella task:** [pytorch/pytorch#92670](https://github.com/pytorch/pytorch/issues/92670)
- **Core IR Opset Analysis:** [Google Doc](https://docs.google.com/document/d/1XR73gknq3gAh6nHuG-jDUjS1_vwjhY6zatn-Tx4zz2c)
- **Op frequency spreadsheet:** [Google Sheet](https://docs.google.com/spreadsheets/d/1sEt0HD-0YAF5lfdOUPPZd2xIvwPL0emE7GaiqgMaTSM)

## 11. Artifacts

| File | Location |
|------|----------|
| Sweep orchestrator | `sweep/run_sweep.py` |
| Sweep worker | `sweep/worker.py` |
| Model enumeration | `sweep/models.py` |
| Corpus (JSON) | `corpus/corpus.json` |
| Graph break analysis | `analysis/graph-break-analysis.md` |
| GitHub issue monitor | `tools/github_issue_monitor.py` |
| Design doc | `design/design-doc.md` |
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
| 29 | 2026-04-02 | Data gap fill + sweep code fixes. Merged dynamic_true into corpus. Added root_cause to all graph_break entries. Backfilled error text for 57 static entries. Fixed Gemma3nModel (graph_break→create_error, 93→92 eval). Resolved 6 mark worker_errors to clean (329→335). Fixed worker.py exception handler (bare except→type checking) and run_sweep.py subprocess command leak. |
| 30 | 2026-04-02 | **Explain deep dive + version trend.** Added all-breaks taxonomy from Pass 2 explain (1,247 breaks, 11 categories). Added Section 4.6: version trend across PyTorch 2.8→2.9→2.10 (12 graph breaks fixed, 0 regressions, clean rate 64%→75%). Added Section 6.1: Graph Break Fix Study use case. Reverted 3 fixes that masked compile-time signal (AutoformerModel, InformerModel, Mamba); added explain_error classification. |
| 31 | 2026-04-02 | **Normalized break trend + analysis script.** Added total graph break counts from explain across all 3 versions. Added apples-to-apples comparison on persistently-broken models. Added `tools/analyze_trend.py` for reproducible version comparison. |
| 32 | 2026-04-03 | **Clean re-run data.** Re-ran all 3 versions (identify + explain) with reverted worker.py to eliminate data contamination. Updated all trend tables with clean data. Key changes: v2.10 eval graph_break 92→90, 6 worker_error models (cuDNN infra). A2A comparison now shows slight break *increase* (419→438) from stricter soundness checks — more correct, not worse. Added `tools/update_corpus.py` for automated corpus merging with changelog generation. |
| 33 | 2026-04-04–05 | **Explain methodology migration.** Replaced deprecated `torch._dynamo.explain()` with `TORCH_LOGS=+graph_breaks` + counting backend (explain API removed in PyTorch 2.10). Re-swept all 3 versions. Key changes: total breaks 1,247→1,254 (v2.10), AutoformerModel/InformerModel train now succeed (were explain_error), RwkvModel now times out. Added explain timeout tracking. Updated Section 3.1 methodology and Section 4.6 trend tables. |
| 34 | 2026-04-05 | **GitHub issue tracking.** Created 4 issues for top graph break findings (deepcopy, frame skip, truncation, compile crashes). Added label taxonomy with `for:*` audience labels for Dynamo team, HF Transformers, and corpus tooling. Added `tools/github_issue_monitor.py` for automated new-issue/comment alerting. Expanded Section 4.4 remaining gaps (breaks without reasons, truncated sub-causes, explain errors, tlparse traces). Added Section 6. |
| 35 | 2026-04-20 | **Phase 3 — correctness testing infra.** Added Section 8 (eager vs compiled forward output comparison). Adopted HF prior art: recursive `ModelOutput` walker with field-type taxonomy (skip int/bool/0-dim/Cache), HF tolerance table verbatim (fp32: atol=1e-6 rtol=1e-4), `set_seed` + less-flaky-test helpers. 5-class failure taxonomy: `match`, `divergence` (continuous severity = max_diff/atol — no arbitrary drift threshold), `nan_inf_introduced`, `shape_mismatch`, `dtype_mismatch`. MVP: HF eval fp32 backend=eager (~352 models). V2: train, dtype matrix, Diffusers, Inductor. Implemented `run_correctness` in `worker.py` (dispatch via `--pass-num 4`). Smoke validated on 5 known-clean models (Bert, GPT2, ViT, ResNet, T5Gemma — all match, max_diff ≤ 1.43e-6). Full Phase 3 spec at `design/phase3-correctness.md`. |
