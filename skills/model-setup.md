# Model Setup Skill

How we configure models for graph break testing.
This captures the design decisions and accumulated learning about model instantiation,
input generation, and configuration — so any agent can set up new models correctly.

---

## 1. Design Principles

### Why batch_size = 2
We use `batch_size=2`, never 0 or 1. Reason: PyTorch's compiler can specialize on batch_size=1,
producing code paths that don't exist at realistic batch sizes. batch_size=2 is the minimum that
avoids this specialization while keeping memory/compute low.

### Why static shapes (default)
Static shapes are the default for all sweeps. Dynamic shapes (`torch._dynamo.mark_dynamic()`)
are a separate, explicitly requested test. Static shapes produce more stable, reproducible
results and represent the simpler compilation path.

### Why 2 layers is enough
Graph break behavior is determined by **model architecture** (which ops are used), not depth.
A model with 2 transformer layers uses the same ops as one with 96 layers. We reduce all
models to 2 layers via `_reduce_model_size()` to keep creation time and memory reasonable.

### Why we test eval AND train modes
Some graph breaks only appear in training (backward pass, gradient computation, dropout).
We test both to catch the full picture.

---

## 2. Config Generation

### Standard models (single config)
Most HF models: `config = ConfigClass()` with zero-argument constructor.
The config class is resolved from `model_class.config_class` (authoritative source during enumeration).

### Variant models (ForCausalLM, ForConditionalGeneration)
These share the **base model's config**. Resolution:
- `CodeGenForCausalLM` → base model `CodeGenModel` → config `CodeGenConfig`
- The `hf_config` field in the spec is set during `enumerate_hf()` from `model_class.config_class`
- **CRITICAL:** When building re-run specs, always include `hf_config`, `hf_class`, and `variant` fields.
  Without `hf_config`, the fallback `name.replace("Model", "Config")` fails for variant names
  (e.g., `"CodeGenForCausalLM".replace("Model", "Config")` = `"CodeGenForCausalLM"` — unchanged!)
- **Always build re-run specs from `enumerate_all()`**, never from checkpoint data.

### Composite models (need sub-configs)
These models can't be created with a zero-argument config constructor. Each needs explicit sub-config
construction in `_create_config()`:

| Model Family | Sub-configs needed | Pattern |
|---|---|---|
| EncoderDecoderModel | encoder + decoder | `from_encoder_decoder_configs(BertConfig, BertConfig)` |
| VisionEncoderDecoderModel | ViT + GPT2 | `from_encoder_decoder_configs(ViTConfig, GPT2Config)` |
| SpeechEncoderDecoderModel | Wav2Vec2 + Bert | `from_encoder_decoder_configs(Wav2Vec2Config, BertConfig)` |
| VisionTextDualEncoder | ViT + Bert | explicit dict construction |
| RagModel | DPR + Bart | explicit dict construction |
| Musicgen/MusicgenMelody | T5 + Encodec + decoder | manual dict + `from_dict()` |

**When adding a new composite model:** If `config_cls()` fails, check if the model's `__init__`
requires sub-models or sub-configs. Add a handler to `_create_config()` following the patterns above.

---

## 3. Size Reduction (`_reduce_model_size`)

Applied to ALL models after config creation:

### Layer reduction
Attributes checked: `num_hidden_layers`, `num_layers`, `n_layer`, `n_layers`,
`encoder_layers`, `decoder_layers`, `num_encoder_layers`, `num_decoder_layers`.
If > 4, set to 2.

### Hidden size reduction
If `hidden_size > 4096`: scale down to 1024, proportionally reduce `intermediate_size`
and `num_attention_heads`.

### Alignment fixes
After reduction, ensure:
- `hidden_size % num_attention_heads == 0`
- `num_attention_heads % num_key_value_heads == 0`

### MoE reduction
`num_local_experts`, `num_experts`, `n_routed_experts`: if > 4, set to 4.
Then ensure `topk <= expert_count`.

### Sub-config reduction
Recursively reduces: `text_config`, `decoder`, `encoder`, `audio_config`, `speech_config`.
Vision config: only reduce layers, NOT hidden_size (pooling depends on spatial dims).

### Known issues
- Some models reject `num_hidden_layers` being set on their config (e.g., FunnelModel uses
  `block_sizes`, ProphetNet uses `num_encoder_layers`/`num_decoder_layers`). These show as
  `create_error` and need model-specific overrides in `_fix_config()`.

---

## 4. Input Type Detection

The worker auto-detects input type from config attributes:

| Input type | Detection signal | Example models |
|---|---|---|
| text | `vocab_size` present, no vision/audio attrs | BertModel, GPT2 |
| vision | `image_size` or `patch_size` present, no `vocab_size` | ViTModel, ResNetModel |
| multimodal | both `vocab_size` and vision attrs | LlavaModel, Qwen2VL |
| seq2seq | `is_encoder_decoder=True` or `decoder_layers` | T5, Bart |
| audio | `feature_size` or `sampling_rate` | Wav2Vec2, Whisper |
| speech_seq2seq | audio + `decoder_layers` | SpeechT5, SeamlessM4T |

ForConditionalGeneration variants often need **multimodal inputs** even when the base model
uses text-only. The worker has extensive special handling for FCG input construction.

---

## 5. Adding New Models

### Step-by-step
1. **Check if the model is in `enumerate_all()`** — if not, determine why (new HF release? special naming?)
2. **Test individually first** with baseline settings (4 workers, 180s timeout)
3. **If create_error:** Check if it needs:
   - A composite config handler in `_create_config()`
   - A config fix in `_fix_config()` (e.g., non-standard `num_hidden_layers`)
   - Special init handling (some models need extra args beyond config)
4. **If eager_error:** Check if the model's `forward()` signature matches our input detection
5. **Record wall_time_s** — if > 120s, add to `large_models.json`
6. **Test both eval and train modes**

### Common failure patterns and fixes
| Error | Likely cause | Fix |
|---|---|---|
| `__init__() missing 1 required positional argument: 'config'` | Missing `hf_config` in spec, or composite model without handler | Build spec from `enumerate_all()`, or add handler to `_create_config()` |
| `does not support num_hidden_layers` | Model uses non-standard layer config attr | Add to skip-list in `_reduce_model_size()`, or add fix to `_fix_config()` |
| `Either a configuration has to be provided, or all three of...` | Composite model needs sub-configs | Add handler to `_create_config()` |
| `object has no attribute 'hidden_size'` | Config structure changed in new transformers version | Update `_fix_config()` with the new attribute name |
| `Cannot import name '...' from 'transformers'` | Model added in newer transformers than our env | Upgrade env, or skip model |

---

## 6. Custom Models

Custom models (non-HF) use a separate worker (`corpora/custom-models/worker.py`) with:
- Model definitions in individual Python files under `corpora/custom-models/`
- Each file provides a `create_model()` function that returns `(model, example_inputs)`
- No generic config generation — each model manages its own setup
- Diagnostics must go to stderr, not stdout (stdout is the JSON protocol)

---

## Revision Log

| Date | What changed | Lesson source |
|------|-------------|---------------|
| 2026-04-14 | Initial draft | v2.10_full sweep: incomplete re-run specs caused 30+ false errors; accumulated config/input knowledge from worker.py |

---

*This document is loaded when adding new models or debugging model setup errors.*
*To invoke: read `skills/model-setup.md` before model setup work.*
