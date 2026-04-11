# Custom Models Corpus

Non-HuggingFace open-source models tested with `torch.compile` / `dynamo.explain`.

These models can't be tested via the main HF sweep — they have custom architectures,
non-standard dependencies, or aren't published as HF model classes. We download their
source directly from GitHub/HuggingFace, mock external deps, and test at the top level.

## Why These Models?

The HuggingFace corpus covers models available via `AutoModel`, but many popular
open-source models have custom architectures that live outside the HF ecosystem.
We selected these models by:

1. **Surveying high-star GitHub repos** in categories with known `torch.compile`
   friction (TTS/voice cloning, diffusion, multimodal, face restoration)
2. **Filtering out models already covered** by the HuggingFace corpus — if a model
   or its architecture family is already tested there, we skip it
3. **Prioritizing diverse graph break patterns** — each model was chosen because it
   exercises a different combination of Dynamo-unfriendly patterns (data-dependent
   branching, `Tensor.item()`, dynamic shape ops)

The result is 6 models (7 entry points) from 5 repos totaling ~160k GitHub stars,
representing real-world code that users actually compile.

## Models

| Model | Entry Point | Category | Repo | Stars | Graph Breaks | Root Cause |
|-------|-------------|----------|------|-------|-------------|------------|
| GFPGAN | forward | Face restoration | TencentARC/GFPGAN | ~36k | 0 | Clean |
| FLUX.1-DiT | forward | Diffusion | black-forest-labs/flux | ~20k | 0 | Clean |
| OpenVoice | voice_conversion | TTS | myshell-ai/OpenVoice | ~30k | 7 | Tensor.item() |
| OpenVoice | infer | TTS | myshell-ai/OpenVoice | — | 22 | nonzero, item, data-dep branch |
| GPT-SoVITS | infer | TTS | RVC-Boss/GPT-SoVITS | ~39k | 4 | Tensor.item() in WaveNet |
| MiniCPM-V Resampler | forward | Multimodal | OpenBMB/MiniCPM-V | ~14k | 5 | data-dep branch, scalar |
| MiniCPM-V ViT | forward | Multimodal | OpenBMB/MiniCPM-V | — | 3 | nonzero in attention |

Note: OpenVoice has two entries because `voice_conversion()` and `infer()` exercise
very different code paths. The `infer()` path includes the StochasticDurationPredictor
with spline transforms (nonzero, data-dependent branching) that `voice_conversion()` skips entirely.

## Usage

```bash
# Test a single model
python worker.py --model-name GFPGAN

# Test all models
python worker.py --all --output results.json

# Without proxy (direct download)
python worker.py --all --no-proxy --output results.json

# List available models
python models.py
```

## How It Works

1. **`models.py`** — Registry of model specs: GitHub URLs, mock requirements, constructor args, input shapes
2. **`worker.py`** — Downloads source files, mocks external deps, instantiates model, runs eager + `dynamo.explain`
3. Source files are cached in `_sources/` (gitignored)

### Adding a New Model

Add an entry to `MODELS` in `models.py`:

```python
{
    "name": "MyModel",
    "source": "custom",
    "category": "tts",  # or diffusion, face_restoration, multimodal, etc.
    "repo": "owner/repo",
    "files": {
        "mymodel/model.py": "https://raw.githubusercontent.com/owner/repo/main/model.py",
    },
    "mocks": ["some_heavy_dep"],
    "model_module": "mymodel.model",
    "model_class": "MyModelClass",
    "model_kwargs": {"hidden_dim": 256},
    "input_shape": [1, 3, 224, 224],  # or use input_fn for complex inputs
    "compile_target": "model",  # or "model.forward" / "model.infer"
}
```

## Key Findings

### Same graph break categories, different solutions

All graph breaks fall into 3 categories: `aten.nonzero`, `Tensor.item()`, and data-dependent branching.
But the **specific functions** causing them differ across repos, so fixes aren't interchangeable:

- **OpenVoice/GPT-SoVITS**: `fused_add_tanh_sigmoid_multiply` → `n_channels[0]` converts tensor to int for slicing.
  Fix: replace with Python int (known at trace time).
- **MiniCPM-V Resampler**: `tgt_sizes[i]` in a for-loop extracts scalars for pos-embed slicing.
  Fix: restructure loop to avoid per-item extraction.
- **OpenVoice/GPT-SoVITS**: spline transforms use `nonzero` for out-of-bounds clamping.
  Fix: `capture_dynamic_output_shape_ops = True` or rewrite without nonzero.

## Requirements

Same as the main corpus: `torch>=2.10.0`. The `transformers` package is only needed for
the MiniCPM-V ViT test (Idefics2VisionTransformer).
