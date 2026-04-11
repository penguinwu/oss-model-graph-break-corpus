# Custom Models Corpus

Non-HuggingFace open-source models tested with `torch.compile` / `dynamo.explain`.

These models can't be tested via the main HF sweep — they have custom architectures,
non-standard dependencies, or aren't published as HF model classes. We download their
source directly from GitHub/HuggingFace, mock external deps, and test at the top level.

## Models

| Model | Category | Repo | Graph Breaks | Root Cause |
|-------|----------|------|-------------|------------|
| GFPGAN | Face restoration | TencentARC/GFPGAN | 0 | Clean |
| FLUX.1-DiT | Diffusion | black-forest-labs/flux | 0 | Clean |
| OpenVoice | TTS | myshell-ai/OpenVoice | 31 | nonzero, item, data-dep branch |
| GPT-SoVITS | TTS | RVC-Boss/GPT-SoVITS | 4 | Tensor.item() in WaveNet |
| MiniCPM-V Resampler | Multimodal | OpenBMB/MiniCPM-V | 5 | data-dep branch, scalar |
| MiniCPM-V ViT | Multimodal | OpenBMB/MiniCPM-V | 3 | nonzero in attention |

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
