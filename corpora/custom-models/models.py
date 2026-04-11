#!/usr/bin/env python3
"""Model registry for custom (non-HuggingFace) open-source models.

These models can't be instantiated via HF AutoModel — they require downloading
source files from GitHub/HuggingFace and mocking external dependencies.

Each entry specifies:
  - name: human-readable model name
  - source: always "custom"
  - category: domain (tts, diffusion, face_restoration, multimodal)
  - repo: GitHub repo (owner/name)
  - files: dict of {local_path: raw_url} for source files to download
  - mocks: list of modules to mock before import
  - model_class: module path to the main model class
  - model_kwargs: constructor arguments
  - input_fn: name of function in this file that generates test inputs
  - compile_target: what to compile ("model" or "model.method_name")

Usage:
  python models.py                    # list all models
  python models.py --count            # just count
  python models.py --output models.json
"""
import argparse
import json
import sys

# GitHub raw content base URL
GH_RAW = "https://raw.githubusercontent.com"
HF_RAW = "https://huggingface.co"


MODELS = [
    # =========================================================================
    # GFPGAN — Face Restoration (U-Net encoder-decoder)
    # =========================================================================
    {
        "name": "GFPGAN",
        "source": "custom",
        "category": "face_restoration",
        "repo": "TencentARC/GFPGAN",
        "files": {
            "gfpgan/gfpganv1_clean_arch.py": f"{GH_RAW}/TencentARC/GFPGAN/master/gfpgan/archs/gfpganv1_clean_arch.py",
            "gfpgan/stylegan2_clean_arch.py": f"{GH_RAW}/TencentARC/GFPGAN/master/gfpgan/archs/stylegan2_clean_arch.py",
        },
        "mocks": ["basicsr"],
        "setup_instructions": """
# gfpganv1_clean_arch.py imports from basicsr which has heavy deps (cv2, etc.)
# We mock basicsr and provide a minimal arch_util with the needed functions.
""",
        "model_module": "gfpgan.gfpganv1_clean_arch",
        "model_class": "GFPGANv1Clean",
        "model_kwargs": {
            "out_size": 64,
            "num_style_feat": 256,
            "channel_multiplier": 1,
            "narrow": 0.5,
        },
        "input_shape": [2, 3, 64, 64],
        "compile_target": "model",
    },

    # =========================================================================
    # FLUX.1 — Diffusion Transformer (DiT)
    # =========================================================================
    {
        "name": "FLUX.1-DiT",
        "source": "custom",
        "category": "diffusion",
        "repo": "black-forest-labs/flux",
        "files": {
            "flux/__init__.py": None,  # create empty
            "flux/model.py": f"{GH_RAW}/black-forest-labs/flux/main/src/flux/model.py",
            "flux/modules/__init__.py": None,
            "flux/modules/layers.py": f"{GH_RAW}/black-forest-labs/flux/main/src/flux/modules/layers.py",
            "flux/math.py": f"{GH_RAW}/black-forest-labs/flux/main/src/flux/math.py",
        },
        "mocks": ["flux.modules.lora"],
        "pip_deps": ["einops"],
        "model_module": "flux.model",
        "model_class": "Flux",
        "model_kwargs_fn": "flux_kwargs",
        "input_fn": "flux_inputs",
        "compile_target": "model",
    },

    # =========================================================================
    # OpenVoice — TTS / Voice Cloning (VITS-based)
    # =========================================================================
    {
        "name": "OpenVoice-SynthesizerTrn",
        "source": "custom",
        "category": "tts",
        "repo": "myshell-ai/OpenVoice",
        "files": {
            "openvoice/__init__.py": None,
            "openvoice/models.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/models.py",
            "openvoice/commons.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/commons.py",
            "openvoice/modules.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/modules.py",
            "openvoice/transforms.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/transforms.py",
            "openvoice/attentions.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/attentions.py",
        },
        "mocks": [],
        "model_module": "openvoice.models",
        "model_class": "SynthesizerTrn",
        "model_kwargs": {
            "n_vocab": 100,
            "spec_channels": 513,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "gin_channels": 256,
        },
        "input_fn": "openvoice_inputs",
        "compile_target": "model.voice_conversion",
        "skip_train": True,  # No forward() — inference-only voice cloning tool
    },

    # OpenVoice infer() — full TTS pipeline (text→audio), exercises more code paths
    {
        "name": "OpenVoice-SynthesizerTrn-infer",
        "source": "custom",
        "category": "tts",
        "repo": "myshell-ai/OpenVoice",
        "files": {
            "openvoice/__init__.py": None,
            "openvoice/models.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/models.py",
            "openvoice/commons.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/commons.py",
            "openvoice/modules.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/modules.py",
            "openvoice/transforms.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/transforms.py",
            "openvoice/attentions.py": f"{GH_RAW}/myshell-ai/OpenVoice/main/openvoice/attentions.py",
        },
        "mocks": [],
        "model_module": "openvoice.models",
        "model_class": "SynthesizerTrn",
        "model_kwargs": {
            "n_vocab": 100,
            "spec_channels": 513,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "gin_channels": 256,
        },
        "input_fn": "openvoice_infer_inputs",
        "compile_target": "model.infer",
        "skip_train": True,  # No forward() — inference-only voice cloning tool
    },

    # =========================================================================
    # GPT-SoVITS — TTS (VITS + GPT autoregressive)
    # =========================================================================
    {
        "name": "GPT-SoVITS-SynthesizerTrn",
        "source": "custom",
        "category": "tts",
        "repo": "RVC-Boss/GPT-SoVITS",
        "files": {
            "gptsovits/__init__.py": None,
            "gptsovits/module/__init__.py": None,
            "gptsovits/module/models.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/models.py",
            "gptsovits/module/commons.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/commons.py",
            "gptsovits/module/modules.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/modules.py",
            "gptsovits/module/attentions.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/attentions.py",
            "gptsovits/module/transforms.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/transforms.py",
        },
        "mocks": ["f5_tts", "module.mrte_model", "module.quantize", "text", "torchmetrics"],
        "extra_sys_paths": ["gptsovits"],  # GPT-SoVITS uses `from module import commons`
        "model_module": "gptsovits.module.models",
        "model_class": "SynthesizerTrn",
        "model_kwargs": {
            "spec_channels": 513,
            "segment_size": 20480,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_speakers": 0,
            "gin_channels": 512,
            "use_sdp": True,
            "semantic_frame_rate": "50hz",
            "version": "v2",
        },
        "input_fn": "gptsovits_inputs",
        "compile_target": "model.infer",
    },

    # GPT-SoVITS forward() — training path with quantizer + encoder + flow + decoder
    {
        "name": "GPT-SoVITS-SynthesizerTrn-forward",
        "source": "custom",
        "category": "tts",
        "repo": "RVC-Boss/GPT-SoVITS",
        "files": {
            "gptsovits/__init__.py": None,
            "gptsovits/module/__init__.py": None,
            "gptsovits/module/models.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/models.py",
            "gptsovits/module/commons.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/commons.py",
            "gptsovits/module/modules.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/modules.py",
            "gptsovits/module/attentions.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/attentions.py",
            "gptsovits/module/transforms.py": f"{GH_RAW}/RVC-Boss/GPT-SoVITS/main/GPT_SoVITS/module/transforms.py",
        },
        "mocks": ["f5_tts", "module.mrte_model", "module.quantize", "text", "torchmetrics"],
        "extra_sys_paths": ["gptsovits"],
        "model_module": "gptsovits.module.models",
        "model_class": "SynthesizerTrn",
        "model_kwargs": {
            "spec_channels": 704,  # match ref_enc v2 channel count
            "segment_size": 20480,
            "inter_channels": 192,
            "hidden_channels": 192,
            "filter_channels": 768,
            "n_heads": 2,
            "n_layers": 6,
            "kernel_size": 3,
            "p_dropout": 0.1,
            "resblock": "1",
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [8, 8, 2, 2],
            "upsample_initial_channel": 512,
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "n_speakers": 0,
            "gin_channels": 512,
            "use_sdp": True,
            "semantic_frame_rate": "50hz",
            "version": "v2",
        },
        "input_fn": "gptsovits_forward_inputs",
        "compile_target": "model",
    },

    # =========================================================================
    # MiniCPM-V — Multimodal LLM (Resampler component)
    # =========================================================================
    {
        "name": "MiniCPM-V-Resampler",
        "source": "custom",
        "category": "multimodal",
        "repo": "OpenBMB/MiniCPM-V",
        "hf_model_id": "openbmb/MiniCPM-Llama3-V-2_5",
        "files": {
            "minicpm/__init__.py": None,
            "minicpm/resampler.py": f"{HF_RAW}/openbmb/MiniCPM-Llama3-V-2_5/raw/main/resampler.py",
        },
        "patches": {
            "minicpm/resampler.py": [
                # Fix missing List import
                {"find": "from typing import Optional, Tuple", "replace": "from typing import List, Optional, Tuple"},
            ],
        },
        "mocks": [],  # Uses real transformers (PreTrainedModel, is_deepspeed_zero3_enabled)
        "model_module": "minicpm.resampler",
        "model_class": "Resampler",
        "model_kwargs": {
            "num_queries": 64,
            "embed_dim": 768,
            "num_heads": 12,
            "kv_dim": 1024,
            "adaptive": False,
            "max_size": [14, 14],
        },
        "input_fn": "minicpm_resampler_inputs",
        "compile_target": "model",
    },

    {
        "name": "MiniCPM-V-ViT",
        "source": "custom",
        "category": "multimodal",
        "repo": "OpenBMB/MiniCPM-V",
        "note": "Uses Idefics2VisionTransformer from transformers — requires transformers>=5.0",
        "files": {},  # No custom files needed — uses HF's Idefics2
        "mocks": [],
        "model_module": "transformers.models.idefics2.modeling_idefics2",
        "model_class": "Idefics2VisionTransformer",
        "model_kwargs_fn": "minicpm_vit_kwargs",
        "input_fn": "minicpm_vit_inputs",
        "compile_target": "model",
    },
]


def enumerate_custom():
    """Return all custom model specs."""
    return MODELS


# =========================================================================
# Input generation functions — called by worker to create synthetic inputs
# =========================================================================

def flux_kwargs():
    """Create FluxParams for FLUX.1 DiT."""
    from flux.model import FluxParams
    params = FluxParams(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=384,
        mlp_ratio=4.0,
        num_heads=6,
        depth=2,
        depth_single_blocks=2,
        axes_dim=[32, 16, 16],
        theta=10000,
        qkv_bias=True,
        guidance_embed=False,
    )
    return {"params": params}


def flux_inputs(batch_size=2):
    """FLUX.1 DiT inputs: (img, timestep, y, txt, txt_ids, img_ids)."""
    import torch
    seq_len = 16
    txt_len = 8
    hidden = 64  # in_channels
    img = torch.randn(batch_size, seq_len, hidden)
    timesteps = torch.rand(batch_size)
    y = torch.randn(batch_size, 768)  # vec_in_dim
    txt = torch.randn(batch_size, txt_len, 4096)  # context_in_dim
    txt_ids = torch.zeros(batch_size, txt_len, 3)
    img_ids = torch.zeros(batch_size, seq_len, 3)
    return {"img": img, "img_ids": img_ids, "txt": txt,
            "txt_ids": txt_ids, "timesteps": timesteps, "y": y}


def openvoice_inputs(batch_size=2):
    """OpenVoice SynthesizerTrn.voice_conversion() inputs."""
    import torch
    T = 32
    y = torch.randn(batch_size, 513, T)                   # spectrogram
    y_lengths = torch.tensor([T] * batch_size)
    sid_src = torch.randn(batch_size, 256, 1)             # speaker embedding
    sid_tgt = torch.randn(batch_size, 256, 1)
    return {"y": y, "y_lengths": y_lengths,
            "sid_src": sid_src, "sid_tgt": sid_tgt}


def openvoice_infer_inputs(batch_size=2):
    """OpenVoice SynthesizerTrn.infer() inputs — full TTS pipeline."""
    import torch
    T_text = 16
    x = torch.randint(0, 100, (batch_size, T_text))        # text token ids
    x_lengths = torch.tensor([T_text] * batch_size)
    sid = torch.randint(0, 256, (batch_size,))              # speaker id
    return {"x": x, "x_lengths": x_lengths, "sid": sid}


def gptsovits_inputs(batch_size=2):
    """GPT-SoVITS SynthesizerTrn.infer() inputs."""
    import torch
    T_ssl = 32
    T_text = 16
    ssl = torch.randn(batch_size, 768, T_ssl)
    y = torch.randn(batch_size, 704, T_ssl)
    y_lengths = torch.tensor([T_ssl] * batch_size)
    text = torch.randint(0, 28, (batch_size, T_text))
    text_lengths = torch.tensor([T_text] * batch_size)
    return {"ssl": ssl, "y": y, "y_lengths": y_lengths,
            "text": text, "text_lengths": text_lengths}


def gptsovits_forward_inputs(batch_size=2):
    """GPT-SoVITS SynthesizerTrn.forward() inputs — training path.

    y needs 513 channels (spec_channels) for enc_q, but ref_enc (v2)
    clips y[:, :704]. In practice y is the full spectrogram (513 ch).
    """
    import torch
    T_ssl = 32
    T_text = 16
    ssl = torch.randn(batch_size, 768, T_ssl)
    y = torch.randn(batch_size, 704, T_ssl)     # matches ref_enc v2 + enc_q
    y_lengths = torch.tensor([T_ssl] * batch_size)
    text = torch.randint(0, 28, (batch_size, T_text))
    text_lengths = torch.tensor([T_text] * batch_size)
    return {"ssl": ssl, "y": y, "y_lengths": y_lengths,
            "text": text, "text_lengths": text_lengths}


def minicpm_resampler_inputs(batch_size=2):
    """MiniCPM-V Resampler inputs."""
    import torch
    x = torch.randn(batch_size, 196, 1024)  # 14x14 patches, vision_dim
    tgt_sizes = torch.tensor([[14, 14]] * batch_size)
    return {"x": x, "tgt_sizes": tgt_sizes}


def minicpm_vit_kwargs():
    """Create Idefics2VisionConfig for MiniCPM-V ViT."""
    from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionConfig
    config = Idefics2VisionConfig(
        hidden_size=384,
        image_size=196,
        intermediate_size=768,
        num_attention_heads=6,
        num_hidden_layers=2,
        patch_size=14,
    )
    return {"config": config}


def minicpm_vit_inputs(batch_size=2):
    """MiniCPM-V ViT inputs."""
    import torch
    return {"pixel_values": torch.randn(batch_size, 3, 196, 196)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom model registry")
    parser.add_argument("--count", action="store_true")
    parser.add_argument("--output", type=str)
    parser.add_argument("--category", type=str, help="Filter by category")
    args = parser.parse_args()

    models = enumerate_custom()
    if args.category:
        models = [m for m in models if m.get("category") == args.category]

    if args.count:
        print(f"Total custom models: {len(models)}")
        by_cat = {}
        for m in models:
            cat = m.get("category", "unknown")
            by_cat[cat] = by_cat.get(cat, 0) + 1
        for cat, count in sorted(by_cat.items()):
            print(f"  {cat}: {count}")
    elif args.output:
        with open(args.output, "w") as f:
            json.dump(models, f, indent=2)
        print(f"Saved {len(models)} models to {args.output}")
    else:
        for m in models:
            print(f"  {m['name']:35s}  {m.get('category','?'):20s}  {m['repo']}")
