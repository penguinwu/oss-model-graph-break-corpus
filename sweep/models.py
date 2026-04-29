#!/usr/bin/env python3
"""Model enumeration — discovers models from TIMM, HF Transformers, Diffusers, and custom repos.

Each model is represented as a JSON-serializable spec dict with at minimum:
  {"name": "...", "source": "timm|hf|diffusers|custom"}

Plus source-specific fields needed by worker.py to instantiate the model.

Usage:
  python models.py --source timm           # list all TIMM models
  python models.py --source hf             # list all HF base Model classes
  python models.py --source diffusers      # list all Diffusers ModelMixin classes
  python models.py --source custom         # list custom (non-HF) models
  python models.py --source all            # list everything
  python models.py --source all --count    # just counts
  python models.py --source all --output models.json  # save to file
"""
import argparse
import inspect
import json
import sys
from pathlib import Path


def enumerate_timm():
    """Enumerate all TIMM models via timm.list_models()."""
    import timm

    models = []
    for name in sorted(timm.list_models()):
        models.append({
            "name": name,
            "source": "timm",
        })
    return models


def enumerate_hf():
    """Enumerate HF Transformers model classes with matching Configs.

    Discovers three tiers of model classes:
    - Base *Model (backbone only — e.g. Gemma4Model)
    - *ForCausalLM (backbone + lm_head — e.g. Gemma4ForCausalLM)
    - *ForConditionalGeneration (backbone + task head + multimodal merge)

    Filters:
    - Must subclass PreTrainedModel
    - Must not be a PreTrainedModel base class
    - Must have a config_class with a matching Config in transformers
    """
    import transformers

    # Accepted suffixes and their variant labels
    _SUFFIXES = [
        ("Model", "base"),
        ("ForCausalLM", "causal_lm"),
        ("ForConditionalGeneration", "conditional_generation"),
    ]

    models = []
    seen = set()

    # NOTE: We use dir() + per-attribute getattr instead of inspect.getmembers()
    # because transformers exposes lazy-import attributes; a broken environment
    # (missing torchvision, etc.) makes a single class raise ModuleNotFoundError
    # at attribute access time, and inspect.getmembers() aborts the entire walk.
    for name in sorted(dir(transformers)):
        try:
            obj = getattr(transformers, name)
        except (ModuleNotFoundError, ImportError, RuntimeError):
            continue
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, transformers.PreTrainedModel):
            continue
        if obj is transformers.PreTrainedModel:
            continue
        if "PreTrained" in name or "Pretrained" in name:
            continue

        # Skip models whose forward() is _forward_unimplemented (abstract base classes)
        fwd = getattr(obj, "forward", None)
        if fwd and "unimplemented" in getattr(fwd, "__name__", ""):
            continue

        # Skip models that explicitly reject construction
        _SKIP_MODELS = {
            "T5GemmaEncoderModel",  # Raises "only supports encoder-only" — use T5GemmaModel instead
            "BarkModel",            # Abstract — forward() is _forward_unimplemented
            "BartPretrainedModel",  # Abstract — forward() is _forward_unimplemented
            "PretrainedBartModel",  # Abstract — forward() is _forward_unimplemented
            "PretrainedFSMTModel",  # Abstract — forward() is _forward_unimplemented
            "Qwen3OmniMoeTalkerForConditionalGeneration",   # Internal: needs inputs_embeds from parent Thinker
            "Qwen3OmniMoeCode2WavTransformerModel",         # Internal: does not accept input_ids
            "Qwen2_5OmniTalkerForConditionalGeneration",    # Internal: needs inputs_embeds from parent Thinker
            "Qwen2_5OmniToken2WavModel",                    # Requires batch_size=1, internal component
            "Sam2VideoModel",       # Stateful: forward() requires inference_session with frame tracking state
            "Sam3VideoModel",       # Stateful: forward() requires inference_session with frame tracking state
            "Sam3TrackerVideoModel",# Stateful: forward() requires inference_session with frame tracking state
            "EdgeTamVideoModel",    # Stateful: forward() requires inference_session with frame tracking state
            "EdgeTamModel",         # Requires timm/repvit_m1.dist_in1k download, blocked on devvm (564s timeout)
            "EdgeTamVisionModel",   # Same timm dependency as EdgeTamModel
            "PI0ForConditionalGeneration",  # image_token_id (257152) == vocab_size; reduced model can't embed image tokens
            "ClvpModelForConditionalGeneration",  # CUDA assert: decoder embedding index out of range from speech encoder token coupling
            # Models requiring deps not installable under our agent:claude_code BPF identity (#91)
            "DinatModel",                   # Requires `natten` library — wheel build fails under our env
            "LayoutLMv2Model",              # Requires `detectron2` from dl.fbaipublicfiles.com — domain not allowlisted
        }
        if name in _SKIP_MODELS:
            continue

        # Determine which variant this is
        variant = None
        for suffix, var_name in _SUFFIXES:
            if name.endswith(suffix):
                # Base models must not contain "For" (avoids FormerModel etc.)
                if var_name == "base" and "For" in name:
                    continue
                variant = var_name
                break
        if variant is None:
            continue

        # Get config class from the model class itself (authoritative source)
        config_cls = getattr(obj, "config_class", None)
        if config_cls is None:
            continue
        config_name = config_cls.__name__
        if not hasattr(transformers, config_name):
            continue

        # Deduplicate (some models are aliased)
        if name in seen:
            continue
        seen.add(name)

        spec = {
            "name": name,
            "source": "hf",
            "hf_class": name,
            "hf_config": config_name,
        }
        if variant != "base":
            spec["variant"] = variant

        # Detect input type using base model name for heuristics
        base_name = name
        if variant == "causal_lm":
            base_name = name.replace("ForCausalLM", "Model")
        elif variant == "conditional_generation":
            base_name = name.replace("ForConditionalGeneration", "Model")
        input_type = _detect_hf_input_type(base_name)
        if input_type != "text":
            spec["input_type"] = input_type

        models.append(spec)

    return models


def _detect_hf_input_type(model_name):
    """Detect input modality from model class name (fast, no config instantiation)."""
    name_lower = model_name.lower()

    # Multimodal models (vision + text)
    multimodal_keywords = [
        "align", "altclip", "aimv2model", "blip", "bridgetower", "clip",
        "chinese_clip", "chineseclip", "flava", "git", "llava", "paligemma",
        "siglip", "colpali", "colqwen", "idefics", "instructblip",
        "kosmos", "mgp", "owlv2", "owl_vit", "owlvit", "pix2struct",
        "vilt", "xclip", "groupvit", "groundingdino", "mmgroundingdino",
    ]
    # Match only composite models (not sub-models like CLIPTextModel, CLIPVisionModel)
    if any(kw in name_lower for kw in multimodal_keywords):
        # Exclude sub-models (TextModel, VisionModel)
        if not any(sub in name_lower for sub in ["text", "vision", "image", "audio"]):
            return "multimodal"

    # Vision models
    vision_keywords = [
        "vit", "swin", "deit", "beit", "convnext", "resnet", "poolformer",
        "pvt", "segformer", "dinat", "nat", "levit", "efficientnet",
        "mobilevit", "mobilenet", "regnet", "van", "dinov2", "hiera",
        "cvt", "focalnet", "bit", "dpt", "glpn", "visionmodel",
    ]
    if any(kw in name_lower for kw in vision_keywords):
        return "vision"

    # Audio models
    audio_keywords = [
        "whisper", "wav2vec", "hubert", "audio", "unispeech", "wavlm",
        "sew", "ast", "clap",
    ]
    if any(kw in name_lower for kw in audio_keywords):
        return "audio"

    # Seq2seq models
    seq2seq_keywords = [
        "t5", "bart", "mbart", "pegasus", "marian", "blenderbot",
        "prophetnet", "led", "longt5", "mt5", "umt5", "nllb", "seamless",
        "m2m", "plbart", "bigbirdpegasus",
    ]
    if any(kw in name_lower for kw in seq2seq_keywords):
        return "seq2seq"

    return "text"


def enumerate_diffusers():
    """Enumerate Diffusers ModelMixin subclasses.

    Diffusers models need per-family constructor args and input shapes.
    We provide minimal configs for known families; unknown models get
    an empty constructor_args (will likely fail — flagged for manual config).
    """
    import diffusers
    from diffusers import ModelMixin

    # Minimal constructor configs for known model families
    FAMILY_CONFIGS = {
        # VAE / Autoencoder family
        "AutoencoderKL": {
            "constructor_args": {
                "in_channels": 3, "out_channels": 3, "latent_channels": 4,
                "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D"),
                "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D"),
                "block_out_channels": (32, 64),
            },
            "inputs": {"sample": [1, 3, 64, 64]},
        },
        "AutoencoderTiny": {
            "constructor_args": {
                "in_channels": 3, "out_channels": 3, "latent_channels": 4,
                "encoder_block_out_channels": (32, 32),
                "decoder_block_out_channels": (32, 32),
                "num_encoder_blocks": (1, 1), "num_decoder_blocks": (1, 1),
            },
            "inputs": {"sample": [3, 3, 64, 64]},
        },
        # UNet family
        "UNet2DModel": {
            "constructor_args": {
                "in_channels": 3, "out_channels": 3, "sample_size": 32,
                "down_block_types": ("DownBlock2D", "DownBlock2D"),
                "up_block_types": ("UpBlock2D", "UpBlock2D"),
                "block_out_channels": (32, 64),
            },
            "inputs": {"sample": [1, 3, 32, 32], "timestep": 1.0},
        },
        "UNet2DConditionModel": {
            "constructor_args": {
                "in_channels": 4, "out_channels": 4, "sample_size": 32,
                "cross_attention_dim": 32,
                "down_block_types": ("CrossAttnDownBlock2D", "DownBlock2D"),
                "up_block_types": ("UpBlock2D", "CrossAttnUpBlock2D"),
                "block_out_channels": (32, 64),
            },
            "inputs": {
                "sample": [1, 4, 32, 32],
                "timestep": 1.0,
                "encoder_hidden_states": [1, 1, 32],
            },
        },
        # DiT / Transformer family
        "DiTTransformer2DModel": {
            "constructor_args": {
                "num_attention_heads": 2, "attention_head_dim": 16,
                "in_channels": 4, "out_channels": 4, "num_layers": 2,
                "sample_size": 8, "num_embeds_ada_norm": 10,
            },
            "inputs": {
                "hidden_states": [3, 4, 8, 8],
                "timestep": 1.0,
                "class_labels": [3],
            },
        },
        # Verified pass on 2026-04-29 with default AutoencoderKL constructor_args
        # — the only AutoencoderKL variant whose __init__ accepts the parent
        # family's kwargs cleanly. Other variants (Allegro, CogVideoX, Cosmos,
        # Hunyuan*, LTX*, Magvit, Mochi, QwenImage, TemporalDecoder, Wan)
        # require per-variant configs (#94).
        "AutoencoderKLFlux2": {
            "constructor_args": {
                "in_channels": 3, "out_channels": 3, "latent_channels": 4,
                "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D"),
                "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D"),
                "block_out_channels": (32, 64),
            },
            "inputs": {"sample": [1, 3, 64, 64]},
        },
    }

    models = []
    skipped_unconstructable = []
    auto_constructed = []
    # See enumerate_hf for rationale — tolerate broken lazy imports.
    for name in sorted(dir(diffusers)):
        try:
            obj = getattr(diffusers, name)
        except (ModuleNotFoundError, ImportError, RuntimeError):
            continue
        if not inspect.isclass(obj):
            continue
        if not issubclass(obj, ModelMixin):
            continue
        if obj is ModelMixin:
            continue
        # Skip multi-model wrappers
        if "Multi" in name:
            continue

        # Path 1: explicit FAMILY_CONFIGS entry (canonical for tested-known shapes)
        config = FAMILY_CONFIGS.get(name)
        if config:
            models.append({
                "name": name,
                "source": "diffusers",
                "hf_class": name,
                "constructor_args": config["constructor_args"],
                "inputs": config["inputs"],
                "has_config": True,
            })
            continue

        # Path 2: signature-only check. Most diffusers ModelMixin subclasses
        # use `register_to_config` decorator so every __init__ arg has a
        # default — we don't need to actually construct (which would be slow,
        # ~10s per model at enumeration). Just check there's no required arg;
        # if `cls()` would succeed, emit the spec, and let the worker's
        # `create_diffusers_model` do the actual construction + auto-synthesize
        # inputs from `forward.__signature__` at create time. If construction
        # actually fails at sweep time, the row goes to gated bucket where the
        # validator's `harness` classifier ("__init__() missing N required
        # positional arguments") catches it.
        try:
            init_sig = inspect.signature(obj.__init__)
            required = [p for p in init_sig.parameters.values()
                        if p.default is p.empty
                        and p.name != "self"
                        and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
            if required:
                skipped_unconstructable.append(name)
                continue
        except (ValueError, TypeError):
            skipped_unconstructable.append(name)
            continue

        models.append({
            "name": name,
            "source": "diffusers",
            "hf_class": name,
            "auto_inputs": True,
        })
        auto_constructed.append(name)

    if auto_constructed:
        print(f"[enumerate_diffusers] Auto-constructed {len(auto_constructed)} "
              f"models via no-arg trial; inputs synthesized at create time from "
              f"forward signature.", file=sys.stderr)
    if skipped_unconstructable:
        print(f"[enumerate_diffusers] Skipped {len(skipped_unconstructable)} "
              f"ModelMixin subclasses that need explicit constructor_args. "
              f"Add a FAMILY_CONFIGS entry to enable. Examples: "
              f"{', '.join(skipped_unconstructable[:3])}, ...",
              file=sys.stderr)

    return models


def enumerate_custom():
    """Enumerate custom (non-HuggingFace) models from the custom-models corpus."""
    import importlib.util
    custom_models_path = Path(__file__).resolve().parent.parent / "corpora" / "custom-models" / "models.py"
    if not custom_models_path.exists():
        print(f"Warning: custom models registry not found at {custom_models_path}")
        return []
    spec = importlib.util.spec_from_file_location("custom_models", custom_models_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.enumerate_custom()


def enumerate_all():
    """Enumerate models from default sources: hf + diffusers + custom.

    Per Peng's standing direction, timm is NOT included by default — request
    it explicitly via `--source timm` if needed. This helper mirrors the
    default --source list in the sweep CLI.

    Diffusers models are included if they either have an explicit
    FAMILY_CONFIGS entry (has_config=True) OR pass the no-required-init-args
    signature check (auto_inputs=True). Models that need explicit constructor
    args but lack a FAMILY_CONFIGS entry are excluded.
    """
    models = []
    # HF first: smaller, more diverse, expose more problems early
    models.extend(enumerate_hf())
    # Include diffusers models with explicit configs OR auto-inputs introspection
    models.extend([m for m in enumerate_diffusers()
                   if m.get("has_config", False) or m.get("auto_inputs", False)])
    # Custom models (non-HF repos)
    models.extend(enumerate_custom())
    return models


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Enumerate models for graph break sweep")
    parser.add_argument("--source", default="all", choices=["timm", "hf", "diffusers", "custom", "all"])
    parser.add_argument("--count", action="store_true", help="Just print counts")
    parser.add_argument("--output", help="Save model list to JSON file")
    parser.add_argument("--has-config-only", action="store_true",
                        help="Only include models with known input configs (diffusers)")
    args = parser.parse_args()

    if args.source == "all":
        models = enumerate_all()
    elif args.source == "timm":
        models = enumerate_timm()
    elif args.source == "hf":
        models = enumerate_hf()
    elif args.source == "diffusers":
        models = enumerate_diffusers()
    elif args.source == "custom":
        models = enumerate_custom()

    if args.has_config_only:
        models = [m for m in models if m.get("has_config", True)]

    if args.count:
        by_source = {}
        for m in models:
            by_source.setdefault(m["source"], 0)
            by_source[m["source"]] += 1
        for source, count in sorted(by_source.items()):
            print(f"  {source}: {count}")
        print(f"  total: {len(models)}")
    elif args.output:
        with open(args.output, "w") as f:
            json.dump(models, f, indent=2)
        print(f"Saved {len(models)} models to {args.output}")
    else:
        for m in models:
            print(json.dumps(m))


if __name__ == "__main__":
    main()
