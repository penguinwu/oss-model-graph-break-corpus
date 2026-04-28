#!/usr/bin/env python3
"""Single-model worker — subprocess entry point for the two-pass sweep.

Called by the orchestrator as:
  python worker.py --model-json '{"name":"resnet50","source":"timm",...}' \
                   --pass 1 --device cuda --mode eval

Identify: fullgraph=True compile → binary pass/fail
Explain: TORCH_LOGS=graph_breaks + counting backend → detailed graph break analysis

Outputs a single JSON line to stdout. All logs go to stderr.
"""
import argparse
import inspect
import json
import os
import sys
import time
import traceback
import warnings


# ─── Chaos mode fast path ────────────────────────────────────────────────────
# Handle chaos mode BEFORE importing torch (saves 3-5s startup).
# This makes stress tests fast and prevents chaos workers from being
# killed by timeout before they even reach the chaos handler.

def _handle_chaos():
    """Fast exit for chaos mode — no heavy imports needed."""
    # Parse model-json from argv without full argparse
    for i, arg in enumerate(sys.argv):
        if arg == "--model-json" and i + 1 < len(sys.argv):
            try:
                spec = json.loads(sys.argv[i + 1])
            except (json.JSONDecodeError, IndexError):
                return  # Not parseable, continue normal path
            chaos = spec.get("_chaos")
            if not chaos:
                return  # Normal model, continue to torch imports

            # Parse mode and pass-num from argv
            mode, pass_num = "eval", 1
            for j, a in enumerate(sys.argv):
                if a == "--mode" and j + 1 < len(sys.argv):
                    mode = sys.argv[j + 1]
                elif a == "--pass-num" and j + 1 < len(sys.argv):
                    pass_num = int(sys.argv[j + 1])

            print(f"PHASE:chaos_{chaos}", file=sys.stderr)
            name = spec["name"]
            source = spec.get("source", "chaos")

            if chaos == "clean":
                time.sleep(1)
                result = {"name": name, "source": source, "mode": mode,
                          "pass": pass_num, "status": "full_graph", "wall_time_s": 1.0}
            elif chaos == "crash":
                import signal as _sig
                os.kill(os.getpid(), _sig.SIGSEGV)
            elif chaos == "hang":
                time.sleep(999999)
            elif chaos == "slow":
                sleep_time = int(spec.get("_chaos_sleep", 200))
                time.sleep(sleep_time)
                result = {"name": name, "source": source, "mode": mode,
                          "pass": pass_num, "status": "full_graph",
                          "wall_time_s": float(sleep_time)}
            elif chaos == "oom":
                # Import torch only for OOM test
                import torch
                torch.randn(1024, 1024, 1024, 1024, device="cuda")
            elif chaos == "exit1":
                sys.exit(1)
            elif chaos == "bad_json":
                print("THIS IS NOT VALID JSON")
                sys.exit(0)
            elif chaos == "gpu_leak":
                import torch
                _leaked = torch.randn(2048, 2048, 512, device="cuda")  # ~8GB
                result = {"name": name, "source": source, "mode": mode,
                          "pass": pass_num, "status": "full_graph", "wall_time_s": 1.0}
            else:
                result = {"name": name, "source": source, "mode": mode,
                          "pass": pass_num, "status": "worker_error",
                          "error": f"Unknown chaos mode: {chaos}"}
            print(json.dumps(result))
            sys.exit(0)


if __name__ == "__main__":
    _handle_chaos()

# ─── Heavy imports (only reached for real models) ────────────────────────────

warnings.filterwarnings("ignore")

import torch
import torch._dynamo

from explain import run_graph_break_analysis


# ─── Model creation ──────────────────────────────────────────────────────────

# Default batch size — avoid 0 and 1 (PyTorch specializes on these)
DEFAULT_BATCH = 2


def create_timm_model(spec, device, batch_size=DEFAULT_BATCH):
    import timm

    model = timm.create_model(spec["name"], pretrained=False)
    cfg = model.default_cfg
    input_size = cfg.get("input_size", (3, 224, 224))
    x = torch.randn(batch_size, *input_size, device=device)
    model = model.to(device)
    return model, None, (x,)


def _create_config(model_name, config_cls):
    """Create config with special handling for composite models.

    Some HF models (encoder-decoder, dual-encoder, musicgen) require
    sub-configs at init time. Their config constructors fail with no args.
    This function creates properly constructed configs for those models.
    """
    name_lower = model_name.lower()

    # --- Composite models that need sub-configs at init ---
    if name_lower == "encoderdecodermodel":
        from transformers import BertConfig, EncoderDecoderConfig
        return EncoderDecoderConfig.from_encoder_decoder_configs(
            BertConfig(num_hidden_layers=2),
            BertConfig(num_hidden_layers=2, is_decoder=True, add_cross_attention=True)
        )

    if name_lower == "visionencoderdecodermodel":
        from transformers import ViTConfig, GPT2Config, VisionEncoderDecoderConfig
        return VisionEncoderDecoderConfig.from_encoder_decoder_configs(
            ViTConfig(num_hidden_layers=2),
            GPT2Config(n_layer=2, add_cross_attention=True)
        )

    if name_lower == "speechencoderdecodermodel":
        from transformers import Wav2Vec2Config, BertConfig, SpeechEncoderDecoderConfig
        return SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
            Wav2Vec2Config(num_hidden_layers=2),
            BertConfig(num_hidden_layers=2, is_decoder=True, add_cross_attention=True)
        )

    if name_lower == "visiontextdualencodermodel":
        from transformers import ViTConfig, BertConfig, VisionTextDualEncoderConfig
        return VisionTextDualEncoderConfig(
            vision_config=ViTConfig(num_hidden_layers=2).to_dict(),
            text_config=BertConfig(num_hidden_layers=2).to_dict()
        )

    if name_lower == "ragmodel":
        from transformers import DPRConfig, BartConfig, RagConfig
        return RagConfig(
            question_encoder=DPRConfig().to_dict(),
            generator=BartConfig(encoder_layers=2, decoder_layers=2).to_dict()
        )

    if name_lower in ("musicgenmodel", "musicgenmelodymodel"):
        from transformers import T5Config, EncodecConfig
        t5 = T5Config(num_layers=2, num_decoder_layers=2)
        enc = EncodecConfig()
        if name_lower == "musicgenmodel":
            from transformers import MusicgenConfig, MusicgenDecoderConfig
            dec = MusicgenDecoderConfig(num_hidden_layers=2)
            d = dec.to_dict()
            d.update(text_encoder=t5.to_dict(), audio_encoder=enc.to_dict(),
                     decoder=dec.to_dict(), model_type="musicgen")
            return MusicgenConfig.from_dict(d)
        else:
            from transformers import MusicgenMelodyConfig, MusicgenMelodyDecoderConfig
            dec = MusicgenMelodyDecoderConfig(num_hidden_layers=2)
            d = dec.to_dict()
            d.update(text_encoder=t5.to_dict(), audio_encoder=enc.to_dict(),
                     decoder=dec.to_dict(), model_type="musicgen_melody")
            return MusicgenMelodyConfig.from_dict(d)

    # --- Sub-model configs: extract from parent config ---
    # Some models (e.g., Aimv2TextModel) are enumerated with the parent config class
    # (Aimv2Config) but need the sub-config (text_config, depth_decoder_config, etc.)
    _sub_config_map = {
        "aimv2textmodel": "text_config",
        "clapaudiomodel": "audio_config",
        "csmdepthdecodermodel": "depth_decoder_config",
        "csmdepthdecoderforcausallm": "depth_decoder_config",
        "paddleocrtextmodel": "text_config",
        "qwen3omnimoecode2wavtransformermodel": "code2wav_config",
    }
    sub_attr = _sub_config_map.get(name_lower)
    if sub_attr:
        parent_config = config_cls()
        sub_config = getattr(parent_config, sub_attr, None)
        if sub_config is not None:
            return sub_config

    # Default: standard constructor
    return config_cls()


def _fix_config(model_name, config):
    """Patch known config bugs so models can be instantiated with default configs.

    Many HF model configs have None or missing values that crash during
    model construction. This fixes them with sensible defaults.
    """
    name_lower = model_name.lower()

    # --- ProphetNet: rejects num_hidden_layers, uses encoder/decoder_layers ---
    if "prophetnet" in name_lower:
        # _reduce_model_size sets num_hidden_layers which ProphetNet's config rejects.
        # Delete it and ensure encoder/decoder layers are set instead.
        if hasattr(config, "num_encoder_layers"):
            config.num_encoder_layers = min(getattr(config, "num_encoder_layers", 2), 2)
        if hasattr(config, "num_decoder_layers"):
            config.num_decoder_layers = min(getattr(config, "num_decoder_layers", 2), 2)

    # --- Time series: need prediction_length and context_length ---
    if name_lower in ("autoformermodel", "informermodel", "timeseriestransformermodel"):
        if not getattr(config, "prediction_length", None):
            config.prediction_length = 24
        if not getattr(config, "context_length", None):
            config.context_length = 96

    # torch.compile materializes attention masks that eager skips — these two need config tweaks
    if name_lower == "autoformermodel":
        d_model = getattr(config, "d_model", 64)
        num_heads = getattr(config, "encoder_attention_heads", 2)
        config.context_length = d_model // num_heads

    if name_lower == "informermodel":
        config.distil = False

    # --- AyaVisionModel: embed_dim must be divisible by num_heads ---
    if name_lower == "ayavisionmodel":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            embed_dim = getattr(vision_cfg, "hidden_size", 1152)
            num_heads = getattr(vision_cfg, "num_attention_heads", 14)
            if embed_dim % num_heads != 0:
                vision_cfg.num_attention_heads = 16  # 1152 % 16 = 0

    # --- ChameleonModel, Emu3Model: vq_config + vocabulary_map ---
    if name_lower in ("chameleonmodel", "emu3model"):
        if getattr(config, "vq_config", None) is None:
            from transformers import ChameleonVQVAEConfig
            config.vq_config = ChameleonVQVAEConfig()
        if getattr(config, "vocabulary_map", None) is None:
            config.vocabulary_map = {"image_token_id": 0}
        if getattr(config, "image_token_id", None) is None:
            config.image_token_id = 0

    # --- ClvpModel: needs vocab_size + hidden_size + max_position_embeddings ---
    if name_lower == "clvpmodel":
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            if not hasattr(config, "vocab_size") or config.vocab_size is None:
                config.vocab_size = getattr(text_cfg, "vocab_size", 256)
            if not hasattr(config, "hidden_size") or getattr(config, "hidden_size", None) is None:
                config.hidden_size = getattr(text_cfg, "hidden_size", 768)
            if not hasattr(config, "max_position_embeddings") or getattr(config, "max_position_embeddings", None) is None:
                config.max_position_embeddings = 512

    # --- DbrxModel: attention config needs rope_theta, clip_qkv, and hidden_size sync ---
    if name_lower == "dbrxmodel":
        # Sync hidden_size with d_model so _reduce_model_size can reduce both
        if hasattr(config, "d_model") and config.d_model:
            config.hidden_size = config.d_model
        attn_cfg = getattr(config, "attn_config", None)
        if attn_cfg and not hasattr(attn_cfg, "rope_theta"):
            attn_cfg.rope_theta = 500000.0
        if attn_cfg and getattr(attn_cfg, "clip_qkv", None) is None:
            attn_cfg.clip_qkv = 8.0

    # --- HunYuan / Ministral: rope_theta + head_dim are None ---
    if name_lower in ("hunyuandensev1model", "hunyuanmoev1model", "ministralmodel"):
        if getattr(config, "rope_theta", None) is None:
            config.rope_theta = 10000.0
        if getattr(config, "head_dim", None) is None:
            hs = getattr(config, "hidden_size", 4096)
            nh = getattr(config, "num_attention_heads", 32)
            config.head_dim = hs // nh

    # --- Table-driven None config fixes ---
    # Models where default config has None for required fields.
    # Format: model_name → {attr: default_value}
    _CONFIG_NONE_DEFAULTS = {
        "ministral3model": {"rope_theta": 10000.0},
        "nemotronmodel": {"partial_rotary_factor": 0.5},
        "moonshinestreamingmodel": {},  # num_key_value_heads handled below
        "pix2structtextmodel": {"initializer_range": 0.02},
        "dots1model": {"hidden_size": 768, "n_routed_experts": 4,
                       "n_shared_experts": 1, "num_experts_per_tok": 2},
        "esmmodel": {"vocab_size": 33, "position_embedding_type": "absolute",
                     "emb_layer_norm_before": False, "pad_token_id": 0},
        "bayesiandetectormodel": {"watermarking_depth": 3},
        "gptneoxmodel": {"rotary_pct": 0.25, "rotary_emb_base": 10000},
        "deepseekv2model": {"num_experts_per_tok": 6, "n_group": 1, "topk_group": 1},
        "lfm2moemodel": {"num_local_experts": 8, "num_experts_per_tok": 2},
    }
    defaults = _CONFIG_NONE_DEFAULTS.get(name_lower, {})
    for attr, default in defaults.items():
        if getattr(config, attr, None) is None:
            setattr(config, attr, default)

    # num_key_value_heads defaults to num_attention_heads for several models
    if name_lower in ("nemotronmodel", "moonshinestreamingmodel"):
        if getattr(config, "num_key_value_heads", None) is None:
            config.num_key_value_heads = getattr(config, "num_attention_heads", 8)

    # --- Idefics3Model, SmolVLMModel: padding_idx within num_embeddings ---
    if name_lower in ("idefics3model", "smolvlmmodel"):
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            vocab = getattr(text_cfg, "vocab_size", 32000)
            pad = getattr(text_cfg, "pad_token_id", None)
            if pad is not None and pad >= vocab:
                text_cfg.vocab_size = pad + 1

    # --- Qwen3OmniMoeTalkerModel: missing pad_token_id + vocab_size + num_hidden_layers ---
    if name_lower in ("qwen3omnioetalkermodel", "qwen3omnimoetalkermodel"):
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 0
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            if getattr(config, "vocab_size", None) is None:
                config.vocab_size = getattr(text_cfg, "vocab_size", 3072)
            if getattr(config, "num_hidden_layers", None) is None:
                config.num_hidden_layers = getattr(text_cfg, "num_hidden_layers", 20)

    # --- Blip2Model / InstructBlipModel / InstructBlipVideoModel: image_token_index is None ---
    if name_lower in ("blip2model", "instructblipmodel", "instructblipvideomodel"):
        if getattr(config, "image_token_index", None) is None:
            config.image_token_index = 2
        if name_lower == "instructblipvideomodel":
            if getattr(config, "image_token_id", None) is None:
                config.image_token_id = getattr(config, "image_token_index", 2)
            if getattr(config, "video_token_index", None) is None:
                config.video_token_index = getattr(config, "image_token_index", 2)

    # --- IdeficsModel: use_resampler must be True when perceiver_embeddings are passed ---
    if name_lower == "ideficsmodel":
        config.use_resampler = True

    # --- Lfm2MoeModel: layer_types needs "attention" entries ---
    if name_lower == "lfm2moemodel":
        num_layers = getattr(config, "num_hidden_layers", 2)
        if getattr(config, "layer_types", None) is None:
            config.layer_types = ["attention"] * num_layers

    # --- DeepseekV2: align moe_intermediate_size for grouped_mm stride constraint ---
    # grouped_mm requires strides to be multiples of 16 bytes (8 elements in bf16).
    if "deepseekv2" in name_lower:
        moe_is = getattr(config, "moe_intermediate_size", None)
        if moe_is is not None and moe_is % 8 != 0:
            config.moe_intermediate_size = ((moe_is + 7) // 8) * 8

    # --- Qwen VL family: mrope_section missing from rope_parameters ---
    # Models that use multi-resolution rotary position embeddings
    _qwen_vl_names = (
        "qwen2vlmodel", "qwen2vltextmodel",
        "qwen2_5_vlmodel", "qwen2_5_vltextmodel",
        "qwen3vlmodel", "qwen3vlmoemodel",
        "qwen3_5moemodel", "qwen3_5moetextmodel",
        "qwen3_5model", "qwen3_5textmodel",
        "qwen2_5omnitalkermodel",
        "paddleocrtextmodel", "paddleocrvlmodel",
    )
    if name_lower in _qwen_vl_names:
        # Fix on top-level config
        rp = getattr(config, "rope_parameters", None)
        if rp and isinstance(rp, dict) and "mrope_section" not in rp:
            rp["mrope_section"] = [16, 24, 24]
        # Fix on text_config (most VL models delegate to text sub-model)
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            rp = getattr(text_cfg, "rope_parameters", None)
            if rp and isinstance(rp, dict) and "mrope_section" not in rp:
                rp["mrope_section"] = [16, 24, 24]

    # --- Florence2Model (train create_error): patch_size is tuple ---
    if name_lower == "florence2model":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            ps = getattr(vision_cfg, "patch_size", None)
            if isinstance(ps, (list, tuple)):
                # DaViT uses list of patch sizes — keep as-is, the model handles it
                pass

    # --- Olmo2/3, OlmoHybrid: reduce size to avoid OOM ---
    if name_lower in ("olmo2model", "olmo3model", "olmohybridmodel"):
        config.num_hidden_layers = 2
        config.hidden_size = 1024
        config.intermediate_size = 2048
        config.num_attention_heads = 8
        config.num_key_value_heads = 4

    # --- DepthProModel: make config self-consistent for small patch_size ---
    # Default patch_size=384 requires 1536x1536 images. Reduce to 16.
    # Must also fix all dependent dims: _reduce_model_size doesn't recurse into
    # image_model_config/patch_model_config/fov_model_config (they're not named
    # vision_config/text_config), so we handle them here.
    if name_lower == "depthpromodel":
        config.patch_size = 16
        config.scaled_images_ratios = [1.0]
        config.scaled_images_overlap_ratios = [0.0]
        # Feature dims must match sub-config hidden_size (768 by default)
        hs = config.patch_model_config.hidden_size
        config.scaled_images_feature_dims = [hs]
        # Reduce sub-config layers (they default to 12, not reached by _reduce_model_size)
        for sub_attr in ("image_model_config", "patch_model_config", "fov_model_config"):
            sub = getattr(config, sub_attr, None)
            if sub is not None:
                if getattr(sub, "num_hidden_layers", 0) > 4:
                    sub.num_hidden_layers = 2
                # Sync image_size with reduced patch_size
                sub.image_size = config.patch_size
        # intermediate_hook_ids must be < num_hidden_layers (now 2)
        config.intermediate_hook_ids = [1]
        config.intermediate_feature_dims = [config.fusion_hidden_size]

    # --- UVDocModel: backbone must output all stages for bridge_connector ---
    # UVDocModel.forward() concatenates ALL backbone feature_maps along channel dim,
    # so the backbone must have out_features for every stage. Default config only
    # outputs the last stage, causing channel count mismatch in bridge_connector.
    if name_lower == "uvdocmodel":
        backbone_cfg = getattr(config, "backbone_config", None)
        if backbone_cfg:
            stage_configs = getattr(backbone_cfg, "stage_configs", None)
            if stage_configs:
                all_stages = [f"stage{i+1}" for i in range(len(stage_configs))]
                backbone_cfg.out_features = all_stages
                backbone_cfg.out_indices = list(range(1, len(stage_configs) + 1))

    # --- Llama4VisionModel: intermediate_size must equal hidden_size / pixel_shuffle_ratio^2 ---
    # Llama4VisionMLP2.fc1 takes pixel-shuffled features as input, whose channel count
    # is hidden_size / pixel_shuffle_ratio^2. But the default config has intermediate_size=5632
    # which doesn't match hidden_size=768 / 0.25 = 3072.
    if name_lower == "llama4visionmodel":
        psr = getattr(config, "pixel_shuffle_ratio", 0.5)
        hs = getattr(config, "hidden_size", 768)
        config.intermediate_size = int(hs / (psr ** 2))
    # Also fix for Llama4Model (ForConditionalGeneration variant)
    if name_lower == "llama4model":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg and hasattr(vision_cfg, "pixel_shuffle_ratio"):
            psr = getattr(vision_cfg, "pixel_shuffle_ratio", 0.5)
            hs = getattr(vision_cfg, "hidden_size", 768)
            vision_cfg.intermediate_size = int(hs / (psr ** 2))
            # Align vision_output_dim to projector_output_dim so
            # multi_modal_projector(vision_adapter_output) has matching dims
            pod = getattr(vision_cfg, "projector_output_dim", None)
            if pod is not None:
                vision_cfg.vision_output_dim = pod

    # --- AriaModel: projector_patch_to_query_dict must include actual patch count ---
    # Default vision config produces 49 patches (224/32=7, 7*7=49), but
    # projector_patch_to_query_dict only has keys {1225, 4900}. Add entry for 49.
    if name_lower == "ariamodel":
        ptqd = getattr(config, "projector_patch_to_query_dict", None)
        if ptqd is not None:
            vision_cfg = getattr(config, "vision_config", None)
            if vision_cfg:
                img_size = getattr(vision_cfg, "image_size", 224)
                ps = getattr(vision_cfg, "patch_size", 32)
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                if isinstance(ps, (list, tuple)):
                    ps = ps[0]
                num_patches = (img_size // ps) ** 2
                if num_patches not in ptqd:
                    # Use smallest existing query count as baseline
                    ptqd[num_patches] = min(ptqd.values())

    # --- LwDetrModel: projector_scale_factors is empty by default → 0 feature levels ---
    # The empty projector produces no feature maps, causing "expected a non-empty list of
    # Tensors" in torch.cat. Also reduce backbone layers and fix out_indices/out_features.
    if name_lower == "lwdetrmodel":
        if not getattr(config, "projector_scale_factors", None):
            config.projector_scale_factors = [1.0]
            config.num_feature_levels = 1
            config.projector_in_channels = [config.d_model]
        backbone_cfg = getattr(config, "backbone_config", None)
        if backbone_cfg:
            n = getattr(backbone_cfg, "num_hidden_layers", 10)
            if n > 4:
                backbone_cfg.num_hidden_layers = 2
                backbone_cfg.out_indices = [1, 2]
                backbone_cfg.out_features = ["stage1", "stage2"]
            backbone_cfg.image_size = 256
        # Reduce num_queries and group_detr to avoid topk overflow
        config.num_queries = 10
        config.group_detr = 1

    # --- Qwen VL family: vision out_hidden_size must match text_config.hidden_size ---
    # Default configs have out_hidden_size=3584 (from pretrained weights) but text_config
    # defaults to a different hidden_size. The merger FC2 projects to out_hidden_size,
    # but inputs_embeds uses text_config.hidden_size → numel mismatch in forward.
    vision_cfg = getattr(config, "vision_config", None)
    text_cfg = getattr(config, "text_config", None)
    if vision_cfg and text_cfg:
        ohs = getattr(vision_cfg, "out_hidden_size", None)
        ths = getattr(text_cfg, "hidden_size", None)
        if ohs is not None and ths is not None and ohs != ths:
            vision_cfg.out_hidden_size = ths

    # --- Gemma3ForConditionalGeneration: mm_tokens_per_image must match vision image_size ---
    # Default mm_tokens_per_image=256 (tokens_per_side=16) with image_size=224 (patches=14)
    # gives kernel_size = patches_per_image // tokens_per_side = 14 // 16 = 0 → AvgPool2d(0) crash.
    # Fix: set mm_tokens_per_image = patches_per_image^2 so kernel=1 (no pooling).
    if name_lower == "gemma3model":
        vision_cfg = getattr(config, "vision_config", None)
        mm_tok = getattr(config, "mm_tokens_per_image", None)
        if vision_cfg and mm_tok:
            img_sz = getattr(vision_cfg, "image_size", 224) or 224
            ps = getattr(vision_cfg, "patch_size", 16)
            patches_per_image = img_sz // ps
            tokens_per_side = int(mm_tok ** 0.5)
            if tokens_per_side > patches_per_image:
                # Reduce mm_tokens_per_image to match image_size
                config.mm_tokens_per_image = patches_per_image ** 2

    # --- Gemma4ForConditionalGeneration: default config has vision_config=None ---
    # Gemma4 is multimodal but its default Gemma4Config omits vision_config.
    # Construct one so ForConditionalGeneration gets proper multimodal inputs.
    if name_lower == "gemma4model":
        if getattr(config, "vision_config", None) is None:
            try:
                import transformers
                config.vision_config = transformers.Gemma4VisionConfig(
                    num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                    num_attention_heads=2, num_key_value_heads=2,
                )
            except Exception:
                pass

    # --- Mistral3/LightOnOcr: default image_size=1540 creates too many patches ---
    # (1540/14)^2 = 12100 patches → 3025 features per image. Cap to 224.
    if name_lower in ("mistral3model", "lightonocrmodel"):
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            vision_cfg.image_size = 224

    # --- Idefics3/SmolVLM/AyaVision/Cohere2Vision: pixel_shuffle needs patches_per_side % scale == 0 ---
    # Default image_size=224, patch_size=32 → 7 patches per side, but scale_factor=2 needs even count.
    # Set image_size=256 → 8 patches per side → 8 % 2 == 0 ✓
    _pixel_shuffle_models = ("idefics3model", "smolvlmmodel", "cohere2visionmodel")
    if name_lower in _pixel_shuffle_models:
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            ps = getattr(vision_cfg, "patch_size", 32)
            if isinstance(ps, (list, tuple)):
                ps = ps[0]
            scale = getattr(config, "scale_factor", getattr(config, "downsample_factor", 2))
            img_sz = getattr(vision_cfg, "image_size", 224) or 224
            patches_per_side = img_sz // ps if ps > 0 else 7
            if patches_per_side % scale != 0:
                # Round up to next multiple of scale * patch_size
                new_patches = ((patches_per_side + scale - 1) // scale) * scale
                vision_cfg.image_size = new_patches * ps

    # --- AyaVision: pixel_shuffle needs patches_per_side % downsample_factor == 0 ---
    # Default image_size=384, patch_size=14 → 27 patches per side, 27 % 2 != 0.
    if name_lower == "ayavisionmodel":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            ps = getattr(vision_cfg, "patch_size", 14)
            if isinstance(ps, (list, tuple)):
                ps = ps[0]
            scale = getattr(config, "downsample_factor", 2)
            img_sz = getattr(vision_cfg, "image_size", 384) or 384
            patches_per_side = img_sz // ps if ps > 0 else 27
            if patches_per_side % scale != 0:
                new_patches = ((patches_per_side + scale - 1) // scale) * scale
                vision_cfg.image_size = new_patches * ps

    # --- Generic size reduction for all models ---
    # For graph break detection, 2 layers is sufficient.
    # This prevents create-phase timeouts for large models.
    _reduce_model_size(config)

    # ── Post-reduction fixes ──
    # These must run after _reduce_model_size() because they depend on final dimensions.

    # --- Glm4v/GlmImage family: mrope_section missing from rope_parameters ---
    # Must be post-reduction: _reduce_model_size may change num_attention_heads
    # (e.g. Glm4vMoe: 96 heads don't divide 4096, gets reduced to 16),
    # which changes head_dim and thus the required mrope_section sum.
    _glm_mrope_names = (
        "glm4vmodel", "glm4vtextmodel", "glm4vmoemodel", "glm4vmoetextmodel",
        "glmimagemodel", "glmimagetextmodel",
        "glm46vmodel",
    )
    if name_lower in _glm_mrope_names:
        for cfg_obj in [config, getattr(config, "text_config", None)]:
            if cfg_obj is None:
                continue
            rp = getattr(cfg_obj, "rope_parameters", None)
            if rp and isinstance(rp, dict):
                # Compute correct mrope_section sum = rotary_dim = head_dim * prf / 2
                head_dim = getattr(cfg_obj, "head_dim", None)
                if head_dim is None:
                    hs = getattr(cfg_obj, "hidden_size", 4096)
                    nh = getattr(cfg_obj, "num_attention_heads", 32)
                    head_dim = hs // nh
                prf = rp.get("partial_rotary_factor", 1.0)
                dim = int(head_dim * prf)
                target = dim // 2  # inv_freq uses arange(0, dim, 2)
                # Scale default [8,12,12] proportionally to target
                rp["mrope_section"] = [target * 8 // 32, target * 12 // 32, target - target * 8 // 32 - target * 12 // 32]


    # --- Jamba: attn_layer_period must be ≤ num_hidden_layers ---
    if name_lower == "jambamodel":
        n = getattr(config, "num_hidden_layers", 2)
        if getattr(config, "attn_layer_period", n + 1) > n:
            config.attn_layer_period = 1
            config.attn_layer_offset = 0

    # --- Lfm2Model: needs 'conv' in layer_types ---
    if name_lower == "lfm2model":
        lt = getattr(config, "layer_types", None)
        if lt and isinstance(lt, list) and "conv" not in lt:
            config.layer_types[0] = "conv"

    # --- Lfm2MoeModel: needs both 'conv' and 'full_attention' in layer_types ---
    if name_lower == "lfm2moemodel":
        lt = getattr(config, "layer_types", None)
        if lt and isinstance(lt, list):
            config.layer_types = ["conv", "full_attention"][:len(lt)]

    # --- Lfm2VlModel: text_config needs 'conv' in layer_types ---
    if name_lower == "lfm2vlmodel":
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            lt = getattr(text_cfg, "layer_types", None)
            if lt and isinstance(lt, list) and "conv" not in lt:
                text_cfg.layer_types[0] = "conv"

    # --- DbrxModel: sync d_model and FFN config with reduced hidden_size ---
    if name_lower == "dbrxmodel":
        hs = getattr(config, "hidden_size", None)
        if hs:
            config.d_model = hs
            ffn_cfg = getattr(config, "ffn_config", None)
            if ffn_cfg:
                ffn_cfg.ffn_hidden_size = hs

    # --- LongcatFlashModel: force num_key_value_heads = num_attention_heads after reduction ---
    # LongcatFlash uses Multi-head Latent Attention (MLA): kv_b_proj always outputs num_attention_heads
    # KV heads (not num_key_value_heads). After _reduce_model_size reduces num_key_value_heads
    # independently (e.g. 64→8), the generic attention function applies repeat_kv incorrectly,
    # turning 16 KV heads into 32 before SDPA which then fails (query has 16 heads, kv has 32).
    # Fix: align num_key_value_heads to num_attention_heads post-reduction.
    if name_lower == "longcatflashmodel":
        nh = getattr(config, "num_attention_heads", None)
        if nh:
            config.num_key_value_heads = nh

    # --- GlmMoeDsa: MLA attention — same fix as LongcatFlash ---
    if "glmmoedsa" in name_lower:
        nh = getattr(config, "num_attention_heads", None)
        if nh:
            config.num_key_value_heads = nh

    # --- RecurrentGemmaModel: ensure block_types includes 'attention' after num_hidden_layers reduction ---
    # The model uses layers_block_type = (block_types * 100)[:num_hidden_layers].
    # Default block_types = ['recurrent', 'recurrent', 'attention'] — with num_hidden_layers=2,
    # all 'attention' blocks are cut, and config.layers_block_type.index('attention') raises ValueError.
    # Fix: set block_types = ['recurrent', 'attention'] so the pattern includes attention at 2 layers.
    if name_lower == "recurrentgemmamodel":
        config.block_types = ["recurrent", "attention"]

    # --- VipLlava: vision_feature_layers indices must be valid after layer reduction ---
    # Default (-2, -5, -8, -11, 6) — all out of range with 2 vision layers (3 hidden_states).
    if name_lower == "vipllavamodel":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            n_layers = getattr(vision_cfg, "num_hidden_layers", 2)
            config.vision_feature_layers = [-1, -2]

    # --- Mllama: intermediate_layers_indices must be valid after layer reduction ---
    # Default [3, 7, 15, 23, 30] — all out of range with 2 vision layers.
    # vision_output_dim = hidden_size * (len(intermediate_layers_indices) + 1)
    if name_lower == "mllamamodel":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            n_layers = getattr(vision_cfg, "num_hidden_layers", 2)
            valid_indices = [i for i in range(n_layers)]  # [0, 1]
            vision_cfg.intermediate_layers_indices = valid_indices
            # Sync vision_output_dim — stored attr, not auto-computed
            v_hs = getattr(vision_cfg, "hidden_size", 1280)
            vision_cfg.vision_output_dim = v_hs * (len(valid_indices) + 1)
            # Also reduce num_global_layers
            if getattr(vision_cfg, "num_global_layers", 0) > 2:
                vision_cfg.num_global_layers = 2

    # --- Gemma3n: list-typed configs and num_kv_shared_layers need post-reduction sync ---
    if name_lower == "gemma3nmodel":
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            n = getattr(text_cfg, "num_hidden_layers", 2)
            # num_kv_shared_layers must be <= num_hidden_layers
            if getattr(text_cfg, "num_kv_shared_layers", 0) > n:
                text_cfg.num_kv_shared_layers = 0
            # Truncate per-layer lists
            for attr in ("intermediate_size", "activation_sparsity_pattern"):
                val = getattr(text_cfg, attr, None)
                if isinstance(val, list) and len(val) > n:
                    setattr(text_cfg, attr, val[:n])

    # --- Qwen2VL / Qwen2_5_VL: merger output must match text_config.hidden_size ---
    # Qwen2VL: merger uses vision_config.hidden_size directly (no out_hidden_size).
    # Qwen2_5_VL: merger uses PatchMerger(dim=out_hidden_size, context_dim=hidden_size).
    # After reduction, text_config.hidden_size may differ.
    if name_lower in ("qwen2vlmodel", "qwen2_5_vlmodel"):
        text_cfg = getattr(config, "text_config", None)
        vision_cfg = getattr(config, "vision_config", None)
        if text_cfg and vision_cfg:
            v_hs = getattr(vision_cfg, "hidden_size", None)
            t_hs = getattr(text_cfg, "hidden_size", None)
            if v_hs and t_hs and v_hs != t_hs:
                # Set text hidden_size to match vision (don't change vision, it controls merger)
                text_cfg.hidden_size = v_hs
                # Qwen2_5_VL: also align out_hidden_size (merger output dim)
                if getattr(vision_cfg, "out_hidden_size", None) is not None:
                    vision_cfg.out_hidden_size = v_hs
                if getattr(text_cfg, "intermediate_size", None):
                    text_cfg.intermediate_size = max(text_cfg.intermediate_size, v_hs * 2)
                # Must preserve original head_dim for mrope_section compatibility.
                # mrope_section is designed for the original head_dim (typically 128).
                # Find num_attention_heads that gives original head_dim.
                rp = getattr(text_cfg, "rope_scaling", None)
                mrope = rp.get("mrope_section", None) if isinstance(rp, dict) else None
                if mrope:
                    target_head_dim = sum(mrope) * 2  # mrope uses head_dim/2
                    heads = v_hs // target_head_dim
                    if heads > 0 and v_hs % target_head_dim == 0:
                        text_cfg.num_attention_heads = heads
                        nkv = getattr(text_cfg, "num_key_value_heads", None)
                        if nkv and nkv > heads:
                            text_cfg.num_key_value_heads = heads
                else:
                    # No mrope — just find valid heads
                    nh = getattr(text_cfg, "num_attention_heads", None)
                    if nh and v_hs % nh != 0:
                        for heads in (16, 8, 4):
                            if v_hs % heads == 0:
                                text_cfg.num_attention_heads = heads
                                nkv = getattr(text_cfg, "num_key_value_heads", None)
                                if nkv and nkv > heads:
                                    text_cfg.num_key_value_heads = heads
                                break
                nkv = getattr(text_cfg, "num_key_value_heads", None)
                nh = getattr(text_cfg, "num_attention_heads", None)
                if nh and nkv and nkv > 0 and nh % nkv != 0:
                    for k in range(nkv, 0, -1):
                        if nh % k == 0:
                            text_cfg.num_key_value_heads = k
                            break

    # --- Cohere2Vision: alignment_intermediate_size must work with reduced dims ---
    if name_lower == "cohere2visionmodel":
        vision_cfg = getattr(config, "vision_config", None)
        text_cfg = getattr(config, "text_config", None)
        if vision_cfg and text_cfg:
            v_hs = getattr(vision_cfg, "hidden_size", 1152)
            t_hs = getattr(text_cfg, "hidden_size", None)
            scale = getattr(config, "downsample_factor", 2)
            # Projector input = v_hs * scale^2, output = alignment_intermediate_size // 2 (SwiGLU)
            # Then final linear: alignment_intermediate_size // 2 → text_hidden_size
            if t_hs:
                config.alignment_intermediate_size = t_hs * 2  # SwiGLU halves it

    # --- Lfm2Vl: vision projector expects vision_config.hidden_size → text_config.hidden_size ---
    if name_lower == "lfm2vlmodel":
        vision_cfg = getattr(config, "vision_config", None)
        text_cfg = getattr(config, "text_config", None)
        if vision_cfg and text_cfg:
            v_hs = getattr(vision_cfg, "hidden_size", None)
            t_hs = getattr(text_cfg, "hidden_size", None)
            if v_hs and t_hs and v_hs != t_hs:
                # Align text to vision since vision encoder is not reduced
                text_cfg.hidden_size = v_hs
                if getattr(text_cfg, "intermediate_size", None):
                    text_cfg.intermediate_size = max(text_cfg.intermediate_size, v_hs * 2)
                nh = getattr(text_cfg, "num_attention_heads", None)
                if nh and v_hs % nh != 0:
                    for heads in (16, 8, 4):
                        if v_hs % heads == 0:
                            text_cfg.num_attention_heads = heads
                            nkv = getattr(text_cfg, "num_key_value_heads", None)
                            if nkv and nkv > heads:
                                text_cfg.num_key_value_heads = heads
                            break

    # --- Ovis2: config.hidden_size (visual embedding table) must match text_config.hidden_size ---
    if name_lower == "ovis2model":
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            t_hs = getattr(text_cfg, "hidden_size", None)
            if t_hs and t_hs != getattr(config, "hidden_size", None):
                config.hidden_size = t_hs

    # --- FastVlm: image_seq_length is wrong for reduced vision output ---
    if name_lower == "fastvlmmodel":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            # With timm backbone after reduction, actual feature count depends on model.
            # Set image_seq_length to a small valid value so token count matches.
            config.image_seq_length = 1

    # --- LlavaNext/LlavaOnevision/LlavaNextVideo: simplify grid_pinpoints ---
    # pack_image_features does spatial unpadding + newline insertion that changes
    # the feature count from the naïve num_patches * patches_per_image.
    # Simplify to single-tile grid so the computation is predictable.
    _llava_grid_models = ("llavanextmodel", "llavaonevisionmodel", "llavanextvideomodel")
    if name_lower in _llava_grid_models:
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            isz = getattr(vision_cfg, "image_size", 336) or 336
            config.image_grid_pinpoints = [[isz, isz]]

    # --- Qwen OmniThinker models: missing vision_start/end_token_id ---
    # Config defaults lack these but the model's position embedding code requires them.
    _qwen_omni = ("qwen3omnimoethinkermodel", "qwen2_5omnithinkermodel")
    if name_lower in _qwen_omni:
        if not hasattr(config, "vision_start_token_id") or config.vision_start_token_id is None:
            config.vision_start_token_id = 151652
            config.vision_end_token_id = 151653
        # Qwen2_5OmniThinker also needs mrope_section in rope_scaling
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            rp = getattr(text_cfg, "rope_scaling", None)
            if isinstance(rp, dict) and "mrope_section" not in rp:
                rp["mrope_section"] = [16, 24, 24]

    # --- Gemma3n: num_kv_shared_layers must be < num_hidden_layers ---
    # Default num_kv_shared_layers=15 but after reduction to 2 layers, first_kv_shared
    # becomes negative → prev_layers is empty → .index() crashes.
    if "gemma3n" in name_lower:
        nkv = getattr(config, "num_kv_shared_layers", 0)
        nhl = getattr(config, "num_hidden_layers", 35)
        if nkv >= nhl:
            config.num_kv_shared_layers = max(nhl - 2, 0)

    # --- Qwen3OmniMoe family: missing attributes on talker sub-config ---
    # When called as Qwen3OmniMoeTalkerModel, config IS the talker config.
    # When called as Qwen3OmniMoeModel, talker is at config.talker_config.
    def _fix_qwen3_omni_talker(talker_cfg):
        text_cfg = getattr(talker_cfg, "text_config", None)
        if text_cfg and not hasattr(text_cfg, "shared_expert_intermediate_size"):
            text_cfg.shared_expert_intermediate_size = getattr(text_cfg, "intermediate_size", 2048)
        if not hasattr(talker_cfg, "spatial_merge_size") or getattr(talker_cfg, "spatial_merge_size", None) is None:
            talker_cfg.spatial_merge_size = 2

    if "qwen3omnimoetalk" in name_lower:
        _fix_qwen3_omni_talker(config)
    if "qwen3omnimoe" in name_lower and hasattr(config, "talker_config"):
        _fix_qwen3_omni_talker(config.talker_config)

    # --- Qwen3OmniMoeForConditionalGeneration: top-level config lacks initializer_range ---
    # Sub-configs (thinker, talker, code2wav) all have it, but the parent doesn't.
    if "qwen3omnimoe" in name_lower:
        if not hasattr(config, "initializer_range") or getattr(config, "initializer_range", None) is None:
            # Use __dict__ to bypass @strict decorator that blocks setattr
            config.__dict__["initializer_range"] = 0.02

    # --- ZambaModel: use_mamba_kernels=False (no causal-conv1d), layers_block_type ---
    if "zamba" in name_lower and "zamba2" not in name_lower:
        if getattr(config, "use_mamba_kernels", True):
            config.use_mamba_kernels = False
        # Zamba's _tied_weights_keys hardcodes layer 2 as shared transformer source.
        # Need at least 4 layers with layer 2 and one other hybrid layer.
        n = getattr(config, "num_hidden_layers", 76)
        lbt = getattr(config, "layers_block_type", None)
        if lbt and n < len(lbt):
            # Already truncated by _reduce_model_size — ensure at least 2 hybrid layers
            hybrid_count = sum(1 for t in lbt if t == "hybrid")
            if hybrid_count < 2:
                # Expand to include second hybrid layer (index 7 in default pattern)
                full_lbt = getattr(type(config)(), "layers_block_type", lbt)
                # Find first two hybrid indices
                hybrids = [i for i, t in enumerate(full_lbt) if t == "hybrid"]
                if len(hybrids) >= 2:
                    needed = hybrids[1] + 1
                    config.num_hidden_layers = needed
                    config.layers_block_type = list(full_lbt[:needed])

    # --- Zamba2Model: needs enough layers to include at least one hybrid layer ---
    # Zamba2's HybridMambaAttentionDynamicCache.get_seq_length accesses
    # self.transformer_layers[0] which is empty if no hybrid layers exist.
    # Default has 54 layers with first hybrid at index 6 — need at least 7 layers.
    # Also: hybrid_layer_ids is a separate config field that Zamba2MLP uses to build
    # layer_dic — after reducing num_hidden_layers we must keep it consistent with
    # layers_block_type or forward() raises KeyError on the new hybrid index.
    if "zamba2" in name_lower:
        lbt = getattr(config, "layers_block_type", None)
        if lbt:
            hybrid_idxs = [i for i, t in enumerate(lbt) if t == "hybrid"]
            if not hybrid_idxs:
                # No hybrid layers after reduction — expand to include first one
                full_lbt = getattr(type(config)(), "layers_block_type", lbt)
                full_hybrid_idxs = [i for i, t in enumerate(full_lbt) if t == "hybrid"]
                if full_hybrid_idxs:
                    needed = full_hybrid_idxs[0] + 1  # include first hybrid
                    config.num_hidden_layers = needed
                    config.layers_block_type = list(full_lbt[:needed])
                    hybrid_idxs = [full_hybrid_idxs[0]]
            if hasattr(config, "hybrid_layer_ids"):
                config.hybrid_layer_ids = hybrid_idxs

    # --- xLSTM: num_blocks must match num_hidden_layers after reduction ---
    if "xlstm" in name_lower:
        config.num_blocks = getattr(config, "num_hidden_layers", 2)

    # --- PPOCRV5ServerDetModel: multiple None configs by default ---
    # Each conv entry is [kernel_size, stride, padding]. All convs in a group are
    # summed together, so they must produce the same output shape → stride=1, pad=(k-1)/2.
    if "ppocrv5" in name_lower and "serverdet" in name_lower.replace("_", ""):
        if getattr(config, "intraclass_block_config", None) is None:
            config.intraclass_block_config = {
                "reduce_channel": [1, 1, 0],
                "vertical_long_to_small_conv_longratio": [[7, 1], 1, [3, 0]],
                "vertical_long_to_small_conv_midratio": [[5, 1], 1, [2, 0]],
                "vertical_long_to_small_conv_shortratio": [[3, 1], 1, [1, 0]],
                "horizontal_small_to_long_conv_longratio": [[1, 7], 1, [0, 3]],
                "horizontal_small_to_long_conv_midratio": [[1, 5], 1, [0, 2]],
                "horizontal_small_to_long_conv_shortratio": [[1, 3], 1, [0, 1]],
                "symmetric_conv_long_longratio": [7, 1, 3],
                "symmetric_conv_long_midratio": [5, 1, 2],
                "symmetric_conv_long_shortratio": [3, 1, 1],
                "return_channel": [1, 1, 0],
            }
        if getattr(config, "scale_factor_list", None) is None:
            config.scale_factor_list = [1, 2, 4, 8]
        if getattr(config, "kernel_list", None) is None:
            config.kernel_list = [7, 5, 3, 3]

    return config


def _reduce_model_size(config):
    """Reduce model size to prevent create-phase timeouts.

    Graph break behavior is determined by the model architecture (ops used),
    not the model depth/width. 2 layers is sufficient for detection.

    NOTE: Some configs encode the layer block layout in *multiple* parallel
    fields (e.g. Zamba2 has both `layers_block_type` and `hybrid_layer_ids`,
    Jamba has `layers_block_type` + `attn_layer_period` + `expert_layer_period`).
    Reducing num_hidden_layers + truncating layers_block_type here will leave
    those parallel fields stale and out of sync. Sync them in `_fix_config`
    AFTER this function runs (search "hybrid_layer_ids" for the Zamba2 example).
    """
    # Reduce layers (skip non-int values like tuples)
    for attr in ("num_hidden_layers", "num_layers", "n_layer", "n_layers",
                 "encoder_layers", "decoder_layers", "num_encoder_layers",
                 "num_decoder_layers"):
        val = getattr(config, attr, None)
        if val is not None and isinstance(val, int) and val > 4:
            try:
                setattr(config, attr, 2)
            except (ValueError, AttributeError, NotImplementedError):
                pass  # Some configs reject certain attributes (e.g. ProphetNet)

    # Truncate layer_types / layers_block_type to match reduced num_hidden_layers.
    # Must include all unique types from the original list (hybrid models need both
    # linear_attention AND full_attention layers to initialize caches correctly).
    for cfg_obj in [config] + [getattr(config, sub, None) for sub in ("text_config",)]:
        if cfg_obj is None:
            continue
        n_layers = getattr(cfg_obj, "num_hidden_layers", None)
        if not isinstance(n_layers, int):
            continue
        for lt_attr in ("layer_types", "layers_block_type"):
            lt = getattr(cfg_obj, lt_attr, None)
            if isinstance(lt, (list, tuple)) and len(lt) > n_layers:
                all_types = list(dict.fromkeys(lt))  # unique, order-preserving
                if len(all_types) > n_layers:
                    # Need more layers to fit all types — expand
                    n_layers = len(all_types)
                    cfg_obj.num_hidden_layers = n_layers
                # One of each type, then repeat last to fill
                new_lt = list(all_types)
                while len(new_lt) < n_layers:
                    new_lt.append(all_types[-1])
                setattr(cfg_obj, lt_attr, new_lt)

    # Funnel-family: block_sizes must sum to num_hidden_layers
    block_sizes = getattr(config, "block_sizes", None)
    if block_sizes is not None and isinstance(block_sizes, (list, tuple)):
        config.block_sizes = [1, 1]  # 2 layers = 2 blocks of 1
        for _attr, _val in [("num_hidden_layers", 2), ("num_blocks", 2)]:
            try:
                if hasattr(config, _attr):
                    setattr(config, _attr, _val)
            except (ValueError, AttributeError, NotImplementedError):
                pass

    # Reduce hidden size for very large models (> 4096)
    # Skip if hidden_size is a tuple/list (Swin-family uses per-stage sizes)
    _hs = getattr(config, "hidden_size", 0)
    if isinstance(_hs, int) and _hs > 4096:
        scale = config.hidden_size / 1024
        config.hidden_size = 1024
        # Scale down dependent dimensions proportionally
        if getattr(config, "intermediate_size", None):
            config.intermediate_size = max(config.intermediate_size // int(scale), 2048)
        if getattr(config, "num_attention_heads", None):
            config.num_attention_heads = max(config.num_attention_heads // int(scale), 4)
        if getattr(config, "num_key_value_heads", None):
            config.num_key_value_heads = min(
                getattr(config, "num_key_value_heads", 4),
                config.num_attention_heads
            )

    # Ensure hidden_size is divisible by num_attention_heads after reduction
    # Skip if either is a tuple/list (Swin-family models use per-stage tuples)
    hs = getattr(config, "hidden_size", None)
    nh = getattr(config, "num_attention_heads", None)
    if hs and nh and isinstance(hs, int) and isinstance(nh, int) and hs % nh != 0:
        # Find largest power-of-2 divisor of hs that's >= 4
        for heads in (16, 8, 4):
            if hs % heads == 0:
                config.num_attention_heads = heads
                nkv = getattr(config, "num_key_value_heads", None)
                if nkv is not None and nkv > heads:
                    config.num_key_value_heads = heads
                break

    # Ensure num_attention_heads is divisible by num_key_value_heads
    nh = getattr(config, "num_attention_heads", None)
    nkv = getattr(config, "num_key_value_heads", None)
    if nh and nkv and isinstance(nh, int) and isinstance(nkv, int) and nkv > 0 and nh % nkv != 0:
        # Find largest divisor of nh that's <= nkv
        for k in range(nkv, 0, -1):
            if nh % k == 0:
                config.num_key_value_heads = k
                break

    # Reduce MoE expert count (each expert is a full FFN layer)
    for attr in ("num_local_experts", "num_experts", "n_routed_experts"):
        val = getattr(config, attr, None)
        if val is not None and isinstance(val, int) and val > 4:
            setattr(config, attr, 4)

    # After reducing experts, ensure topk doesn't exceed available expert count
    for expert_attr in ("num_local_experts", "num_experts", "n_routed_experts"):
        expert_count = getattr(config, expert_attr, None)
        if expert_count is not None and isinstance(expert_count, int):
            for topk_attr in ("num_experts_per_tok", "num_selected_experts", "top_k"):
                topk_val = getattr(config, topk_attr, None)
                if topk_val is not None and isinstance(topk_val, int) and topk_val > expert_count:
                    setattr(config, topk_attr, expert_count)

    # Reduce sub-configs (vision, text, etc.)
    for sub_attr in ("text_config", "decoder", "encoder",
                     "audio_config", "speech_config"):
        sub = getattr(config, sub_attr, None)
        if sub is not None and hasattr(sub, "num_hidden_layers"):
            _reduce_model_size(sub)
    # Vision config: only reduce layers, not hidden_size (pooling depends on spatial dims)
    vision_sub = getattr(config, "vision_config", None)
    if vision_sub is not None:
        for attr in ("num_hidden_layers", "num_layers"):
            val = getattr(vision_sub, attr, None)
            if val is not None and isinstance(val, int) and val > 4:
                setattr(vision_sub, attr, 2)


def create_hf_model(spec, device, batch_size=DEFAULT_BATCH):
    import transformers

    model_name = spec.get("hf_class") or spec["name"]

    # For ForCausalLM / ForConditionalGeneration variants, derive the base model
    # name so config creation, input detection, and model-specific handlers all
    # match against the same keys they use for base models.
    variant = spec.get("variant")
    # Defensive: infer variant from class-name suffix when caller didn't set it.
    # Specs loaded from corpus.json or other downstream sources can drop this
    # field; without it config lookup falls through to the model class itself
    # and creation fails with "missing 1 required positional argument: config".
    if variant is None:
        if model_name.endswith("ForCausalLM"):
            variant = "causal_lm"
        elif model_name.endswith("ForConditionalGeneration"):
            variant = "conditional_generation"
    if variant == "causal_lm":
        base_model_name = model_name.replace("ForCausalLM", "Model")
    elif variant == "conditional_generation":
        base_model_name = model_name.replace("ForConditionalGeneration", "Model")
    else:
        base_model_name = model_name

    # Derive config from spec if available, otherwise from base model name
    config_name = spec.get("hf_config") or base_model_name.replace("Model", "Config")
    # Gemma4VisionModel.config_class incorrectly points to Gemma4Config (composite)
    # instead of Gemma4VisionConfig — use the correct vision-specific config.
    if config_name == "Gemma4Config" and "Vision" in model_name:
        config_name = "Gemma4VisionConfig"
    config_cls = getattr(transformers, config_name)
    model_cls = getattr(transformers, model_name)

    config = _create_config(base_model_name, config_cls)

    # For ForConditionalGeneration: preserve text_config.hidden_size before reduction
    # only if the vision merger uses out_hidden_size (which we've aligned to text hs).
    # Models WITHOUT out_hidden_size (e.g., Qwen2VL) use vision_config.hidden_size
    # as merger output, so text_config.hidden_size should match that after reduction.
    # Exclude Qwen2_5_VL: its out_hidden_size == vision_hidden_size, and the
    # post-reduction fix aligns text to vision — restoring would undo that.
    _orig_text_hs = None
    _skip_restore = base_model_name.lower() in ("qwen2_5_vlmodel",)
    if variant == "conditional_generation" and not _skip_restore:
        text_cfg = getattr(config, "text_config", None)
        vision_cfg_check = getattr(config, "vision_config",
                                   getattr(config, "image_config", None))
        if text_cfg and vision_cfg_check:
            has_out_hs = getattr(vision_cfg_check, "out_hidden_size", None) is not None
            if has_out_hs:
                _orig_text_hs = getattr(text_cfg, "hidden_size", None)

    config = _fix_config(base_model_name, config)

    # Restore text hidden_size for ForConditionalGeneration with out_hidden_size
    if _orig_text_hs is not None:
        text_cfg = getattr(config, "text_config", None)
        if text_cfg and getattr(text_cfg, "hidden_size", None) != _orig_text_hs:
            text_cfg.hidden_size = _orig_text_hs
            # Re-sync dependent dims that _reduce_model_size may have scaled
            if getattr(text_cfg, "intermediate_size", None):
                text_cfg.intermediate_size = max(text_cfg.intermediate_size, _orig_text_hs * 2)

    # PerceptionLM: projector needs model_args["embed_dim"] but TimmWrapper passes
    # model_args as **kwargs to timm.create_model, and ResNet rejects embed_dim.
    # Fix: set model_args for projector, patch timm create to filter unknown kwargs.
    _plm_patched = False
    if base_model_name.lower() == "perceptionlmmodel":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg and getattr(vision_cfg, "model_args", None) is None:
            vision_cfg.model_args = {"embed_dim": 2048}
        try:
            from transformers.models.timm_wrapper import modeling_timm_wrapper as _tw
            _orig_create = _tw._create_timm_model_with_error_handling
            def _safe_create(cfg, **kwargs):
                # Filter out embed_dim — it's for the projector, not timm
                kwargs.pop("embed_dim", None)
                return _orig_create(cfg, **kwargs)
            _tw._create_timm_model_with_error_handling = _safe_create
            _plm_patched = True
        except Exception:
            pass

    model = model_cls(config).to(device)

    if _plm_patched:
        _tw._create_timm_model_with_error_handling = _orig_create

    # Always auto-detect from config — spec's name-based hint can be wrong
    # (e.g., GroundingDino classified as vision but needs text input_ids too)
    input_type = _detect_hf_input_type(base_model_name, config)

    vocab_size = getattr(config, "vocab_size", None) or 1000
    vocab_size = min(vocab_size, 1000)
    B = batch_size

    if input_type == "multimodal":
        # Vision + text multimodal models
        img_size = 224
        vision_cfg = getattr(config, "vision_config", getattr(config, "image_config", None))
        if vision_cfg:
            img_size = getattr(vision_cfg, "image_size", 224)
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
        num_channels = 3
        if vision_cfg:
            num_channels = getattr(vision_cfg, "num_channels", 3)
        # Check if this is a video-multimodal model (needs 5D pixel_values)
        num_frames = getattr(vision_cfg, "num_frames", None) if vision_cfg else None
        if num_frames:
            pixel_values = torch.randn(B, num_frames, num_channels, img_size, img_size, device=device)
        else:
            pixel_values = torch.randn(B, num_channels, img_size, img_size, device=device)
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "pixel_values": pixel_values,
        }
    elif input_type == "seq2seq":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "decoder_input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
        }
    elif input_type == "vision":
        img_size = getattr(config, "image_size", None)
        if img_size is None:
            # Check vision sub-config (e.g., Sam, Sam2)
            vision_cfg = getattr(config, "vision_config", None)
            img_size = getattr(vision_cfg, "image_size", None) if vision_cfg else None
            if img_size is None and vision_cfg:
                # Infer from backbone_feature_sizes (e.g. Sam2: [[256,256],...] → 1024)
                bfs = getattr(vision_cfg, "backbone_feature_sizes", None)
                if bfs and len(bfs) > 0:
                    img_size = bfs[0][0] * 4  # first feature map * stride
            if img_size is None:
                img_size = 224
        if isinstance(img_size, (list, tuple)):
            img_h = img_size[0]
            img_w = img_size[1] if len(img_size) > 1 else img_size[0]
        else:
            img_h = img_w = img_size
        num_channels = getattr(config, "num_channels", 3)
        inputs = {"pixel_values": torch.randn(B, num_channels, img_h, img_w, device=device)}
    elif input_type == "video":
        # Video models expect 5D input: (B, num_frames, C, H, W)
        img_size = getattr(config, "image_size", None) or getattr(config, "crop_size", 224)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        num_channels = getattr(config, "num_channels", None) or getattr(config, "in_chans", 3)
        num_frames = getattr(config, "num_frames", None) or getattr(config, "frames_per_clip", 8)
        pixel_key = "pixel_values"
        # VJEPA2 uses pixel_values_videos
        name_lower = (spec.get("hf_class") or spec["name"]).lower()
        if "vjepa2" in name_lower:
            pixel_key = "pixel_values_videos"
        inputs = {pixel_key: torch.randn(B, num_frames, num_channels, img_size, img_size, device=device)}
    elif input_type == "image_pair":
        # Image pair matching models (e.g., EfficientLoFTR)
        # Input: (B, 2, C, H, W) — two images stacked
        img_size = getattr(config, "image_size", 224)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        num_channels = getattr(config, "num_channels", 3)
        inputs = {"pixel_values": torch.randn(B, 2, num_channels, img_size, img_size, device=device)}
    elif input_type == "video_multimodal":
        # Video-text multimodal (e.g., PeVideo) — needs pixel_values_videos + input_ids
        img_size = 224
        num_channels = 3
        num_frames = 8
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            img_size = getattr(vision_cfg, "image_size", 224)
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
            num_channels = getattr(vision_cfg, "num_channels", 3)
            num_frames = getattr(vision_cfg, "num_frames", 8)
        # Probe model for actual expected image size (e.g., timm-backed encoders)
        for mod in model.modules():
            pe = getattr(mod, "patch_embed", None)
            if pe and hasattr(pe, "img_size"):
                probed = pe.img_size
                img_size = probed[0] if isinstance(probed, (tuple, list)) else probed
                break
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "pixel_values_videos": torch.randn(B, num_frames, num_channels, img_size, img_size, device=device),
        }
    elif input_type == "time_series":
        # Time series models expect past_values
        num_input_channels = getattr(config, "num_input_channels", None)
        context_length = getattr(config, "context_length", 32)
        seq_len = context_length
        name_lower = (spec.get("hf_class") or spec["name"]).lower()
        # PatchTSMixer/PatchTST always need 3D: (batch, seq_len, channels)
        if "patchtsmixer" in name_lower or "patchtst" in name_lower:
            channels = num_input_channels or 1
            inputs = {"past_values": torch.randn(B, seq_len, channels, device=device)}
        elif "timesfm" in name_lower and "2_5" not in name_lower:
            # TimesFmModel: 2D + extra args
            # freq must be (B, 1) so freq_emb output is (B, 1, D) for broadcasting
            inputs = {
                "past_values": torch.randn(B, seq_len, device=device),
                "past_values_padding": torch.ones(B, seq_len, device=device),
                "freq": torch.zeros(B, 1, dtype=torch.long, device=device),
            }
        else:
            # Default: 2D (batch, seq_len)
            inputs = {"past_values": torch.randn(B, seq_len, device=device)}
    elif input_type == "audio_codec":
        # Audio codec models (Encodec, DAC, Xcodec, Mimi) — raw waveform
        inputs = {"input_values": torch.randn(B, 1, 16000, device=device)}
    elif input_type == "audio":
        # Audio models have varied input formats
        model_type = getattr(config, "model_type", "")
        name_lower = (spec.get("hf_class") or spec["name"]).lower()
        if "whisper" in model_type:
            decoder_start = getattr(config, "decoder_start_token_id", 1) or 1
            inputs = {
                "input_features": torch.randn(B, 80, 3000, device=device),
                "decoder_input_ids": torch.tensor([[decoder_start]] * B, device=device),
            }
        elif "speech_to_text" in model_type or "speech2text" in name_lower:
            # Speech2Text: audio encoder-decoder, needs input_features + decoder_input_ids
            num_mel_bins = getattr(config, "num_mel_bins", getattr(config, "input_feat_per_channel", 80))
            vocab_size = getattr(config, "vocab_size", 1000)
            inputs = {
                "input_features": torch.randn(B, 100, num_mel_bins, device=device),
                "decoder_input_ids": torch.randint(0, min(vocab_size, 1000), (B, 32), device=device),
            }
        elif "speecht5" in model_type or "speecht5" in name_lower:
            # SpeechT5 base: no prenet, expects pre-processed (B, T, hidden_size) for both sides
            hidden = getattr(config, "hidden_size", 768)
            inputs = {
                "input_values": torch.randn(B, 100, hidden, device=device),
                "decoder_input_values": torch.randn(B, 50, hidden, device=device),
            }
        elif "fastspeech2" in name_lower:
            # FastSpeech2: TTS, needs phoneme input_ids (not audio features)
            vocab_size = getattr(config, "vocab_size", 78)
            inputs = {"input_ids": torch.randint(0, vocab_size, (B, 32), device=device)}
        elif "wav2vec2" in model_type and "bert" in model_type:
            # Wav2Vec2Bert uses input_features (not input_values)
            inputs = {"input_features": torch.randn(B, 100, 160, device=device)}
        elif "clap" in model_type or "clap" in name_lower:
            # CLAP: treats spectrogram as image → 4D (B, 1, spec_size, num_mel_bins)
            audio_cfg = getattr(config, "audio_config", config)
            num_mel_bins = getattr(audio_cfg, "num_mel_bins", 64)
            spec_size = getattr(audio_cfg, "spec_size", 256)
            inputs = {"input_features": torch.randn(B, 1, spec_size, num_mel_bins, device=device)}
            # Full ClapModel also needs text inputs (contrastive audio-text)
            text_cfg = getattr(config, "text_config", None)
            if text_cfg:
                voc = min(getattr(text_cfg, "vocab_size", 1000), 1000)
                inputs["input_ids"] = torch.randint(0, voc, (B, 32), device=device)
                inputs["attention_mask"] = torch.ones(B, 32, dtype=torch.long, device=device)
        elif "univnet" in model_type or "univnet" in name_lower:
            # UnivNet vocoder: needs mel spectrogram as input_features
            num_mel_bins = getattr(config, "num_mel_channels", getattr(config, "num_mel_bins", 100))
            inputs = {"input_features": torch.randn(B, num_mel_bins, 100, device=device)}
        elif "moonshine" in model_type or "moonshine" in name_lower:
            # Moonshine: speech encoder-decoder, needs input_values + decoder_input_ids
            inputs = {
                "input_values": torch.randn(B, 16000, device=device),
                "decoder_input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            }
        elif "spectrogram" in model_type or hasattr(config, "num_mel_bins"):
            # AST and similar spectrogram models
            num_mel_bins = getattr(config, "num_mel_bins", 128)
            max_length = getattr(config, "max_length", 1024)
            inputs = {"input_values": torch.randn(B, max_length, num_mel_bins, device=device)}
        else:
            # Most audio models use input_values (raw waveform)
            inputs = {"input_values": torch.randn(B, 16000, device=device)}
    else:  # text
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # ── Post-processing: add model-specific required inputs ──
    # Use base model name for handler matching so ForCausalLM/ForConditionalGeneration
    # variants reuse the same input logic as their base model.
    name_lower = base_model_name.lower()
    model_type = getattr(config, "model_type", "")

    # ── Grid-THW flattened-patch vision models ──
    # These vision encoders use pre-patchified 2D tensors + grid_thw, not standard (B,C,H,W).
    # Shape: (total_patches, patch_dim) where patch_dim = channels * [temporal_ps *] ps * ps
    # name → (input_key, include_temporal_patch_size)
    _GRID_THW_PATCH_MODELS = {
        "glm4vvisionmodel":       ("hidden_states", True),
        "glm4vmoevisionmodel":    ("hidden_states", True),
        "glmocrvisionmodel":      ("hidden_states", True),
        "glmimagevisionmodel":    ("pixel_values", False),
        "videollama3visionmodel": ("pixel_values", False),
    }
    _grid_thw_spec = _GRID_THW_PATCH_MODELS.get(name_lower)
    # Ernie vision transformer: same pattern, matched by substring
    if _grid_thw_spec is None and "ernie4_5_vl" in name_lower and "visiontransformer" in name_lower:
        _grid_thw_spec = ("hidden_states", False)
    if _grid_thw_spec is not None:
        _input_key, _use_temporal = _grid_thw_spec
        in_channels = getattr(config, "in_channels", None) or getattr(config, "num_channels", 3)
        patch_size = getattr(config, "patch_size", 14)
        temporal_ps = getattr(config, "temporal_patch_size", 2) if _use_temporal else 1
        patch_dim = in_channels * temporal_ps * patch_size * patch_size
        grid_thw = torch.tensor([[1, 4, 8]] * B, dtype=torch.long, device=device)
        total_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        inputs = {
            _input_key: torch.randn(total_patches, patch_dim, device=device),
            "grid_thw": grid_thw,
        }
        if name_lower == "videollama3visionmodel":
            inputs["merge_sizes"] = torch.tensor([1] * B, dtype=torch.long, device=device)

    # Non-Glm models that need hidden_states
    _other_hidden_states_models = {
        "flavamultimodalmodel",
    }
    if name_lower in _other_hidden_states_models:
        hidden_size = getattr(config, "hidden_size", 768)
        inputs = {"hidden_states": torch.randn(B, 32, hidden_size, device=device)}

    # Phi4MultimodalAudioModel: needs hidden_states (B, seq_len, input_size) + mask
    # input_size defaults to 80 (audio mel features); mask can be None
    if name_lower == "phi4multimodalaudiomodel":
        input_size = getattr(config, "input_size", 80)
        seq_len = 100
        inputs = {
            "hidden_states": torch.randn(B, seq_len, input_size, device=device),
            "mask": None,
        }

    # Siglip2VisionModel needs pre-patchified pixel_values + pixel_attention_mask + spatial_shapes
    # pixel_values shape: (B, num_patches, num_channels * patch_size * patch_size)
    # NOT standard (B, C, H, W) — Siglip2 uses nn.Linear for patch embedding, not Conv2d
    if name_lower == "siglip2visionmodel":
        num_patches = getattr(config, "num_patches", 256)
        patch_size = getattr(config, "patch_size", 16)
        num_channels = getattr(config, "num_channels", 3)
        patch_dim = num_channels * patch_size * patch_size
        num_patches_h = int(num_patches**0.5)
        inputs = {
            "pixel_values": torch.randn(B, num_patches, patch_dim, device=device),
            "pixel_attention_mask": torch.ones(B, num_patches, dtype=torch.long, device=device),
            "spatial_shapes": torch.tensor([[num_patches_h, num_patches_h]] * B, dtype=torch.long, device=device),
        }

    # PI0ForConditionalGeneration: skipped in models.py (image_token_id == vocab_size,
    # can't reduce model and keep image token embedding valid). Keep base PI0Model handler below.

    # InstructBlip needs qformer_input_ids
    if "instructblip" in name_lower and "qformer_input_ids" not in inputs:
        qformer_vocab = getattr(getattr(config, "qformer_config", None), "vocab_size", 1000) if hasattr(config, "qformer_config") else 1000
        inputs["qformer_input_ids"] = torch.randint(0, min(qformer_vocab, 1000), (B, 32), device=device)

    # BarkFineModel: expects 3D input_ids (batch, seq_len, n_codes_total) + codebook_idx
    if name_lower == "barkfinemodel":
        n_codes_total = getattr(config, "n_codes_total", 8)
        inputs = {
            "codebook_idx": 1,
            "input_ids": torch.randint(0, vocab_size, (B, 32, n_codes_total), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # PerceiverModel needs 'inputs' key (not input_ids)
    if name_lower == "perceivermodel":
        num_self_attends = getattr(config, "num_self_attends_per_block", 6)
        inputs = {"inputs": torch.randn(B, 32, getattr(config, "d_model", 768), device=device)}

    # SegGptModel needs prompt_pixel_values + prompt_masks
    # SegGpt concatenates pixel_values + prompt_pixel_values along height,
    # so individual images must be half the model's expected height.
    # prompt_masks must have same num_channels as pixel_values because SegGptEmbeddings
    # passes the concatenated prompt through the same patch_embeddings (which checks num_channels).
    if name_lower == "seggptmodel":
        img_size = getattr(config, "image_size", 896)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        img_size = img_size // 2  # Model expects concatenated 2*H
        num_channels = getattr(config, "num_channels", 3)
        inputs = {
            "pixel_values": torch.randn(B, num_channels, img_size, img_size, device=device),
            "prompt_pixel_values": torch.randn(B, num_channels, img_size, img_size, device=device),
            "prompt_masks": torch.randn(B, num_channels, img_size, img_size, device=device),
        }

    # PeAudioModel needs input_ids + mono audio (text-audio multimodal)
    if name_lower == "peaudiomodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "input_values": torch.randn(B, 1, 16000, device=device),
        }

    # PeAudioVideoModel needs at least 2 of: input_ids, pixel_values_videos, input_values
    if name_lower == "peaudiovideomodel":
        img_size = 224
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            img_size = getattr(vision_cfg, "image_size", 224)
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
        # Probe for actual image size
        for mod in model.modules():
            pe = getattr(mod, "patch_embed", None)
            if pe and hasattr(pe, "img_size"):
                probed = pe.img_size
                img_size = probed[0] if isinstance(probed, (tuple, list)) else probed
                break
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "pixel_values_videos": torch.randn(B, 8, 3, img_size, img_size, device=device),
        }

    # VibeVoiceAcousticTokenizerModel needs input_values (raw audio, 1-channel)
    if name_lower == "vibevoiceacoustictokenizermodel":
        inputs = {"input_values": torch.randn(B, 1, 16000, device=device)}

    # Qwen2_5OmniToken2WavModel needs code + conditioning + reference_mel
    if name_lower == "qwen2_5omnitoken2wavmodel":
        hidden = getattr(config, "hidden_size", 768)
        inputs = {
            "code": torch.randint(0, 100, (B, 32), device=device),
            "conditioning": torch.randn(B, 32, hidden, device=device),
            "reference_mel": torch.randn(B, 80, 100, device=device),
        }

    # VibeVoiceAcousticTokenizerEncoderModel: forward expects hidden_states (not input_values)
    # hidden_states shape: (B, channels, T) where channels defaults to 1
    if name_lower == "vibevoiceacoustictokenizerencodermodel":
        inputs = {"hidden_states": torch.randn(B, 1, 16000, device=device)}

    # VibeVoiceAcousticTokenizerDecoderModel: expects encoded features
    if name_lower == "vibevoiceacoustictokenizerdecodermodel":
        hidden = getattr(config, "hidden_size", 64)
        inputs = {"hidden_states": torch.randn(B, hidden, 70, device=device)}

    # PPOCRV5MobileDetModel: forward expects hidden_states (not pixel_values)
    # hidden_states is the image tensor (B, C, H, W) — same shape, different key name
    if name_lower == "ppocrv5mobiledetmodel":
        img_size = getattr(config, "image_size", 640)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        inputs = {"hidden_states": torch.randn(B, 3, img_size, img_size, device=device)}

    # PI0Model needs action_embeds
    if name_lower == "pi0model":
        hidden = getattr(config, "hidden_size", 768)
        inputs["action_embeds"] = torch.randn(B, 16, hidden, device=device)

    # ── needs_special_input fixes ──

    # BrosModel needs bbox
    if name_lower == "brosmodel":
        inputs["bbox"] = torch.randint(0, 1000, (B, 32, 8), device=device)

    # UdopModel needs bbox (float for mean() operations)
    if name_lower == "udopmodel":
        inputs["bbox"] = torch.rand(B, 32, 4, device=device) * 1000

    # Pix2StructVisionModel needs flattened_patches
    # First 2 columns are row/col position indices fed into nn.Embedding(seq_len, ...).
    # Must be valid non-negative integers; random floats cause CUDA OOB assert.
    if name_lower == "pix2structvisionmodel":
        hidden = getattr(config, "hidden_size", 768)
        patch_embed_dim = getattr(config, "patch_embed_hidden_size", None)
        num_channels = getattr(config, "num_channels", 3)
        patch_size = getattr(config, "patch_size", 16)
        seq_len = getattr(config, "seq_len", 4096)
        flat_dim = patch_size * patch_size * num_channels + 2
        flattened_patches = torch.randn(B, 32, flat_dim, device=device)
        # Columns 0,1 are position indices — must be valid embedding indices
        flattened_patches[:, :, 0] = torch.randint(0, seq_len, (B, 32), device=device).float()
        flattened_patches[:, :, 1] = torch.randint(0, seq_len, (B, 32), device=device).float()
        inputs = {
            "flattened_patches": flattened_patches,
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # ViltModel needs pixel_values (multimodal but detected differently)
    if name_lower == "viltmodel":
        img_size = getattr(config, "image_size", 384)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        inputs["pixel_values"] = torch.randn(B, 3, img_size, img_size, device=device)

    # LxmertModel needs visual_feats
    if name_lower == "lxmertmodel":
        num_visual = 36
        visual_feat_dim = getattr(config, "visual_feat_dim", 2048)
        visual_pos_dim = getattr(config, "visual_pos_dim", 4)
        inputs["visual_feats"] = torch.randn(B, num_visual, visual_feat_dim, device=device)
        inputs["visual_pos"] = torch.randn(B, num_visual, visual_pos_dim, device=device)

    # XmodModel needs set_default_language
    if name_lower == "xmodmodel":
        try:
            model.set_default_language("en_XX")
        except Exception:
            pass

    # ── image_token_mismatch fixes ──
    # Models that check image_token_id in input_ids and count them to match vision encoder output.
    # The correct number of image tokens varies per model due to different pooling/projection.
    # Models with huge feature counts (FastVlm, Janus, DeepseekVL) have flattened tensor comparisons
    # in their model code and cannot be fixed with simple token injection.
    image_token_fix = {
        # model_name: tokens_per_sequence (determined empirically)
        "ariamodel": 128,
        "florence2model": 50,
        "gemma3nmodel": 256,
        "gotocr2model": 256,
        # InternVLModel: vision_tower (448x448, patch=14) → 1024 patches → pixel_shuffle(0.5) → 256
        "internvlmodel": 256,
        "paligemmamodel": 256,
        "vipllavamodel": 576,
    }
    if name_lower in image_token_fix and "input_ids" in inputs:
        image_token_id = getattr(config, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(config, "image_token_index", None)
        if image_token_id is not None:
            num_image_tokens = image_token_fix[name_lower]
            # Resize embeddings if image_token_id >= vocab_size (OOB for default embedding)
            if image_token_id >= vocab_size and hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(image_token_id + 1)
            # Extend input_ids to fit image tokens + some text
            total_len = num_image_tokens + 8
            new_input_ids = torch.randint(0, vocab_size, (B, total_len), device=device)
            new_input_ids[:, :num_image_tokens] = image_token_id
            inputs["input_ids"] = new_input_ids
            inputs["attention_mask"] = torch.ones(B, total_len, dtype=torch.long, device=device)
            # PaliGemma train mode requires token_type_ids
            if name_lower == "paligemmamodel":
                token_type_ids = torch.zeros(B, total_len, dtype=torch.long, device=device)
                token_type_ids[:, num_image_tokens:] = 1
                inputs["token_type_ids"] = token_type_ids

    # ── shape_mismatch fixes ──

    # OwlViT/Owlv2 text models — seq_len must equal max_position_embeddings
    if name_lower in ("owlvittextmodel", "owlv2textmodel"):
        max_pos = getattr(config, "max_position_embeddings", 16)
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, max_pos), device=device),
            "attention_mask": torch.ones(B, max_pos, dtype=torch.long, device=device),
        }

    # OwlViTModel/Owlv2Model — text seq_len must match max_position_embeddings
    if name_lower in ("owlvitmodel", "owlv2model"):
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            max_pos = getattr(text_cfg, "max_position_embeddings", 16)
            inputs["input_ids"] = torch.randint(0, vocab_size, (B, max_pos), device=device)
            inputs["attention_mask"] = torch.ones(B, max_pos, dtype=torch.long, device=device)

    # Siglip2Model — pre-patchified pixel_values + pixel_attention_mask + spatial_shapes
    # Same as Siglip2VisionModel: uses nn.Linear patch embedding, not Conv2d
    if name_lower == "siglip2model":
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            num_patches = getattr(vision_cfg, "num_patches", 256)
            patch_size = getattr(vision_cfg, "patch_size", 16)
            num_channels = getattr(vision_cfg, "num_channels", 3)
            patch_dim = num_channels * patch_size * patch_size
            num_patches_h = int(num_patches**0.5)
            inputs["pixel_values"] = torch.randn(B, num_patches, patch_dim, device=device)
            inputs["pixel_attention_mask"] = torch.ones(B, num_patches, dtype=torch.long, device=device)
            inputs["spatial_shapes"] = torch.tensor([[num_patches_h, num_patches_h]] * B, dtype=torch.long, device=device)

    # DepthProModel — image_size must be >= patch_size * 4 (scaled_images_ratios)
    if name_lower == "depthpromodel":
        patch_size = getattr(config, "patch_size", 384)
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        # Must be >= patch_size / min(scaled_images_ratios)
        # Default ratios: (0.25, 0.5, 1) → need img_size >= patch_size / 0.25 = 4 * patch_size
        ratios = getattr(config, "scaled_images_ratios", (0.25, 0.5, 1.0))
        min_ratio = min(ratios) if ratios else 0.25
        img_size = max(int(patch_size / min_ratio), patch_size)
        num_channels = getattr(config, "num_channels", 3)
        inputs = {"pixel_values": torch.randn(B, num_channels, img_size, img_size, device=device)}

    # DecisionTransformerModel — needs states, actions, rewards, returns_to_go, timesteps
    if name_lower == "decisiontransformermodel":
        state_dim = getattr(config, "state_dim", 17)
        act_dim = getattr(config, "act_dim", 4)
        seq_len = 32
        inputs = {
            "states": torch.randn(B, seq_len, state_dim, device=device),
            "actions": torch.randn(B, seq_len, act_dim, device=device),
            "rewards": torch.randn(B, seq_len, 1, device=device),
            "returns_to_go": torch.randn(B, seq_len, 1, device=device),
            "timesteps": torch.randint(0, 100, (B, seq_len), device=device),
            "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
        }

    # TimesFmModel — past_values_padding + freq required
    # freq must be (B, 1) so freq_emb output is (B, 1, D) for broadcasting with (B, num_patches, D)
    if name_lower == "timesfmmodel":
        context_length = getattr(config, "context_length", 32)
        inputs = {
            "past_values": torch.randn(B, context_length, device=device),
            "past_values_padding": torch.ones(B, context_length, device=device),
            "freq": torch.zeros(B, 1, dtype=torch.long, device=device),
        }

    # ── Additional model-specific fixes ──

    # BarkModel: forward doesn't accept input_ids — uses generate() pattern
    # Skip: not a standard forward() model
    # BarkModel has no forward() — only generate(). Use empty input to trigger error gracefully.
    if name_lower == "barkmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
        }

    # Blip2Model / InstructBlipModel: need pixel_values + input_ids with image tokens
    if name_lower in ("blip2model", "instructblipmodel"):
        num_query_tokens = getattr(config, "num_query_tokens", 32)
        image_token_id = getattr(config, "image_token_index", 2)
        total_len = num_query_tokens + 8
        new_ids = torch.randint(0, vocab_size, (B, total_len), device=device)
        new_ids[:, :num_query_tokens] = image_token_id
        # Build pixel_values from vision_config
        img_size = 224
        num_channels = 3
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            img_size = getattr(vision_cfg, "image_size", 224) or 224
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
            num_channels = getattr(vision_cfg, "num_channels", 3)
        inputs = {
            "input_ids": new_ids,
            "attention_mask": torch.ones(B, total_len, dtype=torch.long, device=device),
            "pixel_values": torch.randn(B, num_channels, img_size, img_size, device=device),
        }
        # InstructBlipModel also needs qformer_input_ids
        if name_lower == "instructblipmodel":
            qformer_vocab = getattr(getattr(config, "qformer_config", None), "vocab_size", 1000) if hasattr(config, "qformer_config") else 1000
            inputs["qformer_input_ids"] = torch.randint(0, min(qformer_vocab, 1000), (B, 32), device=device)

    # InstructBlipVideoModel: needs pixel_values + input_ids + qformer_input_ids
    # pixel_values must always be 5D: (B, num_frames, C, H, W) — the model forward
    # unpacks as (batch_size, frames, channel, height, width).
    if name_lower == "instructblipvideomodel":
        num_query_tokens = getattr(config, "num_query_tokens", 32)
        image_token_id = getattr(config, "image_token_index", 2)
        num_frames = 4  # InstructBlipVideo always expects video frames
        total_len = num_query_tokens * num_frames + 8
        new_ids = torch.randint(0, vocab_size, (B, total_len), device=device)
        new_ids[:, :num_query_tokens * num_frames] = image_token_id
        # Build pixel_values from vision_config
        img_size = 224
        num_channels = 3
        vision_cfg = getattr(config, "vision_config", None)
        if vision_cfg:
            img_size = getattr(vision_cfg, "image_size", 224) or 224
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
            num_channels = getattr(vision_cfg, "num_channels", 3)
        pv = torch.randn(B, num_frames, num_channels, img_size, img_size, device=device)
        qformer_vocab = getattr(getattr(config, "qformer_config", None), "vocab_size", 1000) if hasattr(config, "qformer_config") else 1000
        inputs = {
            "input_ids": new_ids,
            "attention_mask": torch.ones(B, total_len, dtype=torch.long, device=device),
            "pixel_values": pv,
            "qformer_input_ids": torch.randint(0, min(qformer_vocab, 1000), (B, 32), device=device),
        }

    # MusicgenModel/MusicgenMelodyModel: input_ids must be (B*num_codebooks, seq)
    if name_lower in ("musicgenmodel", "musicgenmelodymodel"):
        dec_cfg = getattr(config, "decoder", None)
        num_codebooks = getattr(dec_cfg, "num_codebooks", 4) if dec_cfg else 4
        dec_vocab = getattr(dec_cfg, "vocab_size", 2048) if dec_cfg else 2048
        inputs = {
            "input_ids": torch.randint(0, min(dec_vocab, 1000),
                                       (B * num_codebooks, 32), device=device),
        }

    # AutoformerModel, InformerModel, TimeSeriesTransformerModel: need past_values + past_time_features
    if name_lower in ("autoformermodel", "informermodel", "timeseriestransformermodel"):
        context_length = getattr(config, "context_length", 96)
        prediction_length = getattr(config, "prediction_length", 24)
        num_time_features = getattr(config, "num_time_features", 0)
        # past_values must be long enough for the maximum lag in lags_sequence
        lags = getattr(config, "lags_sequence", None)
        if lags:
            max_lag = max(lags)
        else:
            max_lag = 0
        past_len = context_length + max_lag
        inputs = {
            "past_values": torch.randn(B, past_len, device=device),
            "past_observed_mask": torch.ones(B, past_len, device=device),
        }
        if num_time_features > 0:
            inputs["past_time_features"] = torch.randn(B, past_len, num_time_features, device=device)
            inputs["future_time_features"] = torch.randn(B, prediction_length, num_time_features, device=device)
        else:
            # Model still needs time features tensors, but with 0 features
            inputs["past_time_features"] = torch.randn(B, past_len, 0, device=device)
            inputs["future_time_features"] = torch.randn(B, prediction_length, 0, device=device)

    # Gemma3Model: vision pooling stride error with reduced config — use text-only
    if name_lower == "gemma3model":
        text_cfg = getattr(config, "text_config", config)
        voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
        inputs = {
            "input_ids": torch.randint(0, voc, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            # Train mode requires token_type_ids
            "token_type_ids": torch.zeros(B, 32, dtype=torch.long, device=device),
        }

    # Gemma3Model: stride should not be zero (sliding window config issue)
    if name_lower == "gemma3model":
        if getattr(config, "sliding_window", None) == 0:
            config.sliding_window = 4096

    # Gemma4Model (composite): text-only with mm_token_type_ids for train mode
    # Gemma4 uses pre-patchified vision (nn.Linear, not Conv2d) — text-only avoids
    # the complex multimodal path while still testing the core language model + compile.
    if name_lower == "gemma4model":
        text_cfg = getattr(config, "text_config", config)
        voc = min(getattr(text_cfg, "vocab_size", 262144) or 262144, 1000)
        inputs = {
            "input_ids": torch.randint(0, voc, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            # Train mode requires mm_token_type_ids (0=text, 1=image, 2=audio)
            "mm_token_type_ids": torch.zeros(B, 32, dtype=torch.long, device=device),
        }
        if getattr(config, "sliding_window", None) == 0:
            config.sliding_window = 4096

    # Gemma4VisionModel: pre-patchified pixel_values (nn.Linear patch embed, not Conv2d)
    # pixel_values shape: (B, max_patches, patch_size * patch_size * num_channels)
    # pixel_position_ids shape: (B, max_patches, 2) — row/col position of each patch
    if name_lower == "gemma4visionmodel":
        patch_size = getattr(config, "patch_size", 16)
        num_channels = getattr(config, "num_channels", 3)
        patch_dim = num_channels * patch_size * patch_size  # 768
        # num_patches must be divisible by pooling_kernel_size^2 (default 3^2=9)
        # so the multihead pooling layers can reduce evenly at each stage.
        pool_k = getattr(config, "pooling_kernel_size", 3)
        num_patches = pool_k ** 4  # 81 for k=3: supports 2 pooling stages (81→9→1)
        grid_side = int(num_patches ** 0.5)  # 9 for 81 patches
        inputs = {
            "pixel_values": torch.randn(B, num_patches, patch_dim, device=device),
            "pixel_position_ids": torch.randint(0, grid_side, (B, num_patches, 2), device=device),
        }

    # Gemma4AudioModel: mel spectrogram input_features
    # input_features shape: (B, time_frames, feature_size=128)
    if name_lower == "gemma4audiomodel":
        feature_size = getattr(config, "feature_size", 128)
        inputs = {
            "input_features": torch.randn(B, 100, feature_size, device=device),
            "attention_mask": torch.ones(B, 100, dtype=torch.long, device=device),
        }

    # Hybrid attention+Mamba models: disable cache to avoid transformers 5.5
    # cache_utils.py ValueError ("get_seq_length/has_previous_state can only be
    # called on Attention/LinearAttention layers"). These models mix attention and
    # Mamba/linear-attention layers — the hybrid cache validation is broken in eager.
    if name_lower in ("bambamodel", "granitemoehybridmodel", "jambamodel"):
        inputs["use_cache"] = False

    # KyutaiSpeechToTextModel: input_ids must be 3D (B, seq_len, num_codebooks + 1)
    # The embeddings layer adds audio_tokens_offsets of shape (num_codebooks + 1,) to input_ids,
    # then sums over the last dim. 2D input_ids causes shape mismatch (seq_len vs num_codebooks+1).
    if name_lower == "kyutaispeechtotextmodel":
        num_codebooks = getattr(config, "num_codebooks", 32)
        n_streams = num_codebooks + 1  # +1 for text token stream (padded with 0)
        inputs = {
            "input_ids": torch.randint(0, min(vocab_size, 1000), (B, 32, n_streams), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # ReformerModel: train mode needs seq_len matching axial_pos_shape product
    if name_lower == "reformermodel":
        axial = getattr(config, "axial_pos_shape", (64, 64))
        if isinstance(axial, (list, tuple)):
            target_len = 1
            for d in axial:
                target_len *= d
            inputs["input_ids"] = torch.randint(0, vocab_size, (B, target_len), device=device)
            inputs["attention_mask"] = torch.ones(B, target_len, dtype=torch.long, device=device)

    # Idefics2Model, InstructBlipVideoModel: need pixel_values with num_images dim
    if name_lower == "idefics2model":
        vision_cfg = getattr(config, "vision_config", None)
        img_size = 224
        if vision_cfg:
            img_size = getattr(vision_cfg, "image_size", 224) or 224
            if isinstance(img_size, (list, tuple)):
                img_size = img_size[0]
        # Idefics2 expects pixel_values of shape (B, num_images, C, H, W)
        inputs["pixel_values"] = torch.randn(B, 1, 3, img_size, img_size, device=device)
        inputs["pixel_attention_mask"] = torch.ones(B, 1, img_size, img_size, dtype=torch.long, device=device)

    # Kosmos2Model: image_embeds + image_embeds_position_mask
    if name_lower == "kosmos2model":
        hidden_size = getattr(config, "hidden_size", 2048)
        num_image_tokens = 64
        total_len = num_image_tokens + 8
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, total_len), device=device),
            "attention_mask": torch.ones(B, total_len, dtype=torch.long, device=device),
            "image_embeds": torch.randn(B, num_image_tokens, hidden_size, device=device),
            "image_embeds_position_mask": torch.cat([
                torch.ones(B, num_image_tokens, dtype=torch.bool, device=device),
                torch.zeros(B, 8, dtype=torch.bool, device=device),
            ], dim=1),
        }

    # TvpModel: needs pixel_values explicitly
    # Must use batch_size=1 due to a bug in TvpVisualInputEmbedding.add_2d_positional_embeddings:
    # col_position_embeddings.view(batch_size, 1, width, hidden_dim) fails for batch_size>1
    # because the embedding tensor only has (width, hidden_dim) elements.
    if name_lower == "tvpmodel":
        img_size = getattr(config, "image_size", 448)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        num_frames = getattr(config, "num_frames", 4)
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (1, 32), device=device),
            "attention_mask": torch.ones(1, 32, dtype=torch.long, device=device),
            "pixel_values": torch.randn(1, num_frames, 3, img_size, img_size, device=device),
        }

    # IdeficsModel: requires exactly one of pixel_values/image_encoder_embeddings/perceiver_embeddings
    # When use_resampler=True and perceiver_embeddings is provided, the model unpacks it as
    # (batch_size, num_images, image_seq_len, image_hidden_size) — must be 4D.
    # Also requires image_attention_mask of shape (B, text_seq_len, num_images).
    # image_hidden_size must match vision_config.embed_dim (not config.hidden_size) because the
    # cross-attention k/v projections use kv_input_dim=vision_config.embed_dim for cross-attention.
    if name_lower == "ideficsmodel":
        vision_cfg = getattr(config, "vision_config", None)
        image_hidden_size = getattr(vision_cfg, "embed_dim", 768) if vision_cfg else 768
        perceiver_cfg = getattr(config, "perceiver_config", None)
        n_latents = getattr(perceiver_cfg, "resampler_n_latents", 64) if perceiver_cfg else 64
        num_images = 1
        text_seq_len = 32
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, text_seq_len), device=device),
            "attention_mask": torch.ones(B, text_seq_len, dtype=torch.long, device=device),
            "perceiver_embeddings": torch.randn(B, num_images, n_latents, image_hidden_size, device=device),
            "image_attention_mask": torch.ones(B, text_seq_len, num_images, dtype=torch.long, device=device),
        }

    # PaddleOCRVisionModel: needs 5D pixel_values + cu_seqlens + image_grid_thw
    # pixel_values shape: (1, total_patches, num_channels, patch_size, patch_size)
    # cu_seqlens: cumulative sequence lengths for Flash Attention, shape (num_images + 1,)
    if name_lower == "paddleocrvisionmodel":
        patch_size = getattr(config, "patch_size", 14)
        num_channels = getattr(config, "num_channels", 3)
        image_grid_thw = [(1, 4, 8)] * B
        total_patches = sum(t * h * w for t, h, w in image_grid_thw)
        pixel_values = torch.randn(1, total_patches, num_channels, patch_size, patch_size, device=device)
        # cu_seqlens: cumulative patch counts per temporal slice
        grid_thw_tensor = torch.tensor(image_grid_thw, dtype=torch.long, device=device)
        cu_seqlens = torch.repeat_interleave(
            grid_thw_tensor[:, 1] * grid_thw_tensor[:, 2], grid_thw_tensor[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(cu_seqlens, (1, 0), value=0)
        inputs = {
            "pixel_values": pixel_values,
            "cu_seqlens": cu_seqlens,
            "image_grid_thw": image_grid_thw,
        }

    # UVDocModel: needs proper image size for backbone
    if name_lower == "uvdocmodel":
        img_size = getattr(config, "image_size", 224)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        # UVDoc uses a backbone that expects specific channel count
        num_channels = getattr(config, "num_channels", 3)
        inputs = {"pixel_values": torch.randn(B, num_channels, img_size, img_size, device=device)}

    # XCLIPVisionModel: standalone vision encoder expects 4D (B*num_frames, C, H, W)
    # The full XCLIPModel reshapes (B, num_frames, C, H, W) -> (B*num_frames, C, H, W)
    # before calling the vision model. The encoder layers assume batch_dim = B*num_frames
    # for cross-frame message passing (batch_size = batch_time // num_frames).
    if name_lower == "xclipvisionmodel":
        img_size = getattr(config, "image_size", 224)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        num_channels = getattr(config, "num_channels", 3)
        num_frames = getattr(config, "num_frames", 8)
        inputs = {"pixel_values": torch.randn(B * num_frames, num_channels, img_size, img_size, device=device)}

    # Llama4VisionModel: needs pixel_values as required positional argument
    if name_lower == "llama4visionmodel":
        img_size = getattr(config, "image_size", 560)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        num_channels = getattr(config, "num_channels", 3)
        inputs = {"pixel_values": torch.randn(B, num_channels, img_size, img_size, device=device)}

    # Lfm2Model: architecture issue with conv ordering
    # Lfm2VlModel: shape mismatch in vision projection
    # These may need specific model version — skip for now

    # LwDetrModel: image_size lives in backbone_config, not top-level config
    if name_lower == "lwdetrmodel":
        backbone_cfg = getattr(config, "backbone_config", None)
        img_size = getattr(backbone_cfg, "image_size", 256) if backbone_cfg else 256
        if isinstance(img_size, (list, tuple)):
            h, w = img_size
        else:
            h = w = img_size
        num_channels = 3
        if backbone_cfg:
            num_channels = getattr(backbone_cfg, "num_channels", 3)
        inputs = {"pixel_values": torch.randn(B, num_channels, h, w, device=device)}

    # FastSpeech2ConformerModel: train mode needs all four label types
    # The model forward checks: if self.training and any label is None → ValueError.
    # Shapes: spectrogram_labels (B, spec_len, num_mel), duration_labels (B, text_len) long,
    #         pitch_labels (B, text_len, 1), energy_labels (B, text_len, 1).
    if name_lower == "fastspeech2conformermodel":
        num_mel = getattr(config, "num_mel_channels", 80)
        text_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 32
        # spectrogram_labels length must match length_regulator output.
        # With duration_labels=ones(B, text_len), the regulator outputs seq_len=text_len.
        inputs["spectrogram_labels"] = torch.randn(B, text_len, num_mel, device=device)
        inputs["duration_labels"] = torch.ones(B, text_len, dtype=torch.long, device=device)
        inputs["pitch_labels"] = torch.randn(B, text_len, 1, device=device)
        inputs["energy_labels"] = torch.randn(B, text_len, 1, device=device)

    # CohereAsrModel: encoder-decoder ASR model, needs input_features (mel spectrogram) + decoder_input_ids
    # Encoder is ParakeetEncoder which expects (B, time, num_mel_bins) audio features.
    # The subsampling layer calls input_features.unsqueeze(1), so input_features must not be None.
    if name_lower == "cohereasrmodel":
        encoder_cfg = getattr(config, "encoder_config", None)
        num_mel_bins = getattr(encoder_cfg, "num_mel_bins", 128) if encoder_cfg else 128
        decoder_start = getattr(config, "decoder_start_token_id", None) or getattr(config, "bos_token_id", 1) or 1
        inputs = {
            "input_features": torch.randn(B, 100, num_mel_bins, device=device),
            "decoder_input_ids": torch.tensor([[decoder_start]] * B, device=device),
        }

    # PaliGemmaModel train: needs token_type_ids
    if name_lower == "paligemmamodel":
        inputs["token_type_ids"] = torch.zeros(B, inputs["input_ids"].shape[1],
                                                dtype=torch.long, device=device)

    # ── VL/multimodal models: text-only to avoid NoneType iteration on image inputs ──
    _vl_text_only_models = {
        # NoneType.tolist from image_grid_thw
        "qwen3vlmodel", "qwen3vlmoemodel", "qwen3_5moemodel", "qwen3_5model",
        # NoneType not iterable from image_sizes/grid_thw
        "qwen2vlmodel", "qwen2_5_vlmodel", "llavamodel", "llavanextmodel",
        "llavanextvideomodel",
        # FCG overrides provide proper vision inputs; base models need text-only
        "fastvlmmodel", "lfm2vlmodel", "llama4model",
        "mistral3model", "videollama3model",
        "glm46vmodel", "glm4vmodel", "glm4vmoemodel",
        "glmimagemodel", "glmimagetextmodel", "glmocrmodel",
        # mat1/mat2 mismatch from vision projection — text-only avoids it
        "ernie4_5_vlmoemodel", "ernie4_5_vl_moemodel",
        # NoneType has no len
        "llavaonevisionmodel",
        # Image features / image tokens mismatch
        "cohere2visionmodel", "deepseekvlmodel", "janusmodel",
        # high_res_pixel_values size mismatch (vision_config vs high_res_vision_config)
        "deepseekvlhybridmodel",
        # aspect_ratio_ids required with pixel_values
        "mllamamodel",
        # Not enough values to unpack (image inputs)
        "idefics3model", "smolvlmmodel",
        # Image token/feature mismatch or tuple unpack
        "vipllavamodel", "ayavisionmodel",
        # numel mismatch: token_count * text_hidden_size != token_count * vision_hidden_size
        "ovis2model",
        # PerceptionLM: flatten(0,1) on 4D (B,C,H,W) merges batch+channels; needs 5D via FCG override
        "perceptionlmmodel",
        # Qwen Omni "Thinker" = text decoder only (audio/vision handled by separate "Audio" model)
        "qwen2_5omnithinkermodel", "qwen3omnimoethinkermodel",
        # mrope_section issue or vision path NoneType
        "qwen2_5omnitalkermodel", "paddleocrtextmodel", "paddleocrvlmodel",
        # Multimodal models that just need input_ids (no vision path)
        "higgsaudiov2model", "phi4multimodalmodel", "modernvbertmodel",
        # OCR/vision text-only to avoid NoneType
        "lightonocrmodel",
    }
    if name_lower in _vl_text_only_models:
        text_cfg = getattr(config, "text_config", config)
        voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
        inputs = {
            "input_ids": torch.randint(0, voc, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # VisionEncoderDecoderModel: needs pixel_values + decoder_input_ids
    if name_lower == "visionencoderdecodermodel":
        enc_cfg = getattr(config, "encoder", None)
        img_size = getattr(enc_cfg, "image_size", 224) if enc_cfg else 224
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        num_channels = getattr(enc_cfg, "num_channels", 3) if enc_cfg else 3
        inputs = {
            "pixel_values": torch.randn(B, num_channels, img_size, img_size, device=device),
            "decoder_input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
        }

    # SpeechEncoderDecoderModel: needs input_values + decoder_input_ids
    if name_lower == "speechencoderdecodermodel":
        inputs = {
            "input_values": torch.randn(B, 16000, device=device),
            "decoder_input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
        }

    # BayesianDetectorModel: expects g_values (green-list scores), not input_ids
    if name_lower == "bayesiandetectormodel":
        depth = getattr(config, "watermarking_depth", 3)
        inputs = {
            "g_values": torch.randn(B, 32, depth, device=device),
            "mask": torch.ones(B, 32, dtype=torch.bool, device=device),
        }

    # SeamlessM4TTextToUnitModel: must NOT pass input_ids, pass inputs_embeds instead
    if name_lower in ("seamlessm4ttexttounitymodel", "seamlessm4ttexttounitemodel",
                       "seamlessm4ttexttounitmodel"):
        hidden = getattr(config, "hidden_size", 1024)
        inputs = {
            "inputs_embeds": torch.randn(B, 32, hidden, device=device),
            "decoder_input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
        }

    # WhisperForCausalLM: decoder-only — needs input_ids (no encoder, no decoder_input_ids)
    if name_lower == "whispermodel" and variant == "causal_lm":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # Pix2StructForConditionalGeneration: needs flattened_patches + decoder_input_ids
    if name_lower == "pix2structmodel" and variant == "conditional_generation":
        num_channels = getattr(config, "num_channels", 3)
        patch_size = getattr(config, "patch_size", 16)
        seq_len = getattr(config, "seq_len", 4096)
        flat_dim = patch_size * patch_size * num_channels + 2
        dec_vocab = getattr(getattr(config, "text_config", config), "vocab_size", 1000)
        flattened_patches = torch.randn(B, 32, flat_dim, device=device)
        flattened_patches[:, :, 0] = torch.randint(0, seq_len, (B, 32), device=device).float()
        flattened_patches[:, :, 1] = torch.randint(0, seq_len, (B, 32), device=device).float()
        inputs = {
            "flattened_patches": flattened_patches,
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "decoder_input_ids": torch.randint(0, min(dec_vocab, 1000), (B, 32), device=device),
        }

    # UdopEncoderModel: needs bbox (float, shape B,seq,4) + pixel_values
    if name_lower == "udopencodermodel":
        img_size = getattr(config, "image_size", 224)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "bbox": torch.rand(B, 32, 4, device=device) * 1000,
            "pixel_values": torch.randn(B, 3, img_size, img_size, device=device),
        }

    # RagModel: context_input_ids first dim must be B*n_docs
    if name_lower == "ragmodel":
        n_docs = getattr(config, "n_docs", 5)
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "context_input_ids": torch.randint(0, vocab_size, (B * n_docs, 32), device=device),
            "context_attention_mask": torch.ones(B * n_docs, 32, dtype=torch.long, device=device),
            "doc_scores": torch.randn(B, n_docs, device=device),
        }

    if name_lower == "glmmoedsamodel":
        text_cfg = getattr(config, "text_config", config)
        voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
        inputs = {
            "input_ids": torch.randint(0, voc, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # DiaModel: text-to-speech with byte-level encoder + multi-channel audio decoder
    # decoder_input_ids must be 3D: (B, seq_len, num_channels=9) for 9 audio codebooks
    if "dia" in name_lower and name_lower in ("diamodel", "diamodel"):
        dec_cfg = getattr(config, "decoder_config", None)
        num_channels = getattr(dec_cfg, "num_channels", 9) if dec_cfg else 9
        enc_vocab = getattr(getattr(config, "encoder_config", None), "vocab_size", 256)
        dec_vocab = getattr(dec_cfg, "vocab_size", 1028) if dec_cfg else 1028
        inputs = {
            "input_ids": torch.randint(0, min(enc_vocab, 256), (B, 32), device=device),
            "decoder_input_ids": torch.randint(0, min(dec_vocab, 1028), (B, 32, num_channels), device=device),
        }
    if name_lower == "diaforconditionalgeneration":
        dec_cfg = getattr(config, "decoder_config", None)
        num_channels = getattr(dec_cfg, "num_channels", 9) if dec_cfg else 9
        enc_vocab = getattr(getattr(config, "encoder_config", None), "vocab_size", 256)
        dec_vocab = getattr(dec_cfg, "vocab_size", 1028) if dec_cfg else 1028
        inputs = {
            "input_ids": torch.randint(0, min(enc_vocab, 256), (B, 32), device=device),
            "decoder_input_ids": torch.randint(0, min(dec_vocab, 1028), (B, 32, num_channels), device=device),
        }

    # CsmBackboneModel: audio codebook model — needs 3D input_ids (B, seq_len, num_codebooks)
    # Embedding layer does .sum(dim=2) over codebook axis; 2D input collapses hidden dim.
    if name_lower == "csmbackbonemodel":
        num_codebooks = getattr(config, "num_codebooks", 32)
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32, num_codebooks), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # Qwen2_5OmniTalkerForConditionalGeneration: text-only
    if name_lower == "qwen2_5omnitalkermodel" and variant == "conditional_generation":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # ForCausalLM: add labels for loss computation (tests the loss path too)
    if variant == "causal_lm" and "input_ids" in inputs:
        inputs["labels"] = inputs["input_ids"].clone()

    # Musicgen ForCausalLM: internally expands batch by num_codebooks, so labels
    # shape (B, seq_len) mismatches output (B*num_codebooks, seq_len). Remove labels.
    if variant == "causal_lm" and "musicgen" in name_lower:
        inputs.pop("labels", None)

    # GitForCausalLM: loss path shifts logits by num_image_tokens, causing
    # batch_size mismatch in cross_entropy when no pixel_values provided.
    if variant == "causal_lm" and "git" in name_lower:
        inputs.pop("labels", None)

    # MoshiForConditionalGeneration: needs both user and moshi audio codes
    if name_lower == "moshimodel" and variant == "conditional_generation":
        num_codebooks = getattr(config, "num_codebooks", 8)
        audio_vocab = getattr(config, "audio_vocab_size", 2048)
        seq_len = inputs.get("input_ids", torch.empty(0, 32)).shape[-1]
        inputs["user_audio_codes"] = torch.randint(
            0, min(audio_vocab, 256), (B, num_codebooks, seq_len), device=device
        )
        inputs["moshi_audio_codes"] = torch.randint(
            0, min(audio_vocab, 256), (B, num_codebooks, seq_len), device=device
        )

    # PeAudioFrameLevelModel: dual-encoder, needs both input_ids and 3D input_values
    if name_lower == "peaudioframelevelmodel":
        text_vocab = getattr(config, "vocab_size", 32000)
        inputs = {
            "input_ids": torch.randint(0, min(text_vocab, 256), (B, 32), device=device),
            "input_values": torch.randn(B, 1, 16000, device=device),
        }

    # PeAudioVideoModel: needs at least 2 of input_ids/input_values/pixel_values_videos
    if name_lower == "peaudiovideomodel":
        text_vocab = getattr(config, "vocab_size", 32000)
        inputs = {
            "input_ids": torch.randint(0, min(text_vocab, 256), (B, 32), device=device),
            "input_values": torch.randn(B, 1, 16000, device=device),
        }

    # PI0: robotics model needs state tensor and input_ids (text)
    if "pi0" in name_lower:
        state_dim = getattr(config, "max_action_dim", 32)
        inputs["state"] = torch.randn(B, state_dim, device=device)

    # Musicgen/MusicgenMelody ForConditionalGeneration: encoder gets (B, seq),
    # decoder gets (B * num_codebooks, seq) — different batch sizes.
    if variant == "conditional_generation" and "musicgen" in name_lower:
        dec_cfg = getattr(config, "decoder", None)
        num_codebooks = getattr(dec_cfg, "num_codebooks", 4) if dec_cfg else 4
        dec_vocab = getattr(dec_cfg, "vocab_size", 2048) if dec_cfg else 2048
        enc_vocab = min(getattr(config, "vocab_size", 32000) or 32000, 256)
        inputs = {
            "input_ids": torch.randint(0, enc_vocab, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
            "decoder_input_ids": torch.randint(0, min(dec_vocab, 256), (B * num_codebooks, 32), device=device),
        }

    # ForConditionalGeneration: ensure multimodal inputs for vision-language models.
    # Many base model handlers above force text-only inputs (via _vl_text_only_models
    # or model-specific overrides). For base models that's correct — they crash with
    # raw pixel_values. But ForConditionalGeneration wraps the full pipeline (vision
    # encoder + merge + LM), so it needs actual multimodal inputs to test the
    # integration layer where real graph breaks occur.
    if variant == "conditional_generation" and "pixel_values" not in inputs and "flattened_patches" not in inputs:
        vision_cfg = getattr(config, "vision_config", getattr(config, "image_config", None))
        if vision_cfg:
            model_fwd_params = set(inspect.signature(model.forward).parameters.keys())
            text_cfg = getattr(config, "text_config", config)
            voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)

            # Common: resolve image_token_id
            image_token_id = getattr(config, "image_token_id",
                                     getattr(config, "image_token_index", None))

            # Helper: build input_ids with image token placeholders
            def _build_ids(n_vis, token_id=image_token_id):
                text_len = 8
                if token_id is not None:
                    sl = text_len + n_vis
                    _ids = torch.randint(0, voc, (B, sl), device=device)
                    _ids[:, :n_vis] = token_id
                    if token_id >= voc:
                        model.resize_token_embeddings(token_id + 1)
                else:
                    sl = 32
                    _ids = torch.randint(0, voc, (B, sl), device=device)
                return _ids, sl

            name_lower_fcg = base_model_name.lower()

            # ── Models where FCG should use text-only (vision path broken) ──
            _fcg_text_only = {
                # GlmImage: temporal_patch_size=1 causes grid_thw dimension mismatch
                "glmimagemodel",
                # PerceptionLM: TimmWrapper/ResNet returns 4D features, projector expects 3D
                "perceptionlmmodel",
            }
            if name_lower_fcg in _fcg_text_only:
                pass  # keep text-only inputs from base model handler

            # ── Model-specific overrides (before generic paths) ──

            # Mistral3/LightOnOcr: standard 4D pixel_values + image_sizes,
            # but n_vis must account for spatial_merge_size
            elif name_lower_fcg in ("mistral3model", "lightonocrmodel"):
                img_size = getattr(vision_cfg, "image_size", 224) or 224
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                patch_size = getattr(vision_cfg, "patch_size", 14)
                merge_size = getattr(config, "spatial_merge_size", 2)
                patches_per_side = img_size // patch_size
                # After spatial merging: features = (patches_per_side / merge_size) ^ 2
                n_vis = (patches_per_side // merge_size) ** 2
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, num_ch, img_size, img_size, device=device),
                    "image_sizes": torch.tensor([[img_size, img_size]] * B, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # Idefics3/SmolVLM/AyaVision: 5D pixel_values + pixel_attention_mask,
            # n_vis must account for pixel_shuffle scale_factor
            elif name_lower_fcg in ("idefics3model", "smolvlmmodel"):
                img_size = getattr(vision_cfg, "image_size", 256) or 256
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                patch_size = getattr(vision_cfg, "patch_size", 32)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                scale = getattr(config, "scale_factor", 2)
                patches_per_side = img_size // patch_size
                n_vis = (patches_per_side // scale) ** 2
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, 1, num_ch, img_size, img_size, device=device),
                    "pixel_attention_mask": torch.ones(B, 1, img_size, img_size,
                                                       dtype=torch.long, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # AyaVision/Cohere2Vision: 4D pixel_values + image_sizes,
            # n_vis must account for downsample_factor pixel_shuffle
            elif name_lower_fcg in ("ayavisionmodel", "cohere2visionmodel"):
                img_size = getattr(vision_cfg, "image_size", 256) or 256
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                patch_size = getattr(vision_cfg, "patch_size", 32)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                scale = getattr(config, "downsample_factor", 2)
                patches_per_side = img_size // patch_size
                n_vis = (patches_per_side // scale) ** 2
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, num_ch, img_size, img_size, device=device),
                    "image_sizes": torch.tensor([[img_size, img_size]] * B, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # FastVlm: timm ConvNet vision tower — output token count is
            # unpredictable from config (no patch_size). Use text-only.
            elif "fastvlm" in name_lower_fcg:
                text_cfg_fcg = getattr(config, "text_config", config)
                voc_fcg = min(getattr(text_cfg_fcg, "vocab_size", 32000) or 32000, 1000)
                inputs = {
                    "input_ids": torch.randint(0, voc_fcg, (B, 32), device=device),
                    "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
                }

            # Lfm2Vl: SigLIP2 vision tower expects patchified 3D pixel_values
            # (B, num_patches, C*patch_h*patch_w) + spatial_shapes + pixel_attention_mask
            elif "lfm2vl" in name_lower_fcg:
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                patch_size = getattr(vision_cfg, "patch_size", 16)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                downsample = getattr(config, "downsample_factor", 2)
                feat_h = feat_w = downsample * 2  # 4 patches per side
                num_patches = feat_h * feat_w  # 16
                patch_dim = num_ch * patch_size * patch_size  # 3*16*16 = 768
                n_vis = (feat_h // downsample) * (feat_w // downsample)
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, num_patches, patch_dim, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }
                if "spatial_shapes" in model_fwd_params:
                    inputs["spatial_shapes"] = torch.tensor([[feat_h, feat_w]] * B, device=device)
                if "pixel_attention_mask" in model_fwd_params:
                    # 2D mask (B, num_patches) — get_image_features does
                    # .sum(dim=1) expecting scalar per image for slicing
                    inputs["pixel_attention_mask"] = torch.ones(B, num_patches,
                                                                dtype=torch.long, device=device)

            # Llama4: pixel_shuffle reduces token count by pixel_shuffle_ratio^2
            # pixel_shuffle_ratio can be 0.5 (unshuffle → more channels, fewer patches)
            elif "llama4" in name_lower_fcg:
                img_size = getattr(vision_cfg, "image_size", 224) or 224
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                patch_size = getattr(vision_cfg, "patch_size", 14)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                shuffle_ratio = getattr(vision_cfg, "pixel_shuffle_ratio", 0.5)
                patches_per_side = img_size // patch_size
                n_vis = int(patches_per_side * shuffle_ratio) ** 2
                if n_vis < 1:
                    n_vis = 1
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, num_ch, img_size, img_size, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # DeepseekVLHybrid: needs both pixel_values and high_res_pixel_values
            elif "deepseekvlhybrid" in name_lower_fcg:
                img_size = getattr(vision_cfg, "image_size", 224) or 224
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                high_res_cfg = getattr(config, "high_res_vision_config", None)
                hr_size = getattr(high_res_cfg, "image_size", img_size) if high_res_cfg else img_size
                if isinstance(hr_size, (list, tuple)):
                    hr_size = hr_size[0]
                patch_size = getattr(vision_cfg, "patch_size", 14)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                n_vis = (img_size // patch_size) ** 2 if patch_size > 0 else 196
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, num_ch, img_size, img_size, device=device),
                    "high_res_pixel_values": torch.randn(B, num_ch, hr_size, hr_size, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # PaddleOCR: get_image_features does unsqueeze(0) on pixel_values,
            # so input must be (total_patches, C, patch_size, patch_size) → vision
            # model receives (1, total_patches, C, patch_size, patch_size).
            # total_patches = B * patches_per_image (one image per batch item).
            elif "paddleocr" in name_lower_fcg:
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                merge_size = getattr(vision_cfg, "spatial_merge_size", 2)
                patch_size = getattr(vision_cfg, "patch_size", 14)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                h_patches = w_patches = merge_size * 2  # 4 patches per side
                patches_per_image = h_patches * w_patches  # 16
                total_patches = patches_per_image * B
                n_vis = (h_patches // merge_size) * (w_patches // merge_size)
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(total_patches, num_ch, patch_size, patch_size, device=device),
                    "image_grid_thw": torch.tensor([[1, h_patches, w_patches]] * B, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }
                if "mm_token_type_ids" in model_fwd_params:
                    inputs["mm_token_type_ids"] = torch.zeros(B, seq_len, dtype=torch.long, device=device)

            # PerceptionLM: needs 5D pixel_values (B, num_images, C, H, W)
            elif "perceptionlm" in name_lower_fcg:
                img_size = getattr(vision_cfg, "image_size", 224) or 224
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                n_vis = 16
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, 1, num_ch, img_size, img_size, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # Kosmos2: uses img_input_mask and latent_query_num (default 64)
            # image_to_text_projection outputs (B, latent_query_num, hidden_size)
            elif "kosmos2" in name_lower_fcg:
                img_size = getattr(vision_cfg, "image_size", 224) or 224
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                n_latent = getattr(config, "latent_query_num", 64)
                text_len = 8
                seq_len = text_len + n_latent
                ids = torch.randint(0, voc, (B, seq_len), device=device)
                img_mask = torch.zeros(B, seq_len, dtype=torch.bool, device=device)
                img_mask[:, :n_latent] = True
                inputs = {
                    "pixel_values": torch.randn(B, num_ch, img_size, img_size, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                    "image_embeds_position_mask": img_mask,
                }

            # Ovis2: pixel_values is (num_images, C, H, W), each batch item gets the features
            # Vision model outputs (num_images, seq_len, hidden) where seq_len depends on hidden_stride
            elif "ovis2" in name_lower_fcg:
                img_size = getattr(vision_cfg, "image_size", 224) or 224
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                patch_size = getattr(vision_cfg, "patch_size", 14)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                hidden_stride = getattr(vision_cfg, "hidden_stride", 2) or 1
                patches_per_side = img_size // patch_size
                # After hidden_stride downsampling: (patches_per_side / hidden_stride) ^ 2
                seq_per_image = (patches_per_side // hidden_stride) ** 2
                # Total features = B * seq_per_image. Each batch item gets seq_per_image tokens.
                n_vis = seq_per_image
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, num_ch, img_size, img_size, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # Janus: aligner outputs projection_dim (2048) which gets reshaped
            # to text_config.hidden_size (4096), halving the token count.
            # n_vis = patches_per_image * projection_dim / text_hidden_size
            elif "janus" in name_lower_fcg:
                img_size = getattr(vision_cfg, "image_size", 384) or 384
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_ch = getattr(vision_cfg, "num_channels", 3) or 3
                patch_size = getattr(vision_cfg, "patch_size", 16) or 16
                patches_per_image = (img_size // patch_size) ** 2
                proj_dim = getattr(vision_cfg, "projection_dim", 2048)
                t_hs = getattr(getattr(config, "text_config", config), "hidden_size", 4096)
                n_vis = patches_per_image * proj_dim // t_hs
                ids, seq_len = _build_ids(n_vis)
                inputs = {
                    "pixel_values": torch.randn(B, num_ch, img_size, img_size, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }

            # ── Path 1: Gemma4 pre-patchified vision (nn.Linear patch embed) ──
            elif (hasattr(vision_cfg, "pooling_kernel_size")
                  and "image_position_ids" in model_fwd_params):
                patch_size = getattr(vision_cfg, "patch_size", 16)
                patch_dim = 3 * patch_size * patch_size
                pool_k = getattr(vision_cfg, "pooling_kernel_size", 3)
                num_patches = pool_k ** 4  # 81 for k=3
                grid_side = int(num_patches ** 0.5)
                n_vis_tokens = num_patches // (pool_k ** 2)
                ids, seq_len = _build_ids(n_vis_tokens)
                inputs = {
                    "pixel_values": torch.randn(B, num_patches, patch_dim, device=device),
                    "image_position_ids": torch.randint(0, grid_side, (B, num_patches, 2), device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                    "mm_token_type_ids": torch.zeros(B, seq_len, dtype=torch.long, device=device),
                }

            # ── Path 2: Grid-based vision (Qwen VL, Glm4v, Ernie, PaddleOCR, VideoLlama3) ──
            elif "image_grid_thw" in model_fwd_params:
                patch_size = getattr(vision_cfg, "patch_size", 14)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                temporal_ps = getattr(vision_cfg, "temporal_patch_size", None)
                if temporal_ps is None:
                    temporal_ps = 1
                in_ch = getattr(vision_cfg, "in_channels", 3)
                merge_size = getattr(vision_cfg, "spatial_merge_size", 2)
                # Always use t=1 for image inputs; t>1 is video which creates
                # repeat_interleave issues in models like Glm4v
                t, h, w = 1, merge_size * 2, merge_size * 2
                num_patches_per_image = t * h * w
                patch_dim = in_ch * temporal_ps * patch_size * patch_size
                n_vis_per_image = t * (h // merge_size) * (w // merge_size)
                # Grid models: pixel_values is unbatched (total_patches, patch_dim),
                # image_grid_thw has one entry per image. With B images, we need B entries.
                # Each batch item's input_ids gets n_vis_per_image image tokens.
                total_patches = num_patches_per_image * B
                ids, seq_len = _build_ids(n_vis_per_image)
                inputs = {
                    "pixel_values": torch.randn(total_patches, patch_dim, device=device),
                    "image_grid_thw": torch.tensor([[t, h, w]] * B, device=device),
                    "input_ids": ids,
                    "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                }
                if "mm_token_type_ids" in model_fwd_params:
                    inputs["mm_token_type_ids"] = torch.zeros(B, seq_len, dtype=torch.long, device=device)
                # VideoLlama3: requires image_merge_sizes for pixel_unshuffle split
                if "image_merge_sizes" in model_fwd_params:
                    inputs["image_merge_sizes"] = torch.tensor(
                        [merge_size] * B, device=device)

            # ── Path 3: Standard vision (B, C, H, W) with optional image_sizes ──
            else:
                img_size = getattr(vision_cfg, "image_size", 224) or 224
                if isinstance(img_size, (list, tuple)):
                    img_size = img_size[0]
                num_channels = getattr(vision_cfg, "num_channels",
                                       getattr(vision_cfg, "in_channels", 3)) or 3
                patch_size = getattr(vision_cfg, "patch_size", 14)
                if isinstance(patch_size, (list, tuple)):
                    patch_size = patch_size[0]
                n_vis_tokens = (img_size // patch_size) ** 2 if patch_size > 0 else 196

                # LlavaNext/LlavaOnevision/LlavaNextVideo: need 5D pixel_values
                # (B, num_patches_per_image, C, H, W) where num_patches comes from
                # image_grid_pinpoints. image_sizes is required.
                # pack_image_features adds newline tokens per row for grid patches,
                # so n_vis = base(ps^2) + grid_unpadded(ps*(ps+1)) = 2*ps^2 + ps.
                grid_pinpoints = getattr(config, "image_grid_pinpoints", None)
                if grid_pinpoints and "image_sizes" in model_fwd_params:
                    try:
                        from transformers.models.llava_next.modeling_llava_next import (
                            image_size_to_num_patches,
                        )
                        num_patches = image_size_to_num_patches(
                            torch.tensor([img_size, img_size]),
                            grid_pinpoints, img_size,
                        )
                    except Exception:
                        num_patches = 1
                    ps = img_size // patch_size if patch_size > 0 else 14
                    # With single-tile grid [[img_size, img_size]]:
                    # base = ps^2, grid (1x1 after unpad for square) = ps*(ps+1) with newlines
                    n_vis_tokens = ps * ps + ps * (ps + 1)
                    ids, seq_len = _build_ids(n_vis_tokens)
                    inputs = {
                        "pixel_values": torch.randn(B, num_patches, num_channels,
                                                    img_size, img_size, device=device),
                        "image_sizes": torch.tensor([[img_size, img_size]] * B, device=device),
                        "input_ids": ids,
                        "attention_mask": torch.ones(B, seq_len, dtype=torch.long,
                                                     device=device),
                    }
                else:
                    ids, seq_len = _build_ids(n_vis_tokens)

                    # Idefics3/SmolVLM: expects 5D pixel_values (B, num_images, C, H, W)
                    if "pixel_attention_mask" in model_fwd_params:
                        inputs = {
                            "pixel_values": torch.randn(B, 1, num_channels, img_size,
                                                        img_size, device=device),
                            "pixel_attention_mask": torch.ones(B, 1, img_size, img_size,
                                                               dtype=torch.long, device=device),
                            "input_ids": ids,
                            "attention_mask": torch.ones(B, seq_len, dtype=torch.long,
                                                         device=device),
                        }
                    else:
                        inputs = {
                            "pixel_values": torch.randn(B, num_channels, img_size, img_size,
                                                        device=device),
                            "input_ids": ids,
                            "attention_mask": torch.ones(B, seq_len, dtype=torch.long,
                                                         device=device),
                        }
                    # image_sizes: needed by Mistral3, Cohere2Vision, etc.
                    # Exclude Llava base (crashes with vision_tower.patch_size)
                    # and VipLlava (similar issue)
                    no_image_sizes = ("llavamodel" in name_lower_fcg
                                      or "vipllava" in name_lower_fcg)
                    if ("image_sizes" in model_fwd_params
                            and not no_image_sizes):
                        inputs["image_sizes"] = torch.tensor(
                            [[img_size, img_size]] * B, device=device)
                    # aspect_ratio_ids: Mllama requires it
                    if "aspect_ratio_ids" in model_fwd_params:
                        max_tiles = getattr(vision_cfg, "max_num_tiles", 4)
                        inputs["aspect_ratio_ids"] = torch.tensor([[1]] * B, device=device)
                        inputs["aspect_ratio_mask"] = torch.ones(
                            B, 1, max_tiles, dtype=torch.long, device=device)
                        # Mllama expects 5D pixel_values: (B, num_images, max_tiles, C, H, W)
                        inputs["pixel_values"] = torch.randn(
                            B, 1, max_tiles, num_channels, img_size, img_size,
                            device=device)
                        inputs["cross_attention_mask"] = torch.ones(
                            B, seq_len, 1, max_tiles,
                            dtype=torch.long, device=device)

            # Common post-processing for all paths
            if "token_type_ids" in model_fwd_params and "token_type_ids" not in inputs:
                sl = inputs["input_ids"].shape[1]
                inputs["token_type_ids"] = torch.zeros(B, sl, dtype=torch.long, device=device)

    # ClvpModelForConditionalGeneration: text+speech contrastive model
    if name_lower == "clvpmodel" and variant == "conditional_generation":
        speech_cfg = getattr(config, "speech_config", None)
        text_cfg = getattr(config, "text_config", config)
        text_voc = min(getattr(text_cfg, "vocab_size", 256) or 256, 256)
        num_mel = getattr(speech_cfg, "num_mel_bins", 80) if speech_cfg else 80
        inputs = {
            "input_ids": torch.randint(0, text_voc, (B, 32), device=device),
            "input_features": torch.randn(B, num_mel, 100, device=device),
        }

    # Qwen3OmniMoeTalkerForConditionalGeneration: text-only
    if name_lower in ("qwen3omnioetalkermodel", "qwen3omnimoetalkermodel") and variant == "conditional_generation":
        text_cfg = getattr(config, "text_config", config)
        text_voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
        inputs = {
            "input_ids": torch.randint(0, text_voc, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # ── Model-specific Audio FCG overrides ──
    # Qwen2Audio: audio token count must match encoder output exactly.
    # The encoder pipeline has multiple downsampling stages; use
    # _get_feat_extract_output_lengths to compute the final audio feature count.
    if name_lower == "qwen2audiomodel" and variant == "conditional_generation":
        audio_cfg = getattr(config, "audio_config", config)
        text_cfg = getattr(config, "text_config", config)
        text_voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
        num_mel = getattr(audio_cfg, "num_mel_bins", 128)
        mel_len = 3000
        # Probe encoder to get actual output feature count after all downsampling
        n_feat = mel_len // 16  # conservative fallback
        feat_mask_len = mel_len // 4
        audio_enc = getattr(model, "audio_tower", None)
        if audio_enc is not None and hasattr(audio_enc, "_get_feat_extract_output_lengths"):
            with torch.no_grad():
                probe = torch.randn(1, num_mel, mel_len, device=device)
                probe_out = audio_enc(probe)
                feat_mask_len = probe_out.last_hidden_state.shape[1] if hasattr(probe_out, "last_hidden_state") else probe_out.shape[1]
                _, out_lens = audio_enc._get_feat_extract_output_lengths(
                    torch.tensor([feat_mask_len], device=device)
                )
                n_feat = out_lens[0].item()
        text_len = 8
        seq_len = text_len + n_feat
        ids = torch.randint(0, text_voc, (B, seq_len), device=device)
        audio_token_id = getattr(config, "audio_token_id", None)
        if audio_token_id is not None:
            ids[:, :n_feat] = audio_token_id
            if audio_token_id >= text_voc:
                model.resize_token_embeddings(audio_token_id + 1)
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
            "input_features": torch.randn(B, num_mel, mel_len, device=device),
            "feature_attention_mask": torch.ones(B, feat_mask_len, dtype=torch.long, device=device),
        }

    # Voxtral: text-only avoids broken audio path
    if name_lower == "voxtralmodel" and variant == "conditional_generation":
        text_cfg = getattr(config, "text_config", config)
        text_voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
        inputs = {
            "input_ids": torch.randint(0, text_voc, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # VoxtralRealtime: input_ids seq_len must match audio pooler output length
    if name_lower == "voxtralrealtimemodel" and variant == "conditional_generation":
        text_cfg = getattr(config, "text_config", config)
        audio_cfg = getattr(config, "audio_config", config)
        text_voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
        num_mel = getattr(audio_cfg, "num_mel_bins", getattr(audio_cfg, "feature_size", 128))
        max_src = getattr(audio_cfg, "max_source_positions", 1500)
        mel_len = max_src * 2
        feats = torch.randn(B, num_mel, mel_len, device=device)
        # Probe audio encoder to get pooler output seq_len
        with torch.no_grad():
            audio_out = model.get_audio_features(input_features=feats)
            text_len = audio_out.pooler_output.shape[1]
        inputs = {
            "input_ids": torch.randint(0, text_voc, (B, text_len), device=device),
            "attention_mask": torch.ones(B, text_len, dtype=torch.long, device=device),
            "input_features": feats,
        }

    # ── Audio ForConditionalGeneration: need input_ids + audio features ──
    # Audio FCG models are classified as "audio" by _detect_hf_input_type and get
    # audio-only inputs. But as FCG they need both audio AND text (input_ids with
    # audio token placeholders + input_features mel spectrogram).
    if variant == "conditional_generation" and "input_ids" not in inputs:
        audio_cfg = getattr(config, "audio_config", None)
        text_cfg = getattr(config, "text_config", None)
        if audio_cfg and text_cfg:
            text_voc = min(getattr(text_cfg, "vocab_size", 32000) or 32000, 1000)
            num_mel_bins = getattr(audio_cfg, "num_mel_bins", 128)
            # Most audio FCG models require specific mel lengths:
            # Qwen2Audio/Voxtral require exactly 3000 frames; AudioFlamingo3 needs
            # max_source_positions * 2; GlmAsr uses conv downsampling.
            max_src = getattr(audio_cfg, "max_source_positions", 1500)
            audio_seq_len = max_src * 2  # 3000 for most models
            # Audio token count: use a small fixed count. The audio encoder's
            # output length varies per model; exact matching requires model-specific
            # handlers (Qwen2Audio, Voxtral need ~1500 tokens = encoder output).
            n_audio_tokens = 8
            text_len = 8
            seq_len = text_len + n_audio_tokens
            ids = torch.randint(0, text_voc, (B, seq_len), device=device)
            audio_token_id = getattr(config, "audio_token_id", None)
            if audio_token_id is not None:
                ids[:, :n_audio_tokens] = audio_token_id
                if audio_token_id >= text_voc:
                    model.resize_token_embeddings(audio_token_id + 1)
            inputs = {
                "input_ids": ids,
                "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device=device),
                "input_features": torch.randn(B, num_mel_bins, audio_seq_len, device=device),
            }
            fwd_params = set(inspect.signature(model.forward).parameters.keys())
            if "input_features_mask" in fwd_params or "feature_attention_mask" in fwd_params:
                mask_key = "feature_attention_mask" if "feature_attention_mask" in fwd_params else "input_features_mask"
                inputs[mask_key] = torch.ones(B, audio_seq_len, dtype=torch.long, device=device)

    return model, inputs, None


def _detect_hf_input_type(model_name, config):
    """Detect input modality from model name and config attributes."""
    name_lower = model_name.lower()

    # Check config for multimodal (vision + text sub-configs)
    has_vision_config = hasattr(config, "vision_config") or hasattr(config, "image_config")
    has_text_config = hasattr(config, "text_config")
    if has_vision_config and has_text_config:
        return "multimodal"

    # Vision+text without vision_config (backbone-based vision + text)
    # e.g., GroundingDino, MMGroundingDino
    model_type = getattr(config, "model_type", "")
    grounding_types = {"grounding-dino", "mm-grounding-dino"}
    if model_type in grounding_types or ("groundingdino" in name_lower and has_text_config):
        return "multimodal"

    # Video-text multimodal (needs pixel_values_videos + input_ids)
    # e.g., PeVideoModel — has text_config but video forward
    if ("pevideo" in name_lower or model_type == "pe_video") and has_text_config:
        return "video_multimodal"

    # Vision-only with sub-config (e.g., Sam, Sam2) — has vision_config but no text_config
    if has_vision_config and not has_text_config and not hasattr(config, "vocab_size"):
        return "vision"

    # Check config for audio sub-configs
    has_audio_config = hasattr(config, "audio_config") or hasattr(config, "speech_config")
    if has_audio_config:
        return "audio"

    # Image pair matching models (5D input: B, 2, C, H, W)
    if "efficientloftr" in name_lower or ("loftr" in name_lower and "image" not in name_lower):
        return "image_pair"

    # Video models — check before general vision (they need 5D input)
    video_keywords = ["timesformer", "videomae", "vivit", "xclip", "video"]
    if any(kw in name_lower for kw in video_keywords) and hasattr(config, "num_frames"):
        return "video"
    # VJEPA2 is a video model with unique arg name (pixel_values_videos)
    if "vjepa2" in name_lower and hasattr(config, "frames_per_clip"):
        return "video"

    # Time series models
    time_series_keywords = ["patchtsmixer", "patchtst", "timesfm"]
    if any(kw in name_lower for kw in time_series_keywords):
        return "time_series"
    if hasattr(config, "context_length") and hasattr(config, "patch_length"):
        return "time_series"

    # Audio codec models (encode/decode raw waveforms, NOT text)
    audio_codec_keywords = ["encodec", "dac", "xcodec", "mimi"]
    model_type_lower = getattr(config, "model_type", "").lower()
    if any(kw in name_lower for kw in audio_codec_keywords) or any(kw in model_type_lower for kw in audio_codec_keywords):
        return "audio_codec"

    # Vision models — check config attributes first, then name
    if hasattr(config, "image_size") and not hasattr(config, "vocab_size"):
        return "vision"
    vision_keywords = [
        "vit", "swin", "deit", "beit", "convnext", "resnet", "poolformer",
        "pvt", "segformer", "dinat", "nat", "levit", "efficientnet",
        "mobilevit", "mobilenet", "regnet", "van", "dinov2", "hiera",
        "siglip", "image", "cvt", "focalnet", "bit", "dpt", "glpn",
        # Object detection / segmentation / depth / video models
        "detr", "dfine", "rtdetr", "yolos", "deformable",
        "seggpt", "segformer", "maskformer", "mask2former",
        "depth", "loftr", "uvdoc", "vjepa",
        "ppdoclayout", "ppocr", "timmwrapper",
        "tabletransformer", "groundingdino",
        # Vision models with vocab_size (dVAE codebook, not text)
        "flavaimage",
    ]
    # Only match vision if the model is NOT also a text model
    # Note: some vision models have vocab_size for codebook (BEiT, Flava) — not text
    has_vocab = hasattr(config, "vocab_size") and config.vocab_size is not None
    has_image_size = hasattr(config, "image_size")
    # Models with image_size + vocab_size that are truly vision (dVAE codebook)
    codebook_vision = has_image_size and has_vocab and not hasattr(config, "pad_token_id")
    if any(kw in name_lower for kw in vision_keywords) and (not has_vocab or codebook_vision):
        return "vision"
    # Also check model_type for vision patterns
    model_type = getattr(config, "model_type", "")
    vision_model_types = ["detr", "yolos", "depth", "seggpt", "vjepa", "rt_detr", "d_fine", "sam2"]
    if any(vt in model_type for vt in vision_model_types) and not has_vocab:
        return "vision"

    # Audio models
    model_type = getattr(config, "model_type", "")
    audio_keywords = [
        "whisper", "wav2vec", "hubert", "audio", "unispeech", "wavlm", "sew", "ast",
        "univnet", "fastspeech2", "clapaudio", "speech2text", "speech_to_text", "speecht5",
        "moonshine",
    ]
    if any(kw in name_lower for kw in audio_keywords) or "audio" in model_type:
        return "audio"

    # Seq2seq models
    if getattr(config, "is_encoder_decoder", False):
        return "seq2seq"
    seq2seq_keywords = [
        "t5", "bart", "mbart", "pegasus", "marian", "blenderbot",
        "prophetnet", "led", "longt5", "mt5", "umt5", "nllb", "seamless",
        "m2m", "plbart", "bigbirdpegasus",
    ]
    if any(kw in name_lower for kw in seq2seq_keywords):
        return "seq2seq"

    return "text"


def create_diffusers_model(spec, device):
    import diffusers

    model_name = spec.get("hf_class") or spec["name"]
    model_cls = getattr(diffusers, model_name)

    # Use provided config or try minimal defaults
    constructor_args = spec.get("constructor_args", {})
    model = model_cls(**constructor_args).to(device)

    # Use provided inputs or try to generate from spec
    input_spec = spec.get("inputs", {})
    if input_spec:
        inputs = {}
        for k, shape in input_spec.items():
            if isinstance(shape, list):
                inputs[k] = torch.randn(*shape, device=device)
            else:
                inputs[k] = torch.tensor(shape, device=device)
        # Fix 0D timestep → 1D (diffusion models need 1D timestep array)
        if "timestep" in inputs and inputs["timestep"].ndim == 0:
            batch_size = next(
                (v.shape[0] for v in inputs.values() if isinstance(v, torch.Tensor) and v.ndim > 0), 1
            )
            inputs["timestep"] = torch.randint(0, 1000, (batch_size,), device=device)
        # Fix class_labels: should be long integer indices, not float
        if "class_labels" in inputs and inputs["class_labels"].is_floating_point():
            num_classes = constructor_args.get("num_embeds_ada_norm", constructor_args.get("num_classes", 10))
            inputs["class_labels"] = torch.randint(0, num_classes, inputs["class_labels"].shape, device=device)
        return model, inputs, None

    # Fallback: try positional args based on model type
    return model, {}, None


def create_model(spec, device, batch_size=DEFAULT_BATCH):
    """Factory: create model + inputs based on source."""
    source = spec["source"]
    if source == "timm":
        return create_timm_model(spec, device, batch_size=batch_size)
    elif source == "hf":
        return create_hf_model(spec, device, batch_size=batch_size)
    elif source == "diffusers":
        return create_diffusers_model(spec, device)
    else:
        raise ValueError(f"Unknown source: {source}")


def _create_inputs_only(spec, device, batch_size=DEFAULT_BATCH):
    """Create only inputs (no model retained) for a given spec.

    Used by validation to get shape B inputs without keeping a second model in memory.
    """
    model, inputs_dict, inputs_tuple = create_model(spec, device, batch_size=batch_size)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return inputs_dict, inputs_tuple


# ─── Identify: fullgraph identification ──────────────────────────────────────

def _mark_dynamic_dims(inputs_dict, inputs_tuple, source, input_type):
    """Mark realistic dynamic dimensions on inputs based on model modality.

    - Batch (dim 0): always dynamic
    - Sequence length (dim 1 for NLP/audio): dynamic
    - Spatial dims (dim 2,3 for vision): NOT dynamic (architectures require fixed sizes)
    - Channel/hidden dims: NOT dynamic
    """
    if inputs_tuple:
        for t in inputs_tuple:
            if isinstance(t, torch.Tensor) and t.ndim >= 1:
                torch._dynamo.mark_dynamic(t, 0)  # batch
                # Vision: only batch is dynamic (spatial dims are architecture-fixed)
    elif inputs_dict:
        for key, t in inputs_dict.items():
            if not isinstance(t, torch.Tensor) or t.ndim < 1:
                continue
            torch._dynamo.mark_dynamic(t, 0)  # batch always
            if t.ndim >= 2 and key in (
                "input_ids", "attention_mask", "decoder_input_ids",
                "decoder_attention_mask", "token_type_ids", "position_ids",
                "input_features", "input_values",
            ):
                torch._dynamo.mark_dynamic(t, 1)  # seq_len / time


def run_identify(spec, device, mode, dynamic=False, compile_kwargs=None):
    """Compile a model and check for errors. Returns JSON-serializable result.

    With default compile_kwargs (fullgraph=True, backend="eager"), this is the
    original graph break identification pass. With custom compile_kwargs, it
    tests arbitrary torch.compile configurations.
    """
    t_start = time.perf_counter()
    result = {
        "name": spec["name"],
        "source": spec["source"],
        "mode": mode,
        "pass": "identify",
        "dynamic": dynamic,
        "phase": "create",  # tracks current phase for timeout diagnosis
    }
    if spec.get("variant"):
        result["variant"] = spec["variant"]

    # Phase markers to stderr — orchestrator reads these on timeout
    print("PHASE:create", file=sys.stderr, flush=True)
    try:
        model, inputs_dict, inputs_tuple = create_model(spec, device)
    except Exception as e:
        result["status"] = "create_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        result["create_time_s"] = round(time.perf_counter() - t_start, 3)
        return result
    result["create_time_s"] = round(time.perf_counter() - t_start, 3)

    if mode == "train":
        model.train()
    else:
        model.eval()
    ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

    # Step 1: Eager baseline
    result["phase"] = "eager"
    print("PHASE:eager", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    t_eager = time.perf_counter()
    try:
        with ctx:
            if inputs_tuple:
                model(*inputs_tuple)
            else:
                model(**inputs_dict)
    except Exception as e:
        err_str = str(e)
        err_type = type(e).__name__
        # Image token mismatch: retry with correct token count
        retried = False
        if "Image features and image tokens do not match" in err_str and inputs_dict:
            import re
            match = re.search(r'tokens:\s*(\d+),\s*features:\s*(\d+)', err_str)
            if match:
                need_features = int(match.group(2))
                config = None
                try:
                    import transformers
                    cfg_name = spec.get("hf_config") or spec["name"].replace("Model", "Config")
                    config = getattr(transformers, cfg_name)()
                except Exception:
                    pass
                image_token_id = None
                if config:
                    image_token_id = getattr(config, "image_token_id", None)
                    if image_token_id is None:
                        image_token_id = getattr(config, "image_token_index", None)
                if image_token_id is not None and need_features <= 50000:
                    # Inject exactly need_features image tokens across the batch
                    B = 2
                    vocab_size = getattr(config, "vocab_size", 32000)
                    tokens_per_seq = (need_features + B - 1) // B
                    new_seq_len = tokens_per_seq + 8
                    new_ids = torch.randint(0, min(vocab_size, 1000),
                                            (B, new_seq_len), device=device)
                    new_ids[:, :tokens_per_seq] = image_token_id
                    # Trim excess so total = need_features exactly
                    total = tokens_per_seq * B
                    if total > need_features:
                        excess = total - need_features
                        new_ids[-1, tokens_per_seq - excess:tokens_per_seq] = 2
                    inputs_dict["input_ids"] = new_ids
                    inputs_dict["attention_mask"] = torch.ones(
                        B, new_seq_len, dtype=torch.long, device=device)
                    if "token_type_ids" in inputs_dict:
                        inputs_dict["token_type_ids"] = torch.zeros(
                            B, new_seq_len, dtype=torch.long, device=device)
                    print("PHASE:eager_retry_image_tokens", file=sys.stderr, flush=True)
                    try:
                        with ctx:
                            model(**inputs_dict)
                        retried = True
                        result["eager_time_s"] = round(time.perf_counter() - t_eager, 3)
                        result["image_token_retry"] = need_features
                    except Exception as e2:
                        result["status"] = "eager_error"
                        result["error"] = str(e2)[:500]
                        result["error_type"] = type(e2).__name__
                        result["eager_time_s"] = round(time.perf_counter() - t_eager, 3)
                        result["image_token_retry_failed"] = need_features
                        _cleanup(model, device)
                        return result
        if not retried:
            result["status"] = "eager_error"
            result["error"] = err_str[:500]
            result["error_type"] = err_type
            result["eager_time_s"] = round(time.perf_counter() - t_eager, 3)
            _cleanup(model, device)
            return result
    result["eager_time_s"] = round(time.perf_counter() - t_eager, 3)

    # Step 2: torch.compile
    result["phase"] = "compile"
    print("PHASE:compile", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    t_compile = time.perf_counter()

    # Build compile kwargs: default to fullgraph=True + eager (graph break detection)
    effective_kwargs = {"fullgraph": True, "backend": "eager"}
    if compile_kwargs:
        effective_kwargs.update(compile_kwargs)
    result["compile_kwargs"] = effective_kwargs
    uses_fullgraph = effective_kwargs.get("fullgraph", False)

    try:
        # dynamic modes: True = all dims symbolic, "mark" = realistic dims only, False = static
        compile_dynamic = True if dynamic is True else None
        if dynamic == "mark":
            input_type = spec.get("input_type", "auto")
            _mark_dynamic_dims(inputs_dict, inputs_tuple, spec["source"], input_type)
        # User-supplied compile_kwargs win — they may explicitly set `dynamic`
        # via --compile-kwargs (e.g. fullgraph runs that want dynamic=True regardless
        # of the CLI --dynamic flag).
        if "dynamic" not in effective_kwargs:
            effective_kwargs["dynamic"] = compile_dynamic
        compiled = torch.compile(model, **effective_kwargs)
        with ctx:
            if inputs_tuple:
                compiled(*inputs_tuple)
            else:
                compiled(**inputs_dict)
        result["status"] = "full_graph" if uses_fullgraph else "success"
        if uses_fullgraph:
            result["fullgraph_ok"] = True
    except Exception as e:
        err_str = str(e)[:500]
        err_type = type(e).__name__

        if uses_fullgraph:
            # Distinguish real graph breaks from config/infrastructure errors.
            # Graph breaks come from torch._dynamo internals; config errors are
            # AttributeError, TypeError, or errors with telltale patterns that
            # indicate the model itself failed, not that Dynamo hit a graph break.
            NON_GRAPH_BREAK_TYPES = (AttributeError, TypeError, ImportError)
            NON_GRAPH_BREAK_PATTERNS = [
                "object has no attribute",
                "missing 1 required positional argument",
                "Image features and image tokens do not match",
                "CUDA out of memory",
                "cannot be instantiated",
                "prediction_length",
            ]
            is_infra_error = (
                isinstance(e, NON_GRAPH_BREAK_TYPES)
                or any(p in err_str for p in NON_GRAPH_BREAK_PATTERNS)
            )

            if is_infra_error:
                result["status"] = "compile_error"
                result["fullgraph_ok"] = False
                result["error"] = err_str
                result["error_type"] = err_type
            else:
                result["status"] = "graph_break"
                result["fullgraph_ok"] = False
                result["fullgraph_error"] = err_str
        else:
            # Non-fullgraph mode: any exception is an error
            result["status"] = "error"
            result["error"] = err_str
            result["error_type"] = err_type
    result["compile_time_s"] = round(time.perf_counter() - t_compile, 3)
    result["phase"] = "done"

    if device == "cuda":
        result["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    _cleanup(model, device)
    return result


# ─── Validation: two-shape correctness check ─────────────────────────────────

# Validation shapes — avoid 0, 1 (specialized by PyTorch)
SHAPE_A_BATCH = 2
SHAPE_A_SEQ = 32
SHAPE_B_BATCH = 4
SHAPE_B_SEQ = 48


def _compare_outputs(out_a, out_b, atol=1e-4, rtol=1e-4):
    """Compare two model outputs. Returns (match: bool, max_diff: float, details: str)."""
    def _to_tensors(out):
        if isinstance(out, torch.Tensor):
            return [out]
        if hasattr(out, "values"):  # ModelOutput / dict-like
            return [v for v in out.values() if isinstance(v, torch.Tensor)]
        if isinstance(out, (tuple, list)):
            tensors = []
            for v in out:
                if isinstance(v, torch.Tensor):
                    tensors.append(v)
                elif isinstance(v, (tuple, list)):
                    tensors.extend(t for t in v if isinstance(t, torch.Tensor))
            return tensors
        return []

    tensors_a = _to_tensors(out_a)
    tensors_b = _to_tensors(out_b)

    if len(tensors_a) != len(tensors_b):
        return False, float("inf"), f"output count mismatch: {len(tensors_a)} vs {len(tensors_b)}"

    max_diff = 0.0
    for i, (ta, tb) in enumerate(zip(tensors_a, tensors_b)):
        if ta.shape != tb.shape:
            return False, float("inf"), f"tensor {i} shape mismatch: {ta.shape} vs {tb.shape}"
        if ta.dtype != tb.dtype:
            # Cast to float for comparison
            ta, tb = ta.float(), tb.float()
        diff = (ta - tb).abs().max().item()
        max_diff = max(max_diff, diff)
        if not torch.allclose(ta, tb, atol=atol, rtol=rtol):
            return False, max_diff, f"tensor {i} mismatch: max_diff={diff:.6f}"

    return True, max_diff, "ok"


# ─── Correctness: eager vs compiled output comparison (HF-style) ─────────────


def _is_cache_like(obj):
    """Detect HF Cache objects (DynamicCache, HybridCache, etc.) without hard import."""
    cls_name = type(obj).__name__
    return cls_name.endswith("Cache") and hasattr(obj, "__class__")


def _compare_outputs_recursive(out_eager, out_compiled, atol=1e-6, rtol=1e-4):
    """HF-style recursive walk over ModelOutput/dict/tuple/list structures.

    Returns dict with:
      - status: "match" | "divergence" | "nan_inf_introduced" | "shape_mismatch" | "dtype_mismatch"
      - max_diff: float (worst max-diff across all compared fields)
      - severity_ratio: max_diff / atol (continuous; sort triage by this)
      - compared_fields: list of field paths that were compared
      - skipped_fields: list of (path, reason) tuples
      - first_divergence: path of first failing field (if any)
    """
    state = {
        "status": "match",
        "max_diff": 0.0,
        "compared": [],
        "skipped": [],
        "first_divergence": None,
    }

    def _walk(a, b, path):
        # None
        if a is None and b is None:
            state["skipped"].append((path, "both_none"))
            return
        if a is None or b is None:
            state["skipped"].append((path, f"one_none: eager={a is None}, compiled={b is None}"))
            return

        # Cache objects — skip entirely
        if _is_cache_like(a) or _is_cache_like(b):
            state["skipped"].append((path, f"cache_object: {type(a).__name__}"))
            return

        # Tensors
        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                if state["status"] == "match":
                    state["status"] = "shape_mismatch"
                    state["first_divergence"] = path
                return
            # Shape check
            if a.shape != b.shape:
                if state["status"] == "match":
                    state["status"] = "shape_mismatch"
                    state["first_divergence"] = f"{path} (eager={tuple(a.shape)}, compiled={tuple(b.shape)})"
                return
            # Skip int/bool (HF rationale: argmax-derived, tiny upstream diffs flip results)
            if a.dtype in (torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                state["skipped"].append((path, f"int_or_bool: {a.dtype}"))
                return
            # Skip 0-dim (loss in forward; compared in V2 backward-grad pass)
            if a.dim() == 0:
                state["skipped"].append((path, "zero_dim_scalar"))
                return
            # Dtype must match for float comparison
            if a.dtype != b.dtype:
                if state["status"] == "match":
                    state["status"] = "dtype_mismatch"
                    state["first_divergence"] = f"{path} (eager={a.dtype}, compiled={b.dtype})"
                return
            # NaN/Inf handling
            a_f = a.detach().float()
            b_f = b.detach().float()
            a_nan = torch.isnan(a_f)
            b_nan = torch.isnan(b_f)
            a_inf = torch.isinf(a_f)
            b_inf = torch.isinf(b_f)
            # NaN/Inf introduced by compile (eager clean, compiled dirty)
            if (b_nan & ~a_nan).any() or (b_inf & ~a_inf).any():
                if state["status"] == "match":
                    state["status"] = "nan_inf_introduced"
                    state["first_divergence"] = path
                # Still compute max_diff on finite portion if possible
                finite = ~(a_nan | b_nan | a_inf | b_inf)
                if finite.any():
                    diff = (a_f[finite] - b_f[finite]).abs().max().item()
                    state["max_diff"] = max(state["max_diff"], diff)
                state["compared"].append(path)
                return
            # Strip nan/-inf before max-diff
            finite = ~(a_nan | b_nan | a_inf | b_inf)
            if not finite.any():
                state["skipped"].append((path, "all_nan_or_inf_in_both"))
                return
            diff = (a_f[finite] - b_f[finite]).abs().max().item()
            state["max_diff"] = max(state["max_diff"], diff)
            state["compared"].append(path)
            # Per-field tolerance check (mark divergence but keep walking)
            if not torch.allclose(a_f[finite], b_f[finite], atol=atol, rtol=rtol):
                if state["status"] == "match":
                    state["status"] = "divergence"
                    state["first_divergence"] = f"{path} (max_diff={diff:.3e})"
            return

        # ModelOutput / dict-like (has .keys())
        if hasattr(a, "keys") and hasattr(b, "keys"):
            keys_a = set(a.keys())
            keys_b = set(b.keys())
            if keys_a != keys_b:
                state["skipped"].append((path, f"key_mismatch: only_eager={keys_a - keys_b}, only_compiled={keys_b - keys_a}"))
            for k in keys_a & keys_b:
                _walk(a[k], b[k], f"{path}.{k}" if path else k)
            return

        # tuple / list
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            if len(a) != len(b):
                if state["status"] == "match":
                    state["status"] = "shape_mismatch"
                    state["first_divergence"] = f"{path} (len eager={len(a)}, compiled={len(b)})"
                return
            for i, (ai, bi) in enumerate(zip(a, b)):
                _walk(ai, bi, f"{path}[{i}]")
            return

        # Scalar / unknown — skip with reason
        state["skipped"].append((path, f"unhandled_type: {type(a).__name__}"))

    _walk(out_eager, out_compiled, "")

    return {
        "status": state["status"],
        "max_diff": state["max_diff"],
        "severity_ratio": (state["max_diff"] / atol) if atol > 0 else float("inf"),
        "bitwise_equal": state["status"] == "match" and state["max_diff"] == 0.0,
        "compared_fields": state["compared"],
        "skipped_fields": state["skipped"],
        "first_divergence": state["first_divergence"],
        "tolerance": {"atol": atol, "rtol": rtol},
    }


def run_correctness(spec, device, mode):
    """Compare eager vs compiled forward outputs at the same shape.

    Phase 3 base case: same input, same seed, same shape — only diff is torch.compile wrapper.
    Surfaces every numerical divergence introduced by the compiler. No pass/fail target.
    """
    t_start = time.perf_counter()
    result = {
        "name": spec["name"],
        "source": spec["source"],
        "mode": mode,
        "pass": "correctness",
    }

    # Lazy import HF helpers — fail soft if not available
    try:
        from transformers import set_seed
    except ImportError:
        set_seed = lambda s: torch.manual_seed(s)
    try:
        from transformers.testing_utils import (
            set_config_for_less_flaky_test,
            set_model_for_less_flaky_test,
        )
    except ImportError:
        set_config_for_less_flaky_test = lambda c: None
        set_model_for_less_flaky_test = lambda m: None

    # Phase 1: Create model + apply less-flaky config
    print("PHASE:create", file=sys.stderr, flush=True)
    try:
        if hasattr(spec, "get") and spec.get("source") == "hf":
            # Apply less-flaky config tweaks before model construction if possible
            pass  # set_config_for_less_flaky_test runs on the config; create_model owns config
        model, inputs_dict, inputs_tuple = create_model(spec, device)
    except Exception as e:
        result["status"] = "create_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        return result
    result["create_time_s"] = round(time.perf_counter() - t_start, 3)

    # Apply less-flaky tweaks to instantiated model (disables dropout, fixed init)
    try:
        if hasattr(model, "config"):
            set_config_for_less_flaky_test(model.config)
        set_model_for_less_flaky_test(model)
    except Exception as e:
        # Helpers can fail on unusual model shapes — log but don't abort
        result["less_flaky_warning"] = f"{type(e).__name__}: {str(e)[:200]}"

    if mode == "train":
        model.train()
    else:
        model.eval()
    ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

    # Phase 2: Eager forward
    print("PHASE:eager", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    set_seed(42)
    try:
        with ctx:
            if inputs_tuple:
                out_eager = model(*inputs_tuple)
            else:
                out_eager = model(**inputs_dict)
    except Exception as e:
        result["status"] = "eager_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    # Phase 3: Compile + forward
    print("PHASE:compile", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, fullgraph=True, backend="eager")
    except Exception as e:
        result["status"] = "compile_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    print("PHASE:compiled_forward", file=sys.stderr, flush=True)
    set_seed(42)
    try:
        with ctx:
            if inputs_tuple:
                out_compiled = compiled(*inputs_tuple)
            else:
                out_compiled = compiled(**inputs_dict)
    except Exception as e:
        result["status"] = "compile_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    # Phase 4: Compare (HF tolerance for fp32/CUDA)
    print("PHASE:compare", file=sys.stderr, flush=True)
    # MVP: fp32 only. Tolerance from HF test_modeling_common.py:194-222.
    cmp = _compare_outputs_recursive(out_eager, out_compiled, atol=1e-6, rtol=1e-4)
    result["status"] = cmp["status"]
    result["max_diff"] = round(cmp["max_diff"], 8)
    result["severity_ratio"] = round(cmp["severity_ratio"], 4)
    result["bitwise_equal"] = cmp["bitwise_equal"]
    result["tolerance"] = cmp["tolerance"]
    result["compared_fields"] = cmp["compared_fields"]
    # Compact skipped list — store reasons grouped
    skipped_summary = {}
    for path, reason in cmp["skipped_fields"]:
        key = reason.split(":")[0]
        skipped_summary[key] = skipped_summary.get(key, 0) + 1
    result["skipped_fields_summary"] = skipped_summary
    if cmp["first_divergence"]:
        result["first_divergence"] = cmp["first_divergence"]

    result["wall_time_s"] = round(time.perf_counter() - t_start, 3)

    if device == "cuda":
        result["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    _cleanup(model, device)
    return result


def run_validate(spec, device, mode, dynamic=False):
    """Validate dynamic shape correctness by comparing outputs at two different shapes.

    1. Create model + inputs at shape A (batch=3, seq=32)
    2. Eager forward at shape A → reference
    3. Compile with dynamic=True at shape A
    4. Create inputs at shape B (batch=5, seq=48)
    5. Compiled forward at shape B → test output
    6. Eager forward at shape B → reference B
    7. Compare compiled B vs eager B
    """
    t_start = time.perf_counter()
    result = {
        "name": spec["name"],
        "source": spec["source"],
        "mode": mode,
        "pass": "validate",
        "dynamic": dynamic,
        "shape_a_batch": SHAPE_A_BATCH,
        "shape_b_batch": SHAPE_B_BATCH,
    }

    # Phase 1: Create model
    print("PHASE:create", file=sys.stderr, flush=True)
    try:
        model, inputs_dict_a, inputs_tuple_a = create_model(spec, device, batch_size=SHAPE_A_BATCH)
    except Exception as e:
        result["status"] = "create_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        return result
    result["create_time_s"] = round(time.perf_counter() - t_start, 3)

    if mode == "train":
        model.train()
    else:
        model.eval()
    ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

    # Phase 2: Eager forward at shape A (warmup / baseline)
    print("PHASE:eager_a", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    try:
        with ctx:
            if inputs_tuple_a:
                _ = model(*inputs_tuple_a)
            else:
                _ = model(**inputs_dict_a)
    except Exception as e:
        result["status"] = "eager_error"
        result["error"] = f"shape_a eager: {str(e)[:400]}"
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    # Phase 3: Compile with dynamic=True at shape A
    print("PHASE:compile", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    compile_dynamic = True if dynamic is True else None
    if dynamic == "mark":
        input_type = spec.get("input_type", "auto")
        _mark_dynamic_dims(inputs_dict_a, inputs_tuple_a, spec["source"], input_type)
    try:
        compiled = torch.compile(model, fullgraph=True, backend="eager", dynamic=compile_dynamic)
        with ctx:
            if inputs_tuple_a:
                _ = compiled(*inputs_tuple_a)
            else:
                _ = compiled(**inputs_dict_a)
    except Exception as e:
        result["status"] = "compile_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    # Phase 4: Create inputs at shape B (reuse same model, just new inputs)
    print("PHASE:shape_b", file=sys.stderr, flush=True)
    try:
        inputs_dict_b, inputs_tuple_b = _create_inputs_only(spec, device, batch_size=SHAPE_B_BATCH)
    except Exception as e:
        result["status"] = "create_error_b"
        result["error"] = f"shape_b create: {str(e)[:400]}"
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    # Phase 5: Compiled forward at shape B
    print("PHASE:compiled_b", file=sys.stderr, flush=True)
    try:
        with ctx:
            if inputs_tuple_b:
                compiled_out_b = compiled(*inputs_tuple_b)
            else:
                compiled_out_b = compiled(**inputs_dict_b)
    except Exception as e:
        result["status"] = "compiled_shape_b_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    # Phase 6: Eager forward at shape B
    print("PHASE:eager_b", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    try:
        eager_model = model  # uncompiled
        with ctx:
            if inputs_tuple_b:
                eager_out_b = eager_model(*inputs_tuple_b)
            else:
                eager_out_b = eager_model(**inputs_dict_b)
    except Exception as e:
        result["status"] = "eager_shape_b_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        _cleanup(model, device)
        return result

    # Phase 7: Compare
    print("PHASE:compare", file=sys.stderr, flush=True)
    match, max_diff, details = _compare_outputs(compiled_out_b, eager_out_b)
    result["status"] = "pass" if match else "mismatch"
    result["max_diff"] = round(max_diff, 8) if max_diff != float("inf") else "inf"
    result["compare_details"] = details
    result["wall_time_s"] = round(time.perf_counter() - t_start, 3)

    if device == "cuda":
        result["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    _cleanup(model, device)
    return result


# ─── Explain: detailed analysis ─────────────────────────────────────────────

def run_explain(spec, device, mode, dynamic=False, compile_kwargs=None):
    """Analyze graph breaks using shared methodology (TORCH_LOGS + counting backend).

    Core analysis is delegated to run_graph_break_analysis() from explain.py,
    which is the canonical implementation shared across all suites (HF, diffusers,
    timm, custom). This wrapper handles model creation, TORCH_TRACE, and GPU memory.

    `dynamic` and `compile_kwargs` are forwarded so the explain pass exercises the
    same dynamo path as identify. The analysis function will override `fullgraph`
    (must be False to count breaks) and `backend` (must be the counting backend),
    but kwargs like `dynamic` flow through.
    """
    result = {
        "name": spec["name"],
        "source": spec["source"],
        "mode": mode,
        "pass": "explain",
    }

    try:
        model, inputs_dict, inputs_tuple = create_model(spec, device)
    except Exception as e:
        result["status"] = "create_error"
        result["error"] = str(e)[:500]
        result["error_type"] = type(e).__name__
        return result

    if mode == "train":
        model.train()
    else:
        model.eval()
    ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

    # Core graph break analysis — shared with all suites
    inputs = inputs_tuple if inputs_tuple else inputs_dict
    analysis = run_graph_break_analysis(model, inputs, mode=mode,
                                         dynamic=dynamic, compile_kwargs=compile_kwargs)

    result["explain_time_s"] = analysis["explain_time_s"]
    if "effective_compile_kwargs" in analysis:
        result["compile_kwargs"] = analysis["effective_compile_kwargs"]
    if analysis["status"] == "ok":
        result["status"] = "ok"
        result["graph_count"] = analysis["graph_count"]
        result["graph_break_count"] = analysis["graph_break_count"]
        result["ops_per_graph"] = analysis["ops_per_graph"]
        result["compile_times"] = analysis["compile_times"]
        result["break_reasons"] = analysis["break_reasons"]
    else:
        result["status"] = "explain_error"
        result["error"] = analysis.get("error", "unknown")

    # Step 2: TORCH_TRACE (if env var is set — orchestrator sets it before launch)
    trace_dir = os.environ.get("TORCH_TRACE")
    if trace_dir and result.get("status") == "ok":
        # TORCH_TRACE is already set in env — just compile again
        torch._dynamo.reset()
        start = time.perf_counter()
        try:
            compiled = torch.compile(model, backend="eager")
            with ctx:
                if inputs_tuple:
                    compiled(*inputs_tuple)
                else:
                    compiled(**inputs_dict)
        except Exception:
            pass  # Trace may still have partial data
        result["trace_time_s"] = round(time.perf_counter() - start, 3)

        # Measure trace size
        from pathlib import Path
        if os.path.exists(trace_dir):
            trace_files = list(Path(trace_dir).rglob("*"))
            trace_size = sum(f.stat().st_size for f in trace_files if f.is_file())
            result["trace_size_kb"] = round(trace_size / 1024, 1)
            result["trace_file_count"] = len([f for f in trace_files if f.is_file()])

    if device == "cuda":
        result["gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)

    _cleanup(model, device)
    return result


def _cleanup(model, device):
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Single-model worker for graph break sweep")
    parser.add_argument("--model-json", required=True, help="JSON string with model spec")
    parser.add_argument("--pass-num", type=int, required=True, choices=[1, 2, 3, 4],
                        help="Pass 1 (identify), 2 (explain/analyze), 3 (validate), or 4 (correctness)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--dynamic", nargs="?", const="true", default=None,
                        choices=["true", "mark"],
                        help="Dynamic shapes: 'true' = all dims symbolic, 'mark' = realistic dims only")
    parser.add_argument("--dynamo-flags", default=None,
                        help="JSON dict of torch._dynamo.config flags to set before compilation")
    parser.add_argument("--inductor-flags", default=None,
                        help="JSON dict of torch._inductor.config flags to set before compilation")
    parser.add_argument("--setup-script", default=None,
                        help="Python file exec'd once before pass dispatch. For multi-line "
                             "setup that doesn't fit a key=val (e.g., logging suppression).")
    parser.add_argument("--compile-kwargs", default=None,
                        help="JSON dict of torch.compile() kwargs (backend, fullgraph, etc.)")
    args = parser.parse_args()

    spec = json.loads(args.model_json)

    # Apply dynamo config flags if provided
    if args.dynamo_flags:
        dynamo_flags = json.loads(args.dynamo_flags)
        for flag_name, flag_value in dynamo_flags.items():
            if hasattr(torch._dynamo.config, flag_name):
                setattr(torch._dynamo.config, flag_name, flag_value)
                print(f"DYNAMO_FLAG: {flag_name} = {flag_value}", file=sys.stderr, flush=True)
            else:
                print(f"WARNING: Unknown dynamo config flag: {flag_name}", file=sys.stderr, flush=True)

    # Apply inductor config flags if provided
    if args.inductor_flags:
        import torch._inductor
        inductor_flags = json.loads(args.inductor_flags)
        for flag_name, flag_value in inductor_flags.items():
            if hasattr(torch._inductor.config, flag_name):
                setattr(torch._inductor.config, flag_name, flag_value)
                print(f"INDUCTOR_FLAG: {flag_name} = {flag_value}", file=sys.stderr, flush=True)
            else:
                print(f"WARNING: Unknown inductor config flag: {flag_name}", file=sys.stderr, flush=True)

    # Run user setup script (multi-line config that doesn't fit key=val).
    # Run as a fresh module-like scope: empty globals, so the script does its
    # own imports. Pre-populating `torch` collides with `import torch._dynamo`
    # at the script top, which the bytecode compiler treats as a local rebind.
    if args.setup_script:
        script_path = args.setup_script
        try:
            with open(script_path) as f:
                script_src = f.read()
            exec_globals = {"__name__": "__setup__", "__file__": script_path}
            exec(compile(script_src, script_path, "exec"), exec_globals)
            print(f"SETUP_SCRIPT: ran {script_path}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"ERROR: --setup-script {script_path} failed: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            sys.exit(2)

    # Convert dynamic arg: None→False, "true"→True, "mark"→"mark"
    dynamic_val = {"true": True, "mark": "mark"}.get(args.dynamic, False)

    # Parse compile kwargs (backend, fullgraph, etc.)
    compile_kwargs = json.loads(args.compile_kwargs) if args.compile_kwargs else None

    if args.pass_num == 1:
        result = run_identify(spec, args.device, args.mode, dynamic=dynamic_val,
                              compile_kwargs=compile_kwargs)
    elif args.pass_num == 2:
        # Thread compile_kwargs to explain pass too — break_reasons captured here
        # feed the per-issue plan, so they must reflect the user's actual config
        # (dynamic-shapes path, etc.). The explain pass will override `fullgraph`
        # to False (it must count breaks, not hard-fail) and `backend` to its own
        # counting backend, but other kwargs (dynamic, mode hints) flow through.
        result = run_explain(spec, args.device, args.mode,
                             dynamic=dynamic_val, compile_kwargs=compile_kwargs)
    elif args.pass_num == 3:
        result = run_validate(spec, args.device, args.mode, dynamic=dynamic_val)
    elif args.pass_num == 4:
        result = run_correctness(spec, args.device, args.mode)
    else:
        # Defensive — pass_num is constrained to {1,2,3,4}. If we're here it's
        # equivalent to pass 2 (legacy behavior).
        result = run_explain(spec, args.device, args.mode,
                             dynamic=dynamic_val, compile_kwargs=compile_kwargs)

    # Output single JSON line to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
