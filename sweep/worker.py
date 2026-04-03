#!/usr/bin/env python3
"""Single-model worker — subprocess entry point for the two-pass sweep.

Called by the orchestrator as:
  python worker.py --model-json '{"name":"resnet50","source":"timm",...}' \
                   --pass 1 --device cuda --mode eval

Pass 1: fullgraph=True compile → binary pass/fail
Pass 2: explain() + optional TORCH_TRACE → detailed graph break analysis

Outputs a single JSON line to stdout. All logs go to stderr.
"""
import argparse
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
                          "pass": pass_num, "status": "clean", "wall_time_s": 1.0}
            elif chaos == "crash":
                import signal as _sig
                os.kill(os.getpid(), _sig.SIGSEGV)
            elif chaos == "hang":
                time.sleep(999999)
            elif chaos == "slow":
                sleep_time = int(spec.get("_chaos_sleep", 200))
                time.sleep(sleep_time)
                result = {"name": name, "source": source, "mode": mode,
                          "pass": pass_num, "status": "clean",
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
                          "pass": pass_num, "status": "clean", "wall_time_s": 1.0}
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

    # Default: standard constructor
    return config_cls()


def _fix_config(model_name, config):
    """Patch known config bugs so models can be instantiated with default configs.

    Many HF model configs have None or missing values that crash during
    model construction. This fixes them with sensible defaults.
    """
    name_lower = model_name.lower()

    # --- Time series: need prediction_length and context_length ---
    if name_lower in ("autoformermodel", "informermodel", "timeseriestransformermodel"):
        if not getattr(config, "prediction_length", None):
            config.prediction_length = 24
        if not getattr(config, "context_length", None):
            config.context_length = 96

    # --- AutoformerModel: Dynamo FakeTensor bug — context_length must equal head_dim ---
    # The autocorrelation attention uses irfft(x, n=tgt_len) and later view(bsz, heads, tgt_len, head_dim).
    # Dynamo's symbolic shape inference incorrectly uses tgt_len for head_dim when they differ,
    # causing "shape '[2, 2, 96, 96]' invalid for input of size 12288".
    # Fix: set context_length = head_dim = d_model // encoder_attention_heads = 64 // 2 = 32.
    if name_lower == "autoformermodel":
        d_model = getattr(config, "d_model", 64)
        enc_heads = getattr(config, "encoder_attention_heads", 2)
        head_dim = d_model // enc_heads if enc_heads > 0 else 32
        config.context_length = head_dim
        config.prediction_length = head_dim

    # --- InformerModel: distil=True causes downsampling between encoder layers ---
    # The attention mask is created for the full sequence length (context_length) but after
    # InformerConvLayer downsampling, hidden_states shrink (96→48) while mask stays at 96×96.
    # The attention check raises ValueError which explain() cannot handle as a graph break.
    # Fix: disable distillation so no conv downsampling occurs.
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

    # --- Ministral3Model: rope_theta is None ---
    if name_lower == "ministral3model":
        if getattr(config, "rope_theta", None) is None:
            config.rope_theta = 10000.0

    # --- NemotronModel: partial_rotary_factor and num_key_value_heads are None ---
    if name_lower == "nemotronmodel":
        if getattr(config, "partial_rotary_factor", None) is None:
            config.partial_rotary_factor = 0.5
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

    # --- MoonshineStreamingModel: missing num_key_value_heads ---
    if name_lower == "moonshinestreamingmodel":
        if not hasattr(config, "num_key_value_heads") or config.num_key_value_heads is None:
            config.num_key_value_heads = getattr(config, "num_attention_heads", 8)

    # --- Pix2StructTextModel: missing initializer_range ---
    if name_lower == "pix2structtextmodel":
        if not hasattr(config, "initializer_range"):
            config.initializer_range = 0.02

    # --- Qwen3OmniMoeTalkerModel: missing pad_token_id + vocab_size + num_hidden_layers ---
    if name_lower in ("qwen3omnioetalkermodel", "qwen3omnimoetalkermodel"):
        if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
            config.pad_token_id = 0
        text_cfg = getattr(config, "text_config", None)
        if text_cfg:
            if not hasattr(config, "vocab_size") or getattr(config, "vocab_size", None) is None:
                config.vocab_size = getattr(text_cfg, "vocab_size", 3072)
            if not hasattr(config, "num_hidden_layers") or getattr(config, "num_hidden_layers", None) is None:
                config.num_hidden_layers = getattr(text_cfg, "num_hidden_layers", 20)

    # --- Dots1Model: hidden_size, MoE expert counts are None ---
    if name_lower == "dots1model":
        if getattr(config, "hidden_size", None) is None:
            config.hidden_size = 768
        if getattr(config, "n_routed_experts", None) is None:
            config.n_routed_experts = 4
        if getattr(config, "n_shared_experts", None) is None:
            config.n_shared_experts = 1
        if getattr(config, "num_experts_per_tok", None) is None:
            config.num_experts_per_tok = 2

    # --- EsmModel: vocab_size + position_embedding_type + emb_layer_norm + pad_token_id ---
    if name_lower == "esmmodel":
        if getattr(config, "vocab_size", None) is None:
            config.vocab_size = 33
        if getattr(config, "position_embedding_type", None) is None:
            config.position_embedding_type = "absolute"
        if getattr(config, "emb_layer_norm_before", None) is None:
            config.emb_layer_norm_before = False
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = 0

    # --- BayesianDetectorModel: watermarking_depth is None ---
    if name_lower == "bayesiandetectormodel":
        if getattr(config, "watermarking_depth", None) is None:
            config.watermarking_depth = 3

    # --- Blip2Model / InstructBlipModel / InstructBlipVideoModel: image_token_index is None ---
    if name_lower in ("blip2model", "instructblipmodel", "instructblipvideomodel"):
        if getattr(config, "image_token_index", None) is None:
            config.image_token_index = 2
        # InstructBlipVideoModel.forward() references config.image_token_id (not mapped
        # in attribute_map, which only has video_token_id → video_token_index).
        # Set it directly so the model can find it.
        if name_lower == "instructblipvideomodel":
            if not hasattr(config, "image_token_id") or getattr(config, "image_token_id", None) is None:
                config.image_token_id = getattr(config, "image_token_index", 2)
            # Also ensure video_token_index is set (used by ForConditionalGeneration)
            if getattr(config, "video_token_index", None) is None:
                config.video_token_index = getattr(config, "image_token_index", 2)

    # --- IdeficsModel: use_resampler must be True when perceiver_embeddings are passed ---
    if name_lower == "ideficsmodel":
        config.use_resampler = True

    # --- Lfm2MoeModel: layer_types + expert counts are None ---
    if name_lower == "lfm2moemodel":
        if getattr(config, "num_local_experts", None) is None:
            config.num_local_experts = 8
        if getattr(config, "num_experts_per_tok", None) is None:
            config.num_experts_per_tok = 2
        num_layers = getattr(config, "num_hidden_layers", 2)
        if getattr(config, "layer_types", None) is None:
            config.layer_types = ["attention"] * num_layers

    # --- DeepseekV2Model: num_experts_per_tok + n_group are None ---
    if name_lower == "deepseekv2model":
        if getattr(config, "num_experts_per_tok", None) is None:
            config.num_experts_per_tok = 6
        if getattr(config, "n_group", None) is None:
            config.n_group = 1
        if getattr(config, "topk_group", None) is None:
            config.topk_group = 1

    # --- GPTNeoXModel: rotary_pct is None ---
    if name_lower == "gptneoxmodel":
        if getattr(config, "rotary_pct", None) is None:
            config.rotary_pct = 0.25
        if getattr(config, "rotary_emb_base", None) is None:
            config.rotary_emb_base = 10000

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

    # --- Generic: truncate layer_types to match reduced num_hidden_layers ---
    for cfg_obj in [config, getattr(config, "text_config", None)]:
        if cfg_obj is None:
            continue
        n = getattr(cfg_obj, "num_hidden_layers", None)
        if n is not None:
            lt = getattr(cfg_obj, "layer_types", None)
            if lt is not None and isinstance(lt, list) and len(lt) > n:
                # Collect unique types from full list to ensure all are represented
                all_types = list(dict.fromkeys(lt))  # Deduplicated, preserves order
                if n >= len(all_types):
                    # Fill with one of each type, then repeat last
                    new_lt = all_types[:n]
                    while len(new_lt) < n:
                        new_lt.append(all_types[-1])
                else:
                    # Not enough layers for all types — take first n
                    new_lt = lt[:n]
                cfg_obj.layer_types = new_lt

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

    # --- RecurrentGemmaModel: ensure block_types includes 'attention' after num_hidden_layers reduction ---
    # The model uses layers_block_type = (block_types * 100)[:num_hidden_layers].
    # Default block_types = ['recurrent', 'recurrent', 'attention'] — with num_hidden_layers=2,
    # all 'attention' blocks are cut, and config.layers_block_type.index('attention') raises ValueError.
    # Fix: set block_types = ['recurrent', 'attention'] so the pattern includes attention at 2 layers.
    if name_lower == "recurrentgemmamodel":
        config.block_types = ["recurrent", "attention"]

    return config


def _reduce_model_size(config):
    """Reduce model size to prevent create-phase timeouts.

    Graph break behavior is determined by the model architecture (ops used),
    not the model depth/width. 2 layers is sufficient for detection.
    """
    # Reduce layers
    for attr in ("num_hidden_layers", "num_layers", "n_layer", "n_layers",
                 "encoder_layers", "decoder_layers", "num_encoder_layers",
                 "num_decoder_layers"):
        val = getattr(config, attr, None)
        if val is not None and isinstance(val, int) and val > 4:
            setattr(config, attr, 2)

    # Reduce hidden size for very large models (> 4096)
    if getattr(config, "hidden_size", 0) > 4096:
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
    hs = getattr(config, "hidden_size", None)
    nh = getattr(config, "num_attention_heads", None)
    if hs and nh and hs % nh != 0:
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
    if nh and nkv and nkv > 0 and nh % nkv != 0:
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

    config_name = spec.get("hf_config") or spec["name"].replace("Model", "Config")
    model_name = spec.get("hf_class") or spec["name"]

    config_cls = getattr(transformers, config_name)
    model_cls = getattr(transformers, model_name)

    config = _create_config(model_name, config_cls)
    config = _fix_config(model_name, config)
    model = model_cls(config).to(device)

    # Always auto-detect from config — spec's name-based hint can be wrong
    # (e.g., GroundingDino classified as vision but needs text input_ids too)
    input_type = _detect_hf_input_type(model_name, config)

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
    name_lower = (spec.get("hf_class") or spec["name"]).lower()
    model_type = getattr(config, "model_type", "")

    # Sub-models that need hidden_states instead of/in addition to pixel_values
    # These are internal vision encoders (Glm4v, GlmOcr, PaddleOCR, etc.)
    #
    # Glm vision models have a special input format: hidden_states/pixel_values are
    # 2D tensors of FLATTENED PIXEL PATCHES, not transformer hidden states.
    # Shape: (total_patches, in_channels * temporal_patch_size * patch_size * patch_size)
    # where total_patches = sum(t*h*w for each row of grid_thw).
    _glm_vision_models = {
        "glm4vvisionmodel", "glm4vmoevisionmodel", "glmocrvisionmodel",
    }
    if name_lower in _glm_vision_models:
        in_channels = getattr(config, "in_channels", 3)
        temporal_patch_size = getattr(config, "temporal_patch_size", 2)
        patch_size = getattr(config, "patch_size", 14)
        patch_pixel_dim = in_channels * temporal_patch_size * patch_size * patch_size
        grid_thw = torch.tensor([[1, 4, 8]] * B, dtype=torch.long, device=device)
        total_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        inputs = {
            "hidden_states": torch.randn(total_patches, patch_pixel_dim, device=device),
            "grid_thw": grid_thw,
        }

    # GlmImageVisionModel: pixel_values are 2D flattened patches (no temporal dim)
    if name_lower == "glmimagevisionmodel":
        in_channels = getattr(config, "in_channels", 3)
        patch_size = getattr(config, "patch_size", 16)
        patch_pixel_dim = in_channels * patch_size * patch_size
        grid_thw = torch.tensor([[1, 4, 8]] * B, dtype=torch.long, device=device)
        total_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        inputs = {
            "pixel_values": torch.randn(total_patches, patch_pixel_dim, device=device),
            "grid_thw": grid_thw,
        }

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

    # VideoLlama3VisionModel: pre-patchified pixel_values + grid_thw + merge_sizes
    # pixel_values shape: (total_patches, num_channels * patch_size * patch_size)
    # merge_sizes shape: (num_images_or_videos,) — scalar per image/video
    if name_lower == "videollama3visionmodel":
        num_channels = getattr(config, "num_channels", 3)
        patch_size = getattr(config, "patch_size", 16)
        patch_pixel_dim = num_channels * patch_size * patch_size
        grid_thw = torch.tensor([[1, 4, 8]] * B, dtype=torch.long, device=device)
        total_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())
        inputs = {
            "pixel_values": torch.randn(total_patches, patch_pixel_dim, device=device),
            "grid_thw": grid_thw,
            "merge_sizes": torch.tensor([1] * B, dtype=torch.long, device=device),
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

    # Video inference session models — need inference_session (stateful, not compilable)
    # Skip: EdgeTamVideoModel, Sam2VideoModel, Sam3TrackerVideoModel, Sam3VideoModel

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
    if name_lower == "pix2structvisionmodel":
        hidden = getattr(config, "hidden_size", 768)
        patch_embed_dim = getattr(config, "patch_embed_hidden_size", None)
        num_channels = getattr(config, "num_channels", 3)
        patch_size = getattr(config, "patch_size", 16)
        # flattened_patches: (B, seq_len, patch_size*patch_size*channels + 2)
        flat_dim = patch_size * patch_size * num_channels + 2
        inputs = {
            "flattened_patches": torch.randn(B, 32, flat_dim, device=device),
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

    # HiggsAudioV2Model — needs audio_input_ids or input_ids
    if name_lower == "higgsaudiov2model":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # Phi4MultimodalModel needs input_ids explicitly
    if name_lower == "phi4multimodalmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # XmodModel needs set_default_language
    if name_lower == "xmodmodel":
        try:
            model.set_default_language("en_XX")
        except Exception:
            pass

    # MambaModel / FalconMambaModel: eval mode creates MambaCache / FalconMambaCache,
    # which calls torch._dynamo.mark_static_address() — a forbidden callable during explain().
    # In train mode, cache creation is skipped (use_cache=False by default in training).
    # Fix: pass use_cache=False explicitly to prevent cache initialization.
    if name_lower in ("mambamodel", "falconmambamodel"):
        inputs["use_cache"] = False

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

    # ModernVBertModel: needs image features not pixel_values
    if name_lower == "modernvbertmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

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

    # Qwen2_5OmniTalkerModel: mrope_section issue — text-only inputs
    if name_lower == "qwen2_5omnitalkermodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # PaddleOCRTextModel: mrope_section issue — text-only
    if name_lower == "paddleocrtextmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
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

    # PaddleOCRVLModel: text-only inputs to avoid vision path NoneType
    if name_lower == "paddleocrvlmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
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

    # LlavaNextVideoModel: text-only to avoid vision NoneType
    if name_lower == "llavanextvideomodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # LightOnOcrModel: text-only inputs
    if name_lower == "lightonocrmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # GlmOcrModel: text-only inputs
    if name_lower == "glmocrmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # GlmImageModel: text-only inputs to avoid vision NoneType
    if name_lower == "glmimagemodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # GlmImageTextModel: text-only (split_with_sizes mismatch is from vision path)
    if name_lower == "glmimagetextmodel":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

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

    # Qwen3_5Model: text-only, avoid NoneType.tolist in train
    if name_lower == "qwen3_5model":
        inputs = {
            "input_ids": torch.randint(0, vocab_size, (B, 32), device=device),
            "attention_mask": torch.ones(B, 32, dtype=torch.long, device=device),
        }

    # ── VL/multimodal models: text-only to avoid NoneType iteration on image inputs ──
    _vl_text_only_models = {
        # NoneType.tolist from image_grid_thw
        "qwen2_5_vlmodel", "qwen3vlmodel", "qwen3vlmoemodel", "qwen3_5moemodel",
        # NoneType not iterable from image_sizes/grid_thw
        "qwen2vlmodel", "llavamodel", "llavanextmodel",
        "mistral3model", "videollama3model",
        "glm46vmodel", "glm4vmodel", "glm4vmoemodel",
        # mat1/mat2 mismatch from vision projection — text-only avoids it
        "ernie4_5_vlmoemodel", "ernie4_5_vl_moemodel", "lfm2vlmodel",
        # NoneType has no len
        "llavaonevisionmodel",
        # Image features / image tokens mismatch
        "cohere2visionmodel", "fastvlmmodel", "deepseekvlmodel", "janusmodel",
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


# ─── Pass 1: fullgraph identification ────────────────────────────────────────

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


def run_pass1(spec, device, mode, dynamic=False):
    """Try fullgraph=True compile. Returns JSON-serializable result."""
    t_start = time.perf_counter()
    result = {
        "name": spec["name"],
        "source": spec["source"],
        "mode": mode,
        "pass": 1,
        "dynamic": dynamic,
        "phase": "create",  # tracks current phase for timeout diagnosis
    }

    # Phase markers to stderr — orchestrator reads these on timeout
    print("PHASE:create", file=sys.stderr, flush=True)
    try:
        model, inputs_dict, inputs_tuple = create_model(spec, device)
    except Exception as e:
        result["status"] = "create_error"
        result["error"] = str(e)[:500]
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
                    cfg_name = spec["name"].replace("Model", "Config")
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
                        result["eager_time_s"] = round(time.perf_counter() - t_eager, 3)
                        result["image_token_retry_failed"] = need_features
                        _cleanup(model, device)
                        return result
        if not retried:
            result["status"] = "eager_error"
            result["error"] = err_str[:500]
            result["eager_time_s"] = round(time.perf_counter() - t_eager, 3)
            _cleanup(model, device)
            return result
    result["eager_time_s"] = round(time.perf_counter() - t_eager, 3)

    # Step 2: fullgraph=True
    result["phase"] = "compile"
    print("PHASE:compile", file=sys.stderr, flush=True)
    torch._dynamo.reset()
    t_compile = time.perf_counter()
    try:
        # dynamic modes: True = all dims symbolic, "mark" = realistic dims only, False = static
        compile_dynamic = True if dynamic is True else None
        if dynamic == "mark":
            input_type = spec.get("input_type", "auto")
            _mark_dynamic_dims(inputs_dict, inputs_tuple, spec["source"], input_type)
        compiled = torch.compile(model, fullgraph=True, backend="eager", dynamic=compile_dynamic)
        with ctx:
            if inputs_tuple:
                compiled(*inputs_tuple)
            else:
                compiled(**inputs_dict)
        result["status"] = "clean"
        result["fullgraph_ok"] = True
    except Exception as e:
        err_str = str(e)[:500]
        err_type = type(e).__name__

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
        _cleanup(model, device)
        return result

    # Phase 4: Create inputs at shape B (reuse same model, just new inputs)
    print("PHASE:shape_b", file=sys.stderr, flush=True)
    try:
        inputs_dict_b, inputs_tuple_b = _create_inputs_only(spec, device, batch_size=SHAPE_B_BATCH)
    except Exception as e:
        result["status"] = "create_error_b"
        result["error"] = f"shape_b create: {str(e)[:400]}"
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


# ─── Pass 2: detailed analysis ───────────────────────────────────────────────

def run_pass2(spec, device, mode):
    """Run explain() + TORCH_TRACE on a model known to have graph breaks."""
    result = {
        "name": spec["name"],
        "source": spec["source"],
        "mode": mode,
        "pass": 2,
    }

    try:
        model, inputs_dict, inputs_tuple = create_model(spec, device)
    except Exception as e:
        result["status"] = "create_error"
        result["error"] = str(e)[:500]
        return result

    if mode == "train":
        model.train()
    else:
        model.eval()
    ctx = torch.no_grad() if mode == "eval" else torch.enable_grad()

    # Step 1: explain()
    torch._dynamo.reset()
    start = time.perf_counter()
    try:
        with ctx:
            if inputs_tuple:
                explanation = torch._dynamo.explain(model)(*inputs_tuple)
            else:
                explanation = torch._dynamo.explain(model)(**inputs_dict)

        result["explain_time_s"] = round(time.perf_counter() - start, 3)
        result["graph_count"] = explanation.graph_count
        result["graph_break_count"] = explanation.graph_break_count
        result["ops_per_graph"] = [len(g) for g in explanation.ops_per_graph]

        if hasattr(explanation, "compile_times"):
            try:
                result["compile_times"] = [
                    round(float(t), 3) if not isinstance(t, str) else t
                    for t in explanation.compile_times
                ]
            except Exception:
                result["compile_times"] = [str(t) for t in explanation.compile_times]

        # Extract break reasons
        if hasattr(explanation, "break_reasons") and explanation.break_reasons:
            result["break_reasons"] = []
            for br in explanation.break_reasons:
                try:
                    info = {"reason": str(getattr(br, "reason", str(br)))[:300]}
                    if hasattr(br, "user_stack") and br.user_stack:
                        top = br.user_stack[-1]
                        info["file"] = str(getattr(top, "filename", ""))
                        info["line"] = getattr(top, "lineno", 0)
                    result["break_reasons"].append(info)
                except Exception:
                    result["break_reasons"].append({"reason": str(br)[:300]})

        result["status"] = "ok"
    except Exception as e:
        result["status"] = "explain_error"
        result["error"] = str(e)[:500]
        result["explain_time_s"] = round(time.perf_counter() - start, 3)

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
    parser.add_argument("--pass-num", type=int, required=True, choices=[1, 2, 3],
                        help="Pass 1 (fullgraph), 2 (explain), or 3 (validate)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--mode", default="eval", choices=["eval", "train"])
    parser.add_argument("--dynamic", nargs="?", const="true", default=None,
                        choices=["true", "mark"],
                        help="Dynamic shapes: 'true' = all dims symbolic, 'mark' = realistic dims only")
    args = parser.parse_args()

    spec = json.loads(args.model_json)

    # Convert dynamic arg: None→False, "true"→True, "mark"→"mark"
    dynamic_val = {"true": True, "mark": "mark"}.get(args.dynamic, False)

    if args.pass_num == 1:
        result = run_pass1(spec, args.device, args.mode, dynamic=dynamic_val)
    elif args.pass_num == 3:
        result = run_validate(spec, args.device, args.mode, dynamic=dynamic_val)
    else:
        result = run_pass2(spec, args.device, args.mode)

    # Output single JSON line to stdout
    print(json.dumps(result))


if __name__ == "__main__":
    main()
