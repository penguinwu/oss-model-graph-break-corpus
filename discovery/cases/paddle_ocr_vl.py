"""Per-case config: PaddleOCRVL multimodal data-dependent graph breaks.

Reproduces graph breaks from `PaddleOCRVLForConditionalGeneration` under
torch.compile(fullgraph=True). Multiple distinct user-code shapes fire
within this one case:

  1. repeat_interleave on data-dep input in transformers/models/paddleocr_vl/
     modeling_paddleocr_vl.py:1234 (cu_seqlens construction from image_grid_thw)
  2. is_causal sdpa branch in transformers/integrations/sdpa_attention.py:77
     (`is_causal = query.shape[2] > 1 and ...`) firing inside the vision
     encoder layers
  3. Tensor.item() / scalar_outputs in the vision embeddings path
     (image_embeddings[start:end, :] indexing using item-derived sizes)

Per Phase 3 frame: the case = the model with its FULL break set. The agent
attacks the whole stack; BS-108 is the dominant tag (repeat_interleave on
image_grid_thw) but the agent is not scope-restricted to it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

CASE_ID = "paddle_ocr_vl"
MODEL_NAME = "PaddleOCRVLForConditionalGeneration"
BREAK_SHAPE_ID = "BS-108"  # data-dep dynamic-shape op (repeat_interleave); sdpa is_causal + item() also live in this case

# Locations on disk for the runner.
WORK_DIR = Path("/tmp/discovery-runs/paddle_ocr_vl")
PADDLE_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/paddleocr_vl/modeling_paddleocr_vl.py")
BASELINE_SCRIPT = WORK_DIR / "baseline_paddle.py"
VALIDATE_SCRIPT = WORK_DIR / "validate.py"


def _build_config():
    """Tiny PaddleOCRVL config. Vision uses spatial_merge_size=2;
    we keep h=w=4 patches so n_vis_per_image = (4/2)*(4/2) = 4 tokens."""
    from transformers import PaddleOCRVLConfig

    config = PaddleOCRVLConfig()
    t = config.text_config
    t.num_hidden_layers = 2
    t.hidden_size = 256
    t.intermediate_size = 256
    t.num_attention_heads = 2
    t.num_key_value_heads = 2
    t.head_dim = 128
    t.vocab_size = 1024
    # PaddleOCR text uses M-RoPE; need mrope_section that sums to head_dim/2 = 64.
    if t.rope_parameters is None:
        t.rope_parameters = {}
    t.rope_parameters.setdefault("mrope_section", [16, 24, 24])
    t.rope_parameters.setdefault("rope_type", "default")
    t.rope_parameters.setdefault("rope_theta", t.default_theta)

    v = config.vision_config
    v.num_hidden_layers = 2
    v.hidden_size = 128               # head_dim handled internally by RoPE
    v.intermediate_size = 256
    v.num_attention_heads = 2
    v.patch_size = 14
    v.spatial_merge_size = 2
    return config


def _build_inputs(config, B: int):
    """Mirror sweep/worker.py:2457-2475 (PaddleOCR conditional-generation recipe).

    PaddleOCR vision input is unbatched: pixel_values is
    (total_patches, num_channels, patch_size, patch_size) where
    total_patches = patches_per_image * B.
    """
    v = config.vision_config
    t = config.text_config
    num_ch = getattr(v, "num_channels", 3) or 3
    merge_size = getattr(v, "spatial_merge_size", 2)
    patch_size = getattr(v, "patch_size", 14)
    if isinstance(patch_size, (list, tuple)):
        patch_size = patch_size[0]
    h_patches = w_patches = merge_size * 2  # 4 patches per side
    patches_per_image = h_patches * w_patches  # 16
    total_patches = patches_per_image * B
    n_vis = (h_patches // merge_size) * (w_patches // merge_size)  # 4

    image_token_id = getattr(config, "image_token_id", None)
    text_len = 8
    seq_len = text_len + n_vis
    input_ids = torch.randint(0, t.vocab_size, (B, seq_len), device="cuda")
    if image_token_id is not None:
        input_ids[:, :n_vis] = image_token_id

    inputs = {
        "pixel_values": torch.randn(total_patches, num_ch, patch_size, patch_size, device="cuda"),
        "image_grid_thw": torch.tensor([[1, h_patches, w_patches]] * B, device="cuda"),
        "input_ids": input_ids,
        "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device="cuda"),
        "mm_token_type_ids": torch.zeros(B, seq_len, dtype=torch.long, device="cuda"),
    }
    return inputs


def make_model() -> torch.nn.Module:
    """Tier-1 (fast). Tiny config, B=1."""
    from transformers import PaddleOCRVLForConditionalGeneration

    config = _build_config()
    torch.manual_seed(0)
    model = PaddleOCRVLForConditionalGeneration(config).cuda().eval()
    image_token_id = getattr(config, "image_token_id", None)
    if image_token_id is not None and image_token_id >= config.text_config.vocab_size:
        model.resize_token_embeddings(image_token_id + 1)
    return model


def make_inputs(model: torch.nn.Module) -> dict[str, Any]:
    """Tier-1 (fast). B=1, h=w=4 patches (n_vis=4, seq_len=12).

    Optimized for discovery turnaround. Tier-2 should bump batch + n_images for
    fix-quality numbers."""
    torch.manual_seed(0)
    return _build_inputs(model.config, B=1)


# ----- Tier-2 (realistic) -----

def make_model_realistic() -> torch.nn.Module:
    """Tier-2 model. Identical config to tier-1 — we vary batch, not architecture."""
    return make_model()


def make_inputs_realistic(model: torch.nn.Module) -> dict[str, Any]:
    """Tier-2 (realistic). B=4 — keeps the multimodal glue exercised at non-trivial scale.

    h_patches/w_patches stays at 4; bumping past that exercises real ViT FLOPs but
    vision encoder time dominates and obscures the strategy-cost differences
    we care about."""
    torch.manual_seed(0)
    return _build_inputs(model.config, B=4)


# Perf is measured in a SUBPROCESS (see _measure_case.py).
PERF_CMD = [
    "/home/pengwu/envs/torch211/bin/python",
    "-m", "discovery._measure_case",
    "--case", CASE_ID,
    "--tier", "fast",
]
PERF_CMD_TIER2 = [
    "/home/pengwu/envs/torch211/bin/python",
    "-m", "discovery._measure_case",
    "--case", CASE_ID,
    "--tier", "realistic",
]
BASELINE_PATH = Path(__file__).parent / f"{CASE_ID}.baseline.json"
BASELINE_PATH_TIER2 = Path(__file__).parent / f"{CASE_ID}.baseline.tier2.json"


# Case body — full-model framing. Agent attacks the whole break set.
CASE_BODY = f"""I have a script that hits graph breaks under torch.compile. The script is at {BASELINE_SCRIPT} and currently reports graph_break_count=19. The model is a multimodal vision-language OCR model (PaddleOCRVLForConditionalGeneration); the vision encoder is the PaddleOCR Vision Transformer with grid-based patch input.

Please:
1. Detect the graph breaks
2. Diagnose the root cause(s) — there are multiple distinct break shapes in this model (dynamic-shape repeat_interleave on image_grid_thw, data-dependent guards from sdpa is_causal in the vision encoder, and Tensor.item() / data-dep indexing in the vision embeddings path)
3. Fix them. You may edit any of:
   - {BASELINE_SCRIPT} (the test script — e.g. shape constraints)
   - {PADDLE_SRC} (the PaddleOCRVL model source — covers PaddleOCRVLForConditionalGeneration, PaddleOCRVisionTransformer, PaddleOCRVisionEmbeddings, PaddleOCRVisionEncoder)
   Do NOT edit shared infrastructure outside these files (e.g. sdpa_attention.py, decomposition tables).
4. Verify by re-running the script: graph_break_count should drop (ideally to 0 so fullgraph=True works) and the model output should still match the original eager output within 1e-3.

The python interpreter is `python` (already on PATH at /home/pengwu/envs/torch211/bin/python). Do NOT use conda. Do NOT try to fetch external documentation — diagnose from inline output only.

When you have a fix you believe is correct, save it (in place — modify the existing files, do not write new files) and exit."""


def get_case_spec():
    """Build CaseSpec for the runner. Lazy import to avoid circular dep."""
    from discovery.runner import CaseSpec, WatchedFile

    return CaseSpec(
        case_id=CASE_ID,
        case_body=CASE_BODY,
        watched_files=[
            WatchedFile(path=PADDLE_SRC, original_backup=WORK_DIR / "modeling_paddleocr_vl.py.original"),
            WatchedFile(path=BASELINE_SCRIPT, original_backup=WORK_DIR / "baseline_paddle.py.original"),
        ],
        validate_cmd=["/home/pengwu/envs/torch211/bin/python", str(VALIDATE_SCRIPT)],
        perf_cmd=PERF_CMD,
        perf_cmd_tier2=PERF_CMD_TIER2,
        baseline_path=BASELINE_PATH,
        baseline_path_tier2=BASELINE_PATH_TIER2,
    )


# Smoke run: measure baseline perf and print.
if __name__ == "__main__":
    import json

    from discovery.perf import measure_perf

    result = measure_perf(make_model, make_inputs, n_warmup=5, n_repeat=20)
    print(json.dumps({"case_id": CASE_ID, "baseline": result.to_dict()}, indent=2))
