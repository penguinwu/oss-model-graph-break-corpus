"""Per-case config: Aria multimodal data-dependent graph breaks.

Reproduces graph breaks from `AriaForConditionalGeneration` under
torch.compile(fullgraph=True). Aria is a Mixture-of-Experts vision-language
model (Rhymes-AI/Aria) — text backbone is an MoE Llama-style decoder, vision
encoder is `Idefics3VisionTransformer` (SigLIP-style ViT).

Multiple distinct user-code shapes fire within this one case (corpus reports
graph_break_count=19, graph_count=20 in eval mode):

  1. `output_capturing.capture_outputs` lock context-manager — Dynamo cannot
     enter the `lock` ctx, breaking variadic CALL_FUNCTION_EX paths repeatedly
     (transformers/utils/output_capturing.py).
  2. unfold decomposition in transformers/models/aria/modeling_aria.py
     (`torch.nn.functional.unfold` in the patch-level vision projector — fed
     by `vision_tower.config.patch_size`).
  3. `AutoModel.from_config` glue + image-token positional handling →
     data-dep ops (`_local_scalar_dense` / `nonzero`) in the multimodal
     projector that maps SigLIP patches into the MoE text stream.

Per Phase 3 frame: the case = the model with its FULL break set. The agent
attacks the whole stack; BS-107 is the dominant tag (capture_outputs lock
ctx) but the agent is not scope-restricted to it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

CASE_ID = "aria_data_dep"
MODEL_NAME = "AriaForConditionalGeneration"
BREAK_SHAPE_ID = "BS-107"  # capture_outputs lock ctx mgr; unfold + scalar_dense also live in this case

# Locations on disk for the runner.
WORK_DIR = Path("/tmp/discovery-runs/aria_data_dep")
ARIA_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/aria/modeling_aria.py")
IDEFICS3_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/idefics3/modeling_idefics3.py")
BASELINE_SCRIPT = WORK_DIR / "baseline_aria.py"
VALIDATE_SCRIPT = WORK_DIR / "validate.py"


def _build_config():
    """Tiny Aria config. Aria text is MoE — keep moe_num_experts at default
    (8) but reduce hidden/layers; vision is Idefics3 (SigLIP-like) at
    image_size=224 / patch_size=32 → 49 patches. Worker.py adds {49: 128}
    to projector_patch_to_query_dict; we mirror that here."""
    from transformers import AriaConfig

    config = AriaConfig()
    t = config.text_config
    t.num_hidden_layers = 2
    t.hidden_size = 128
    t.intermediate_size = 256
    t.num_attention_heads = 4
    t.num_key_value_heads = 2
    t.vocab_size = 1024
    # MoE: shrink expert intermediate too if exposed (Aria uses moe_intermediate_size or
    # falls back to intermediate_size). Leave moe_num_experts at default; tracer needs
    # to see the MoE routing.

    v = config.vision_config
    v.image_size = 224                # default; patch_size=32 → 7x7=49 patches
    v.num_hidden_layers = 2
    v.hidden_size = 128               # = 2 * 64 (head_dim)
    v.intermediate_size = 256
    v.num_attention_heads = 2

    # Aria-specific: projector_patch_to_query_dict default has keys {1225, 4900}
    # only. With image_size=224 / patch_size=32 we get 49 patches — add an entry
    # so the projector can map 49 → 128 query tokens (mirrors worker.py:464-478).
    ptqd = getattr(config, "projector_patch_to_query_dict", None)
    if ptqd is not None:
        patches_per_side = v.image_size // v.patch_size
        num_patches = patches_per_side ** 2
        if num_patches not in ptqd:
            ptqd[num_patches] = min(ptqd.values())  # 49 → 128

    return config


def _build_inputs(config, B: int):
    """Mirror sweep/worker.py:1587-1612 (Aria image_token_fix recipe).

    Aria expects `num_image_tokens=128` image-token placeholders in input_ids
    plus 8 text tokens (worker.py uses image_token_fix['ariamodel'] = 128)."""
    v = config.vision_config
    t = config.text_config
    img_size = v.image_size
    num_ch = getattr(v, "num_channels", 3) or 3

    image_token_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
    num_image_tokens = 128             # worker.py image_token_fix['ariamodel']
    text_len = 8
    seq_len = num_image_tokens + text_len
    input_ids = torch.randint(0, t.vocab_size, (B, seq_len), device="cuda")
    if image_token_id is not None:
        input_ids[:, :num_image_tokens] = image_token_id

    return {
        "pixel_values": torch.randn(B, num_ch, img_size, img_size, device="cuda"),
        "input_ids": input_ids,
        "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device="cuda"),
    }


def make_model() -> torch.nn.Module:
    """Tier-1 (fast). Tiny config, B=1."""
    from transformers import AriaForConditionalGeneration

    config = _build_config()
    torch.manual_seed(0)
    model = AriaForConditionalGeneration(config).cuda().eval()
    image_token_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
    if image_token_id is not None and image_token_id >= config.text_config.vocab_size:
        model.resize_token_embeddings(image_token_id + 1)
    return model


def make_inputs(model: torch.nn.Module) -> dict[str, Any]:
    """Tier-1 (fast). B=1, image_size=224 (49 patches → 128 image tokens, seq_len=136).

    Optimized for discovery turnaround. Tier-2 should bump batch + n_images for
    fix-quality numbers."""
    torch.manual_seed(0)
    return _build_inputs(model.config, B=1)


# ----- Tier-2 (realistic) -----

def make_model_realistic() -> torch.nn.Module:
    """Tier-2 model. Identical config to tier-1 — we vary batch, not architecture."""
    return make_model()


def make_inputs_realistic(model: torch.nn.Module) -> dict[str, Any]:
    """Tier-2 (realistic). B=4 — keeps the multimodal glue + MoE routing exercised
    at non-trivial scale.

    image_size stays at 224; bumping past that exercises real ViT FLOPs but the
    Idefics3 vision encoder time dominates and obscures the strategy-cost
    differences we care about."""
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
CASE_BODY = f"""I have a script that hits graph breaks under torch.compile. The script is at {BASELINE_SCRIPT} and currently reports a non-zero graph_break_count. The model is a multimodal MoE vision-language model (AriaForConditionalGeneration); the vision encoder is Idefics3VisionTransformer (SigLIP-style ViT) and the text backbone is an MoE Llama-style decoder.

Please:
1. Detect the graph breaks
2. Diagnose the root cause(s) — there are multiple distinct break shapes in this model (a `lock` context manager around `output_capturing.capture_outputs`, decomposition issues from torch.nn.functional.unfold in the patch-level vision projector, and data-dep operators in the multimodal glue / MoE routing)
3. Fix them. You may edit any of:
   - {BASELINE_SCRIPT} (the test script — e.g. shape constraints)
   - {ARIA_SRC} (the Aria model source)
   - {IDEFICS3_SRC} (the Idefics3 vision-encoder source)
   Do NOT edit shared infrastructure outside these files (e.g. sdpa_attention.py, decomposition tables, output_capturing.py).
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
            WatchedFile(path=ARIA_SRC, original_backup=WORK_DIR / "modeling_aria.py.original"),
            WatchedFile(path=IDEFICS3_SRC, original_backup=WORK_DIR / "modeling_idefics3.py.original"),
            WatchedFile(path=BASELINE_SCRIPT, original_backup=WORK_DIR / "baseline_aria.py.original"),
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
