"""Per-case config: Mistral3 multimodal data-dependent graph breaks.

Reproduces graph breaks from `Mistral3ForConditionalGeneration` under
torch.compile(fullgraph=True). Multiple distinct user-code shapes fire
within this one case:

  1. is_causal sdpa branch in transformers/models/pixtral/modeling_pixtral.py
     (`is_causal = query.shape[2] > 1 and ...`)
  2. unfold decomposition in transformers/models/mistral3/modeling_mistral3.py
     (`torch.nn.functional.unfold` produces unhinted symbol)
  3. _local_scalar_dense / nonzero data-dep ops elsewhere in the multimodal
     glue (image-token positional handling)

Per Phase 3 frame: the case = the model with its FULL break set. The agent
attacks the whole stack; BS-105 is the dominant tag (sdpa is_causal) but the
agent is not scope-restricted to it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

CASE_ID = "mistral3_data_dep"
MODEL_NAME = "Mistral3ForConditionalGeneration"
BREAK_SHAPE_ID = "BS-105"  # data-dep guard on dynamic shape (sdpa is_causal); unfold + scalar_dense also live in this case

# Locations on disk for the runner.
WORK_DIR = Path("/tmp/discovery-runs/mistral3_data_dep")
MISTRAL3_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/mistral3/modeling_mistral3.py")
PIXTRAL_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/pixtral/modeling_pixtral.py")
BASELINE_SCRIPT = WORK_DIR / "baseline_mistral3.py"
VALIDATE_SCRIPT = WORK_DIR / "validate.py"


def _build_config():
    """Tiny Mistral3 config. Vision RoPE is hard-wired to head_dim=64,
    so vision_hidden_size must equal num_attention_heads * 64."""
    from transformers import Mistral3Config

    config = Mistral3Config()
    t = config.text_config
    t.num_hidden_layers = 2
    t.hidden_size = 128
    t.intermediate_size = 256
    t.num_attention_heads = 4
    t.num_key_value_heads = 2
    t.vocab_size = 1024

    v = config.vision_config
    v.image_size = 224                # capped from default 1540
    v.num_hidden_layers = 2
    v.hidden_size = 128               # = 2 * 64 (head_dim)
    v.intermediate_size = 256
    v.num_attention_heads = 2
    return config


def _build_inputs(config, B: int):
    """Mirror sweep/worker.py:2310-2328 (Mistral3 conditional-generation recipe)."""
    v = config.vision_config
    t = config.text_config
    img_size = v.image_size
    num_ch = getattr(v, "num_channels", 3) or 3
    patch_size = getattr(v, "patch_size", 14)
    merge_size = getattr(config, "spatial_merge_size", 2)
    patches_per_side = img_size // patch_size
    n_vis = (patches_per_side // merge_size) ** 2

    image_token_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
    text_len = 8
    seq_len = text_len + n_vis
    input_ids = torch.randint(0, t.vocab_size, (B, seq_len), device="cuda")
    if image_token_id is not None:
        input_ids[:, :n_vis] = image_token_id

    return {
        "pixel_values": torch.randn(B, num_ch, img_size, img_size, device="cuda"),
        "image_sizes": torch.tensor([[img_size, img_size]] * B, device="cuda"),
        "input_ids": input_ids,
        "attention_mask": torch.ones(B, seq_len, dtype=torch.long, device="cuda"),
    }


def make_model() -> torch.nn.Module:
    """Tier-1 (fast). Tiny config, B=1."""
    from transformers import Mistral3ForConditionalGeneration

    config = _build_config()
    torch.manual_seed(0)
    model = Mistral3ForConditionalGeneration(config).cuda().eval()
    image_token_id = getattr(config, "image_token_id", getattr(config, "image_token_index", None))
    if image_token_id is not None and image_token_id >= config.text_config.vocab_size:
        model.resize_token_embeddings(image_token_id + 1)
    return model


def make_inputs(model: torch.nn.Module) -> dict[str, Any]:
    """Tier-1 (fast). B=1, image_size=224 (n_vis=64, seq_len=72).

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

    image_size stays at 224; bumping past that exercises real ViT FLOPs but
    pixtral encoder time dominates and obscures the strategy-cost differences
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
CASE_BODY = f"""I have a script that hits graph breaks under torch.compile. The script is at {BASELINE_SCRIPT} and currently reports graph_break_count=16. The model is a multimodal vision-language model (Mistral3ForConditionalGeneration); the vision encoder is Pixtral.

Please:
1. Detect the graph breaks
2. Diagnose the root cause(s) — there are multiple distinct break shapes in this model (data-dependent guards from sdpa is_causal, decomposition issues from torch.nn.functional.unfold, and data-dep operators in the multimodal glue)
3. Fix them. You may edit any of:
   - {BASELINE_SCRIPT} (the test script — e.g. shape constraints)
   - {MISTRAL3_SRC} (the Mistral3 model source)
   - {PIXTRAL_SRC} (the Pixtral vision-encoder source)
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
            WatchedFile(path=MISTRAL3_SRC, original_backup=WORK_DIR / "modeling_mistral3.py.original"),
            WatchedFile(path=PIXTRAL_SRC, original_backup=WORK_DIR / "modeling_pixtral.py.original"),
            WatchedFile(path=BASELINE_SCRIPT, original_backup=WORK_DIR / "baseline_mistral3.py.original"),
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
