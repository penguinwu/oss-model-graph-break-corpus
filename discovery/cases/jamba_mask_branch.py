"""Per-case config: Jamba data-dependent attention-mask branch graph break.

Reproduces transformers/models/jamba/modeling_jamba.py:755 —
`torch.all(attention_mask == 1)` data-dependent branch in `_update_mamba_mask`.

Tests a different break-shape dimension than the Dbrx case:
- Dbrx: data-dependent for-loop (control flow on tensor count)
- Jamba: data-dependent branch (control flow on tensor predicate)

Mirrors Pilot 2 baseline_jamba.py / validate.py config exactly so that perf
numbers are comparable to pilot 2 results.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

CASE_ID = "jamba_mask_branch"
MODEL_NAME = "JambaForCausalLM"
BREAK_SHAPE_ID = "BS-104"  # data-dependent branch on tensor predicate

WORK_DIR = Path("/tmp/discovery-runs/jamba_mask_branch")
JAMBA_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/jamba/modeling_jamba.py")
BASELINE_SCRIPT = WORK_DIR / "baseline_jamba.py"
VALIDATE_SCRIPT = WORK_DIR / "validate.py"


def _make_config(use_mamba_kernels: bool = False):
    from transformers import JambaConfig
    return JambaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=2,
        expert_layer_period=2,
        expert_layer_offset=1,
        attn_layer_period=2,
        attn_layer_offset=0,
        use_mamba_kernels=use_mamba_kernels,
    )


def make_model() -> torch.nn.Module:
    """Tier-1 model. Same config as Pilot 2 baseline."""
    from transformers import JambaForCausalLM
    torch.manual_seed(0)
    return JambaForCausalLM(_make_config()).eval().cuda()


def make_inputs(_model: torch.nn.Module) -> dict[str, Any]:
    """Tier-1 (fast). (1, 16) = 16 tokens. Mirrors Pilot 2."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (1, 16), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ----- Tier-2 (realistic) -----

def make_model_realistic() -> torch.nn.Module:
    """Tier-2 model. Same architecture as tier-1 — Jamba's relevant config knob
    is num_hidden_layers + use_mamba_kernels, neither of which we change.
    Sequence length is the input axis we vary."""
    return make_model()


def make_inputs_realistic(_model: torch.nn.Module) -> dict[str, Any]:
    """Tier-2 (realistic). batch=4, seq=512 = 2048 tokens (~128x tier-1).

    Same shape as the Dbrx tier-2 to keep cross-case comparisons sane.
    """
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (4, 512), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


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


CASE_BODY = f"""I have a script that hits graph breaks under torch.compile. The script is at {BASELINE_SCRIPT} and currently reports graph_break_count=3.

Please:
1. Detect the graph breaks
2. Diagnose the root cause
3. Fix it. You may edit either the script ({BASELINE_SCRIPT}) or the model source file at {JAMBA_SRC} — pick whichever you believe is the right place to fix.
4. Verify by re-running the script: graph_break_count should drop (ideally to 0 so fullgraph=True works) and the model output should still match the original eager output within 1e-4.

The python interpreter is `python` (already on PATH at /home/pengwu/envs/torch211/bin/python). Do NOT use conda. Do NOT try to fetch external documentation — diagnose from inline output only.

When you have a fix you believe is correct, save it (in place — modify the existing files, do not write new files) and exit."""


def get_case_spec():
    from discovery.runner import CaseSpec, WatchedFile

    return CaseSpec(
        case_id=CASE_ID,
        case_body=CASE_BODY,
        watched_files=[
            WatchedFile(path=JAMBA_SRC, original_backup=WORK_DIR / "modeling_jamba.py.original"),
            WatchedFile(path=BASELINE_SCRIPT, original_backup=WORK_DIR / "baseline_jamba.py.original"),
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
    print(json.dumps({"case_id": CASE_ID, "baseline_tier1": result.to_dict()}, indent=2))
