"""Per-case config: Dbrx MoE data-dependent expert dispatch graph break.

Reproduces transformers/models/dbrx/modeling_dbrx.py:313 — `for expert_idx in expert_hit:`
data-dependent loop in DbrxExperts.forward (MoE expert dispatch).

Mirrors the pilot 4 baseline_dbrx.py / validate.py config exactly so that perf
numbers are comparable to pilot 4 results.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

CASE_ID = "dbrx_moe_data_dep"
MODEL_NAME = "DbrxModel"
BREAK_SHAPE_ID = "BS-103"  # data-dependent for-loop in MoE expert dispatch

# Locations on disk for the runner.
WORK_DIR = Path("/tmp/discovery-runs/dbrx_moe_data_dep")
DBRX_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/dbrx/modeling_dbrx.py")
BASELINE_SCRIPT = WORK_DIR / "baseline_dbrx.py"
VALIDATE_SCRIPT = WORK_DIR / "validate.py"


def make_model() -> torch.nn.Module:
    """Build the same tiny Dbrx config used in pilot 4."""
    from transformers import DbrxConfig, DbrxForCausalLM

    config = DbrxConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        num_hidden_layers=2,
        max_seq_len=64,
        attn_config={"kv_n_heads": 1, "rope_theta": 500000.0, "clip_qkv": 8.0},
        ffn_config={"ffn_hidden_size": 64, "moe_num_experts": 4, "moe_top_k": 2},
    )
    config.hidden_size = config.d_model
    config.ffn_config.hidden_size = config.d_model

    torch.manual_seed(0)
    return DbrxForCausalLM(config).eval().cuda()


def make_inputs(_model: torch.nn.Module) -> dict[str, Any]:
    """Tier-1 (fast). Same input shape as pilot 4 baseline: (1, 16) = 16 tokens.

    Optimized for discovery turnaround. At this size, GPU is overhead-bound and
    FLOP differences (e.g. masked-dense's 2x compute) are invisible. Use tier-2
    for fix-quality numbers.
    """
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (1, 16), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# ----- Tier-2 (realistic) -----

def make_model_realistic() -> torch.nn.Module:
    """Tier-2 model. Same architecture as tier-1, but with a larger max_seq_len
    so realistic-shaped inputs fit. Other config matches tier-1 to keep the
    only changed knob input size."""
    from transformers import DbrxConfig, DbrxForCausalLM

    config = DbrxConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        num_hidden_layers=2,
        max_seq_len=1024,                   # bumped from 64 — RoPE buffer only
        attn_config={"kv_n_heads": 1, "rope_theta": 500000.0, "clip_qkv": 8.0},
        ffn_config={"ffn_hidden_size": 64, "moe_num_experts": 4, "moe_top_k": 2},
    )
    config.hidden_size = config.d_model
    config.ffn_config.hidden_size = config.d_model

    torch.manual_seed(0)
    return DbrxForCausalLM(config).eval().cuda()


def make_inputs_realistic(_model: torch.nn.Module) -> dict[str, Any]:
    """Tier-2 (realistic). batch=4, seq=512 = 2048 tokens (~128x tier-1).

    At this size MoE FLOPs dominate over kernel-launch overhead — masked-dense's
    2x compute should be visible. This is where strategy ranking happens.
    """
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (4, 512), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# Perf is measured in a SUBPROCESS (see _measure_case.py). This avoids
# module-state contamination — sys.modules holds the agent-modified dbrx code,
# and reimporting in-process is fragile. A fresh Python is the simple fix.
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


# Case body — the IDENTICAL prompt body Pilot 4 used, minus the constraint sentence.
# (Constraint is appended by the variant catalog.)
CASE_BODY = f"""I have a script that hits graph breaks under torch.compile. The script is at {BASELINE_SCRIPT} and currently reports graph_break_count=9.

Please:
1. Detect the graph breaks
2. Diagnose the root cause
3. Fix it. You may edit either the script ({BASELINE_SCRIPT}) or the model source file at {DBRX_SRC} — pick whichever you believe is the right place to fix.
4. Verify by re-running the script: graph_break_count should drop (ideally to 0 so fullgraph=True works) and the model output should still match the original eager output within 1e-4.

The python interpreter is `python` (already on PATH at /home/pengwu/envs/torch211/bin/python). Do NOT use conda. Do NOT try to fetch external documentation — diagnose from inline output only.

When you have a fix you believe is correct, save it (in place — modify the existing files, do not write new files) and exit."""


def get_case_spec():
    """Build CaseSpec for the runner. Lazy import to avoid circular dep."""
    from discovery.runner import CaseSpec, WatchedFile

    return CaseSpec(
        case_id=CASE_ID,
        case_body=CASE_BODY,
        watched_files=[
            WatchedFile(path=DBRX_SRC, original_backup=WORK_DIR / "modeling_dbrx.py.original"),
            WatchedFile(path=BASELINE_SCRIPT, original_backup=WORK_DIR / "baseline_dbrx.py.original"),
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
