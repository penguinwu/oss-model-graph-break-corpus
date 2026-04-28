"""Per-case config: VitsModel (text-to-speech) train-mode graph breaks.

Reproduces graph breaks from `VitsModel` (facebook/mms-tts style TTS) under
torch.compile(fullgraph=True) with `model.train()` set. Multiple distinct
user-code shapes fire within this one case:

  1. find_spec / import_utils skipped-function calls in transformers/utils/import_utils.py
     (importlib.util.find_spec marked as skipped by Dynamo)
  2. data-dependent guard from `np.random.uniform` in the
     VitsResidualCouplingBlock dropout path (modeling_vits.py:1116) — fires only
     in train mode because layerdrop is gated by self.training
  3. as_proxy() failure on a `ValueError` arg/kwarg in the duration_predictor
     stochastic flow (modeling_vits.py:1350 / :801 / :672)
  4. callable() builtin operator un-traceable on StringFormatVariable in
     transformers/utils/import_utils.py:1487
  5. data-dependent F.conv1d guard from output_padding_mask shape
     (torch/nn/modules/conv.py:370 — `u0 < 1`) in the flow + decoder path

Per Phase 3 frame: the case = the model with its FULL break set. The agent
attacks the whole stack; BS-106 is the dominant tag (data-dep guard from
np.random.uniform under train-mode layerdrop) but the agent is not
scope-restricted to it.

NOTE — train mode: VitsModel has `if labels is not None: raise NotImplementedError`
in forward, so we cannot do real loss/backward. "Train mode" here means
`model.train()` is set so layerdrop / stochastic-dropout / Dropout layers are
active and the train-only break shapes (np.random.uniform layerdrop) fire.
The discovery harness's perf path still wraps in torch.no_grad() for timing —
this is expected; the train-flag is what surfaces the extra break shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

CASE_ID = "vits_model_train"
MODEL_NAME = "VitsModel"
BREAK_SHAPE_ID = "BS-106"  # data-dep guard from np.random.uniform layerdrop under train mode

# Locations on disk for the runner.
WORK_DIR = Path("/tmp/discovery-runs/vits_model_train")
VITS_SRC = Path("/home/pengwu/envs/torch211/lib/python3.12/site-packages/transformers/models/vits/modeling_vits.py")
BASELINE_SCRIPT = WORK_DIR / "baseline_vits.py"
VALIDATE_SCRIPT = WORK_DIR / "validate.py"


def _build_config():
    """Tiny VitsConfig. Defaults match facebook/mms-tts-eng but we shrink to
    keep the model small enough for fast iteration."""
    from transformers import VitsConfig

    config = VitsConfig(
        vocab_size=38,
        hidden_size=64,           # default 192
        num_hidden_layers=2,      # default 6
        num_attention_heads=2,
        ffn_dim=128,              # default 768
        flow_size=64,             # match hidden_size
        spectrogram_bins=64,      # default 513 — keep decoder small
        upsample_initial_channel=128,  # default 512
        prior_encoder_num_flows=2,     # default 4
        prior_encoder_num_wavenet_layers=2,  # default 4
        posterior_encoder_num_wavenet_layers=2,  # default 16
        duration_predictor_num_flows=2,  # default 4
        duration_predictor_filter_channels=64,  # default 256
        use_stochastic_duration_prediction=True,
        num_speakers=1,
    )
    return config


def _build_inputs(config, B: int, seq_len: int):
    """Mirror sweep/worker.py generic text recipe (no VitsModel-specific
    config exists in worker.py). VitsModel takes input_ids + attention_mask."""
    input_ids = torch.randint(0, config.vocab_size, (B, seq_len), device="cuda")
    attention_mask = torch.ones(B, seq_len, dtype=torch.long, device="cuda")
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_model() -> torch.nn.Module:
    """Tier-1 (fast). Tiny config, B=1, model.train() to surface train-mode breaks."""
    from transformers import VitsModel

    config = _build_config()
    torch.manual_seed(0)
    model = VitsModel(config).cuda().train()  # train(), not eval(), per case definition
    return model


def make_inputs(model: torch.nn.Module) -> dict[str, Any]:
    """Tier-1 (fast). B=1, seq_len=16. Optimized for discovery turnaround."""
    torch.manual_seed(0)
    return _build_inputs(model.config, B=1, seq_len=16)


# ----- Tier-2 (realistic) -----

def make_model_realistic() -> torch.nn.Module:
    """Tier-2 model. Identical config to tier-1 — we vary input length, not arch."""
    return make_model()


def make_inputs_realistic(model: torch.nn.Module) -> dict[str, Any]:
    """Tier-2 (realistic). B=2, seq_len=64 — non-trivial but keeps the
    HiFi-GAN decoder cost (which dominates) bounded."""
    torch.manual_seed(0)
    return _build_inputs(model.config, B=2, seq_len=64)


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
CASE_BODY = f"""I have a script that hits graph breaks under torch.compile. The script is at {BASELINE_SCRIPT} and reports a non-trivial graph_break_count. The model is VitsModel — a text-to-speech model (VITS architecture) with a text encoder, a flow-based prior, a stochastic duration predictor, and a HiFi-GAN decoder. The model is in TRAIN MODE (model.train()), which exercises layerdrop and stochastic dropout paths that don't fire in eval.

Please:
1. Detect the graph breaks
2. Diagnose the root cause(s) — there are multiple distinct break shapes in this model, including:
   - find_spec / importlib skipped-function calls in transformers/utils/import_utils.py
   - data-dependent guard from `np.random.uniform` in the layerdrop path (modeling_vits.py:1116) — train-mode only
   - as_proxy() failure on ValueError args in the stochastic duration predictor flow path
   - callable() builtin un-traceable in import_utils.py
   - data-dependent F.conv1d guard from output_padding_mask in the flow + decoder
3. Fix them. You may edit any of:
   - {BASELINE_SCRIPT} (the test script — e.g. shape constraints, replace np with torch)
   - {VITS_SRC} (the VitsModel source)
   Do NOT edit shared infrastructure outside these files (e.g. import_utils.py, conv.py).
4. Verify by re-running the script: graph_break_count should drop (ideally to 0 so fullgraph=True works) and the model output (waveform) should still match the original eager output within 1e-3.

IMPORTANT — what counts as a real fix: graph_break_count from your own {BASELINE_SCRIPT} is NOT the trial verdict. The trial uses a canonical-input check that loads VitsModel with stock test inputs and runs torch.compile WITHOUT your {BASELINE_SCRIPT} setup overrides applied (no `torch._dynamo.config` flips, no `backend=` swap from your baseline). Run `/home/pengwu/envs/torch211/bin/python /tmp/discovery-runs/vits_model_train/validate.py` to see the canonical `gb_under_canonical_inputs` count after any model edit you make. The trial only earns `fix_status: general` if canonical gb == 0. If canonical gb > 0, your fix relies on setup-script overrides — that's `setup-required` (or `none` under V8/V9 which forbid setup edits).

Methodology rules (declared-fallback — see design.md §4.7):
- Backend: default is `inductor` (the existing torch.compile call). `backend="eager"` is permitted ONLY with a one-line declaration in your final summary or as a `# DECLARED-OVERRIDE: backend=eager — <inductor numerics issue>` comment in the diff. Eager hides Inductor's noise floor; switching is a measurement-affecting change and will be tagged M post-hoc.
- Capture flags (`torch._dynamo.config.capture_scalar_outputs`, `capture_dynamic_output_shape_ops`): permitted ONLY with a one-line declaration naming the specific data-dependent op the flag is guarding (e.g. `# DECLARED-OVERRIDE: capture_scalar_outputs=True — guards u0<1 in conv1d output_padding_mask`). Undeclared flag flips will be tagged S (shortcut) post-hoc.

Both rules are intent specifications, not hard bans — declare and proceed if the override is genuinely needed.

The python interpreter is `python` (already on PATH at /home/pengwu/envs/torch211/bin/python). Do NOT use conda. Do NOT try to fetch external documentation — diagnose from inline output only.

When you have a fix you believe is correct, save it (in place — modify the existing files, do not write new files) and exit."""


def get_case_spec():
    """Build CaseSpec for the runner. Lazy import to avoid circular dep."""
    from discovery.runner import CaseSpec, WatchedFile

    return CaseSpec(
        case_id=CASE_ID,
        case_body=CASE_BODY,
        watched_files=[
            WatchedFile(path=VITS_SRC, original_backup=WORK_DIR / "modeling_vits.py.original"),
            WatchedFile(path=BASELINE_SCRIPT, original_backup=WORK_DIR / "baseline_vits.py.original"),
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
