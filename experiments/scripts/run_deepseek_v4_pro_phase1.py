"""DeepSeek V4 Pro evaluation, Phase 1 runner.

Reads config from experiments/configs/deepseek-v4-pro-phase1.json.
Writes results to experiments/results/deepseek_v4_pro/phase1-tiny-<date>/results.json.

Requires the DeepSeek V4 PR branch of transformers (HF #45643). Run as:

    PYTHONPATH=/tmp/transformers-v4/src ~/envs/torch211/bin/python \\
        experiments/scripts/run_deepseek_v4_pro_phase1.py

See experiments/deepseek_v4_pro_eval_plan.md for the experiment plan.
"""
from __future__ import annotations

import datetime as _dt
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from discovery.perf import patch_torch_manual_seed, measure_perf  # noqa: E402

CONFIG_PATH = REPO_ROOT / "experiments" / "configs" / "deepseek-v4-pro-phase1.json"


def main() -> None:
    cfg = json.load(open(CONFIG_PATH))
    print(f"Loaded config: {cfg['name']}")
    print(f"  description: {cfg['description'][:100]}...")

    # Apply RNG patch BEFORE any model construction.
    patch_torch_manual_seed(seed=cfg["settings"]["seed"])
    torch.set_float32_matmul_precision(cfg["settings"]["matmul_precision"])

    # Lazy import — needs PYTHONPATH override for the PR branch
    try:
        from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM
        import transformers
    except ImportError as e:
        print(f"FATAL: cannot import transformers.models.deepseek_v4 — {e}")
        print("Hint: PYTHONPATH=/tmp/transformers-v4/src python ...")
        sys.exit(1)

    print(f"  transformers: {transformers.__version__} from {transformers.__file__}")

    cfg_kw = cfg["model"]["config_kw"].copy()
    dtype = getattr(torch, cfg["settings"]["dtype"])
    device = cfg["settings"]["device"]
    seed = cfg["settings"]["seed"]
    batch = cfg["settings"]["input_shape"]["batch"]
    seq_len = cfg["settings"]["input_shape"]["seq_len"]
    vocab = cfg_kw["vocab_size"]
    tolerance = cfg["settings"]["tolerance_max_abs_diff"]

    def make_config():
        return DeepseekV4Config(**cfg_kw)

    def make_model():
        torch.manual_seed(seed)  # patched, will always seed 1337 (or configured)
        m = DeepseekV4ForCausalLM(make_config()).to(dtype=dtype, device=device).eval()
        return m

    def make_inputs(_model=None):
        torch.manual_seed(seed)
        return {"input_ids": torch.randint(0, vocab, (batch, seq_len), device=device)}

    started_at = _dt.datetime.utcnow().isoformat() + "Z"
    results: dict = {
        "config_name": cfg["name"],
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "model": {
            "architecture": cfg["model"]["architecture"],
            "transformers_branch": cfg["model"]["transformers_branch"],
            "transformers_branch_sha": cfg["model"]["transformers_branch_sha"],
            "transformers_version": transformers.__version__,
            "config_kw": cfg_kw,
        },
        "settings": cfg["settings"],
        "compile_kwargs": cfg["compile"],
        "torch_version": torch.__version__,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "started_at": started_at,
        "rows": [],
    }

    # ---------- Step 1: instantiate + count params + eager forward ----------
    print()
    print("=== Step 1: instantiate + eager ===")
    t0 = time.time()
    m = make_model()
    inst_s = time.time() - t0
    n_params = sum(p.numel() for p in m.parameters())
    print(f"  instantiation: {inst_s:.1f}s")
    print(f"  total params:  {n_params/1e9:.3f}B  ({n_params:,})")
    bf16_gb = n_params * 2 / (1024**3)
    print(f"  weights at bf16: {bf16_gb:.2f} GB")

    inputs = make_inputs(m)
    with torch.no_grad():
        eager_out = m(**inputs)
    print(f"  eager output: {type(eager_out).__name__}, logits {tuple(eager_out.logits.shape)} {eager_out.logits.dtype}")
    eager_logits = eager_out.logits.detach().clone()

    eager_peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
    print(f"  cuda max memory allocated: {eager_peak_alloc_gb:.2f} GB")

    del m, eager_out, inputs
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- Step 2: dynamo.explain (graph break analysis) ----------
    print()
    print("=== Step 2: graph break analysis (torch._dynamo.explain) ===")
    torch._dynamo.reset()
    m_explain = make_model()
    x_explain = make_inputs(m_explain)
    explain_t0 = time.time()
    try:
        with torch.no_grad():
            expl = torch._dynamo.explain(m_explain)(**x_explain)
        gb_count = expl.graph_break_count
        graph_count = expl.graph_count
        op_count = expl.op_count
        break_reasons_unique = sorted({str(r)[:200] for r in expl.break_reasons})[:20]
        print(f"  graph_count:       {graph_count}")
        print(f"  graph_break_count: {gb_count}")
        print(f"  op_count:          {op_count}")
        if gb_count > 0:
            print("  --- break reasons ---")
            for r in break_reasons_unique:
                print(f"    {r}")
        explain_status = "ok"
        explain_error = None
    except Exception as e:
        gb_count = -1
        graph_count = -1
        op_count = -1
        break_reasons_unique = []
        explain_status = "error"
        explain_error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  EXPLAIN FAILED: {explain_error}")
    explain_s = time.time() - explain_t0
    print(f"  explain wall: {explain_s:.1f}s")

    del m_explain, x_explain
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- Step 3: correctness (compiled vs eager) ----------
    print()
    print("=== Step 3: correctness check ===")
    torch._dynamo.reset()
    m_compile = make_model()
    x_compile = make_inputs(m_compile)
    m_compiled = torch.compile(m_compile, **cfg["compile"])
    correctness_t0 = time.time()
    try:
        with torch.no_grad():
            comp_out = m_compiled(**x_compile)
        comp_logits = comp_out.logits.detach().clone()
        max_abs_diff = (eager_logits.float() - comp_logits.float()).abs().max().item()
        bitwise_equal = bool(torch.equal(eager_logits, comp_logits))
        passes_tolerance = max_abs_diff <= tolerance
        print(f"  max_abs_diff vs eager: {max_abs_diff:.3e}")
        print(f"  bitwise_equal:         {bitwise_equal}")
        print(f"  passes {tolerance:.0e} tol:    {passes_tolerance}")
        correctness_status = "ok"
        correctness_error = None
    except Exception as e:
        max_abs_diff = float("nan")
        bitwise_equal = False
        passes_tolerance = False
        correctness_status = "error"
        correctness_error = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  CORRECTNESS FAILED: {correctness_error}")
    correctness_s = time.time() - correctness_t0
    print(f"  correctness wall: {correctness_s:.1f}s")

    del m_compile, m_compiled, x_compile
    gc.collect()
    torch.cuda.empty_cache()

    # ---------- Step 4: tier-1 perf (measure_perf) ----------
    print()
    print("=== Step 4: tier-1 perf measurement ===")
    perf_settings = cfg["settings"]["perf"]
    perf_t0 = time.time()
    perf_result = measure_perf(
        model_fn=lambda: make_model(),
        inputs_fn=lambda m: make_inputs(m),
        n_warmup=perf_settings["n_warmup"],
        n_repeat=perf_settings["n_repeat"],
        device=device,
        compile_kwargs=cfg["compile"],
        warm_peak_mem=perf_settings["warm_peak_mem"],
        eager_self_check=True,
    )
    perf_d = perf_result.to_dict()
    print(f"  eager_ms:    {perf_d['eager_ms']:.3f}")
    print(f"  compiled_ms: {perf_d['compiled_ms']:.3f}")
    print(f"  speedup:     {perf_d['speedup']:.2f}x")
    print(f"  compile_s:   {perf_d['compile_s']:.2f} s")
    print(f"  eager peak (cold): {perf_d['eager_peak_mem_mb']:.0f} MB")
    print(f"  compiled peak (cold): {perf_d['compiled_peak_mem_mb']:.0f} MB")
    print(f"  eager_self_diff: {perf_d['eager_self_diff']}, deterministic: {perf_d['eager_deterministic']}")
    perf_s = time.time() - perf_t0
    print(f"  perf wall: {perf_s:.1f}s")

    # ---------- Compose results.json ----------
    finished_at = _dt.datetime.utcnow().isoformat() + "Z"
    results["finished_at"] = finished_at
    results["instantiation_s"] = inst_s
    results["total_params"] = n_params
    results["weights_bf16_gb"] = bf16_gb
    results["eager_peak_alloc_gb_first_pass"] = eager_peak_alloc_gb

    results["rows"].append({
        "dimension": "graph_breaks",
        "status": explain_status,
        "graph_count": graph_count,
        "graph_break_count": gb_count,
        "op_count": op_count,
        "break_reasons_unique": break_reasons_unique,
        "wall_s": explain_s,
        "error": explain_error,
    })
    results["rows"].append({
        "dimension": "correctness",
        "status": correctness_status,
        "max_abs_diff_vs_eager": max_abs_diff,
        "bitwise_equal": bitwise_equal,
        "passes_tolerance": passes_tolerance,
        "tolerance": tolerance,
        "wall_s": correctness_s,
        "error": correctness_error,
    })
    results["rows"].append({
        "dimension": "performance_tier1",
        "status": "ok" if perf_d.get("error") is None else "error",
        **perf_d,
    })

    # ---------- Write output ----------
    date_stamp = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_dir = REPO_ROOT / cfg["output_dir"].format(date=date_stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print()
    print(f"=== Results written to {out_path.relative_to(REPO_ROOT)} ===")

    # ---------- One-line summary ----------
    print()
    print("=== Headline ===")
    print(f"  gb_count: {gb_count}")
    print(f"  max_abs_diff: {max_abs_diff:.3e}  (tol {tolerance})")
    print(f"  speedup tier-1: {perf_d['speedup']:.2f}x")
    print(f"  compile_s: {perf_d['compile_s']:.2f}s")


if __name__ == "__main__":
    main()
