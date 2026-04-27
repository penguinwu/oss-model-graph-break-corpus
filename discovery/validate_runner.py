"""Generic per-trial validator producing the validation_v2 schema.

Per-case `validate.py` shims into this so future runs produce structured
fix_status verdicts natively (no post-hoc revalidate.py needed).

Schema produced (printed as JSON to stdout):

```
{
  "integrity": {"import_ok", "eager_ok", "compile_ok"},
  "fix_status": "general" | "setup-required" | "none" | "unknown",
  "details": {
    "gb_in_agent_run": int | None,
    "gb_under_canonical_inputs": int | None,
    "gb_call_sites": [{"reason": str, "type": str, "file": str | None, "line": int | None, "location": str | None}, ...] | None,
    "eager_self_diff": float | None,
    "eager_deterministic": bool | None,
    "max_diff_compiled_vs_eager": float | None,
    "max_diff_vs_baseline": float | None
  },
  "error": str | None
}
```

Usage from a per-case validate.py shim:

```python
import sys
from discovery.validate_runner import main
sys.exit(main(case_id="mistral3_data_dep"))
```
"""
from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path

import torch

# sweep.explain is the corpus-canonical TORCH_LOGS-based graph-break analysis.
# Replaces deprecated torch._dynamo.explain (known to segfault on VITS in nightly,
# per corpora/custom-models/repro_segfault_gptsovits.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweep.explain import run_graph_break_analysis

# HF's set_seed covers ALL RNG sources: torch (CPU+CUDA) + numpy + python random.
# torch.manual_seed alone misses numpy/python.random, which causes intermittent
# non-determinism for any model that uses np.random or python.random in forward
# (e.g. VITS layerdrop uses np.random.uniform). Verified empirically: with full
# set_seed, VITS train mode is bit-identical across forwards; with torch-only
# seed, only 3-6 of 7 forwards match.
from transformers.trainer_utils import set_seed


def _extract_tensor(out):
    """Pull a comparable tensor out of a model output. Handles HuggingFace
    Output objects with various tensor field names, and bare tensors."""
    if isinstance(out, torch.Tensor):
        return out
    # HuggingFace ModelOutput convention: try common field names in order of
    # specificity. First field that's a tensor wins.
    for field in ("logits", "last_hidden_state", "waveform", "pooler_output", "prediction_logits"):
        if hasattr(out, field):
            v = getattr(out, field)
            if isinstance(v, torch.Tensor):
                return v
    # Fallback: scan all attributes for the first tensor (deterministic via dir order)
    for name in dir(out):
        if name.startswith("_"):
            continue
        try:
            v = getattr(out, name)
        except Exception:
            continue
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"can't extract a tensor from {type(out).__name__}")


def _safe_max_diff(a, b):
    """Compare two tensors with prefix-clamp on first OR last dim if shapes
    differ. First-dim clamp covers batch-variable models; last-dim clamp covers
    seq2seq / TTS models (e.g. VitsModel waveform output) where sequence length
    varies between calls."""
    if a.shape == b.shape:
        return (a - b).abs().max().item()
    if a.dim() == 0 or a.dim() != b.dim():
        return None
    # Try first-dim clamp.
    if a.shape[1:] == b.shape[1:]:
        n = min(a.shape[0], b.shape[0])
        if n > 0:
            return (a[:n] - b[:n]).abs().max().item()
    # Try last-dim clamp (TTS / seq2seq with variable output length).
    if a.shape[:-1] == b.shape[:-1]:
        n = min(a.shape[-1], b.shape[-1])
        if n > 0:
            return (a[..., :n] - b[..., :n]).abs().max().item()
    return None  # shape mismatch beyond simple prefix; can't compare


def _run_canonical_check(case) -> dict:
    """Run model with case-spec canonical inputs; return integrity + gb + max_diff."""
    out = {
        "import_ok": False,
        "eager_ok": False,
        "compile_ok": False,
        "graph_count": None,
        "graph_break_count": None,
        "graph_break_call_sites": None,
        "eager_self_diff": None,
        "eager_deterministic": None,
        "max_diff_vs_eager_baseline": None,
        "max_diff_compiled_vs_eager_now": None,
        "error": None,
    }
    try:
        # Use case_spec's make_model + make_inputs (the canonical regime)
        case_mod = importlib.import_module(f"discovery.cases.{case.case_id}")
        # Reset any cached modules the agent may have edited
        for m in list(sys.modules):
            # Heuristic: drop module families that might overlap with watched files
            for wf in case.watched_files:
                stem = Path(wf.path).stem
                if stem in m:
                    del sys.modules[m]
                    break
        # Re-import case (and via it, fresh model code)
        case_mod = importlib.reload(case_mod)
        out["import_ok"] = True

        # Reseed before each model invocation. Stochastic ops in train mode
        # (layerdrop, dropout, stochastic samplers) advance the global RNG state
        # during each call, so eager / explain / compile must each start from
        # the same seed for a deterministic apples-to-apples comparison.
        set_seed(0)
        model = case_mod.make_model()
        inputs = case_mod.make_inputs(model)

        set_seed(0)
        with torch.no_grad():
            eager_raw = model(**inputs)
        eager_out = _extract_tensor(eager_raw)
        out["eager_ok"] = True

        # Second eager forward WITH set_seed reseed (matching the first call).
        # A non-zero diff here means HF's set_seed is INSUFFICIENT for this
        # model — there's an RNG source set_seed doesn't cover (e.g. custom
        # generator, hardware non-determinism, time-based ops). This is the
        # diagnostic for "is our reseed pattern adequate?" rather than "does
        # the model use RNG at all" (the latter is trivially yes for any HF
        # model in train mode).
        inputs_b = {
            k: (v.detach().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        set_seed(0)
        with torch.no_grad():
            eager_raw_b = model(**inputs_b)
        eager_out_b = _extract_tensor(eager_raw_b)
        self_diff = _safe_max_diff(eager_out, eager_out_b)
        if self_diff is not None:
            out["eager_self_diff"] = self_diff
            out["eager_deterministic"] = (self_diff == 0.0)

        # Compare to saved baseline output if present
        baseline_eager_pt = None
        if hasattr(case_mod, "WORK_DIR"):
            cand = case_mod.WORK_DIR / "baseline_eager_output.pt"
            if cand.exists():
                baseline_eager_pt = cand
        if baseline_eager_pt:
            baseline_loaded = torch.load(baseline_eager_pt, map_location=eager_out.device)
            baseline_out = _extract_tensor(baseline_loaded) if not isinstance(baseline_loaded, torch.Tensor) else baseline_loaded
            md = _safe_max_diff(eager_out, baseline_out)
            if md is not None:
                out["max_diff_vs_eager_baseline"] = md

        torch._dynamo.reset()
        set_seed(0)
        # sweep.explain.run_graph_break_analysis is the corpus-canonical
        # methodology (TORCH_LOGS=graph_breaks + counting backend). Returns
        # a dict with graph_count, graph_break_count, and break_reasons
        # already classified by type (Tensor.item() / data-dependent-branch /
        # aten.nonzero / other).
        analysis = run_graph_break_analysis(model, inputs, mode="eval")
        if analysis["status"] == "ok":
            out["graph_count"] = analysis["graph_count"]
            out["graph_break_count"] = analysis["graph_break_count"]
            out["graph_break_call_sites"] = [
                {
                    "reason": br.get("reason"),
                    "type": br.get("type"),
                    "file": br.get("file"),
                    "line": br.get("line"),
                    "location": br.get("location"),
                }
                for br in analysis["break_reasons"]
            ]
        else:
            # explain crashed (e.g. PendingUnbackedSymbolNotFound). Surface the
            # error in graph_break_call_sites so the validator's outer try/except
            # doesn't swallow the methodology signal.
            out["graph_break_call_sites"] = [
                {"reason": f"explain_error: {analysis.get('error', 'unknown')}",
                 "type": "explain_error", "file": None, "line": None}
            ]

        torch._dynamo.reset()
        set_seed(0)
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled_raw = compiled(**inputs)
        compiled_out = _extract_tensor(compiled_raw)
        out["compile_ok"] = True
        md = _safe_max_diff(eager_out, compiled_out)
        if md is not None:
            out["max_diff_compiled_vs_eager_now"] = md

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
    return out


def _run_agent_baseline(case) -> int | None:
    """Run the agent's edited baseline_*.py via subprocess; parse `graph_break_count=N`."""
    baseline_wf = None
    for wf in case.watched_files:
        if "baseline" in Path(wf.path).name:
            baseline_wf = wf
            break
    if baseline_wf is None:
        return None
    try:
        res = subprocess.run(
            [sys.executable, str(baseline_wf.path)],
            capture_output=True, text=True, timeout=300,
        )
        text = res.stdout + "\n" + res.stderr
        m = re.search(r"graph_break_count\s*=\s*(\d+)", text)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _derive_fix_status(agent_gb: int | None, canonical_gb: int | None) -> str:
    if agent_gb is None:
        if canonical_gb == 0:
            return "general"
        if canonical_gb is not None and canonical_gb > 0:
            return "none"
        return "unknown"
    if agent_gb == 0 and canonical_gb == 0:
        return "general"
    if agent_gb == 0 and canonical_gb is not None and canonical_gb > 0:
        return "setup-required"
    if agent_gb > 0:
        return "none"
    return "unknown"


def main(case_id: str) -> int:
    """Entry point for per-case validate.py shims. Prints validation_v2 JSON to stdout."""
    case_mod = importlib.import_module(f"discovery.cases.{case_id}")
    case = case_mod.get_case_spec()

    canonical = _run_canonical_check(case)
    agent_gb = _run_agent_baseline(case)
    fix_status = _derive_fix_status(agent_gb, canonical["graph_break_count"])

    validation_v2 = {
        "integrity": {
            "import_ok": canonical["import_ok"],
            "eager_ok": canonical["eager_ok"],
            "compile_ok": canonical["compile_ok"],
        },
        "fix_status": fix_status,
        "details": {
            "gb_in_agent_run": agent_gb,
            "gb_under_canonical_inputs": canonical["graph_break_count"],
            "gb_call_sites": canonical["graph_break_call_sites"],
            "eager_self_diff": canonical["eager_self_diff"],
            "eager_deterministic": canonical["eager_deterministic"],
            "max_diff_compiled_vs_eager": canonical["max_diff_compiled_vs_eager_now"],
            "max_diff_vs_baseline": canonical["max_diff_vs_eager_baseline"],
        },
        "error": canonical["error"],
    }
    print(json.dumps(validation_v2, indent=2))
    return 0
