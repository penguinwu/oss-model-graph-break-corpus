#!/usr/bin/env python3
"""Worker for custom model corpus — tests a single model with torch.compile.

Workflow per model:
  1. Download source files from GitHub/HuggingFace
  2. Mock external dependencies
  3. Apply patches (if any)
  4. Import and instantiate model
  5. Run eager baseline
  6. Run graph break analysis (shared methodology across all suites)

Usage:
  python worker.py --model-json '{"name": "GFPGAN", ...}'
  python worker.py --model-name GFPGAN
  python worker.py --all
  python worker.py --all --output results.json
"""
import argparse
import contextlib
import importlib
import json
import os
import re
import sys
import time
import traceback
import types
from pathlib import Path

import torch

# Shared explain module — same methodology across all suites
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "sweep"))
from explain import run_graph_break_analysis


# Where to download source files
DOWNLOAD_DIR = Path(__file__).parent / "_sources"


def download_file(url, local_path, proxy_url=None):
    """Download a file from URL. Uses proxy if available (for Meta devvm)."""
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if local_path.exists():
        return True  # already downloaded

    if url is None:
        # Create empty file (e.g., __init__.py)
        local_path.touch()
        return True

    if proxy_url:
        import urllib.request
        req_data = json.dumps({"url": url, "max_size": 500000}).encode()
        req = urllib.request.Request(
            proxy_url,
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = json.loads(resp.read())
            if data.get("ok") and data.get("status") == 200:
                local_path.write_text(data["content"])
                return True
            else:
                print(f"  Proxy download failed: {data.get('status')}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"  Proxy error: {e}", file=sys.stderr)
            return False
    else:
        import urllib.request
        try:
            urllib.request.urlretrieve(url, str(local_path))
            return True
        except Exception as e:
            print(f"  Direct download failed: {e}", file=sys.stderr)
            return False


def download_model_sources(spec, proxy_url=None):
    """Download all source files for a model."""
    model_dir = DOWNLOAD_DIR / spec["name"].lower().replace(" ", "_").replace("-", "_")
    files = spec.get("files", {})

    for local_rel, url in files.items():
        local_path = model_dir / local_rel
        if not download_file(url, local_path, proxy_url):
            raise RuntimeError(f"Failed to download {local_rel} from {url}")

    return model_dir


def apply_patches(spec, model_dir):
    """Apply source patches (e.g., fixing missing imports)."""
    patches = spec.get("patches", {})
    for rel_path, patch_list in patches.items():
        fpath = model_dir / rel_path
        if not fpath.exists():
            continue
        content = fpath.read_text()
        for patch in patch_list:
            content = content.replace(patch["find"], patch["replace"])
        fpath.write_text(content)


def setup_mocks(spec, model_dir):
    """Set up mock modules for external dependencies."""
    mocks = spec.get("mocks", [])
    model_name = spec["name"]

    for mock_name in mocks:
        if mock_name in sys.modules:
            continue

        if mock_name == "basicsr":
            _mock_basicsr()
        elif mock_name == "f5_tts":
            _mock_f5_tts()
        elif mock_name == "module.mrte_model":
            _mock_mrte(model_dir)
        elif mock_name == "module.quantize":
            _mock_quantize(model_dir)
        elif mock_name == "text":
            _mock_text()
        elif mock_name == "torchmetrics":
            _mock_torchmetrics()
        elif mock_name == "flux.modules.lora":
            _mock_flux_lora()
        elif mock_name == "transformers_mock_for_resampler":
            _mock_transformers_minimal()
        else:
            # Generic empty mock
            mod = types.ModuleType(mock_name)
            sys.modules[mock_name] = mod


def _mock_basicsr():
    """Mock basicsr (GFPGAN dependency) with minimal arch_util."""
    from torch import nn

    basicsr = types.ModuleType("basicsr")
    arch_util = types.ModuleType("basicsr.archs.arch_util")

    def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
        for m in (module_list if isinstance(module_list, list) else [module_list]):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                m.weight.data *= scale
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.fill_(bias_fill)

    class ResBlock(nn.Module):
        def __init__(self, num_feat=64, res_scale=1, **kwargs):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True),
                nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.res_scale = res_scale
        def forward(self, x):
            return x + self.body(x) * self.res_scale

    arch_util.default_init_weights = default_init_weights
    arch_util.ResBlock = ResBlock

    basicsr.archs = types.ModuleType("basicsr.archs")
    basicsr.archs.arch_util = arch_util
    basicsr.utils = types.ModuleType("basicsr.utils")
    basicsr.utils.scandir = lambda *a, **kw: iter([])  # stub for file scanning
    basicsr.utils.registry = types.ModuleType("basicsr.utils.registry")

    class MockRegistry:
        def __init__(self, name):
            self.name = name
        def register(self, cls=None):
            if cls: return cls
            def wrapper(c): return c
            return wrapper

    basicsr.utils.registry.ARCH_REGISTRY = MockRegistry("arch")

    sys.modules["basicsr"] = basicsr
    sys.modules["basicsr.archs"] = basicsr.archs
    sys.modules["basicsr.archs.arch_util"] = arch_util
    sys.modules["basicsr.utils"] = basicsr.utils
    sys.modules["basicsr.utils.registry"] = basicsr.utils.registry


def _mock_f5_tts():
    """Mock f5_tts.model.DiT for GPT-SoVITS."""
    from torch import nn

    class MockDiT(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            dim = kwargs.get("dim", 256)
            self.proj = nn.Linear(dim, dim)
        def forward(self, x, *args, **kwargs):
            return self.proj(x)

    f5_tts = types.ModuleType("f5_tts")
    f5_tts_model = types.ModuleType("f5_tts.model")
    f5_tts_model.DiT = MockDiT
    f5_tts.model = f5_tts_model
    sys.modules["f5_tts"] = f5_tts
    sys.modules["f5_tts.model"] = f5_tts_model


def _mock_mrte(model_dir):
    """Mock module.mrte_model.MRTE for GPT-SoVITS."""
    from torch import nn

    class MockMRTE(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.proj = nn.Conv1d(192, 192, 1)
        def forward(self, ssl, ssl_mask, text, text_mask, ge):
            return self.proj(ssl) * ssl_mask

    mod = types.ModuleType("module.mrte_model")
    mod.MRTE = MockMRTE
    sys.modules["module.mrte_model"] = mod


def _mock_quantize(model_dir):
    """Mock module.quantize.ResidualVectorQuantizer for GPT-SoVITS."""
    from torch import nn

    class MockRVQ(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.dim = kwargs.get("dimension", 768)
        def forward(self, x, *args, **kwargs):
            return x, torch.zeros(1), torch.tensor(0.0), [x]
        def decode(self, codes):
            return codes

    mod = types.ModuleType("module.quantize")
    mod.ResidualVectorQuantizer = MockRVQ
    sys.modules["module.quantize"] = mod


def _mock_text():
    """Mock text.symbols for GPT-SoVITS."""
    text_mod = types.ModuleType("text")
    sv1 = types.ModuleType("text.symbols")
    sv1.symbols = list("abcdefghijklmnopqrstuvwxyz") + ["_"]
    sv2 = types.ModuleType("text.symbols2")
    sv2.symbols = list("abcdefghijklmnopqrstuvwxyz") + ["_", " "]
    text_mod.symbols = sv1
    text_mod.symbols2 = sv2
    sys.modules["text"] = text_mod
    sys.modules["text.symbols"] = sv1
    sys.modules["text.symbols2"] = sv2


def _mock_torchmetrics():
    """Mock torchmetrics for GPT-SoVITS."""
    tm = types.ModuleType("torchmetrics")
    tmc = types.ModuleType("torchmetrics.classification")

    class MockMCA:
        def __init__(self, *a, **kw): pass
        def __call__(self, *a): return torch.tensor(0.0)

    tmc.MulticlassAccuracy = MockMCA
    tm.classification = tmc
    sys.modules["torchmetrics"] = tm
    sys.modules["torchmetrics.classification"] = tmc


def _mock_flux_lora():
    """Mock flux.modules.lora for FLUX.1 (LoRA adapters not needed for testing)."""
    from torch import nn

    class LinearLora(nn.Linear):
        def __init__(self, *args, **kwargs):
            kwargs.pop("lora_rank", None)
            kwargs.pop("lora_alpha", None)
            super().__init__(*args, **kwargs)

    def replace_linear_with_lora(*args, **kwargs):
        pass

    mod = types.ModuleType("flux.modules.lora")
    mod.LinearLora = LinearLora
    mod.replace_linear_with_lora = replace_linear_with_lora
    sys.modules["flux.modules.lora"] = mod


def _mock_transformers_minimal():
    """Mock transformers with minimal PreTrainedModel for MiniCPM-V Resampler."""
    from torch import nn

    tm = types.ModuleType("transformers")

    class MockPTM(nn.Module):
        def __init__(self, *a, **kw): super().__init__()
        def _init_weights(self, m): pass

    tm.PreTrainedModel = MockPTM
    integ = types.ModuleType("transformers.integrations")
    integ.is_deepspeed_zero3_enabled = lambda: False
    tm.integrations = integ
    sys.modules["transformers"] = tm
    sys.modules["transformers.integrations"] = integ


def create_model(spec, model_dir):
    """Import and instantiate the model from downloaded sources."""
    # Add model_dir to path so imports work
    if str(model_dir) not in sys.path:
        sys.path.insert(0, str(model_dir))

    # Some models use internal relative imports (e.g., GPT-SoVITS: `from module import commons`)
    # that need a subdirectory on the path too
    extra_paths = spec.get("extra_sys_paths", [])
    for ep in extra_paths:
        full = str(model_dir / ep)
        if full not in sys.path:
            sys.path.insert(0, full)

    module_path = spec["model_module"]
    class_name = spec["model_class"]

    # Get kwargs
    if "model_kwargs_fn" in spec:
        from models import __dict__ as models_ns
        # Import input fn from models.py
        import models as models_mod
        kwargs_fn = getattr(models_mod, spec["model_kwargs_fn"])
        kwargs = kwargs_fn()
    else:
        kwargs = spec.get("model_kwargs", {})

    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    model = cls(**kwargs)
    model.eval()
    return model


def get_inputs(spec):
    """Generate test inputs for the model."""
    if "input_fn" in spec:
        import models as models_mod
        fn = getattr(models_mod, spec["input_fn"])
        return fn()
    elif "input_shape" in spec:
        return torch.randn(*spec["input_shape"])
    else:
        raise ValueError(f"No input specification for {spec['name']}")


def run_test(spec, proxy_url=None, mode="eval"):
    """Full test pipeline for one model."""
    name = spec["name"]
    result = {"name": name, "source": "custom", "category": spec.get("category", "unknown"),
              "mode": mode}

    t0 = time.time()

    # Phase 1: Download
    print(f"PHASE:download {name}", file=sys.stderr)
    try:
        model_dir = download_model_sources(spec, proxy_url)
        apply_patches(spec, model_dir)
        result["download_ok"] = True
    except Exception as e:
        result["status"] = "download_error"
        result["error"] = str(e)
        return result

    # Phase 2: Mock + Import
    print(f"PHASE:create {name}", file=sys.stderr)
    try:
        setup_mocks(spec, model_dir)
        model = create_model(spec, model_dir)
        if mode == "train":
            model.train()
        params = sum(p.numel() for p in model.parameters())
        result["params"] = params
        result["params_M"] = round(params / 1e6, 1)
    except Exception as e:
        result["status"] = "create_error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result

    # Phase 3: Eager
    print(f"PHASE:eager {name}", file=sys.stderr)
    try:
        inputs = get_inputs(spec)
        compile_target = spec.get("compile_target", "model")

        if compile_target.startswith("model."):
            method_name = compile_target.split(".", 1)[1]
            target_fn = getattr(model, method_name)
        else:
            target_fn = model

        ctx = torch.no_grad() if mode == "eval" else contextlib.nullcontext()
        with ctx:
            if isinstance(inputs, dict):
                eager_out = target_fn(**inputs)
            elif isinstance(inputs, tuple):
                eager_out = target_fn(*inputs)
            else:
                eager_out = target_fn(inputs)

        result["eager_ok"] = True
    except Exception as e:
        result["status"] = "eager_error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        return result

    # Phase 4: graph break analysis (shared methodology with HF/diffusers/timm suites)
    print(f"PHASE:explain {name}", file=sys.stderr)
    try:
        analysis = run_graph_break_analysis(target_fn, inputs, mode=mode)

        if analysis["status"] == "ok":
            result["status"] = "full_graph" if analysis["graph_break_count"] == 0 else "graph_break"
            result["graph_break_count"] = analysis["graph_break_count"]
            result["graph_count"] = analysis["graph_count"]
            result["break_reasons"] = analysis["break_reasons"]
            result["ops_per_graph"] = analysis["ops_per_graph"]
            result["compile_times"] = analysis["compile_times"]
            result["explain_time_s"] = analysis["explain_time_s"]
        else:
            result["status"] = "explain_error"
            result["error"] = analysis.get("error", "unknown")

    except Exception as e:
        result["status"] = "explain_error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    result["wall_time_s"] = round(time.time() - t0, 1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Custom model worker")
    parser.add_argument("--model-json", type=str, help="Model spec as JSON")
    parser.add_argument("--model-name", type=str, help="Model name from registry")
    parser.add_argument("--all", action="store_true", help="Test all models")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--proxy", type=str, default="http://localhost:7824/fetch",
                        help="Web proxy URL for downloads")
    parser.add_argument("--no-proxy", action="store_true", help="Skip proxy, download directly")
    parser.add_argument("--mode", choices=["eval", "train", "both"], default="eval",
                        help="Test mode: eval, train, or both")
    # Compatibility args for sweep runner integration
    parser.add_argument("--pass-num", type=int, default=1, help="Sweep pass number (ignored, for compatibility)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (ignored, for compatibility)")
    args = parser.parse_args()

    proxy_url = None if args.no_proxy else args.proxy

    # Add this directory to path so models.py is importable
    sys.path.insert(0, str(Path(__file__).parent))

    from models import enumerate_custom

    if args.model_json:
        specs = [json.loads(args.model_json)]
    elif args.model_name:
        all_models = enumerate_custom()
        specs = [m for m in all_models if m["name"] == args.model_name]
        if not specs:
            print(f"Unknown model: {args.model_name}", file=sys.stderr)
            sys.exit(1)
    elif args.all:
        specs = enumerate_custom()
    else:
        parser.print_help()
        sys.exit(1)

    modes = ["eval", "train"] if args.mode == "both" else [args.mode]

    results = []
    for spec in specs:
        for mode in modes:
            # Skip train mode for models without forward()
            if mode == "train" and spec.get("skip_train"):
                result = {"name": spec["name"], "source": "custom",
                          "category": spec.get("category", "unknown"),
                          "mode": mode, "status": "skipped",
                          "error": "skip_train=True (no forward())"}
                results.append(result)
                continue

            print(f"\n{'='*60}")
            print(f"Testing: {spec['name']} [{mode}]")
            print(f"{'='*60}")

            result = run_test(spec, proxy_url, mode=mode)
            results.append(result)

            status = result.get("status", "unknown")
            breaks = result.get("graph_break_count", "N/A")
            print(f"  Status: {status} | Graph breaks: {breaks}")

        if result.get("break_reasons"):
            for br in result["break_reasons"][:5]:
                print(f"    [{br['type']}] at {br['location']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = r.get("status", "unknown")
        breaks = r.get("graph_break_count", "N/A")
        params = r.get("params_M", "?")
        print(f"  {r['name']:35s} | {status:15s} | breaks={breaks} | {params}M")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"models": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if args.model_json and len(results) == 1:
        # Single model mode (sweep runner integration) — output one JSON line
        # The sweep runner reads the last line of stdout as JSON
        result = results[0]
        result["pass"] = args.pass_num
        print(json.dumps(result))
    else:
        # Multi-model mode — full formatted output
        print("\n--- JSON ---")
        print(json.dumps({"models": results}, indent=2))


if __name__ == "__main__":
    main()
