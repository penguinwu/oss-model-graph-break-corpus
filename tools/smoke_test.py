#!/usr/bin/env python3
"""Smoke test for sweep infrastructure.

Runs 8 known models through the real sweep pipeline and verifies each
produces the expected status. Catches regressions from code changes to
worker.py, models.py, or run_sweep.py in ~2 minutes.

Usage:
    python tools/smoke_test.py
    python tools/smoke_test.py --device cpu
    python tools/smoke_test.py --python /path/to/venv/bin/python
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = REPO_ROOT / "sweep" / "worker.py"
MODELLIBS_CONFIG = REPO_ROOT / "sweep" / "modellibs.json"
MODELLIBS_ROOT = Path.home() / "envs" / "modellibs"
TIMEOUT_PER_MODEL = 120  # generous; models take 10-30s


def _venv_name_from_python(python_bin: str):
    """Extract venv name from a python interpreter path. Mirrors run_sweep.py.

    Uses lexical path (NOT .resolve()) since venvs symlink to system python.
    Returns None if path doesn't look like a ~/envs/<name>/ venv.
    """
    try:
        p = Path(python_bin).absolute()
        if p.parent.name == "bin" and p.parent.parent.parent.name == "envs":
            return p.parent.parent.name
    except (OSError, IndexError):
        pass
    return None


def build_modellibs_env(python_bin):
    """Return env dict with PYTHONPATH injecting modellibs per defaults_per_venv.

    Mirrors sweep/run_sweep.py's _resolve_modellib_paths + _ensure_modellib_pythonpath
    so that subprocess invocations from smoke_test (which bypass run_sweep)
    still get the modellibs after the Phase 5 cleanup.

    Returns dict suitable for subprocess.run(env=...). Always includes the
    full current os.environ so other vars (CUDA, HF_HOME, ...) carry through.
    """
    env = dict(os.environ)
    venv_name = _venv_name_from_python(python_bin)
    if not venv_name:
        return env
    try:
        with open(MODELLIBS_CONFIG) as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        return env
    venv_defaults = cfg.get("defaults_per_venv", {}).get(venv_name, {})
    if not venv_defaults:
        return env
    paths = []
    for pkg in ("transformers", "diffusers", "timm"):
        ver = venv_defaults.get(pkg)
        if not ver:
            continue
        tree = MODELLIBS_ROOT / f"{pkg}-{ver}"
        if tree.exists():
            paths.append(str(tree))
    if paths:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join(paths + ([existing] if existing else []))
    return env

# Fixed set of 8 models covering all status categories.
# Each entry: (spec_dict, expected_eval_status)
# These models are stable across pt2.8-pt2.11 sweeps.
SMOKE_MODELS = [
    # full_graph (3) — text decoder, text encoder, vision
    ({"name": "GPT2Model", "source": "hf", "hf_class": "GPT2Model",
      "hf_config": "GPT2Config"}, "full_graph"),
    ({"name": "DistilBertModel", "source": "hf", "hf_class": "DistilBertModel",
      "hf_config": "DistilBertConfig"}, "full_graph"),
    ({"name": "ViTModel", "source": "hf", "hf_class": "ViTModel",
      "hf_config": "ViTConfig", "input_type": "vision"}, "full_graph"),
    # graph_break (3) — LSH attention, sliding window, encoder-decoder
    ({"name": "ReformerModel", "source": "hf", "hf_class": "ReformerModel",
      "hf_config": "ReformerConfig"}, "graph_break"),
    ({"name": "LongformerModel", "source": "hf", "hf_class": "LongformerModel",
      "hf_config": "LongformerConfig"}, "graph_break"),
    ({"name": "T5Model", "source": "hf", "hf_class": "T5Model",
      "hf_config": "T5Config", "input_type": "seq2seq"}, "graph_break"),
    # graph_break (4th) — was eager_error, fixed upstream in transformers 5.4.0
    ({"name": "OlmoHybridModel", "source": "hf", "hf_class": "OlmoHybridModel",
      "hf_config": "OlmoHybridConfig"}, "graph_break"),
    # create_error (1) — missing detectron2 dependency
    ({"name": "LayoutLMv2Model", "source": "hf", "hf_class": "LayoutLMv2Model",
      "hf_config": "LayoutLMv2Config"}, "create_error"),
]

# Map "clean" -> "full_graph" (worker uses "clean" internally)
STATUS_ALIASES = {"clean": "full_graph"}


def run_single_model(python_bin, spec, device):
    """Run one model through worker.py identify pass (eval mode). Returns result dict."""
    cmd = [
        python_bin, str(WORKER_SCRIPT),
        "--model-json", json.dumps(spec),
        "--pass-num", "1",
        "--device", device,
        "--mode", "eval",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=TIMEOUT_PER_MODEL,
            env=build_modellibs_env(python_bin),
        )
        if proc.returncode != 0 and not proc.stdout.strip():
            return {"status": "worker_crash", "error": proc.stderr[-500:]}
        # Worker outputs a single JSON line to stdout
        last_line = proc.stdout.strip().split("\n")[-1]
        result = json.loads(last_line)
        # Normalize status aliases
        result["status"] = STATUS_ALIASES.get(result["status"], result["status"])
        return result
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {TIMEOUT_PER_MODEL}s"}
    except (json.JSONDecodeError, IndexError) as e:
        return {"status": "parse_error", "error": str(e)}


def check_enumerate_count(python_bin):
    """Verify enumerate_all() returns a reasonable model count."""
    script = (
        "import sys; sys.path.insert(0, 'sweep'); "
        "from models import enumerate_all; "
        "specs = enumerate_all(); "
        "print(len(specs))"
    )
    try:
        proc = subprocess.run(
            [python_bin, "-c", script],
            capture_output=True, text=True, timeout=60,
            cwd=str(REPO_ROOT),
            env=build_modellibs_env(python_bin),
        )
        if proc.returncode != 0:
            return False, 0, proc.stderr[-200:]
        count = int(proc.stdout.strip().split("\n")[-1])
        return count > 700, count, ""
    except Exception as e:
        return False, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="Sweep infrastructure smoke test")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--python", default=sys.executable,
                        help="Python binary (default: current interpreter)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    # Sanity check: smoke_test needs a venv-aware python (not bare system python)
    # so that build_modellibs_env can resolve PYTHONPATH for the post-Phase-5
    # cleanup. If user passes /usr/local/fbcode/.../python3 they'll just see
    # 8 worker_crash failures from `import torch` failing.
    venv_name = _venv_name_from_python(args.python)
    if not venv_name:
        print(f"ERROR: --python must point at a torch venv (e.g. ~/envs/torch211/bin/python3),",
              file=sys.stderr)
        print(f"       not bare system python. Got: {args.python}", file=sys.stderr)
        print(f"       (After Phase 5 modellibs cleanup, smoke_test relies on resolving",
              file=sys.stderr)
        print(f"        venv name → sweep/modellibs.json defaults_per_venv → PYTHONPATH.",
              file=sys.stderr)
        print(f"        System python is not in any venv so no defaults can be resolved.)",
              file=sys.stderr)
        sys.exit(2)

    n_models = len(SMOKE_MODELS)
    print("=" * 60)
    print(f"Smoke test: {n_models} models through sweep pipeline")
    print(f"  Device: {args.device}, Python: {args.python}")
    print("=" * 60)
    print()

    # Phase 1: enumerate_all() check
    enum_ok, enum_count, enum_err = check_enumerate_count(args.python)
    tag = "PASS" if enum_ok else "FAIL"
    print(f"enumerate_all() count: {enum_count} (>700) {'.' * 10} {tag}")
    if enum_err and args.verbose:
        print(f"  error: {enum_err}")
    print()

    # Phase 2: model status checks
    results = []
    t_total = time.perf_counter()
    for i, (spec, expected) in enumerate(SMOKE_MODELS, 1):
        name = spec["name"]
        t0 = time.perf_counter()
        result = run_single_model(args.python, spec, args.device)
        elapsed = time.perf_counter() - t0
        actual = result["status"]
        passed = actual == expected

        tag = "PASS" if passed else "FAIL"
        dots = "." * max(1, 40 - len(name))
        print(f"[{i}/{n_models}] {name} {dots} {tag} ({actual}, {elapsed:.1f}s)")
        if not passed:
            print(f"  expected: {expected}")
            print(f"  actual:   {actual}")
            if "error" in result and args.verbose:
                print(f"  error:    {result['error'][:200]}")
        results.append(passed)

    total_elapsed = time.perf_counter() - t_total

    # Phase 3: summary
    print()
    print("=" * 60)
    n_passed = sum(results) + (1 if enum_ok else 0)
    n_total = len(results) + 1
    n_failed = n_total - n_passed
    print(f"Results: {n_passed}/{n_total} passed, {n_failed} failed ({total_elapsed:.0f}s)")
    print("=" * 60)

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
