#!/usr/bin/env python3
"""Resolve kernels-community kernel paths for HF transformers models that need them.

Future-proof for any torch version: scans the local HF Hub cache for the closest-
matching build variant and emits the LOCAL_KERNELS env value worker.py should set.

Usage (from worker.py / sweep launcher):

    from sweep.kernel_resolver import resolve_kernels_for_model
    local_kernels_value = resolve_kernels_for_model("MraModel", torch_version="2.12.0.dev20260407+cu128")
    if local_kernels_value:
        env["LOCAL_KERNELS"] = local_kernels_value

CLI for diagnostic use:
    python -m sweep.kernel_resolver MraModel               # uses installed torch
    python -m sweep.kernel_resolver MraModel 2.12.0+cu128
    python -m sweep.kernel_resolver --list                 # list known mappings
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# ────────────────────────────────────────────────────────────────────────
# Model → kernels-community repo mapping. Add new entries as new models
# adopt the kernels package. Auto-discovered today by grep of transformers
# source for `get_kernel("kernels-community/...")`.
# ────────────────────────────────────────────────────────────────────────
MODEL_KERNEL_MAP: dict[str, str] = {
    # MRA: 7 transformers models found, but only MRA verified end-to-end (2026-05-01).
    # Others (RWKV, YOSO, Glm Moe DSA, GptOss, OpenAIPrivacyFilter, Sam3Video) need
    # similar verification; add when tested.
    "MraModel": "kernels-community/mra",
    "MraForMaskedLM": "kernels-community/mra",
    "MraForMultipleChoice": "kernels-community/mra",
    "MraForQuestionAnswering": "kernels-community/mra",
    "MraForSequenceClassification": "kernels-community/mra",
    "MraForTokenClassification": "kernels-community/mra",
}

HF_CACHE_DIR = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


# ────────────────────────────────────────────────────────────────────────
# Variant matching
# ────────────────────────────────────────────────────────────────────────

def _parse_torch_version(version: str) -> tuple[int, int, str | None]:
    """Parse torch version string into (major, minor, cuda_tag).

    Examples:
        "2.12.0+cu128"          → (2, 12, "cu128")
        "2.12.0.dev20260407+cu128" → (2, 12, "cu128")
        "2.13.0a0+gitf8d66d2"   → (2, 13, None)
    """
    m = re.match(r"(\d+)\.(\d+)", version)
    major, minor = (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    cuda_match = re.search(r"\+(cu\d+)", version)
    cuda = cuda_match.group(1) if cuda_match else None
    return major, minor, cuda


def _detect_arch_os() -> str:
    """Return e.g. 'x86_64-linux'."""
    import platform
    return f"{platform.machine()}-{platform.system().lower()}"


def _candidate_variant_dirs(repo_id: str) -> list[Path]:
    """Find all build/<variant>/ dirs in the local HF cache for this repo."""
    cache_subdir = "models--" + repo_id.replace("/", "--")
    cache_root = HF_CACHE_DIR / cache_subdir / "snapshots"
    if not cache_root.exists():
        return []
    variants: list[Path] = []
    for snap in cache_root.iterdir():
        build_dir = snap / "build"
        if build_dir.is_dir():
            variants.extend(p for p in build_dir.iterdir() if p.is_dir())
    return variants


def _score_variant(variant_name: str, target_major: int, target_minor: int, target_cuda: str | None, arch: str) -> tuple[int, ...]:
    """Higher score = better match. Used to pick the best available variant when
    no exact torch version match exists.

    Score order (most → least important):
      1. Architecture/OS match (must)
      2. CUDA tag match
      3. Torch major-minor match (exact > one-minor-older > one-minor-newer)
      4. Newer torch preferred over older when no exact match
    """
    if not variant_name.endswith(arch):
        return (0,)

    # variant pattern: torch<MMnn>-cxx11-cu<NNN>-<arch>  e.g. torch211-cxx11-cu128-x86_64-linux
    m = re.match(r"torch(\d+)-cxx\d+-(cu\d+)-", variant_name)
    if not m:
        return (1,)
    torch_tag, cuda_tag = m.group(1), m.group(2)

    # torch_tag like "211" → major=2, minor=11; "27" → major=2, minor=7
    if len(torch_tag) >= 3:
        v_major, v_minor = int(torch_tag[0]), int(torch_tag[1:])
    elif len(torch_tag) == 2:
        v_major, v_minor = int(torch_tag[0]), int(torch_tag[1])
    else:
        return (1,)

    cuda_match = 1 if cuda_tag == target_cuda else 0
    # Closeness on minor: 100 for exact, 90 for one-older, 80 for one-newer, 50 for two-off, ...
    minor_diff = v_minor - target_minor
    if v_major != target_major:
        torch_score = 0
    elif minor_diff == 0:
        torch_score = 100
    elif minor_diff < 0:
        torch_score = max(0, 90 + minor_diff * 5)  # older: 85, 80, 75 ...
    else:
        torch_score = max(0, 80 - minor_diff * 5)  # newer: 75, 70, 65 ...

    return (10, cuda_match, torch_score, v_minor)


def resolve_kernel_path(repo_id: str, torch_version: str) -> Path | None:
    """Return the path to the best-matching variant's package dir, or None.

    The returned path is what LOCAL_KERNELS expects on the right side of `=`.
    Returns None if no variant matches (e.g., new torch version not yet built).
    """
    variants = _candidate_variant_dirs(repo_id)
    if not variants:
        return None
    target_major, target_minor, target_cuda = _parse_torch_version(torch_version)
    arch = _detect_arch_os()

    scored = [
        (_score_variant(v.name, target_major, target_minor, target_cuda, arch), v)
        for v in variants
    ]
    scored = [(s, v) for s, v in scored if s and s[0] > 0]
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[0][1]
    # Verify the package dir exists inside the variant
    pkg_name = repo_id.split("/")[-1]  # "mra" for "kernels-community/mra"
    if (best / pkg_name / "__init__.py").exists():
        return best
    return None


def resolve_kernels_for_model(model_class_name: str, torch_version: str | None = None) -> str | None:
    """Top-level entry: given a model class name, return the LOCAL_KERNELS env value
    (suitable for `LOCAL_KERNELS=<value>`), or None if no kernel needed/available.
    """
    if model_class_name not in MODEL_KERNEL_MAP:
        return None
    repo_id = MODEL_KERNEL_MAP[model_class_name]
    if torch_version is None:
        try:
            import torch
            torch_version = torch.__version__
        except Exception:
            return None
    path = resolve_kernel_path(repo_id, torch_version)
    if path is None:
        return None
    return f"{repo_id}={path}"


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", nargs="?", help="model class name (e.g. MraModel)")
    p.add_argument("torch_version", nargs="?", help="torch version string; default = installed torch")
    p.add_argument("--list", action="store_true", help="list all known model→kernel mappings")
    args = p.parse_args()

    if args.list:
        print("Known model → kernels-community repo mappings:")
        for m, r in sorted(MODEL_KERNEL_MAP.items()):
            print(f"  {m:40s} → {r}")
        print(f"\nHF cache dir: {HF_CACHE_DIR}")
        return 0

    if not args.model:
        p.print_help()
        return 1

    value = resolve_kernels_for_model(args.model, args.torch_version)
    if value:
        print(f"LOCAL_KERNELS={value}")
        return 0
    print(f"No kernel resolved for {args.model} (torch={args.torch_version or '<installed>'}). "
          f"Reasons: model not in mapping, OR no matching variant in local HF cache, "
          f"OR cache not populated (run sweep/setup_hf_kernels_cache.sh first).",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
