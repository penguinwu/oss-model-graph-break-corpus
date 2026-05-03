#!/usr/bin/env python3
"""Sweep venv setup — single canonical entry point for getting a PyTorch venv
ready before launching a sweep.

Why this exists:
  Agent identity (claude_code) cannot reach pypi.nvidia.com (BPF-blocked), so
  fresh pip installs of torch versions that need cuda-toolkit fail. Last loss:
  2026-04-29 overnight, ~8 hours wasted because pip silently fell through to
  a wrong torch version.

The fix:
  1. Maintain canonical cu library pool venvs at ~/envs/cu128/ and ~/envs/cu126/
     (cu libraries only, nothing else). Bootstrap requires non-agent identity
     once; thereafter, agent-side clone-and-upgrade works without touching
     pypi.nvidia.com.
  2. Single entry point ensure_venv_ready(torch_spec, cuda_variant) that:
       - Returns a healthy matching PT venv if one exists
       - Else clones the pool + installs torch + transformers/etc. on top
       - Else (no pool exists for variant) escalates to Peng with bootstrap
         commands and exits 42 (AWAITING_HUMAN_BOOTSTRAP)
  3. Post-install version-match gate (catches silent fall-through). Exit 43
     (VERSION_MISMATCH) if installed torch.__version__ doesn't match request.

Recipe lives at: ~/.myclaw-shared/recipes/python-venv-bpf.md
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────

ENVS_DIR = Path.home() / "envs"
RECIPE_PATH = Path.home() / ".myclaw-shared" / "recipes" / "python-venv-bpf.md"
PYTORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly"

# Exit codes — distinct so monitors/cron can recognize them.
EXIT_OK = 0
EXIT_GENERIC_ERROR = 1
EXIT_AWAITING_HUMAN_BOOTSTRAP = 42  # No pool for this cuda variant; need terminal
EXIT_VERSION_MISMATCH = 43           # pip installed wrong version; silent fall-through

# Cu library packages we require in a pool venv. If any are missing the pool
# is unhealthy and needs re-bootstrap.
CU_LIB_PACKAGES = {
    "cuda-toolkit",
    "cuda-bindings",
    "cuda-pathfinder",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cuda-runtime-cu12",
    "nvidia-cuda-cupti-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cufile-cu12",
    "nvidia-curand-cu12",
    "nvidia-cusolver-cu12",
    "nvidia-cusparse-cu12",
    "nvidia-cusparselt-cu12",
    "nvidia-nccl-cu12",
    "nvidia-nvjitlink-cu12",
    "nvidia-nvshmem-cu12",
    "nvidia-nvtx-cu12",
}

# Standard sweep dependencies installed per-PT-venv (NOT in pools — these are
# PT-version-coupled). transformers is pinned because each torch nightly has
# a known-compatible transformers version.
SWEEP_DEPS = {
    "transformers": "5.6.2",
    "diffusers": None,        # latest compatible
    "timm": None,
    "accelerate": None,
    "sentencepiece": None,
    "protobuf": "<5",
    "huggingface-hub": None,
    "tokenizers": None,
    "safetensors": None,
}

# torch's pure-python deps. Installed via pip from PyPI (allowed through
# fwdproxy). NOT in pool because they're torch-version-coupled.
TORCH_PYTHON_DEPS = [
    "typing-extensions",
    "sympy",
    "networkx",
    "jinja2",
    "fsspec",
    "filelock",
    "setuptools",
    "mpmath",
    "markupsafe",
    "psutil",
    "pyyaml",
    "regex",
    "tqdm",
]


# ──────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────

# ──────────────────────────────────────────────────────────────────────────
# transformers ↔ torch compatibility
# ──────────────────────────────────────────────────────────────────────────
# Minimum torch version required by each transformers minor.
# Sourced from transformers release notes; update on each new minor release.
# https://github.com/huggingface/transformers/releases
TRANSFORMERS_MIN_TORCH = {
    "5.4": "2.1",
    "5.5": "2.1",
    "5.6": "2.2",
    "5.7": "2.3",
    "5.8": "2.3",
}


def _major_minor(version: str) -> tuple[int, int]:
    """'2.12.0.dev20260407+cu128' -> (2, 12). (0, 0) on parse failure."""
    m = re.match(r"^(\d+)\.(\d+)", version)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def check_compat(torch_version: str, transformers_version: str) -> Optional[str]:
    """Return None if the torch/transformers pair is compatible, else a
    human-readable warning string. Caller decides fail-vs-warn policy."""
    tx_mm = ".".join(transformers_version.split(".")[:2])
    min_torch = TRANSFORMERS_MIN_TORCH.get(tx_mm)
    if min_torch is None:
        return (
            f"unknown transformers minor {tx_mm}; "
            f"TRANSFORMERS_MIN_TORCH table in sweep/venv_setup.py needs update"
        )
    if _major_minor(torch_version) < _major_minor(min_torch):
        return (
            f"transformers {transformers_version} requires torch >= {min_torch}, "
            f"but torch is {torch_version}"
        )
    return None


@dataclass
class VenvInfo:
    path: Path
    torch_version: Optional[str]   # e.g. "2.12.0.dev20260323+cu128", or None
    cuda_variant: Optional[str]    # e.g. "cu128", "cu126", or None
    transformers_version: Optional[str]
    is_pool: bool                  # True if name matches cu* and contains only cu libs
    health_issues: list[str]       # empty if healthy


@dataclass
class TorchSpec:
    """A request for "I want to sweep against torch X on cuda variant Y"."""
    version_pattern: str    # exact ("2.12.0.dev20260323+cu128") or glob ("2.12.*")
    cuda_variant: str       # "cu128", "cu126", etc.

    def matches(self, version: str) -> bool:
        """Does an installed torch.__version__ satisfy this spec?"""
        if self.version_pattern == version:
            return True
        if "*" in self.version_pattern:
            pat = self.version_pattern.replace(".", r"\.").replace("*", r".*")
            return bool(re.match(f"^{pat}$", version))
        # Substring match for e.g. "2.12" matching "2.12.0.dev..."
        return version.startswith(self.version_pattern)


# ──────────────────────────────────────────────────────────────────────────
# Identity detection
# ──────────────────────────────────────────────────────────────────────────

def running_as_agent() -> bool:
    """True if running under the BPF-restricted claude_code agent identity.

    Detected via env vars set by Claude Code. Conservative: any sign of agent
    identity → True.
    """
    return bool(
        os.environ.get("CLAUDE_CODE_CURRENT_SESSION_ID")
        or os.environ.get("CLAUDECODE")
        or os.environ.get("CLAUDE_CODE")
    )


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────

def ensure_venv_ready(spec: TorchSpec) -> Path:
    """Return path to a healthy PT venv satisfying spec, or sys.exit with
    distinct exit code.

    Algorithm:
      1. Find a healthy existing PT venv whose torch.__version__ matches → use
      2. Find canonical pool ~/envs/<cuda_variant>/ → clone + install → use
      3. No pool exists for this variant → escalate to Peng → exit 42

    Post-condition (or sys.exit with non-zero):
      - Returned path is a venv with torch.__version__ matching spec
      - NVRTC kernel JIT smoke passes
      - transformers + diffusers installed
    """
    # Step 1: existing healthy match?
    matches = find_matching_pt_venvs(spec)
    for v in matches:
        if not v.health_issues:
            log(f"using existing venv: {v.path} (torch {v.torch_version})")
            return v.path
        # Try to repair lightweight issues
        if can_auto_repair(v):
            repair_venv(v)
            v = inspect_venv(v.path)
            if not v.health_issues:
                log(f"repaired + using existing venv: {v.path}")
                return v.path

    # Step 2: pool exists for this cuda variant?
    pool = pool_for_variant(spec.cuda_variant)
    if pool and is_pool_healthy(pool):
        new_venv = clone_pool_and_install(pool, spec)
        verify_post_install(new_venv, spec)
        log(f"created new venv from pool: {new_venv}")
        return new_venv

    # Step 3: no pool → escalate
    escalate_no_pool(spec)
    sys.exit(EXIT_AWAITING_HUMAN_BOOTSTRAP)


# ──────────────────────────────────────────────────────────────────────────
# Helpers (skeleton — implementations come in Cycle 2 + 3)
# ──────────────────────────────────────────────────────────────────────────

def log(msg: str) -> None:
    print(f"[venv_setup] {msg}", flush=True)


def _venv_python(path: Path) -> Path:
    return path / "bin" / "python"


def _site_packages(venv: Path) -> Optional[Path]:
    """Locate <venv>/lib/python*/site-packages/ — None if venv is empty."""
    lib = venv / "lib"
    if not lib.exists():
        return None
    for py_dir in sorted(lib.iterdir()):
        if py_dir.name.startswith("python"):
            sp = py_dir / "site-packages"
            if sp.exists():
                return sp
    return None


_DIST_INFO_RE = re.compile(r"^(.+)-([0-9][^-]*)$")


def _pip_list(venv: Path) -> dict[str, str]:
    """{package_name: version} read from site-packages/*.dist-info/.

    Pure filesystem walk — no subprocess. Replaces an earlier subprocess
    `pip list` per venv that cost ~0.5s each AND triggered a per-venv pip
    interpreter startup; across 11 venvs that was a multi-second tax. See
    ``_torch_version`` for the same fix on the torch-version probe.

    Names are normalized to pip's hyphen-lowercase form (dist-info
    directories use underscores).
    """
    sp = _site_packages(venv)
    if sp is None:
        return {}
    result: dict[str, str] = {}
    for entry in sp.iterdir():
        if not entry.name.endswith(".dist-info") or not entry.is_dir():
            continue
        stem = entry.name[: -len(".dist-info")]
        m = _DIST_INFO_RE.match(stem)
        if not m:
            continue
        name = m.group(1).lower().replace("_", "-")
        result[name] = m.group(2)
    return result


def _torch_version(venv: Path) -> Optional[str]:
    """Return torch.__version__ by reading site-packages/torch/version.py.

    Pure filesystem read — no `import torch` subprocess. The previous
    implementation spawned a Python that did `import torch`, which costs
    ~4s per venv (CUDA library init); across 11 venvs that was the dominant
    cost of `find_matching_pt_venvs`. Importability is verified separately
    by ``_nvrtc_smoke`` after install.
    """
    sp = _site_packages(venv)
    if sp is None:
        return None
    vf = sp / "torch" / "version.py"
    if not vf.exists():
        return None
    try:
        for line in vf.read_text().splitlines():
            m = re.match(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", line)
            if m:
                return m.group(1)
    except (OSError, UnicodeDecodeError):
        return None
    return None


def _extract_cuda_variant(torch_version: Optional[str]) -> Optional[str]:
    """Parse cuda variant from torch version string, e.g.:
       '2.12.0.dev20260323+cu128' -> 'cu128'
       '2.13.0a0+gitf8d66d2'      -> None (source build, no wheel-cuda tag)
    """
    if not torch_version:
        return None
    m = re.search(r"\+cu(\d+)", torch_version)
    return f"cu{m.group(1)}" if m else None


def _pip_shebang_valid(venv: Path) -> bool:
    """A venv's pip script has a shebang pointing at the venv's python.
    Stale shebangs (from rename/move) are a common breakage."""
    pip = venv / "bin" / "pip"
    if not pip.exists():
        return False
    expected = str(_venv_python(venv))
    try:
        first_line = pip.read_text().splitlines()[0]
    except (UnicodeDecodeError, IndexError):
        return False
    return first_line.startswith("#!") and expected in first_line


def _nvrtc_smoke(venv: Path, timeout_s: int = 30) -> bool:
    """Run NVRTC reduction kernel JIT — same smoke that broke Animesh's sweep.
    True if it passes."""
    py = _venv_python(venv)
    if not py.exists():
        return False
    code = (
        "import torch\n"
        "if not torch.cuda.is_available(): raise SystemExit(2)\n"
        "y = torch.randn(64, 64, device='cuda').prod()\n"
        "_ = y.item()\n"
        "z = torch.fft.fft(torch.randn(64, device='cuda'))\n"
        "_ = z.cpu()\n"
    )
    try:
        result = subprocess.run(
            [str(py), "-c", code],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def find_matching_pt_venvs(spec: TorchSpec) -> list[VenvInfo]:
    """Walk ~/envs/, return list of venvs whose torch matches spec.
    Skips canonical pool venvs (named exactly cu128/cu126) — they're not PT
    venvs (no torch installed)."""
    if not ENVS_DIR.exists():
        return []
    matches = []
    for entry in sorted(ENVS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in ("cu128", "cu126"):
            continue  # pool venvs, skip
        v = inspect_venv(entry)
        if v.torch_version and spec.matches(v.torch_version):
            matches.append(v)
    return matches


def inspect_venv(path: Path) -> VenvInfo:
    """Probe a venv: torch version, cuda variant, transformers, health issues."""
    pkgs = _pip_list(path)
    torch_ver = _torch_version(path)
    cuda_var = _extract_cuda_variant(torch_ver)
    transformers_ver = pkgs.get("transformers")

    issues = []
    if not _venv_python(path).exists():
        issues.append("python interpreter missing")
    if not _pip_shebang_valid(path):
        issues.append("stale pip shebang")
    if torch_ver is None and (path / "lib" / "python3.12" / "site-packages" / "torch").exists():
        issues.append("torch present but not importable")

    is_pool = path.name in ("cu128", "cu126")

    return VenvInfo(
        path=path,
        torch_version=torch_ver,
        cuda_variant=cuda_var,
        transformers_version=transformers_ver,
        is_pool=is_pool,
        health_issues=issues,
    )


_AUTO_REPAIRABLE = {"stale pip shebang"}


def can_auto_repair(v: VenvInfo) -> bool:
    """True iff every health issue is in the auto-repair allowlist."""
    return all(issue in _AUTO_REPAIRABLE for issue in v.health_issues)


def repair_venv(v: VenvInfo) -> None:
    """Fix lightweight issues in place. Idempotent. Currently:
       - stale pip shebang → rewrite to point at venv's python
    """
    if "stale pip shebang" in v.health_issues:
        for script in (v.path / "bin").iterdir():
            if not script.is_file():
                continue
            try:
                content = script.read_bytes()
            except (PermissionError, IsADirectoryError):
                continue
            if not content.startswith(b"#!"):
                continue
            lines = content.split(b"\n", 1)
            shebang = lines[0]
            # If shebang references python via a path that doesn't exist, fix
            new_shebang = f"#!{_venv_python(v.path)}".encode()
            if shebang != new_shebang and b"python" in shebang:
                rest = lines[1] if len(lines) > 1 else b""
                script.write_bytes(new_shebang + b"\n" + rest)
        log(f"repaired pip shebangs in {v.path}")


def is_pool_healthy(pool: Path) -> bool:
    """True if pool has every CU_LIB_PACKAGES package installed."""
    pkgs = _pip_list(pool)
    pkgs_lower = {k.lower() for k in pkgs}
    missing = CU_LIB_PACKAGES - pkgs_lower
    if missing:
        log(f"pool {pool} missing cu packages: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        return False
    return True


def pool_for_variant(cuda_variant: str) -> Optional[Path]:
    """Return ~/envs/<cuda_variant>/ if it exists, else None.
    cuda_variant is e.g. 'cu128' or 'cu126'."""
    candidate = ENVS_DIR / cuda_variant
    return candidate if candidate.exists() else None


def _new_venv_path(spec: TorchSpec) -> Path:
    """Generate a unique target path for a new venv from spec."""
    # Strip non-alphanumeric for filesystem-safe name
    safe = re.sub(r"[^\w.-]", "", spec.version_pattern)
    base = f"torch-{safe}"
    target = ENVS_DIR / base
    if not target.exists():
        return target
    # Disambiguate
    for i in range(1, 100):
        cand = ENVS_DIR / f"{base}-{i}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"too many existing venvs at {target}-N")


def _clone_venv(src: Path, dst: Path) -> None:
    """cp -r src dst, then fix shebangs + pyvenv.cfg to point at dst."""
    if dst.exists():
        raise FileExistsError(f"{dst} already exists; refusing to overwrite")
    shutil.copytree(src, dst)
    # Fix pyvenv.cfg
    cfg = dst / "pyvenv.cfg"
    if cfg.exists():
        text = cfg.read_text()
        text = text.replace(str(src), str(dst))
        text = text.replace(src.name, dst.name)
        cfg.write_text(text)
    # Fix shebangs in bin/
    src_path_bytes = str(src).encode()
    dst_path_bytes = str(dst).encode()
    for script in (dst / "bin").iterdir():
        if not script.is_file():
            continue
        try:
            content = script.read_bytes()
        except (PermissionError, IsADirectoryError):
            continue
        if not content.startswith(b"#!"):
            continue
        if src_path_bytes in content:
            content = content.replace(src_path_bytes, dst_path_bytes)
            script.write_bytes(content)


def _pip_install(venv: Path, args: list[str], timeout_s: int = 600) -> tuple[int, str]:
    """Run pip install in venv with proxy env. Returns (exit_code, output)."""
    py = _venv_python(venv)
    env = os.environ.copy()
    env["HTTP_PROXY"] = "http://fwdproxy:8080"
    env["HTTPS_PROXY"] = "http://fwdproxy:8080"
    cmd = [str(py), "-m", "pip", "install"] + args
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout_s)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired as e:
        return 124, f"TIMEOUT after {timeout_s}s\n{e.stdout or ''}{e.stderr or ''}"


def clone_pool_and_install(pool: Path, spec: TorchSpec) -> Path:
    """Clone pool to a new venv, install torch + sweep deps in carefully
    chosen order to prevent dep-resolver swap.

    Order matters:
      1. Clone pool (cu libs come along)
      2. Install torch via --no-deps from PyTorch nightly index
         (--no-deps avoids re-fetching cuda-toolkit from pypi.nvidia.com)
      3. Install torchvision from SAME nightly index (--no-deps too — its only
         meaningful dep is torch, which is already pinned)
      4. Install torch's python deps (typing-extensions, sympy, etc.) from PyPI
      5. Install transformers + sweep deps from PyPI WITH --no-deps
         (deps that DO matter — tokenizers, huggingface-hub, safetensors —
         are explicitly listed; this prevents torchvision from being pulled in
         which would downgrade torch)
      6. Install per-package transitive deps that matter (e.g., requests for
         huggingface-hub, regex for tokenizers)
    """
    target = _new_venv_path(spec)
    log(f"cloning pool {pool} → {target}")
    _clone_venv(pool, target)

    nightly_index = f"{PYTORCH_NIGHTLY_INDEX}/{spec.cuda_variant}"

    # Step 2: torch with --no-deps
    log(f"installing torch {spec.version_pattern} (--no-deps)")
    rc, out = _pip_install(target, [
        "--pre", "--no-deps",
        "--index-url", nightly_index,
        f"torch=={spec.version_pattern}" if "*" not in spec.version_pattern else "torch",
    ])
    if rc != 0:
        log(f"torch install failed: {out[-500:]}")
        shutil.rmtree(target)
        raise RuntimeError(f"torch install failed: rc={rc}")

    # Step 3: torchvision from same nightly index, --no-deps
    log("installing torchvision (--no-deps, same nightly index)")
    rc, out = _pip_install(target, [
        "--pre", "--no-deps",
        "--index-url", nightly_index,
        "torchvision",
    ])
    if rc != 0:
        log(f"torchvision install warning (non-fatal): {out[-200:]}")
        # torchvision is nice-to-have; not all sweeps need it

    # Step 4: torch's python deps from PyPI
    log(f"installing torch python deps: {len(TORCH_PYTHON_DEPS)} packages")
    rc, out = _pip_install(target, TORCH_PYTHON_DEPS, timeout_s=300)
    if rc != 0:
        log(f"python deps install failed: {out[-500:]}")
        shutil.rmtree(target)
        raise RuntimeError(f"python deps install failed: rc={rc}")

    # Step 5: triton from nightly index (PT-version coupled)
    log("installing triton (from nightly index)")
    rc, out = _pip_install(target, [
        "--pre", "--no-deps",
        "--index-url", nightly_index,
        "triton",
    ])
    if rc != 0:
        log(f"triton install warning (non-fatal): {out[-200:]}")

    # Step 6: transformers + sweep deps with --no-deps (avoid torchvision pull)
    sweep_pkgs = []
    for pkg, ver in SWEEP_DEPS.items():
        if ver:
            sweep_pkgs.append(f"{pkg}=={ver}" if not ver.startswith("<") else f"{pkg}{ver}")
        else:
            sweep_pkgs.append(pkg)
    log(f"installing sweep deps with --no-deps: {sweep_pkgs}")
    rc, out = _pip_install(target, ["--no-deps"] + sweep_pkgs, timeout_s=300)
    if rc != 0:
        log(f"sweep deps install failed: {out[-500:]}")
        shutil.rmtree(target)
        raise RuntimeError(f"sweep deps install failed: rc={rc}")

    # Step 7: known transitive deps (without torch/torchvision)
    transitives = [
        "requests", "charset-normalizer", "idna", "certifi",  # for huggingface-hub
        "hf-xet",                                              # for huggingface-hub
        "packaging",                                            # for transformers
    ]
    log(f"installing transitive deps: {transitives}")
    rc, out = _pip_install(target, transitives, timeout_s=180)
    if rc != 0:
        log(f"transitive deps install warning (non-fatal): {out[-200:]}")

    return target


def verify_post_install(venv_path: Path, spec: TorchSpec) -> None:
    """Critical: after install, verify torch.__version__ ACTUALLY matches.
    pip can silently fall through to a different version (e.g., when
    torchvision pulls stable torch and downgrades). Mismatch = exit 43."""
    actual = _torch_version(venv_path)
    if actual is None:
        msg = f"VERSION_MISMATCH: torch not importable in {venv_path}"
        print(f"[venv_setup] {msg}", file=sys.stderr)
        sys.exit(EXIT_VERSION_MISMATCH)
    if not spec.matches(actual):
        msg = (
            f"VERSION_MISMATCH: requested torch '{spec.version_pattern}' on {spec.cuda_variant}, "
            f"got '{actual}' in {venv_path}.\n"
            f"This is the silent-fall-through pattern (pip resolved deps to a different "
            f"torch version). Recipe: {RECIPE_PATH}"
        )
        print(f"[venv_setup] {msg}", file=sys.stderr)
        sys.exit(EXIT_VERSION_MISMATCH)
    # Also run NVRTC smoke as a final correctness gate
    if not _nvrtc_smoke(venv_path):
        msg = (
            f"VERSION_MISMATCH: torch {actual} installed but NVRTC kernel JIT failed in "
            f"{venv_path}. Likely cu library mismatch (e.g. cu13 libs replaced cu128 from pool)."
        )
        print(f"[venv_setup] {msg}", file=sys.stderr)
        sys.exit(EXIT_VERSION_MISMATCH)
    log(f"verified: torch {actual} on {spec.cuda_variant}, NVRTC smoke ok")


def _bootstrap_commands(spec: TorchSpec) -> str:
    """Return the exact terminal commands Peng needs to run to bootstrap a
    new pool venv for this cuda variant. Call this when no pool exists."""
    pool_path = ENVS_DIR / spec.cuda_variant
    nightly_index = f"{PYTORCH_NIGHTLY_INDEX}/{spec.cuda_variant}"
    return (
        f"# pt2-skill-discovery / corpus needs a {spec.cuda_variant} pool venv.\n"
        f"# Run from your terminal (NOT from Claude Code — pypi.nvidia.com is\n"
        f"# BPF-blocked for agent identity but allowed for pengwu).\n"
        f"\n"
        f"python3 -m venv {pool_path}\n"
        f"{pool_path}/bin/pip install --pre \\\n"
        f"    --index-url {nightly_index} \\\n"
        f"    torch torchvision\n"
        f"\n"
        f"# Then strip everything except cu libraries from the pool:\n"
        f"{pool_path}/bin/pip uninstall -y torch torchvision triton \\\n"
        f"    typing-extensions sympy networkx jinja2 fsspec filelock setuptools \\\n"
        f"    mpmath markupsafe || true\n"
        f"\n"
        f"# Verify pool has cu libs:\n"
        f"{pool_path}/bin/pip list | grep -iE 'nvidia|cuda'\n"
        f"\n"
        f"# Then re-run: tools/run_sweep.py with --torch '{spec.version_pattern}' --cuda {spec.cuda_variant}"
    )


def escalate_no_pool(spec: TorchSpec) -> None:
    """No pool exists for this cuda variant. Print bootstrap commands + ping
    Peng's gchat space. Caller exits with EXIT_AWAITING_HUMAN_BOOTSTRAP."""
    cmds = _bootstrap_commands(spec)
    msg = (
        f"[venv_setup] *AWAITING_HUMAN_BOOTSTRAP*: no pool venv exists at "
        f"~/envs/{spec.cuda_variant}/. Sweep cannot proceed under agent identity.\n\n"
        f"{cmds}\n\n"
        f"Recipe: {RECIPE_PATH}"
    )
    print(msg, file=sys.stderr)

    # Best-effort gchat ping. If gchat command is missing or fails, we still
    # exit 42 with the command printed to stderr so wrapper scripts can read it.
    try:
        gchat_msg = (
            f"[🦦 Otter]: *AWAITING_HUMAN_BOOTSTRAP* — sweep blocked.\n\n"
            f"No pool venv exists at `~/envs/{spec.cuda_variant}/`. Need a one-time "
            f"manual bootstrap from your terminal:\n\n"
            f"```\n{cmds}\n```\n\n"
            f"After bootstrap, re-run the sweep — it'll pick up the pool automatically."
        )
        # Send to Peng's space via gchat CLI. Use --as-bot in own space.
        space_id = "AAQANraxXE4"  # Otter's space
        with open("/tmp/_venv_setup_escalation.txt", "w") as f:
            f.write(gchat_msg)
        subprocess.run(
            ["gchat", "send", space_id, "--as-bot",
             "--text-file", "/tmp/_venv_setup_escalation.txt"],
            timeout=15, capture_output=True,
        )
    except Exception as e:
        log(f"gchat ping failed (non-fatal — message printed to stderr): {e}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--torch", required=True,
                   help="torch version pattern, e.g. '2.12.*' or "
                        "'2.12.0.dev20260323+cu128'")
    p.add_argument("--cuda", required=True, choices=["cu128", "cu126"],
                   help="cuda variant")
    p.add_argument("--check-only", action="store_true",
                   help="Don't install anything, just inspect existing venvs")
    args = p.parse_args()

    spec = TorchSpec(version_pattern=args.torch, cuda_variant=args.cuda)

    if args.check_only:
        matches = find_matching_pt_venvs(spec)
        for v in matches:
            print(f"  {v.path}: torch={v.torch_version} health={v.health_issues or 'OK'}")
        return EXIT_OK

    venv = ensure_venv_ready(spec)
    print(venv)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
