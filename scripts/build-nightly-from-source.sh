#!/bin/bash
# Build PyTorch from source on devvm2166 — proven recipe (2026-04-18).
#
# Structure: preflight → clone → deps → canary → build → fixup → verify
# Each stage checkpoints — resume after failure without redoing prior stages.
#
# Usage:
#   bash scripts/build-nightly-from-source.sh [~/envs/torch-nightly]
#
# Requires: ~15GB disk, ~30-45 min first build, sudo for GitHub clone.
#
# devvm2166-specific facts baked into this script:
#   - CUDA 12.4 is the only complete toolkit (12.8 has no headers)
#   - USE_NCCL=0 (no nccl.h, BPF blocks cmake fetch)
#   - platform010 Python needs libgomp.so.1 copied into torch/lib/
#   - torchaudio must be removed (cu128 mismatch with cu124 build)
#   - torchvision reinstalled CPU-only (no CUDA version conflict)
#   - GitHub clone via sudo when running under agent identity
#   - When running from cron (as pengwu), no sudo needed

set -euo pipefail

VENV="${1:-$HOME/envs/torch-nightly}"
PYTORCH_DIR="/tmp/pytorch-source"
CUDA_HOME="/usr/local/cuda-12.4"
CHECKPOINT_FILE="/tmp/pytorch-build-checkpoint"

log() { echo "[$(date +%H:%M:%S)] $*"; }
fail() { log "FATAL: $*"; exit 1; }

is_agent() {
    [[ "${CLAUDECODE:-}" == "1" ]] || [[ "${AGENT:-}" == "claude-code" ]]
}

run_git() {
    if is_agent; then
        sudo bash -c "HTTPS_PROXY=http://fwdproxy:8080 $*"
    else
        HTTPS_PROXY=http://fwdproxy:8080 bash -c "$*"
    fi
}

# ─── Stage 0: Preflight ───────────────────────────────────────────────
preflight() {
    log "=== PREFLIGHT ==="
    local errors=0

    command -v cmake >/dev/null 2>&1 || { log "FAIL: cmake not found"; ((errors++)); }
    command -v ninja >/dev/null 2>&1 || { log "FAIL: ninja not found"; ((errors++)); }
    command -v gcc   >/dev/null 2>&1 || { log "FAIL: gcc not found"; ((errors++)); }

    if [ ! -f "$CUDA_HOME/targets/x86_64-linux/include/cuda.h" ]; then
        log "FAIL: cuda.h not found at $CUDA_HOME (incomplete toolkit)"
        ((errors++))
    else
        log "  OK: CUDA $CUDA_HOME (headers present)"
    fi

    if [ ! -f "$CUDA_HOME/targets/x86_64-linux/lib/libcudart.so" ]; then
        log "FAIL: libcudart.so not found at $CUDA_HOME"
        ((errors++))
    else
        log "  OK: CUDA runtime present"
    fi

    if [ ! -f "$VENV/bin/python" ]; then
        log "FAIL: venv not found at $VENV"
        ((errors++))
    else
        log "  OK: venv $VENV"
    fi

    local avail_gb
    avail_gb=$(df --output=avail /tmp 2>/dev/null | tail -1 | awk '{printf "%.0f", $1/1024/1024}')
    if [ "$avail_gb" -lt 15 ]; then
        log "FAIL: only ${avail_gb}GB free on /tmp (need 15GB)"
        ((errors++))
    else
        log "  OK: ${avail_gb}GB free on /tmp"
    fi

    if is_agent; then
        if ! sudo -n true 2>/dev/null; then
            log "FAIL: sudo not available (needed for GitHub clone under agent identity)"
            ((errors++))
        else
            log "  OK: sudo available (agent identity detected)"
        fi
    else
        log "  OK: running as $(whoami) (no sudo needed)"
    fi

    if [ "$errors" -gt 0 ]; then
        fail "$errors preflight check(s) failed"
    fi
    log "=== PREFLIGHT PASSED ==="
}

# ─── Stage 1: Clone or fetch ──────────────────────────────────────────
stage_clone() {
    log "=== STAGE 1: Clone/Fetch ==="
    if [ -d "$PYTORCH_DIR/.git" ]; then
        log "Updating existing clone..."
        run_git "cd $PYTORCH_DIR && git fetch origin main"
        cd "$PYTORCH_DIR"
        git checkout FETCH_HEAD
    else
        log "Cloning pytorch (shallow)..."
        [ -d "$PYTORCH_DIR" ] && rm -rf "$PYTORCH_DIR"
        run_git "git clone --depth 1 https://github.com/pytorch/pytorch.git $PYTORCH_DIR"
        if is_agent; then
            sudo chown -R "$(whoami)" "$PYTORCH_DIR"
        fi
        cd "$PYTORCH_DIR"
    fi
    # `git submodule update` spawns one git child per submodule; the simple
    # HTTPS_PROXY env may not propagate to all of them. Force the proxy via
    # git config inside the sudo shell so every child inherits it.
    if is_agent; then
        sudo bash -c "cd $PYTORCH_DIR && git -c http.proxy=http://fwdproxy:8080 -c https.proxy=http://fwdproxy:8080 submodule update --init --recursive"
        # Submodule files just got created by root — chown back so subsequent
        # build steps (run as agent) can write into them.
        sudo chown -R "$(whoami)" "$PYTORCH_DIR"
    else
        cd "$PYTORCH_DIR"
        git -c http.proxy=http://fwdproxy:8080 -c https.proxy=http://fwdproxy:8080 submodule update --init --recursive
    fi
    echo "clone" > "$CHECKPOINT_FILE"
    log "Clone complete — $(git log -1 --format='%h %s')"
}

# ─── Stage 2: Install deps ────────────────────────────────────────────
stage_deps() {
    log "=== STAGE 2: Dependencies ==="
    cd "$PYTORCH_DIR"
    source "$VENV/bin/activate"
    pip install -r requirements.txt 2>&1 | tail -5
    echo "deps" > "$CHECKPOINT_FILE"
    log "Dependencies installed"
}

# ─── Stage 3: Canary build ────────────────────────────────────────────
stage_canary() {
    log "=== STAGE 3: Canary (cmake configure) ==="
    cd "$PYTORCH_DIR"
    source "$VENV/bin/activate"

    export CUDA_HOME="$CUDA_HOME"
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
    export CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
    export PATH="$CUDA_HOME/bin:$PATH"
    export USE_CUDA=1 USE_CUDNN=1 USE_NCCL=0
    export BUILD_TYPE=Release MAX_JOBS=$(nproc)
    export CMAKE_PREFIX_PATH="${VENV}"

    python setup.py build --cmake-only 2>&1 | tail -10
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        fail "cmake configure failed — check CUDA paths and build env"
    fi

    echo "canary" > "$CHECKPOINT_FILE"
    log "Canary passed"
}

# ─── Stage 4: Full build ──────────────────────────────────────────────
stage_build() {
    log "=== STAGE 4: Full Build (~30 min) ==="
    cd "$PYTORCH_DIR"
    source "$VENV/bin/activate"

    export CUDA_HOME="$CUDA_HOME"
    export CUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME"
    export CMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc"
    export PATH="$CUDA_HOME/bin:$PATH"
    export USE_CUDA=1 USE_CUDNN=1 USE_NCCL=0
    export BUILD_TYPE=Release MAX_JOBS=$(nproc)
    export CMAKE_PREFIX_PATH="${VENV}"

    python setup.py develop 2>&1 | tail -10
    echo "build" > "$CHECKPOINT_FILE"
    log "Build complete"
}

# ─── Stage 5: Post-build fixes ────────────────────────────────────────
stage_fixup() {
    log "=== STAGE 5: Post-Build Fixes ==="
    source "$VENV/bin/activate"

    # Fix 1: platform010 Python can't find system libgomp via its restricted linker.
    # Copy from a working pip-installed torch wheel.
    local libgomp_src=""
    for env in torch211 torch210 torch29 torch28; do
        local candidate="$HOME/envs/$env/lib/python3.12/site-packages/torch/lib/libgomp.so.1"
        if [ -f "$candidate" ]; then
            libgomp_src="$candidate"
            break
        fi
    done
    if [ -n "$libgomp_src" ]; then
        cp "$libgomp_src" "$PYTORCH_DIR/torch/lib/libgomp.so.1"
        log "  OK: libgomp.so.1 copied from $libgomp_src"
    else
        log "  WARN: no libgomp source found — torch import may fail"
    fi

    # Fix 2: torchvision/torchaudio pip wheels are ABI-locked to a specific
    # pre-built torch. Our source-built torch ships a different ABI (Meta's
    # platform010 toolchain), and torch's HEAD moves daily — pip wheels can't
    # register their custom ops (`torchvision::nms` etc.), and Python imports
    # cascade-fail anywhere transformers eagerly resolves attributes.
    #
    # Uninstall the pip wheels here — they get rebuilt from source in
    # stage_build_torchvision / stage_build_torchaudio below.
    pip uninstall torchaudio -y 2>/dev/null || true
    pip uninstall torchvision -y 2>/dev/null || true
    log "  OK: torchaudio/torchvision pip wheels uninstalled (will rebuild from source)"

    # Fix 3 (2026-04-28): pip-wheel torch bundles nvidia-cuda-* runtime libs
    # (nvrtc, cudart, cupti) under site-packages/nvidia/. Source-built torch
    # doesn't pull these — at first JIT compile of a CUDA reduce kernel,
    # nvrtc tries to load `libnvrtc-builtins.so.<TORCH_CUDA_VER>` from the
    # search path and fails. The Animesh sweep on 2026-04-28 hit this on 52 of
    # 78 eager_errors before the truncated error message obscured the cause.
    # Match the pip-wheel layout by installing the same nvidia-cuda-* metas.
    local TORCH_CUDA_VER
    TORCH_CUDA_VER=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")
    if [ -n "$TORCH_CUDA_VER" ]; then
        log "  Installing nvidia-cuda-* runtime libs (CUDA ${TORCH_CUDA_VER}) to match pip-wheel layout..."
        # Use --no-deps to keep this surgical: just the runtime libs, no
        # transitive surprises. The version constraint pins to torch's CUDA
        # major.minor (e.g. 12.4.* picks up the latest 12.4.x patch).
        pip install --no-deps \
            "nvidia-cuda-nvrtc-cu12==${TORCH_CUDA_VER}.*" \
            "nvidia-cuda-runtime-cu12==${TORCH_CUDA_VER}.*" \
            "nvidia-cuda-cupti-cu12==${TORCH_CUDA_VER}.*" \
            2>&1 | tail -3 || log "  WARN: nvidia-cuda-* install failed — JIT kernels (e.g. reductions) may fail at runtime"
        log "  OK: nvidia-cuda-* runtime libs installed for CUDA ${TORCH_CUDA_VER}"

        # Fix 4 (2026-04-28): force-preload nvidia-cuda-* via sitecustomize.py.
        # Source-built torch links against system libcudart, so torch's own
        # `_preload_cuda_deps()` (in torch/__init__.py) sees libcudart already
        # loaded in /proc/self/maps and bails out — never preloads the bundled
        # nvrtc-builtins. Without explicit preloading, the lib sits in
        # site-packages/nvidia/cuda_nvrtc/lib/ but nvrtc's runtime loader
        # can't find it. sitecustomize.py runs at Python startup and force-loads
        # all the bundled nvidia-cuda-* libs via ctypes.CDLL with RTLD_GLOBAL.
        local site_pkgs
        site_pkgs=$(python -c "import site; print(site.getsitepackages()[0])")
        cat > "${site_pkgs}/sitecustomize.py" <<'SITECUSTOMIZE_EOF'
"""sitecustomize.py — preload nvidia-cuda-* runtime libs for source-built torch.

Source-built torch links against the *system* libcudart. PyTorch's normal
`_preload_cuda_deps()` checks `/proc/self/maps` for libcudart and bails early
when it's already loaded. That works for pip-wheel torch (ships its own
libcudart) but not for source-built torch (system libcudart is loaded but the
matching nvrtc-builtins / cupti versions live in site-packages/nvidia/, never
preloaded). This file force-loads them at Python startup, before torch is
imported. Catches the failure mode that bit the Animesh sweep on 2026-04-28
(52 of 78 eager_errors were nvrtc JIT failing to find libnvrtc-builtins.so.X.Y).

Generated by scripts/build-nightly-from-source.sh stage_fixup.
"""
import ctypes
import glob
import os


def _preload(libname: str, dirname: str) -> None:
    try:
        import nvidia
        nvidia_dir = os.path.dirname(nvidia.__file__)
        pattern = os.path.join(nvidia_dir, dirname, "lib", libname)
        for path in sorted(glob.glob(pattern)):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue
    except (ImportError, OSError):
        pass


_preload("libcudart.so.*[0-9]", "cuda_runtime")
_preload("libnvrtc.so.*[0-9]", "cuda_nvrtc")
_preload("libnvrtc-builtins.so.*[0-9]", "cuda_nvrtc")
_preload("libcupti.so.*[0-9]", "cuda_cupti")
SITECUSTOMIZE_EOF
        log "  OK: sitecustomize.py written to ${site_pkgs}/ (force-preloads nvidia-cuda-*)"
    else
        log "  WARN: could not detect torch CUDA version — skipping nvidia-cuda-* install + sitecustomize"
    fi

    echo "fixup" > "$CHECKPOINT_FILE"
    log "Post-build fixes applied"
}

# ─── Stage 5b: Build torchvision from source ───────────────────────────
#
# We must build from source (not pip wheel) because:
#   1. pip torchvision wheels are ABI-locked to one specific pytorch.org
#      pre-built torch; our source torch has a different ABI (Meta toolchain)
#      and a daily-moving SHA. Wheels silently fail to register ops.
#   2. Building from source binds the C++ extension to *this* torch's
#      symbols, so torchvision::nms et al. resolve cleanly.
#
# Critical env: CC/CXX must point at /usr/local/bin/clang.par (Meta's clang
# wrapper that understands `--platform platform010`). The default sysconfig CC
# is `clang.par --platform platform010` as a single string, but ccache strips
# the prefix and falls through to /usr/bin/c++ (system gcc) which rejects the
# `--platform` flag — so we set CC/CXX explicitly to bypass ccache.
build_torch_extension_from_source() {
    local repo="$1"; local url="$2"; local label="$3"
    local src_dir="/tmp/${label}-source"
    log "Cloning ${label}..."
    if [ -d "$src_dir/.git" ]; then
        log "  using existing clone at $src_dir"
    else
        run_git "git clone --depth 1 $url $src_dir"
        if is_agent; then sudo chown -R "$(whoami)" "$src_dir"; fi
    fi
    log "Building ${label} from source (CC=clang.par, no ccache)..."
    (
        source "$VENV/bin/activate"
        cd "$src_dir"
        export CC=/usr/local/bin/clang.par
        export CXX=/usr/local/bin/clang++.par
        export CUDA_HOME="$CUDA_HOME"
        export FORCE_CUDA=1 USE_CUDA=1
        export TORCHVISION_USE_VIDEO_CODEC=0 TORCHVISION_USE_FFMPEG=0
        export USE_FFMPEG=0 USE_SOX=0
        export NO_CCACHE=1
        "$VENV/bin/pip" install --no-build-isolation --no-deps -e . 2>&1 | tail -15
    )
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        fail "${label} source build failed — see output above"
    fi
}

stage_build_torchvision() {
    log "=== STAGE 5b: Build torchvision from source ==="
    build_torch_extension_from_source "vision" "https://github.com/pytorch/vision.git" "torchvision"
    echo "torchvision" > "$CHECKPOINT_FILE"
    log "torchvision build complete"
}

stage_build_torchaudio() {
    log "=== STAGE 5c: Build torchaudio from source ==="
    build_torch_extension_from_source "audio" "https://github.com/pytorch/audio.git" "torchaudio"
    echo "torchaudio" > "$CHECKPOINT_FILE"
    log "torchaudio build complete"
}

# ─── Stage 6: Verify ──────────────────────────────────────────────────
stage_verify() {
    log "=== STAGE 6: Verify ==="
    source "$VENV/bin/activate"

    local output
    output=$(python -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} git={torch.version.git_version[:12]}')" 2>&1)
    if [ $? -ne 0 ]; then
        fail "torch import failed: $output"
    fi
    log "  $output"

    local compile_ok
    compile_ok=$(python -c "
import torch
model = torch.nn.Linear(10, 5).cuda()
out = torch.compile(model, fullgraph=True)(torch.randn(2, 10).cuda())
print(f'torch.compile OK: {out.shape}')
" 2>&1)
    if [ $? -ne 0 ]; then
        fail "torch.compile test failed: $compile_ok"
    fi
    log "  $compile_ok"

    # CUDA JIT-codegen smoke (added 2026-04-28 after Animesh sweep nvrtc bug).
    # Pure eager forward of a CUDA reduce kernel exercises the nvrtc JIT path
    # that needs libnvrtc-builtins.so.<TORCH_CUDA_VER>. If nvidia-cuda-* runtime
    # libs are missing (or the system path doesn't have the right CUDA major.minor),
    # this fails fast — before a multi-hour sweep launches against a half-broken
    # venv. Failure mode this catches:
    #   nvrtc: error: failed to open libnvrtc-builtins.so.12.4
    local nvrtc_ok
    nvrtc_ok=$(python -c "
import torch
x = torch.randn(64, 64, device='cuda')
_ = x.prod()      # triggers reduction_prod_kernel JIT compile via nvrtc
_ = torch.fft.fft(x)  # exercises another JIT path
torch.cuda.synchronize()
print('CUDA JIT (nvrtc reduce + fft) OK')
" 2>&1)
    if [ $? -ne 0 ]; then
        fail "CUDA JIT smoke failed (likely missing nvidia-cuda-nvrtc-cu12 — see stage_fixup): $nvrtc_ok"
    fi
    log "  $nvrtc_ok"

    # Verify the venv can actually enumerate the corpus end-to-end. This is the
    # real downstream check — it caught the torchvision-poisons-enumeration trap
    # of 2026-04-28 (where torch alone was fine but the corpus walk crashed on
    # FuyuProcessor's lazy torchvision import). If a future env regression
    # silently breaks enumeration, this fails the build instead of failing the
    # sweep 3 hours later.
    local repo_root
    repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    local enum_ok
    enum_ok=$(python -c "
import sys
sys.path.insert(0, '$repo_root/sweep')
from models import enumerate_hf, enumerate_diffusers, enumerate_custom
hf = enumerate_hf()
df = enumerate_diffusers()
cu = enumerate_custom()
total = len(hf) + len(df) + len(cu)
assert total > 700, f'corpus enumeration too small ({total}); env may be broken'
print(f'corpus enumerable: {total} models (hf:{len(hf)} diffusers:{len(df)} custom:{len(cu)})')
" 2>&1 | tail -3)
    if [ $? -ne 0 ]; then
        fail "corpus enumeration failed: $enum_ok"
    fi
    log "  $enum_ok"

    rm -f "$CHECKPOINT_FILE"
    log "=== BUILD FROM SOURCE COMPLETE ==="
}

# ─── Main ─────────────────────────────────────────────────────────────
main() {
    log "=== Build PyTorch from source (devvm2166 recipe) ==="
    log "Venv: $VENV | CUDA: $CUDA_HOME"

    local checkpoint=""
    if [ -f "$CHECKPOINT_FILE" ]; then
        checkpoint=$(cat "$CHECKPOINT_FILE")
        log "Resuming from checkpoint: $checkpoint"
    fi

    preflight

    case "$checkpoint" in
        "")             stage_clone; stage_deps; stage_canary; stage_build; stage_fixup; stage_build_torchvision; stage_build_torchaudio; stage_verify ;;
        "clone")        stage_deps; stage_canary; stage_build; stage_fixup; stage_build_torchvision; stage_build_torchaudio; stage_verify ;;
        "deps")         stage_canary; stage_build; stage_fixup; stage_build_torchvision; stage_build_torchaudio; stage_verify ;;
        "canary")       stage_build; stage_fixup; stage_build_torchvision; stage_build_torchaudio; stage_verify ;;
        "build")        stage_fixup; stage_build_torchvision; stage_build_torchaudio; stage_verify ;;
        "fixup")        stage_build_torchvision; stage_build_torchaudio; stage_verify ;;
        "torchvision")  stage_build_torchaudio; stage_verify ;;
        "torchaudio")   stage_verify ;;
        *) log "Unknown checkpoint '$checkpoint' — starting fresh"
           rm -f "$CHECKPOINT_FILE"
           stage_clone; stage_deps; stage_canary; stage_build; stage_fixup; stage_build_torchvision; stage_build_torchaudio; stage_verify ;;
    esac
}

main
