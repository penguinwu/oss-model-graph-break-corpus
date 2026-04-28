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

    # Fix 2: torchaudio/torchvision compiled against cu128, our build is cu124.
    pip uninstall torchaudio -y 2>/dev/null || true
    pip uninstall torchvision -y 2>/dev/null || true
    pip install torchvision --no-deps --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2
    log "  OK: torchvision reinstalled (CPU-only)"

    echo "fixup" > "$CHECKPOINT_FILE"
    log "Post-build fixes applied"
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
        "")       stage_clone; stage_deps; stage_canary; stage_build; stage_fixup; stage_verify ;;
        "clone")  stage_deps; stage_canary; stage_build; stage_fixup; stage_verify ;;
        "deps")   stage_canary; stage_build; stage_fixup; stage_verify ;;
        "canary") stage_build; stage_fixup; stage_verify ;;
        "build")  stage_fixup; stage_verify ;;
        "fixup")  stage_verify ;;
        *) log "Unknown checkpoint '$checkpoint' — starting fresh"
           rm -f "$CHECKPOINT_FILE"
           stage_clone; stage_deps; stage_canary; stage_build; stage_fixup; stage_verify ;;
    esac
}

main
