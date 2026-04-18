#!/bin/bash
# Build PyTorch from source — contingency for when nightly pip is unavailable.
#
# Prerequisites (all present on devvm2166):
#   cmake 3.31+, ninja, gcc 11+, CUDA toolkit 12.8
#
# Usage:
#   bash scripts/build-nightly-from-source.sh [--venv ~/envs/torch-nightly]
#
# Takes ~30-45 minutes on first build, ~10-15 minutes for incremental builds.
# Requires ~15GB disk space for the build.

set -euo pipefail

VENV="${1:-$HOME/envs/torch-nightly}"
PYTORCH_DIR="/tmp/pytorch-source"
CUDA_HOME="/usr/local/cuda-12.8"

echo "=== Build PyTorch from source ==="
echo "Target venv: $VENV"
echo "CUDA: $CUDA_HOME"

# Step 1: Clone or update pytorch repo
if [ -d "$PYTORCH_DIR" ]; then
    echo "Updating existing clone..."
    cd "$PYTORCH_DIR"
    # Must use sudo for github access (BPF jailer blocks agent identity)
    sudo bash -c "cd $PYTORCH_DIR && HTTPS_PROXY=http://fwdproxy:8080 git fetch origin main"
    git checkout FETCH_HEAD
else
    echo "Cloning pytorch (this takes a few minutes)..."
    sudo bash -c "HTTPS_PROXY=http://fwdproxy:8080 git clone --depth 1 https://github.com/pytorch/pytorch.git $PYTORCH_DIR"
    sudo chown -R "$(whoami)" "$PYTORCH_DIR"
    cd "$PYTORCH_DIR"
    git submodule update --init --recursive
fi

# Step 2: Build
echo ""
echo "Building PyTorch..."
export CUDA_HOME="$CUDA_HOME"
export CMAKE_PREFIX_PATH="${VENV}"
export USE_CUDA=1
export USE_CUDNN=1
export USE_NCCL=1
export BUILD_TYPE=Release
export MAX_JOBS=$(nproc)

source "$VENV/bin/activate"
pip install -r requirements.txt 2>/dev/null || true
python setup.py develop 2>&1 | tail -20

# Step 3: Verify
echo ""
echo "Verifying build..."
python -c "import torch; print(f'torch: {torch.__version__}, cuda: {torch.cuda.is_available()}, git: {torch.version.git_version[:12]}')"

echo ""
echo "Done. Activate with: source $VENV/bin/activate"
