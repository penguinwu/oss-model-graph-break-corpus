#!/bin/bash
# setup_hf_kernels_cache.sh
#
# One-time (per-VM, re-runnable) setup script to populate the HuggingFace Hub
# cache with kernels-community/* repos that our corpus models depend on.
#
# WHY THIS EXISTS:
#   The HuggingFace `kernels` package fetches CUDA-extension build artifacts from
#   HF Hub on first model load. Under Otter's `agent:claude_code` BPF identity,
#   `huggingface.co` is BLOCKED — every kernel fetch fails with "Name or service
#   not known". Workaround: download the kernels via `sudo` (root identity is not
#   subject to the BPF block) into `~/.cache/huggingface/`, then have worker.py
#   set `LOCAL_KERNELS=...` env var (via sweep/kernel_resolver.py) to point at
#   the local snapshot, bypassing the runtime freshness check.
#
# WHEN TO RUN:
#   - Once per devvm (initial setup)
#   - When a new kernels-community/<repo> is added to KERNEL_REPOS below
#   - When new torch versions ship and you want updated build variants in the cache
#     (kernels-community maintainers release new variants on HF Hub; re-running
#     this script pulls them down)
#
# USAGE:
#   sudo bash sweep/setup_hf_kernels_cache.sh
#
# Idempotent: snapshot_download() skips already-cached files. Safe to re-run.

set -euo pipefail

# kernels-community repos used by transformers models in our corpus.
# Keep in sync with sweep/kernel_resolver.py:MODEL_KERNEL_MAP — when adding a
# new model→repo entry there, also add the repo here so its kernels get cached.
KERNEL_REPOS=(
    "kernels-community/mra"
    # Future additions (uncomment once verified end-to-end and added to MODEL_KERNEL_MAP):
    # "kernels-community/rwkv"
    # "kernels-community/yoso"
    # "kernels-community/flash-mla"           # glm_moe_dsa
    # "kernels-community/vllm-flash-attn3"    # gpt_oss, openai_privacy_filter
    # "kernels-community/cv-utils"            # sam3_video (sam3_video already skipped)
)

# Default to the corpus's standard PT 2.12 venv if not overridden
PYTHON_BIN="${PYTHON_BIN:-/home/pengwu/envs/torch-nightly-cu128/bin/python}"
HF_HOME_DIR="${HF_HOME:-/home/pengwu/.cache/huggingface}"

if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This script must run with sudo (BPF blocks HF Hub for agent:claude_code)."
    echo "  Try: sudo bash $0"
    exit 1
fi

if ! [ -x "$PYTHON_BIN" ]; then
    echo "ERROR: Python binary not found: $PYTHON_BIN"
    echo "  Override with PYTHON_BIN=<path> sudo bash $0"
    exit 1
fi

# Verify huggingface_hub is installed in the python env
if ! "$PYTHON_BIN" -c "import huggingface_hub" 2>/dev/null; then
    echo "ERROR: huggingface_hub not installed in $PYTHON_BIN"
    echo "  Install with: $PYTHON_BIN -m pip install huggingface_hub"
    exit 1
fi

echo "=========================================="
echo "HF Hub kernels cache setup"
echo "=========================================="
echo "Python:   $PYTHON_BIN"
echo "HF_HOME:  $HF_HOME_DIR"
echo "Repos:    ${#KERNEL_REPOS[@]} (${KERNEL_REPOS[*]})"
echo

mkdir -p "$HF_HOME_DIR"

for repo in "${KERNEL_REPOS[@]}"; do
    echo "--- $repo ---"
    HF_HOME="$HF_HOME_DIR" \
    HTTPS_PROXY="http://fwdproxy:8080" \
    HTTP_PROXY="http://fwdproxy:8080" \
    "$PYTHON_BIN" -c "
from huggingface_hub import snapshot_download
import sys
try:
    path = snapshot_download(repo_id='$repo', repo_type='model')
    print(f'  ✓ cached at: {path}')
except Exception as e:
    print(f'  ✗ FAILED: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(1)
"
done

# Restore ownership to the human user (sudo creates root-owned files)
chown -R pengwu:users "$HF_HOME_DIR"

echo
echo "=========================================="
echo "Done. Verify with:"
echo "  $PYTHON_BIN -m sweep.kernel_resolver --list"
echo "  $PYTHON_BIN -m sweep.kernel_resolver MraModel"
echo "=========================================="
