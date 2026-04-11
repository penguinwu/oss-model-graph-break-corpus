#!/bin/bash
# Weekly nightly sweep — runs every Sunday
# Covers both HF/diffusers models AND custom models
# Usage: bash sweep/run_nightly.sh
set -euo pipefail

PROJ_DIR=/home/pengwu/projects/oss-model-graph-break-corpus
DATE=$(date +%Y-%m-%d)
NIGHTLY_DIR=$PROJ_DIR/sweep_results/nightly/$DATE
PYTHON=/home/pengwu/envs/torch-nightly/bin/python

echo "=== Nightly sweep $DATE ==="
echo "Start: $(date)"

# Check torch nightly is available
$PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "ERROR: torch-nightly env not available"
    exit 1
}

mkdir -p "$NIGHTLY_DIR"

# --- Phase 1: HF + diffusers models (static shapes, identify-only) ---
echo ""
echo "=== Phase 1: HF + diffusers identify sweep ==="
$PYTHON $PROJ_DIR/sweep/run_sweep.py \
  --device cuda \
  --workers 4 \
  --python $PYTHON \
  --output-dir $NIGHTLY_DIR \
  --source hf+diffusers --identify-only \
  > $NIGHTLY_DIR/sweep_static.log 2>&1
echo "Phase 1 done at $(date)"

# --- Phase 2: Dynamic shapes (true) ---
echo ""
echo "=== Phase 2: Dynamic=true sweep ==="
$PYTHON $PROJ_DIR/sweep/run_sweep.py \
  --device cuda \
  --workers 4 \
  --python $PYTHON \
  --output-dir $NIGHTLY_DIR/dynamic_true \
  --source hf+diffusers --identify-only \
  --dynamic true \
  > $NIGHTLY_DIR/sweep_dynamic_true.log 2>&1
echo "Phase 2 done at $(date)"

# --- Phase 3: Dynamic shapes (mark) ---
echo ""
echo "=== Phase 3: Dynamic=mark sweep ==="
$PYTHON $PROJ_DIR/sweep/run_sweep.py \
  --device cuda \
  --workers 4 \
  --python $PYTHON \
  --output-dir $NIGHTLY_DIR/dynamic_mark \
  --source hf+diffusers --identify-only \
  --dynamic mark \
  > $NIGHTLY_DIR/sweep_dynamic_mark.log 2>&1
echo "Phase 3 done at $(date)"

# --- Phase 4: Custom models ---
echo ""
echo "=== Phase 4: Custom models sweep ==="
$PYTHON $PROJ_DIR/sweep/run_sweep.py \
  --device cuda \
  --workers 4 \
  --python $PYTHON \
  --output-dir $NIGHTLY_DIR/custom \
  --source custom --identify-only \
  > $NIGHTLY_DIR/sweep_custom.log 2>&1
echo "Phase 4 done at $(date)"

# --- Phase 5: Analysis ---
echo ""
echo "=== Phase 5: Generate analysis report ==="
if [ -f "$PROJ_DIR/tools/daily_summary.py" ]; then
    $PYTHON $PROJ_DIR/tools/daily_summary.py > $NIGHTLY_DIR/summary.txt 2>&1 || true
fi

# Record versions
$PYTHON -c "
import json, torch
d = {'torch': torch.__version__}
try:
    import transformers; d['transformers'] = transformers.__version__
except: pass
try:
    import diffusers; d['diffusers'] = diffusers.__version__
except: pass
print(json.dumps(d, indent=2))
" > $NIGHTLY_DIR/versions.json

echo ""
echo "=== Nightly sweep complete ==="
echo "End: $(date)"
echo "Results: $NIGHTLY_DIR"
