#!/bin/bash
# Weekly nightly sweep — runs every Sunday
# Covers both HF/diffusers models AND custom models
# Usage: SWEEP_PYTHON=/path/to/python bash sweep/run_nightly.sh
set -euo pipefail

PROJ_DIR=/home/pengwu/projects/oss-model-graph-break-corpus
DATE=$(date +%Y-%m-%d)
NIGHTLY_DIR=$PROJ_DIR/sweep_results/nightly/$DATE

# Use SWEEP_PYTHON env var (matches run_sweep.py convention)
export SWEEP_PYTHON=${SWEEP_PYTHON:-/home/pengwu/envs/torch-nightly/bin/python}

echo "=== Nightly sweep $DATE ==="
echo "Start: $(date)"

# Check torch nightly is available
$SWEEP_PYTHON -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "ERROR: torch-nightly env not available"
    exit 1
}

mkdir -p "$NIGHTLY_DIR"

# --- Phase 1: HF + diffusers models (static shapes, identify-only) ---
echo ""
echo "=== Phase 1: HF + diffusers identify sweep ==="
$SWEEP_PYTHON $PROJ_DIR/sweep/run_sweep.py sweep \
  --source hf diffusers \
  --identify-only \
  --output-dir $NIGHTLY_DIR \
  --resume \
  > $NIGHTLY_DIR/sweep_static.log 2>&1
echo "Phase 1 done at $(date)"

# --- Phase 2: Dynamic shapes (all dims) ---
echo ""
echo "=== Phase 2: Dynamic=all sweep ==="
$SWEEP_PYTHON $PROJ_DIR/sweep/run_sweep.py sweep \
  --source hf diffusers \
  --identify-only \
  --dynamic-dim all \
  --output-dir $NIGHTLY_DIR/dynamic_true \
  --resume \
  > $NIGHTLY_DIR/sweep_dynamic_true.log 2>&1
echo "Phase 2 done at $(date)"

# --- Phase 3: Dynamic shapes (batch dim only) ---
echo ""
echo "=== Phase 3: Dynamic=batch sweep ==="
$SWEEP_PYTHON $PROJ_DIR/sweep/run_sweep.py sweep \
  --source hf diffusers \
  --identify-only \
  --dynamic-dim batch \
  --output-dir $NIGHTLY_DIR/dynamic_mark \
  --resume \
  > $NIGHTLY_DIR/sweep_dynamic_mark.log 2>&1
echo "Phase 3 done at $(date)"

# --- Phase 4: Custom models ---
echo ""
echo "=== Phase 4: Custom models sweep ==="
$SWEEP_PYTHON $PROJ_DIR/sweep/run_sweep.py sweep \
  --source custom \
  --identify-only \
  --output-dir $NIGHTLY_DIR/custom \
  --resume \
  > $NIGHTLY_DIR/sweep_custom.log 2>&1
echo "Phase 4 done at $(date)"

# --- Phase 5: Analysis ---
echo ""
echo "=== Phase 5: Generate analysis report ==="
if [ -f "$PROJ_DIR/tools/daily_summary.py" ]; then
    $SWEEP_PYTHON $PROJ_DIR/tools/daily_summary.py > $NIGHTLY_DIR/summary.txt 2>&1 || true
fi

# Generate nightly summary markdown for results/
echo "Generating nightly summary..."
python3 $PROJ_DIR/tools/generate_nightly_summary.py --date $DATE 2>&1 || true

echo ""
echo "=== Nightly sweep complete ==="
echo "End: $(date)"
echo "Results: $NIGHTLY_DIR"
