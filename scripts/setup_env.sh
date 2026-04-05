#!/usr/bin/env bash
# Set up a virtual environment for the OSS Model Compiler Quality Corpus.
# Usage: bash scripts/setup_env.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="${1:-${REPO_ROOT}/env}"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
    echo "To recreate, remove it first: rm -rf $VENV_DIR"
    exit 1
fi

echo "Creating virtual environment at $VENV_DIR ..."
python3 -m venv "$VENV_DIR"

echo "Installing dependencies ..."
"$VENV_DIR/bin/pip" install --upgrade pip -q
"$VENV_DIR/bin/pip" install -r "$REPO_ROOT/requirements.txt"

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then try:"
echo "  python3 tools/query.py                    # browse the corpus"
echo "  python3 tools/reproduce.py BartModel      # reproduce a graph break"
echo "  python3 tools/analyze_explain.py           # root cause taxonomy"
