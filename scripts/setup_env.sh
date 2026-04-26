#!/usr/bin/env bash
# Set up a virtual environment for the OSS Model Compiler Quality Corpus.
# Usage: bash scripts/setup_env.sh VENV_PATH
# Convention: venvs live under ~/envs/<name>, never inside the repo.
# Examples:
#   bash scripts/setup_env.sh ~/envs/torch211       # current stable
#   bash scripts/setup_env.sh ~/envs/torch-nightly  # nightly
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Error: VENV_PATH is required." >&2
    echo "Usage: bash scripts/setup_env.sh VENV_PATH" >&2
    echo "Convention: use ~/envs/<name> (e.g., ~/envs/torch211, ~/envs/torch-nightly)." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$1"

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
