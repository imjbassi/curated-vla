#!/usr/bin/env bash
# Phase 0: create the Python environment for curated-vla inside WSL2.
#
# System packages (needs sudo, run once yourself):
#   sudo apt-get update && sudo apt-get install -y ffmpeg git-lfs
#
# Everything heavy (venv, HF cache, outputs) lives on the WSL filesystem under
# $CVLA_HOME, not on /mnt/c, which is far slower for many small files.
set -euo pipefail

LEROBOT_VERSION="${LEROBOT_VERSION:-0.6.1}"
CVLA_HOME="${CVLA_HOME:-$HOME/cvla}"
VENV="$CVLA_HOME/venv"

mkdir -p "$CVLA_HOME"

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

if [ ! -d "$VENV" ]; then
    uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install "lerobot[smolvla,libero]==${LEROBOT_VERSION}"
# Phase 1 curation/audit tools
uv pip install tabulate scikit-learn matplotlib

# LIBERO asks interactively for a dataset path on first import, which crashes
# non-interactive evals. Answer "n" once to write the default ~/.libero/config.yaml.
if [ ! -f "$HOME/.libero/config.yaml" ]; then
    echo n | python -c "import libero.libero" >/dev/null
fi

python "$(dirname "$0")/check_env.py"
echo
echo "Activate with: source $VENV/bin/activate"
