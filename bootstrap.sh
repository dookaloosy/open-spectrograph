#!/bin/bash
# Bootstrap a working spectrograph-sim dev environment.
#
# Creates .venv/, installs raysect + scientific Python deps, and editable-
# installs the sibling repo (evolutionary-solver) and this repo.
#
# Prerequisites (one-time):
#   sudo apt install python3-venv
#
# Clone the sibling repo next to this repo:
#   git clone <evolutionary-solver-url> ../evolutionary-solver
#
# Usage:
#   ./bootstrap.sh
#
# After it finishes, activate with:
#   source .venv/bin/activate

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Sanity checks ────────────────────────────────────────────────────────────

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 not found on PATH" >&2
    exit 1
fi

if ! python3 -c "import venv" >/dev/null 2>&1; then
    echo "Error: python3 venv module not available." >&2
    echo "Install with:  sudo apt install python3-venv" >&2
    exit 1
fi

if [ ! -d "../evolutionary-solver" ]; then
    echo "Error: ../evolutionary-solver not found." >&2
    echo "Clone it as a sibling of this repo:" >&2
    echo "    git clone <evolutionary-solver-url> ../evolutionary-solver" >&2
    exit 1
fi

# ── Create venv ──────────────────────────────────────────────────────────────

if [ ! -d .venv ]; then
    echo "Creating .venv/ ..."
    python3 -m venv .venv
else
    echo ".venv/ already exists — reusing"
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ── Install Python dependencies ──────────────────────────────────────────────

pip install --upgrade pip
pip install -e ../evolutionary-solver
pip install -e ".[cad]"

echo
echo "Done. Activate with:"
echo "    source .venv/bin/activate"
