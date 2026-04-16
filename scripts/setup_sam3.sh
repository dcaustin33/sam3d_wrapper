#!/usr/bin/env bash
#
# Install the optional sam3 extra (pulls sam3-wrapper from GitHub) and
# download the SAM 3 checkpoint from Hugging Face.
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - Hugging Face CLI login: `huggingface-cli login`
#   - Access approved at https://huggingface.co/facebook/sam3
#
# Usage:
#   bash scripts/setup_sam3.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================================"
echo "Installing sam3-wrapper extra ..."
echo "============================================================"
uv sync --extra sam3

echo ""
echo "============================================================"
echo "Downloading SAM 3 checkpoint ..."
echo "============================================================"
uv run sam3-download

echo ""
echo "Done. You can now run:"
echo "  PYOPENGL_PLATFORM=egl uv run python scripts/infer_with_mask.py \\"
echo "      --images ./photos --output ./results"
