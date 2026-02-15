#!/usr/bin/env bash
#
# Full setup script for sam3d_wrapper.
#
# This script:
#   1. Installs the sam3d_wrapper package with uv
#   2. Clones the upstream sam-3d-body repository
#   3. Installs detectron2 (required for human detection)
#   4. Downloads model checkpoints from Hugging Face
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - git
#   - Hugging Face CLI login: `huggingface-cli login`
#   - Access approved at https://huggingface.co/facebook/sam-3d-body-dinov3
#
# Usage:
#   bash scripts/setup.sh                  # Setup with default (dinov3) model
#   bash scripts/setup.sh --variant vith   # Setup with ViT-H model
#   bash scripts/setup.sh --all            # Setup with both model variants
#   bash scripts/setup.sh --with-sam3      # Also install SAM3 for segmentation
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VARIANT="dinov3"
DOWNLOAD_ALL=false
WITH_SAM3=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --all)
            DOWNLOAD_ALL=true
            shift
            ;;
        --with-sam3)
            WITH_SAM3=true
            shift
            ;;
        -h|--help)
            head -25 "$0" | tail -20
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "SAM 3D Body Wrapper - Full Setup"
echo "============================================================"

# Step 1: Install the package
echo ""
echo "[1/4] Installing sam3d_wrapper with uv ..."
cd "$PROJECT_DIR"
uv sync

# Step 2: Clone the upstream repo + install detectron2
echo ""
echo "[2/4] Cloning sam-3d-body repo and installing detectron2 ..."
SAM3_FLAG=""
if [ "$WITH_SAM3" = true ]; then
    SAM3_FLAG="--with-sam3"
fi
uv run sam3d-setup $SAM3_FLAG

# Step 3: Download checkpoints
echo ""
echo "[3/4] Downloading model checkpoints ..."
if [ "$DOWNLOAD_ALL" = true ]; then
    uv run sam3d-download --all
else
    uv run sam3d-download --variant "$VARIANT"
fi

# Step 4: Verify
echo ""
echo "[4/4] Verifying installation ..."
uv run python -c "
from sam3d_wrapper.repo import ensure_repo, get_repo_path
from sam3d_wrapper.download import verify_checkpoint, get_checkpoint_path
import sys

repo = ensure_repo()
print(f'  Repo path: {repo}')

for variant in ['dinov3', 'vith']:
    ckpt_path = get_checkpoint_path(variant)
    if verify_checkpoint(variant):
        print(f'  Checkpoint {variant}: OK ({ckpt_path})')
    else:
        print(f'  Checkpoint {variant}: not downloaded')

print()
print('Verifying upstream imports ...')
try:
    from sam_3d_body import load_sam_3d_body_hf, SAM3DBodyEstimator
    print('  sam_3d_body: OK')
except ImportError as e:
    print(f'  sam_3d_body: FAILED ({e})')
    sys.exit(1)

print()
print('Setup verified successfully!')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "Quick start:"
echo "  # Python API"
echo "  uv run python -c '"
echo "  from sam3d_wrapper import Sam3DBody"
echo "  model = Sam3DBody(model_variant=\"dinov3\")"
echo "  result = model.predict(\"your_image.jpg\")"
echo "  '"
echo ""
echo "  # CLI inference"
echo "  uv run sam3d-infer --images ./my_images --output ./results"
echo "============================================================"
