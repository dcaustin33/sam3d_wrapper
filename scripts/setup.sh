#!/usr/bin/env bash
#
# Full setup script for sam3d_wrapper.
#
# This script:
#   1. Installs system-level OpenGL/EGL libraries for headless rendering
#   2. Ensures the current user has GPU render device access
#   3. Installs the sam3d_wrapper package with uv
#   4. Clones the upstream sam-3d-body repository
#   5. Installs detectron2 (required for human detection, installed outside
#      uv lock to avoid conflicts — uv sync will remove it otherwise)
#   6. Downloads model checkpoints from Hugging Face
#
# Prerequisites:
#   - uv (https://docs.astral.sh/uv/)
#   - git
#   - sudo access (for apt and group changes)
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
            head -30 "$0" | tail -25
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

# Step 1: System dependencies for headless OpenGL rendering (pyrender)
echo ""
echo "[1/6] Installing system OpenGL/EGL libraries ..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq libegl1-mesa libegl1-mesa-dev libosmesa6 > /dev/null 2>&1
    echo "  EGL and OSMesa libraries installed."
else
    echo "  WARNING: apt-get not found. Manually install libegl1-mesa and libosmesa6."
fi

# Step 2: GPU render device access (/dev/dri/renderD*)
echo ""
echo "[2/6] Checking GPU render device permissions ..."
NEEDS_NEWGRP=false
if [ -d /dev/dri ]; then
    if ! groups | grep -qw "video"; then
        echo "  Adding $(whoami) to 'video' group ..."
        sudo usermod -aG video "$(whoami)"
        NEEDS_NEWGRP=true
    fi
    if ! groups | grep -qw "render"; then
        echo "  Adding $(whoami) to 'render' group ..."
        sudo usermod -aG render "$(whoami)"
        NEEDS_NEWGRP=true
    fi
    if [ "$NEEDS_NEWGRP" = true ]; then
        echo "  Groups updated. You may need to log out and back in (or run 'newgrp render')"
        echo "  for visualization/mesh rendering to work."
    else
        echo "  GPU device access OK."
    fi
else
    echo "  No /dev/dri found — headless rendering may not work."
fi

# Step 3: Install the package
echo ""
echo "[3/6] Installing sam3d_wrapper with uv ..."
cd "$PROJECT_DIR"
uv sync

# Step 4: Clone the upstream repo + install detectron2
echo ""
echo "[4/6] Cloning sam-3d-body repo and installing detectron2 ..."
SAM3_FLAG=""
if [ "$WITH_SAM3" = true ]; then
    SAM3_FLAG="--with-sam3"
fi
uv run sam3d-setup $SAM3_FLAG

# IMPORTANT: detectron2 is installed with --no-deps outside of uv's lock file.
# Running `uv sync` again will REMOVE detectron2. If that happens, re-run:
#   uv pip install "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" --no-build-isolation --no-deps

# Step 5: Download checkpoints
echo ""
echo "[5/6] Downloading model checkpoints ..."
if [ "$DOWNLOAD_ALL" = true ]; then
    uv run sam3d-download --all
else
    uv run sam3d-download --variant "$VARIANT"
fi

# Step 6: Verify
echo ""
echo "[6/6] Verifying installation ..."
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

try:
    import detectron2
    print('  detectron2: OK')
except ImportError:
    print('  detectron2: MISSING (run: uv pip install \"git+https://github.com/facebookresearch/detectron2.git@a1ce2f9\" --no-build-isolation --no-deps)')
    sys.exit(1)

try:
    import moge
    print('  moge (FOV): OK')
except ImportError:
    print('  moge (FOV): MISSING')

print()
print('Setup verified successfully!')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo ""
echo "IMPORTANT: Visualization/mesh rendering requires:"
echo "  export PYOPENGL_PLATFORM=egl"
echo ""
echo "Quick start:"
echo "  # Python API"
echo "  PYOPENGL_PLATFORM=egl uv run python -c '"
echo "  from sam3d_wrapper import Sam3DBody"
echo "  model = Sam3DBody(model_variant=\"dinov3\")"
echo "  result = model.predict(\"your_image.jpg\")"
echo "  '"
echo ""
echo "  # CLI inference"
echo "  PYOPENGL_PLATFORM=egl uv run sam3d-infer --images ./my_images --output ./results"
echo "============================================================"
