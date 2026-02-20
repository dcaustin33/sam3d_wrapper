#!/usr/bin/env bash
#
# Run SAM 3D Body inference.
#
# Usage:
#   bash scripts/inference.sh --images ./photos --output ./results
#   bash scripts/inference.sh --images ./photos --output ./results --save-meshes
#   bash scripts/inference.sh --images ./photos --output ./results --shape-override shape.npy
#   bash scripts/inference.sh --images ./photo.jpg --output ./results --no-fov --no-detector
#
# All arguments are forwarded directly to `sam3d-infer`. Run with --help to see all options.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

export PYOPENGL_PLATFORM=egl

exec uv run sam3d-infer \
    --images /home/ubuntu/a100-gaussian-avatar/sam3d_wrapper/datasets/inference_images \
    --output ./output/inference_results \
    --no-vis
