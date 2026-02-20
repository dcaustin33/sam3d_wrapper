#!/usr/bin/env bash
#
# Run SAM 3D Body for gaussian avatar pipelines.
#
# Estimates a consistent body shape from a random sample, then runs inference
# on all images using that fixed shape. Outputs per-frame .npz files, mesh
# overlay visualizations, and the estimated shape.npy / focal_length.npy.
#
# Usage:
#   bash scripts/gaussian_avatar.sh --images /path/to/images --output /path/to/output
#   bash scripts/gaussian_avatar.sh --images ./photos --output ./results --n-shape-samples 20
#   bash scripts/gaussian_avatar.sh --images ./photos --output ./results --no-vis
#
# All arguments are forwarded directly to gaussian_avatar.py.
# Run with --help to see all options.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

export PYOPENGL_PLATFORM=egl

exec uv run python scripts/gaussian_avatar.py \
    --images /home/ubuntu/a100-gaussian-avatar/sam3d_wrapper/datasets/inference_images \
    --output /home/ubuntu/a100-gaussian-avatar/sam3d_wrapper/output/inference_results --no-vis;
