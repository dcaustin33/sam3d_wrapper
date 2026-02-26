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

BASE="/home/ubuntu/a100-gaussian-avatar/gaussian_avatar/datasets/people_snapshot_corrected"

for SUBJECT in female-3-casual female-4-casual male-4-casual; do
    echo "=== Running inference for $SUBJECT ==="
    uv run sam3d-infer \
        --images "$BASE/$SUBJECT/cam000/images" \
        --output "$BASE/$SUBJECT/cam000/results" \
        --no-vis \
        --shape-calibration-frames 20 \
        --focal-length 2664
done
