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

export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}"

uv run python scripts/infer_coreview386.py
