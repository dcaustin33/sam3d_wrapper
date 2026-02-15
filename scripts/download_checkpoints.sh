#!/usr/bin/env bash
#
# Download SAM 3D Body checkpoints from Hugging Face.
#
# Prerequisites:
#   - huggingface-cli login (authenticate first)
#   - Access approved on the HF repo pages:
#     https://huggingface.co/facebook/sam-3d-body-dinov3
#     https://huggingface.co/facebook/sam-3d-body-vith
#
# Usage:
#   bash scripts/download_checkpoints.sh                    # Download dinov3 (default)
#   bash scripts/download_checkpoints.sh --variant vith     # Download ViT-H variant
#   bash scripts/download_checkpoints.sh --all              # Download both variants
#   bash scripts/download_checkpoints.sh --dir ./my_ckpts   # Custom directory
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

VARIANT="dinov3"
DOWNLOAD_ALL=false
CHECKPOINT_DIR=""

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
        --dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            head -16 "$0" | tail -12
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

cd "$PROJECT_DIR"

DIR_FLAG=""
if [ -n "$CHECKPOINT_DIR" ]; then
    DIR_FLAG="--checkpoint-dir $CHECKPOINT_DIR"
fi

if [ "$DOWNLOAD_ALL" = true ]; then
    uv run sam3d-download --all $DIR_FLAG
else
    uv run sam3d-download --variant "$VARIANT" $DIR_FLAG
fi
