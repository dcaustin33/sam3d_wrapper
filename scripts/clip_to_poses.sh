#!/usr/bin/env bash
#
# End-to-end: download a YouTube clip, extract frames at 1fps, and run SAM 3D Body
# inference filtering to the single largest-bbox person per frame.
#
# Usage:
#   bash scripts/clip_to_poses.sh <youtube_url> <work_dir> [--start TIME] [--end TIME] [--fps N]
#
# Example:
#   bash scripts/clip_to_poses.sh "https://youtu.be/abc" ./runs/myclip --start 0:10 --end 0:40
#
# Outputs inside <work_dir>:
#   video/   - downloaded mp4
#   frames/  - extracted frames (one per second by default)
#   poses/   - SAM 3D Body results (largest bbox only)
#

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <youtube_url> <work_dir> [--start TIME] [--end TIME] [--fps N] [--max-persons N]" >&2
    exit 1
fi

URL="$1"
WORK_DIR="$2"
shift 2

START=""
END=""
FPS="1"
MAX_PERSONS="1"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --max-persons) MAX_PERSONS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

VIDEO_DIR="$WORK_DIR/video"
FRAMES_DIR="$WORK_DIR/frames"
POSES_DIR="$WORK_DIR/poses"

mkdir -p "$VIDEO_DIR" "$FRAMES_DIR" "$POSES_DIR"

echo "[1/3] Downloading clip from $URL"
DOWNLOAD_ARGS=("$URL" "--output" "$VIDEO_DIR")
[ -n "$START" ] && DOWNLOAD_ARGS+=("--start" "$START")
[ -n "$END" ] && DOWNLOAD_ARGS+=("--end" "$END")
uv run python tools/download_youtube.py "${DOWNLOAD_ARGS[@]}"

VIDEO_FILE="$(ls -t "$VIDEO_DIR"/*.mp4 2>/dev/null | head -n 1)"
if [ -z "$VIDEO_FILE" ]; then
    echo "No .mp4 found in $VIDEO_DIR after download" >&2
    exit 1
fi
echo "Downloaded: $VIDEO_FILE"

echo "[2/3] Extracting frames at ${FPS} fps -> $FRAMES_DIR"
uv run python tools/extract_frames.py "$VIDEO_FILE" \
    --output "$FRAMES_DIR" \
    --fps "$FPS"

echo "[3/3] Running SAM 3D Body inference -> $POSES_DIR (max-persons=$MAX_PERSONS)"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_TOKEN="${HF_TOKEN:-${HF_API_KEY:-}}"

uv run sam3d-infer \
    --images "$FRAMES_DIR" \
    --output "$POSES_DIR" \
    --max-persons "$MAX_PERSONS"

echo "Done. Results in $POSES_DIR"
