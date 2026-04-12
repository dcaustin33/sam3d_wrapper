#!/usr/bin/env bash
#
# End-to-end: download a YouTube clip, extract frames, and run SAM 3D Body
# inference. Produces a per-clip folder named after the video:
#
#   <output_dir>/<clip_name>/
#       <clip_name>.mp4     - downloaded clip
#       frames/             - extracted frames (1 fps by default)
#       mhr/                - SAM 3D Body / MHR results (raw .npz, vis, etc.)
#
# Usage:
#   bash scripts/clip_to_poses.sh <youtube_url> <output_dir> \
#       [--start TIME] [--end TIME] [--fps N] \
#       [--max-persons N] [--shape-calibration-frames N] [--limit N]
#
# --limit N: only run inference on the first N extracted frames. Useful for
# a quick visualization test before processing the whole clip. Results go
# into <clip_dir>/mhr_test/ instead of mhr/ so they don't clobber a real run.
#
# Example:
#   bash scripts/clip_to_poses.sh \
#       "https://www.youtube.com/watch?v=Nleku8x8CfM" ./runs \
#       --shape-calibration-frames 10
#
# Note on MHR shape: SAM 3D Body solves each frame independently, so the
# identity (shape) parameters drift frame-to-frame unless you pass
# --shape-calibration-frames, which averages shape over N random frames and
# locks it in for the full sequence.
#

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <youtube_url> <output_dir> [--start TIME] [--end TIME] [--fps N] [--max-persons N] [--shape-calibration-frames N] [--limit N]" >&2
    exit 1
fi

URL="$1"
OUTPUT_DIR="$2"
shift 2

START=""
END=""
FPS="1"
MAX_PERSONS="1"
SHAPE_CAL=""
LIMIT=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --start) START="$2"; shift 2 ;;
        --end) END="$2"; shift 2 ;;
        --fps) FPS="$2"; shift 2 ;;
        --max-persons) MAX_PERSONS="$2"; shift 2 ;;
        --shape-calibration-frames) SHAPE_CAL="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

mkdir -p "$OUTPUT_DIR"
STAGING_DIR="$(mktemp -d "$OUTPUT_DIR/.download.XXXXXX")"
trap 'rm -rf "$STAGING_DIR"' EXIT

echo "[1/3] Downloading clip from $URL"
DOWNLOAD_ARGS=("$URL" "--output" "$STAGING_DIR")
[ -n "$START" ] && DOWNLOAD_ARGS+=("--start" "$START")
[ -n "$END" ] && DOWNLOAD_ARGS+=("--end" "$END")
uv run python tools/download_youtube.py "${DOWNLOAD_ARGS[@]}"

STAGED_VIDEO="$(ls -t "$STAGING_DIR"/*.mp4 2>/dev/null | head -n 1)"
if [ -z "$STAGED_VIDEO" ]; then
    echo "No .mp4 found in $STAGING_DIR after download" >&2
    exit 1
fi

CLIP_NAME="$(basename "$STAGED_VIDEO" .mp4)"
CLIP_DIR="$OUTPUT_DIR/$CLIP_NAME"
FRAMES_DIR="$CLIP_DIR/frames"
MHR_DIR="$CLIP_DIR/mhr"
VIDEO_PATH="$CLIP_DIR/$CLIP_NAME.mp4"

mkdir -p "$CLIP_DIR" "$FRAMES_DIR" "$MHR_DIR"
mv "$STAGED_VIDEO" "$VIDEO_PATH"
echo "Clip: $VIDEO_PATH"

echo "[2/3] Extracting frames at ${FPS} fps -> $FRAMES_DIR"
uv run python tools/extract_frames.py "$VIDEO_PATH" \
    --output "$FRAMES_DIR" \
    --fps "$FPS"

INFER_INPUT="$FRAMES_DIR"
INFER_OUTPUT="$MHR_DIR"
if [ -n "$LIMIT" ]; then
    INFER_INPUT="$CLIP_DIR/frames_test"
    INFER_OUTPUT="$CLIP_DIR/mhr_test"
    rm -rf "$INFER_INPUT"
    mkdir -p "$INFER_INPUT"
    # Symlink the first N frames so inference runs on a bounded subset
    count=0
    for f in "$FRAMES_DIR"/*; do
        [ -f "$f" ] || continue
        count=$((count + 1))
        [ "$count" -gt "$LIMIT" ] && break
        ln -sf "$f" "$INFER_INPUT/$(basename "$f")"
    done
    echo "Test mode: limited to first $LIMIT frames -> $INFER_INPUT"
fi

echo "[3/3] Running SAM 3D Body inference -> $INFER_OUTPUT (max-persons=$MAX_PERSONS)"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export HF_TOKEN="${HF_TOKEN:-${HF_API_KEY:-}}"

INFER_ARGS=(
    "--images" "$INFER_INPUT"
    "--output" "$INFER_OUTPUT"
    "--max-persons" "$MAX_PERSONS"
)
[ -n "$SHAPE_CAL" ] && INFER_ARGS+=("--shape-calibration-frames" "$SHAPE_CAL")

uv run sam3d-infer "${INFER_ARGS[@]}"

echo "Done. Results in $CLIP_DIR"
