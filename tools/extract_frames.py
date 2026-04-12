"""Extract frames from a video at a fixed frames-per-second rate via ffmpeg."""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def extract_frames(
    video: Path,
    output_dir: Path,
    fps: float = 1.0,
    image_format: str = "jpg",
    quality: int = 2,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    if not video.exists():
        raise FileNotFoundError(video)

    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = output_dir / f"{video.stem}_%06d.{image_format}"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video),
        "-vf", f"fps={fps}",
        "-qscale:v", str(quality),
        str(pattern),
    ]
    subprocess.run(cmd, check=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Input video file")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: <video_stem>_frames next to the video)",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract (default: 1)")
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], help="Image format")
    parser.add_argument("--quality", type=int, default=2, help="ffmpeg -qscale:v value (1=best, 31=worst)")
    args = parser.parse_args()

    output = args.output or args.video.parent / f"{args.video.stem}_frames"
    out = extract_frames(args.video, output, fps=args.fps, image_format=args.format, quality=args.quality)
    print(f"Frames written to {out}")


if __name__ == "__main__":
    main()
