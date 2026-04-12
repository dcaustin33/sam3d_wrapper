"""Download a YouTube clip (optionally trimmed to a time range) using yt-dlp.

Default format selector ``bv*+ba/b`` pulls the highest-resolution video stream
and best audio stream available, then muxes them — i.e. native resolution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from yt_dlp import YoutubeDL


def _parse_time(value: str) -> float:
    if value is None:
        return None
    parts = value.split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(f"Invalid time: {value}")


def download(
    url: str,
    output_dir: Path,
    start: str | None = None,
    end: str | None = None,
    fmt: str = "bv*+ba/b",
    container: str = "mp4",
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        "format": fmt,
        "merge_output_format": container,
        "outtmpl": str(output_dir / "%(title)s [%(id)s].%(ext)s"),
        "noplaylist": True,
    }

    if start is not None or end is not None:
        s = _parse_time(start) if start else 0.0
        e = _parse_time(end) if end else None

        def _ranges(info_dict, ydl):
            return [{"start_time": s, "end_time": e if e is not None else info_dict.get("duration")}]

        opts["download_ranges"] = _ranges
        opts["force_keyframes_at_cuts"] = True

    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))

    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Download a YouTube clip via yt-dlp.")
    p.add_argument("url")
    p.add_argument("--output", "-o", default="./downloads", type=Path)
    p.add_argument("--start", help="Start time (e.g. 0, 30, 1:05, 0:01:05)")
    p.add_argument("--end", help="End time (e.g. 60, 1:00, 0:01:00)")
    p.add_argument("--format", dest="fmt", default="bv*+ba/b",
                   help="yt-dlp format selector (default: best video+audio = native resolution)")
    p.add_argument("--container", default="mp4")
    args = p.parse_args()

    path = download(args.url, args.output, args.start, args.end, args.fmt, args.container)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
