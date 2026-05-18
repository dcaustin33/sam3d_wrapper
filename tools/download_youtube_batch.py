"""Batch-download YouTube clips listed in a text file.

Each non-comment line:  URL  [START]  [END]   (whitespace-separated, "-" skips a field)
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from download_youtube import download


def _field(value: str | None) -> str | None:
    if value is None or value == "-" or value == "":
        return None
    return value


def parse_list(path: Path) -> list[tuple[str, str | None, str | None]]:
    entries: list[tuple[str, str | None, str | None]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = shlex.split(line)
        if len(parts) == 0:
            continue
        if len(parts) > 3:
            raise ValueError(f"{path}:{lineno}: expected up to 3 fields, got {len(parts)}")
        url = parts[0]
        start = _field(parts[1]) if len(parts) > 1 else None
        end = _field(parts[2]) if len(parts) > 2 else None
        entries.append((url, start, end))
    return entries


def main() -> None:
    p = argparse.ArgumentParser(description="Batch YouTube downloader.")
    p.add_argument("--list", "-l", default=Path(__file__).parent / "youtube_list.txt", type=Path)
    p.add_argument("--output", "-o", default="./downloads", type=Path)
    p.add_argument("--format", dest="fmt", default="bv*+ba/b")
    p.add_argument("--container", default="mp4")
    args = p.parse_args()

    entries = parse_list(args.list)
    if not entries:
        print(f"No entries in {args.list}")
        return

    for url, start, end in entries:
        trim = f"  [{start or 0} → {end or 'end'}]" if (start or end) else ""
        print(f"→ {url}{trim}")
        path = download(url, args.output, start, end, args.fmt, args.container)
        print(f"  saved: {path}")


if __name__ == "__main__":
    main()
