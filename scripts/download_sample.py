#!/usr/bin/env python3
"""Download the indoor pedestrian demo clip and optional drone weights."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
DRONE_WEIGHTS = ROOT / "drone-yolo.pt"
USER_AGENT = "TundraNVR/0.13"

INDOOR_URL = (
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"
)
DRONE_URL = "https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt"


def _get(url: str, dest: Path, min_bytes: int) -> int:
    if dest.is_file() and dest.stat().st_size > min_bytes:
        print(f"Exists {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, tmp.open("wb") as handle:
            shutil.copyfileobj(resp, handle)
    except Exception as exc:
        print(f"Failed {dest.name}: {exc}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return 1
    if not tmp.is_file() or tmp.stat().st_size < min_bytes:
        print(f"Download looks too small: {dest.name}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return 1
    tmp.replace(dest)
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")
    return 0


def main() -> int:
    samples = _get(INDOOR_URL, SAMPLES / "indoor.mp4", 10_000)
    weights = _get(DRONE_URL, DRONE_WEIGHTS, 10_000)
    return 1 if samples or weights else 0


if __name__ == "__main__":
    raise SystemExit(main())
