#!/usr/bin/env python3
"""Download a short public people-detection clip into data/sample.mp4."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "sample.mp4"
URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    print(f" -> {DEST}")
    try:
        urllib.request.urlretrieve(URL, DEST)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    size = DEST.stat().st_size
    if size < 10_000:
        print(f"Downloaded file looks too small ({size} bytes)", file=sys.stderr)
        return 1
    print(f"Saved {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
