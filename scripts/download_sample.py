#!/usr/bin/env python3
"""Download drone detector weights."""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRONE_WEIGHTS = ROOT / "drone-yolo.pt"
URL = "https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt"
USER_AGENT = "TundraNVR/0.7"


def main() -> int:
    if DRONE_WEIGHTS.is_file() and DRONE_WEIGHTS.stat().st_size > 10_000:
        print(f"Exists {DRONE_WEIGHTS}")
        return 0
    print(f"Downloading {DRONE_WEIGHTS.name}")
    req = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, DRONE_WEIGHTS.open("wb") as handle:
            shutil.copyfileobj(resp, handle)
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        if DRONE_WEIGHTS.exists():
            DRONE_WEIGHTS.unlink()
        return 1
    if not DRONE_WEIGHTS.is_file() or DRONE_WEIGHTS.stat().st_size < 10_000:
        print("Download looks too small", file=sys.stderr)
        return 1
    print(f"Saved {DRONE_WEIGHTS.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
