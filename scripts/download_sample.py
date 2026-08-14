#!/usr/bin/env python3
"""Download the three demo clips and drone detector weights."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "data" / "samples"
DRONE_WEIGHTS = ROOT / "drone-yolo.pt"
USER_AGENT = "TundraNVR/0.7"
CAVIAR1 = "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1"
CAVIAR2 = "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA2"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath"
DRONE_WEIGHTS_URL = "https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt"

KEEP = {"entrance.mp4", "drone.mp4", "package.mp4"}


@dataclass(frozen=True)
class Sample:
    name: str
    url: str
    transcode: bool = False
    max_seconds: int | None = None


SAMPLES = [
    Sample(
        "entrance.mp4",
        f"{CAVIAR2}/EnterExitCrossingPaths1front/EnterExitCrossingPaths1front.mpg",
        transcode=True,
        max_seconds=40,
    ),
    Sample(
        "package.mp4",
        f"{CAVIAR1}/LeftBag/LeftBag.mpg",
        transcode=True,
        max_seconds=40,
    ),
    Sample(
        "drone.mp4",
        f"{COMMONS}/{quote('Quadcopter_20200202.webm')}",
        transcode=True,
        max_seconds=10,
    ),
]


def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=180) as resp, dest.open("wb") as handle:
        shutil.copyfileobj(resp, handle)


def _transcode(src: Path, dest: Path, max_seconds: int | None) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to convert this sample")
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src)]
    if max_seconds:
        cmd.extend(["-t", str(max_seconds)])
    cmd.extend(
        [
            "-vf",
            "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ]
    )
    subprocess.run(cmd, check=True)


def _ok(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 10_000


def download_one(sample: Sample) -> int:
    dest = SAMPLES_DIR / sample.name
    if _ok(dest):
        print(f"Exists {dest}")
        return 0
    print(f"Downloading {sample.name}")
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if sample.transcode:
            with tempfile.TemporaryDirectory() as tmp:
                raw = Path(tmp) / "source"
                _fetch(sample.url, raw)
                _transcode(raw, dest, sample.max_seconds)
        else:
            _fetch(sample.url, dest)
    except Exception as exc:
        print(f"Failed {sample.name}: {exc}", file=sys.stderr)
        if dest.exists():
            dest.unlink()
        return 1
    if not _ok(dest):
        print(f"Downloaded file looks too small: {dest}", file=sys.stderr)
        return 1
    print(f"Saved {dest.stat().st_size} bytes")
    return 0


def download_drone_weights() -> int:
    if _ok(DRONE_WEIGHTS):
        print(f"Exists {DRONE_WEIGHTS}")
        return 0
    print(f"Downloading {DRONE_WEIGHTS.name}")
    try:
        _fetch(DRONE_WEIGHTS_URL, DRONE_WEIGHTS)
    except Exception as exc:
        print(f"Failed drone-yolo.pt: {exc}", file=sys.stderr)
        if DRONE_WEIGHTS.exists():
            DRONE_WEIGHTS.unlink()
        return 1
    if not _ok(DRONE_WEIGHTS):
        print("Drone weights look too small", file=sys.stderr)
        return 1
    print(f"Saved {DRONE_WEIGHTS.stat().st_size} bytes")
    return 0


def _remove_extra() -> None:
    if not SAMPLES_DIR.is_dir():
        return
    for path in SAMPLES_DIR.glob("*.mp4"):
        if path.name not in KEEP:
            path.unlink()
            print(f"Removed {path}")


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _remove_extra()
    failed = download_drone_weights()
    for sample in SAMPLES:
        failed += download_one(sample)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
