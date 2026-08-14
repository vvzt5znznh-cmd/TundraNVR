#!/usr/bin/env python3
"""Download static building-CCTV sample clips and the drone detector weights."""

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
LEGACY_SAMPLE = ROOT / "data" / "sample.mp4"
DRONE_WEIGHTS = ROOT / "drone-yolo.pt"
USER_AGENT = "TundraNVR/0.5 (sample downloader)"
INTEL = "https://github.com/intel-iot-devkit/sample-videos/raw/master"
CAVIAR1 = "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1"
CAVIAR2 = "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA2"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath"
DRONE_WEIGHTS_URL = "https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt"

OBSOLETE = [
    "city.mp4",
    "street.mp4",
    "cars.mp4",
    "wildlife.mp4",
    "livestock.mp4",
    "aircraft.mp4",
    "people.mp4",
]


@dataclass(frozen=True)
class Sample:
    name: str
    url: str
    label: str
    transcode: bool = False
    max_seconds: int | None = None
    alias_legacy: bool = False


SAMPLES = [
    Sample(
        "entrance.mp4",
        f"{CAVIAR2}/EnterExitCrossingPaths1front/EnterExitCrossingPaths1front.mpg",
        "Building entrance — people at the door (CAVIAR)",
        transcode=True,
        max_seconds=40,
    ),
    Sample(
        "corridor.mp4",
        f"{CAVIAR2}/EnterExitCrossingPaths1cor/EnterExitCrossingPaths1cor.mpg",
        "Indoor corridor — people walking (CAVIAR)",
        transcode=True,
        max_seconds=40,
    ),
    Sample(
        "lobby.mp4",
        f"{CAVIAR1}/Meet_Crowd/Meet_Crowd.mpg",
        "Indoor lobby — people meeting (CAVIAR)",
        transcode=True,
        max_seconds=35,
    ),
    Sample(
        "indoor.mp4",
        f"{INTEL}/people-detection.mp4",
        "Indoor hall — pedestrians",
        alias_legacy=True,
    ),
    Sample(
        "aisle.mp4",
        f"{INTEL}/store-aisle-detection.mp4",
        "Indoor aisle — retail CCTV",
    ),
    Sample(
        "parking.mp4",
        f"{INTEL}/person-bicycle-car-detection.mp4",
        "Building parking — people, bicycles, cars",
    ),
    Sample(
        "package.mp4",
        f"{CAVIAR1}/LeftBag/LeftBag.mpg",
        "Indoor — bag left behind (CAVIAR)",
        transcode=True,
        max_seconds=40,
    ),
    Sample(
        "drone.mp4",
        f"{COMMONS}/{quote('Quadcopter_20200202.webm')}",
        "Quadcopter in view",
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
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
    ]
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
    print(f"Downloading {sample.label}")
    print(f" -> {dest}")
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
    if sample.alias_legacy and not _ok(LEGACY_SAMPLE):
        shutil.copy2(dest, LEGACY_SAMPLE)
        print(f"Copied {LEGACY_SAMPLE}")
    return 0


def download_drone_weights() -> int:
    if _ok(DRONE_WEIGHTS):
        print(f"Exists {DRONE_WEIGHTS}")
        return 0
    print("Downloading drone detector weights")
    print(f" -> {DRONE_WEIGHTS}")
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


def _remove_obsolete() -> None:
    for name in OBSOLETE:
        path = SAMPLES_DIR / name
        if path.is_file():
            path.unlink()
            print(f"Removed old sample {path}")


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _remove_obsolete()
    failed = download_drone_weights()
    for sample in SAMPLES:
        failed += download_one(sample)
    if failed:
        print(f"{failed} download(s) failed", file=sys.stderr)
        return 1
    print("Building CCTV samples ready under data/samples/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
