#!/usr/bin/env python3
"""Download public sample clips into data/samples/."""

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
USER_AGENT = "TundraNVR/0.3 (sample downloader)"
INTEL = "https://github.com/intel-iot-devkit/sample-videos/raw/master"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath"


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
        "city.mp4",
        "https://videos.pexels.com/video-files/1721294/1721294-hd_1280_720_25fps.mp4",
        "City street — people, buses, cars",
    ),
    Sample(
        "street.mp4",
        f"{INTEL}/person-bicycle-car-detection.mp4",
        "Parking lot — people, bicycles, cars",
    ),
    Sample(
        "cars.mp4",
        f"{INTEL}/car-detection.mp4",
        "Overhead cars",
    ),
    Sample(
        "people.mp4",
        f"{INTEL}/people-detection.mp4",
        "Indoor pedestrians",
        alias_legacy=True,
    ),
    Sample(
        "wildlife.mp4",
        f"{COMMONS}/Metskitsed_autoteel.webm",
        "Deer on a road",
        transcode=True,
        max_seconds=25,
    ),
    Sample(
        "livestock.mp4",
        f"{COMMONS}/Cattle_drive_on_southern_Oregon_road_(41211547371).webm",
        "Cattle on a road",
        transcode=True,
        max_seconds=30,
    ),
    Sample(
        "aircraft.mp4",
        f"{COMMONS}/{quote('Plane_Spotting_Atlas_Air_Cargo_Boeing_747_Runway_at_RCTP_with_ATC_桃園機場起降.webm')}",
        "Aircraft at a runway",
        transcode=True,
        max_seconds=25,
    ),
    Sample(
        "drone.mp4",
        f"{COMMONS}/Quadcopter_(drone).webm",
        "Quadcopter",
        transcode=True,
        max_seconds=20,
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


def main() -> int:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    failed = 0
    for sample in SAMPLES:
        failed += download_one(sample)
    if failed:
        print(f"{failed} sample(s) failed", file=sys.stderr)
        return 1
    print("All samples ready under data/samples/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
