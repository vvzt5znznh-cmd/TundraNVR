#!/usr/bin/env python3
"""Download YOLO-readable demo clips and optional drone detector weights.

Provenance (see also app/samples.py and README):

- Intel IoT DevKit sample-videos — street / indoor / parking.
- CAVIAR (EC IST 2001 37540, CC BY-SA) — lobby dwell + LeftBag; cite
  http://homepages.inf.ed.ac.uk/rbf/CAVIAR/
- Pexels License — longer courtyard / street stock (no endorsement).
- Wikimedia Commons CC BY — ground camera filming a quadcopter (UAV in
  frame). Not drone-POV / elevated gimbal footage.
- Hugging Face TomSmail/drone-yolo-v1 — optional secondary weights; stock
  YOLOv8n has airplane but no drone class.

entrance.mp4 (CAVIAR mall) is intentionally not downloaded — YOLO false persons.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.samples import CLIPS, samples_dir  # noqa: E402

SAMPLES = samples_dir(ROOT)
DRONE_WEIGHTS = ROOT / "drone-yolo.pt"
DRONE_URL = "https://huggingface.co/TomSmail/drone-yolo-v1/resolve/main/best.pt"
USER_AGENT = "TundraNVR/0.15"


def _get(url: str, dest: Path, min_bytes: int) -> int:
    if dest.is_file() and dest.stat().st_size > min_bytes:
        print(f"Exists {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as handle:
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
            "scale=1280:720:force_original_aspect_ratio=decrease,"
            "pad=1280:720:(ow-iw)/2:(oh-ih)/2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(dest),
        ]
    )
    subprocess.run(cmd, check=True)


def _fetch_clip(name: str, url: str, *, transcode: bool, max_seconds: int | None) -> int:
    dest = SAMPLES / name
    if dest.is_file() and dest.stat().st_size > 10_000:
        print(f"Exists {dest}")
        return 0
    SAMPLES.mkdir(parents=True, exist_ok=True)
    if not transcode and max_seconds is None:
        return _get(url, dest, 10_000)
    with tempfile.TemporaryDirectory(prefix="tundra-sample-") as tmp:
        suffix = Path(url.split("?", 1)[0]).suffix or ".bin"
        if len(suffix) > 5:
            suffix = ".bin"
        raw = Path(tmp) / f"raw{suffix}"
        if _get(url, raw, 10_000):
            return 1
        try:
            print(f"Transcoding {name}" + (f" (≤{max_seconds}s)" if max_seconds else ""))
            _transcode(raw, dest, max_seconds)
        except Exception as exc:
            print(f"Failed {name}: {exc}", file=sys.stderr)
            if dest.exists():
                dest.unlink()
            return 1
    print(f"Saved {dest} ({dest.stat().st_size} bytes)")
    return 0


def main() -> int:
    failed = 0
    for clip in CLIPS:
        print(f"— {clip.label}: {clip.blurb}")
        print(f"  license: {clip.license}")
        if clip.page_review:
            print("  showcase: page_review allowlisted (needs demo.page_review: true)")
        failed += _fetch_clip(
            clip.name,
            clip.url,
            transcode=clip.transcode or clip.max_seconds is not None,
            max_seconds=clip.max_seconds,
        )
    print("— drone-yolo.pt (secondary UAV detector; stock YOLOv8n has no drone class)")
    failed += _get(DRONE_URL, DRONE_WEIGHTS, 10_000)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
