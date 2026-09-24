"""Bundled demo clips: provenance, presets, and showcase Review paging.

Sample / fixture files are never a live site Pattern of Life. They do not
absorb into the occupancy map. By default they also do not page Review.

Opt in with ``demo.page_review: true`` in config.yaml. Only clips marked
``page_review=True`` below may then page (still with provenance ``sample``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleClip:
    """One downloadable demo file under ``data/samples/``."""

    name: str
    label: str
    blurb: str
    url: str
    license: str
    page_review: bool = False
    preset: bool = True
    transcode: bool = False
    max_seconds: int | None = None
    default_fallback: bool = False


# Provenance notes live in scripts/download_sample.py and README.
INTEL = "https://github.com/intel-iot-devkit/sample-videos/raw/master"
CAVIAR1 = "https://groups.inf.ed.ac.uk/vision/DATASETS/CAVIAR/CAVIARDATA1"
COMMONS = "https://commons.wikimedia.org/wiki/Special:FilePath"

CLIPS: tuple[SampleClip, ...] = (
    SampleClip(
        name="street.mp4",
        label="Street",
        blurb="Intel parking lot — people, bicycles, cars (default fallback).",
        url=f"{INTEL}/person-bicycle-car-detection.mp4",
        license="Intel IoT DevKit sample-videos (see upstream repo)",
        default_fallback=True,
    ),
    SampleClip(
        name="indoor.mp4",
        label="Indoor",
        blurb="Intel indoor pedestrians — tracks and dwell.",
        url=f"{INTEL}/people-detection.mp4",
        license="Intel IoT DevKit sample-videos (see upstream repo)",
    ),
    SampleClip(
        name="parking.mp4",
        label="Parking",
        blurb="Intel overhead / lot cars — vehicle namer.",
        url=f"{INTEL}/car-detection.mp4",
        license="Intel IoT DevKit sample-videos (see upstream repo)",
    ),
    SampleClip(
        name="courtyard.mp4",
        label="Courtyard",
        blurb="Pexels outdoor street — longer mixed foot/vehicle traffic.",
        url="https://videos.pexels.com/video-files/3571264/3571264-hd_1280_720_30fps.mp4",
        license="Pexels License (free use; no endorsement implied)",
        page_review=True,
        max_seconds=180,
    ),
    SampleClip(
        name="lobby.mp4",
        label="Lobby",
        blurb="CAVIAR INRIA lobby Browse_WhileWaiting2 — longer indoor dwell.",
        url=f"{CAVIAR1}/Browse_WhileWaiting2/Browse_WhileWaiting2.mpg",
        license="CAVIAR EC IST 2001 37540 — CC BY-SA; cite CAVIAR",
        page_review=True,
        transcode=True,
    ),
    SampleClip(
        name="package.mp4",
        label="Left bag",
        blurb="CAVIAR LeftBag — unattended bag rule (dwell ≥8s).",
        url=f"{CAVIAR1}/LeftBag/LeftBag.mpg",
        license="CAVIAR EC IST 2001 37540 — CC BY-SA; cite CAVIAR",
        page_review=True,
        transcode=True,
    ),
    SampleClip(
        name="drones.mp4",
        label="Drones",
        blurb=(
            "Wikimedia ground/static camera filming a quadcopter in airspace "
            "(UAV as object — not drone-POV / gimbal footage)."
        ),
        url=f"{COMMONS}/Quadcopter_(drone).webm",
        license="CC BY 3.0 — Sounds of Changes / Mikael Maffei (Wikimedia)",
        page_review=True,
        transcode=True,
        max_seconds=90,
    ),
)

# Fallback order when camera index/URL is unavailable.
FALLBACK_ORDER = (
    "street.mp4",
    "indoor.mp4",
    "parking.mp4",
    "courtyard.mp4",
    "lobby.mp4",
    "package.mp4",
    "drones.mp4",
    "drone.mp4",  # legacy name if present
    "sample.mp4",
    # entrance.mp4 intentionally omitted from default search preference;
    # still accepted last if somehow the only file on disk.
    "entrance.mp4",
)

PAGEABLE_NAMES = frozenset(c.name for c in CLIPS if c.page_review)


def clip_by_name(name: str) -> SampleClip | None:
    key = Path(name).name.lower()
    for clip in CLIPS:
        if clip.name == key:
            return clip
    return None


def samples_dir(root: Path) -> Path:
    return root / "data" / "samples"


def is_sample_path(root: Path, source: str | int | Path) -> bool:
    """True when source resolves under data/samples/ (file or relative path)."""
    if isinstance(source, int):
        return False
    text = str(source).strip()
    if not text or "://" in text:
        return False
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    try:
        resolved = path.resolve()
        base = samples_dir(root).resolve()
    except OSError:
        return False
    return resolved == base or base in resolved.parents


def page_review_allowed(name: str) -> bool:
    return Path(name).name.lower() in PAGEABLE_NAMES


def bundled_sample(root: Path) -> Path | None:
    dirs = (samples_dir(root), root / "data")
    for folder in dirs:
        for name in FALLBACK_ORDER:
            path = folder / name
            if path.is_file() and path.stat().st_size > 10_000:
                return path
    return None


def demo_clips(root: Path) -> list[dict]:
    out: list[dict] = []
    for clip in CLIPS:
        if not clip.preset:
            continue
        path = samples_dir(root) / clip.name
        out.append(
            {
                "id": Path(clip.name).stem,
                "label": clip.label,
                "path": f"data/samples/{clip.name}",
                "present": path.is_file() and path.stat().st_size > 10_000,
                "page_review": bool(clip.page_review),
                "blurb": clip.blurb,
            }
        )
    return out
