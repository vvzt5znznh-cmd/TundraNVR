from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"
SETTINGS_PATH = ROOT / "data" / "settings.json"

SUGGESTED_MODELS = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
]

SAMPLE_LABELS = {
    "city.mp4": "City street — people, buses, cars",
    "street.mp4": "Parking lot — people, bicycles, cars",
    "cars.mp4": "Overhead cars",
    "wildlife.mp4": "Deer on a road",
    "livestock.mp4": "Cattle on a road",
    "aircraft.mp4": "Aircraft at a runway",
    "drone.mp4": "Quadcopter",
    "people.mp4": "Indoor pedestrians",
}

OUTDOOR_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
]


class SettingsError(ValueError):
    """Invalid live-settings payload."""


def _as_source(value: Any) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


@dataclass
class CameraConfig:
    source: str | int = "data/samples/city.mp4"
    width: int = 1280
    height: int = 720
    loop_file: bool = True


@dataclass
class PipelineConfig:
    detect_fps: float = 5.0
    live_fps: float = 10.0
    jpeg_quality: int = 80


@dataclass
class MotionConfig:
    min_area: int = 1500
    threshold: int = 25
    blur_ksize: int = 21


@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"
    conf: float = 0.4
    classes: list[str] = field(default_factory=lambda: list(OUTDOOR_CLASSES))
    device: str = "cpu"


@dataclass
class EventsConfig:
    pre_seconds: float = 2.0
    post_seconds: float = 4.0
    cooldown_seconds: float = 8.0
    retention_days: int = 7
    clip_fps: float = 10.0


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class VisionConfig:
    enabled: bool = True
    provider: str = "auto"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "moondream"
    openai_model: str = "gpt-4o-mini"
    timeout_seconds: float = 25.0


PUBLIC_WEBCAMS = [
    {
        "path": "https://webcams.nyctmc.org/api/cameras/8a6bc417-4877-4ebe-8052-88c1b261baf1/image",
        "label": "NYC Central Park West",
    },
    {
        "path": "https://webcams.nyctmc.org/api/cameras/ecba28cb-ac70-4d25-abcb-6506111ea120/image",
        "label": "NYC FDR at Brooklyn Bridge",
    },
    {
        "path": "https://webcams.nyctmc.org/api/cameras/332f161d-47cb-4c8a-b6b6-5ad48a55c978/image",
        "label": "NYC Central Park South",
    },
    {
        "path": "https://webcams.nyctmc.org/api/cameras/7d06c900-a5e5-49ca-96b9-93a0662a2069/image",
        "label": "NYC Verrazano Bridge",
    },
    {
        "path": "https://webcams.nyctmc.org/api/cameras/0f3b6031-fe36-43df-b2c7-6120e0580309/image",
        "label": "NYC Brooklyn Bridge walkway",
    },
]


@dataclass
class AppConfig:
    camera: CameraConfig
    pipeline: PipelineConfig
    motion: MotionConfig
    detection: DetectionConfig
    events: EventsConfig
    server: ServerConfig
    vision: VisionConfig
    root: Path = ROOT

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def clips_dir(self) -> Path:
        return self.data_dir / "clips"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "events.db"

    @property
    def web_dir(self) -> Path:
        return self.root / "web"

    def resolved_source(self) -> str | int:
        source = self.camera.source
        if isinstance(source, int):
            return source
        path = Path(source)
        if path.is_absolute():
            return str(path)
        # Treat URLs as-is; relative paths are relative to project root.
        if "://" in source:
            return source
        return str(self.root / path)


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

    camera_raw = raw.get("camera") or {}
    pipeline_raw = raw.get("pipeline") or {}
    motion_raw = raw.get("motion") or {}
    detection_raw = raw.get("detection") or {}
    events_raw = raw.get("events") or {}
    server_raw = raw.get("server") or {}
    vision_raw = raw.get("vision") or {}

    cfg = AppConfig(
        camera=CameraConfig(
            source=_as_source(camera_raw.get("source", "data/samples/city.mp4")),
            width=int(camera_raw.get("width", 1280)),
            height=int(camera_raw.get("height", 720)),
            loop_file=bool(camera_raw.get("loop_file", True)),
        ),
        pipeline=PipelineConfig(
            detect_fps=float(pipeline_raw.get("detect_fps", 5)),
            live_fps=float(pipeline_raw.get("live_fps", 10)),
            jpeg_quality=int(pipeline_raw.get("jpeg_quality", 80)),
        ),
        motion=MotionConfig(
            min_area=int(motion_raw.get("min_area", 1500)),
            threshold=int(motion_raw.get("threshold", 25)),
            blur_ksize=int(motion_raw.get("blur_ksize", 21)),
        ),
        detection=DetectionConfig(
            model=str(detection_raw.get("model", "yolov8n.pt")),
            conf=float(detection_raw.get("conf", 0.4)),
            classes=list(detection_raw.get("classes") or OUTDOOR_CLASSES),
            device=str(detection_raw.get("device", "cpu")),
        ),
        events=EventsConfig(
            pre_seconds=float(events_raw.get("pre_seconds", 2)),
            post_seconds=float(events_raw.get("post_seconds", 4)),
            cooldown_seconds=float(events_raw.get("cooldown_seconds", 8)),
            retention_days=int(events_raw.get("retention_days", 7)),
            clip_fps=float(events_raw.get("clip_fps", 10)),
        ),
        server=ServerConfig(
            host=str(server_raw.get("host", "0.0.0.0")),
            port=int(server_raw.get("port", 8000)),
        ),
        vision=VisionConfig(
            enabled=bool(vision_raw.get("enabled", True)),
            provider=str(vision_raw.get("provider", "auto")),
            ollama_url=str(vision_raw.get("ollama_url", "http://127.0.0.1:11434")),
            ollama_model=str(vision_raw.get("ollama_model", "moondream")),
            openai_model=str(vision_raw.get("openai_model", "gpt-4o-mini")),
            timeout_seconds=float(vision_raw.get("timeout_seconds", 25)),
        ),
        root=ROOT,
    )
    overlay = _load_overlay()
    camera_overlay = overlay.get("camera") or {}
    detection_overlay = overlay.get("detection") or {}
    if "source" in camera_overlay:
        cfg.camera.source = _as_source(camera_overlay["source"])
    if "model" in detection_overlay:
        cfg.detection.model = str(detection_overlay["model"]).strip() or cfg.detection.model
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.clips_dir.mkdir(parents=True, exist_ok=True)
    cfg.thumbs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _load_overlay() -> dict[str, Any]:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_runtime_settings(*, source: str | int, model: str) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera": {"source": source},
        "detection": {"model": model},
    }
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_settings_update(source: str, model: str) -> tuple[str | int, str]:
    source_text = (source or "").strip()
    model_text = (model or "").strip()
    if not source_text:
        raise SettingsError("source is required")
    if not model_text:
        raise SettingsError("model is required")
    if any(ch in model_text for ch in "\n\r"):
        raise SettingsError("invalid model")
    parsed = _as_source(source_text)
    if isinstance(parsed, str) and "://" not in parsed:
        path = Path(parsed)
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise SettingsError(f"source file not found: {resolved}")
    return parsed, model_text


def listed_samples() -> list[dict[str, str]]:
    folder = ROOT / "data" / "samples"
    items: list[dict[str, str]] = []
    for name, label in SAMPLE_LABELS.items():
        path = folder / name
        if path.is_file() and path.stat().st_size > 10_000:
            items.append({"path": f"data/samples/{name}", "label": label})
    return items


def public_settings(cfg: AppConfig) -> dict[str, Any]:
    return {
        "source": str(cfg.camera.source),
        "model": cfg.detection.model,
        "suggested_models": SUGGESTED_MODELS,
        "suggested_sources": listed_samples(),
        "suggested_webcams": PUBLIC_WEBCAMS,
        "vision": {
            "enabled": cfg.vision.enabled,
            "provider": cfg.vision.provider,
        },
    }
