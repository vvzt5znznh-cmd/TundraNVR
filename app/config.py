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
]

DETECTOR_MODELS = [
    {
        "id": "yolov8n.pt",
        "name": "Fast",
        "family": "YOLOv8 Nano",
        "blurb": "People, cars, and bags at a doorway. Built for a laptop CPU and a live view.",
        "choose_when": "Default. One building camera, you want it snappy.",
        "recommended": True,
    },
    {
        "id": "yolov8s.pt",
        "name": "Balanced",
        "family": "YOLOv8 Small",
        "blurb": "Better at smaller or farther objects — a person down the corridor, a bag on the floor.",
        "choose_when": "Fast is missing things and the machine is not struggling.",
        "recommended": False,
    },
    {
        "id": "yolov8m.pt",
        "name": "Accurate",
        "family": "YOLOv8 Medium",
        "blurb": "Fewer misses on busy or overlapping frames. Boxes are a bit cleaner.",
        "choose_when": "Catching matters more than a smooth live view. Slow on CPU.",
        "recommended": False,
    },
    {
        "id": "yolo11n.pt",
        "name": "Newer fast",
        "family": "YOLO11 Nano",
        "blurb": "Same job as Fast, newer weights. Often a little sharper for the same speed class.",
        "choose_when": "You already like Fast and want the newer nano.",
        "recommended": False,
    },
]

SAMPLE_LABELS = {
    "entrance.mp4": "Building entrance — people at the door",
    "corridor.mp4": "Indoor corridor — people walking",
    "lobby.mp4": "Indoor lobby — people meeting",
    "indoor.mp4": "Indoor hall — pedestrians",
    "aisle.mp4": "Indoor aisle — retail CCTV",
    "parking.mp4": "Building parking — people, bicycles, cars",
    "package.mp4": "Indoor — bag left behind",
    "drone.mp4": "Quadcopter in view",
}

BUILDING_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bird",
    "cat",
    "dog",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "skateboard",
    "airplane",
    "drone",
]

EXPECTED_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "backpack",
    "handbag",
    "suitcase",
    "umbrella",
    "dog",
    "cat",
    "bird",
    "skateboard",
]

ALERT_CLASSES = ["drone", "airplane"]


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
    source: str | int = "data/samples/entrance.mp4"
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
    classes: list[str] = field(default_factory=lambda: list(BUILDING_CLASSES))
    device: str = "cpu"
    drone_model: str = "drone-yolo.pt"
    drone_conf: float = 0.55


@dataclass
class MonitoringConfig:
    expected_classes: list[str] = field(default_factory=lambda: list(EXPECTED_CLASSES))
    alert_classes: list[str] = field(default_factory=lambda: list(ALERT_CLASSES))
    unattended_bags: bool = True


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
    monitoring: MonitoringConfig
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
    def pol_dir(self) -> Path:
        return self.data_dir / "pol"

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
    monitoring_raw = raw.get("monitoring") or {}

    cfg = AppConfig(
        camera=CameraConfig(
            source=_as_source(camera_raw.get("source", "data/samples/entrance.mp4")),
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
            classes=list(detection_raw.get("classes") or BUILDING_CLASSES),
            device=str(detection_raw.get("device", "cpu")),
            drone_model=str(detection_raw.get("drone_model", "drone-yolo.pt")),
            drone_conf=float(detection_raw.get("drone_conf", 0.55)),
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
        monitoring=MonitoringConfig(
            expected_classes=list(monitoring_raw.get("expected_classes") or EXPECTED_CLASSES),
            alert_classes=list(monitoring_raw.get("alert_classes") or ALERT_CLASSES),
            unattended_bags=bool(monitoring_raw.get("unattended_bags", True)),
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
    cfg.pol_dir.mkdir(parents=True, exist_ok=True)
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
            title, _, detail = label.partition(" — ")
            items.append(
                {
                    "path": f"data/samples/{name}",
                    "label": label,
                    "title": title,
                    "detail": detail,
                }
            )
    return items


def public_settings(cfg: AppConfig) -> dict[str, Any]:
    vision_provider = cfg.vision.provider if cfg.vision.enabled else "off"
    if vision_provider == "auto":
        vision_blurb = (
            "Writes the log line. Tries a local caption model, then OpenAI if a key is set, "
            "otherwise a sentence from the detector labels."
        )
    elif vision_provider == "ollama":
        vision_blurb = "Local caption model (Ollama) describes the still after each event."
    elif vision_provider == "openai":
        vision_blurb = "OpenAI vision describes the still after each event."
    else:
        vision_blurb = "Scene notes use the detector labels only — no extra caption model."
    return {
        "source": str(cfg.camera.source),
        "model": cfg.detection.model,
        "suggested_models": SUGGESTED_MODELS,
        "detectors": DETECTOR_MODELS,
        "suggested_sources": listed_samples(),
        "suggested_webcams": PUBLIC_WEBCAMS,
        "vision": {
            "enabled": cfg.vision.enabled,
            "provider": vision_provider,
        },
        "stack": [
            {
                "name": "Drone finder",
                "status": "always on",
                "seat": "node",
                "blurb": (
                    "Always on next to the detector. Stock YOLO has no drone class; "
                    "this extra model looks for quadcopters and winged UAVs. "
                    "A drone prior can skip a quiet Edge and still reach Node."
                ),
            },
            {
                "name": "Scene notes",
                "status": vision_provider,
                "seat": "hub",
                "blurb": vision_blurb,
            },
        ],
        "monitoring": {
            "expected_classes": cfg.monitoring.expected_classes,
            "alert_classes": cfg.monitoring.alert_classes,
        },
        "escalation": [
            {
                "id": "edge",
                "name": "Raspberry",
                "question": "Anomaly or not?",
                "blurb": "Motion plus this camera’s Pattern of Life. Usual frames stay on the Pi.",
            },
            {
                "id": "node",
                "name": "Node",
                "question": "What is it?",
                "blurb": "Names people and things on Edge trips. Drones skip a quiet Edge.",
            },
            {
                "id": "hub",
                "name": "Hub",
                "question": "What is it doing?",
                "blurb": "Captions activity only when Node cannot close the packet.",
            },
        ],
    }
