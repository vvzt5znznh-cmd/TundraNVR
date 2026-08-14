from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def _as_source(value: Any) -> str | int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    return text


@dataclass
class CameraConfig:
    source: str | int = "data/sample.mp4"
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
    classes: list[str] = field(default_factory=lambda: ["person", "car", "dog", "cat"])
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
class AppConfig:
    camera: CameraConfig
    pipeline: PipelineConfig
    motion: MotionConfig
    detection: DetectionConfig
    events: EventsConfig
    server: ServerConfig
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

    cfg = AppConfig(
        camera=CameraConfig(
            source=_as_source(camera_raw.get("source", "data/sample.mp4")),
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
            classes=list(detection_raw.get("classes") or ["person", "car", "dog", "cat"]),
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
        root=ROOT,
    )
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.clips_dir.mkdir(parents=True, exist_ok=True)
    cfg.thumbs_dir.mkdir(parents=True, exist_ok=True)
    return cfg
