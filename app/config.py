from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import yaml

from app.security import ALLOWED_SOURCE_SCHEMES, redact_source, source_scheme

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"
SETTINGS_PATH = ROOT / "data" / "settings.json"

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
    source: str | int = 0
    width: int = 1280
    height: int = 720
    loop_file: bool = False
    name: str = "building"


@dataclass
class PipelineConfig:
    detect_fps: float = 5.0
    live_fps: float = 10.0
    jpeg_quality: int = 80
    idle_detect_seconds: float = 8.0


@dataclass
class MotionConfig:
    min_area: int = 1500
    threshold: int = 25
    blur_ksize: int = 21


@dataclass
class DetectionConfig:
    model: str = "yolov8n.pt"
    conf: float = 0.5
    classes: list[str] = field(default_factory=lambda: list(BUILDING_CLASSES))
    device: str = "cpu"
    drone_model: str = ""
    drone_conf: float = 0.55


@dataclass
class MonitoringConfig:
    expected_classes: list[str] = field(default_factory=lambda: list(EXPECTED_CLASSES))
    alert_classes: list[str] = field(default_factory=lambda: list(ALERT_CLASSES))
    unattended_bags: bool = True
    bag_dwell_seconds: float = 8.0
    bag_person_radius: float = 120.0


@dataclass
class TrackingConfig:
    max_age_s: float = 15.0
    min_hits: int = 2
    iou_match: float = 0.3
    dedup_seconds: float = 8.0


@dataclass
class EventsConfig:
    pre_seconds: float = 2.0
    post_seconds: float = 4.0
    cooldown_seconds: float = 8.0
    retention_days: int = 7
    clip_fps: float = 10.0
    write_media: bool = True


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    api_token: str = ""


@dataclass
class VisionConfig:
    enabled: bool = True
    provider: str = "local"
    allow_cloud: bool = False
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "moondream"
    openai_model: str = "gpt-4o-mini"
    timeout_seconds: float = 25.0
    audit_rate: float = 0.03
    verify_fresh_seconds: float = 120.0
    policy: str = (
        "Fixed building camera. Alert on unattended bags and after-hours people "
        "with no badge in the last minute. Ordinary doorway traffic is normal."
    )


@dataclass
class FusionConfig:
    enabled: bool = True
    badge_window_seconds: float = 60.0
    fixture: str = "data/fusion/badges.jsonl"


@dataclass
class MqttSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    prefix: str = "tundranvr"


@dataclass
class EmbedConfig:
    enabled: bool = True
    knn: int = 5
    gate_alerts: bool = False


@dataclass
class EscalationConfig:
    """auto = recall while Verify is healthy, else pol_score.

    recall = any plausible Detect trip goes to Verify (eval / healthy suppressor).
    pol_score = legacy PoL gate (operational fallback when Verify is down).
    """

    mode: str = "auto"
    pol_score_min: float = 0.7


@dataclass
class TargetModels:
    """Roadmap model at each seat vs what this process actually loads."""

    edge: str = "OpenCV + Pattern of Life (no neural net)"
    node: str = "RF-DETR"
    hub: str = "Moondream 3"


@dataclass
class ZoneConfig:
    name: str
    polygon: list[list[float]]


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
    tracking: TrackingConfig
    fusion: FusionConfig
    mqtt: MqttSettings
    embed: EmbedConfig
    escalation: EscalationConfig
    targets: TargetModels
    zones: list[ZoneConfig] = field(default_factory=list)
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
        if "://" in source:
            return source
        return str(self.root / path)


def _escalation_mode(value: Any) -> str:
    mode = str(value or "auto").strip().lower()
    return mode if mode in {"auto", "recall", "pol_score"} else "auto"


def _zones(raw: list) -> list[ZoneConfig]:
    out: list[ZoneConfig] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        poly = item.get("polygon") or []
        if not name or not isinstance(poly, list):
            continue
        pts = []
        for p in poly:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                pts.append([float(p[0]), float(p[1])])
        if len(pts) >= 3:
            out.append(ZoneConfig(name=name, polygon=pts))
    return out


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
    tracking_raw = raw.get("tracking") or {}
    fusion_raw = raw.get("fusion") or {}
    mqtt_raw = raw.get("mqtt") or {}
    embed_raw = raw.get("embed") or {}
    escalation_raw = raw.get("escalation") or {}
    targets_raw = raw.get("targets") or {}

    provider = str(vision_raw.get("provider", "local")).strip().lower()
    if provider in {"auto", "none", "false"}:
        provider = "local"

    cfg = AppConfig(
        camera=CameraConfig(
            source=_as_source(camera_raw.get("source", 0)),
            width=int(camera_raw.get("width", 1280)),
            height=int(camera_raw.get("height", 720)),
            loop_file=bool(camera_raw.get("loop_file", False)),
            name=str(camera_raw.get("name") or "building"),
        ),
        pipeline=PipelineConfig(
            detect_fps=float(pipeline_raw.get("detect_fps", 5)),
            live_fps=float(pipeline_raw.get("live_fps", 10)),
            jpeg_quality=int(pipeline_raw.get("jpeg_quality", 80)),
            idle_detect_seconds=float(pipeline_raw.get("idle_detect_seconds", 8)),
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
            drone_model=str(detection_raw.get("drone_model") or ""),
            drone_conf=float(detection_raw.get("drone_conf", 0.55)),
        ),
        events=EventsConfig(
            pre_seconds=float(events_raw.get("pre_seconds", 2)),
            post_seconds=float(events_raw.get("post_seconds", 4)),
            cooldown_seconds=float(events_raw.get("cooldown_seconds", 8)),
            retention_days=int(events_raw.get("retention_days", 7)),
            clip_fps=float(events_raw.get("clip_fps", 10)),
            write_media=bool(events_raw.get("write_media", True)),
        ),
        server=ServerConfig(
            host=str(server_raw.get("host", "0.0.0.0")),
            port=int(server_raw.get("port", 8000)),
            api_token=str(
                os.environ.get("TUNDRANVR_API_TOKEN")
                or server_raw.get("api_token")
                or ""
            ).strip(),
        ),
        vision=VisionConfig(
            enabled=bool(vision_raw.get("enabled", True)),
            provider=provider,
            allow_cloud=bool(vision_raw.get("allow_cloud", False)),
            ollama_url=str(vision_raw.get("ollama_url", "http://127.0.0.1:11434")),
            ollama_model=str(vision_raw.get("ollama_model", "moondream")),
            openai_model=str(vision_raw.get("openai_model", "gpt-4o-mini")),
            timeout_seconds=float(vision_raw.get("timeout_seconds", 25)),
            audit_rate=float(vision_raw.get("audit_rate", 0.03)),
            verify_fresh_seconds=float(vision_raw.get("verify_fresh_seconds", 120)),
            policy=str(vision_raw.get("policy") or VisionConfig.policy),
        ),
        monitoring=MonitoringConfig(
            expected_classes=list(monitoring_raw.get("expected_classes") or EXPECTED_CLASSES),
            alert_classes=list(monitoring_raw.get("alert_classes") or ALERT_CLASSES),
            unattended_bags=bool(monitoring_raw.get("unattended_bags", True)),
            bag_dwell_seconds=float(monitoring_raw.get("bag_dwell_seconds", 8)),
            bag_person_radius=float(monitoring_raw.get("bag_person_radius", 120)),
        ),
        tracking=TrackingConfig(
            max_age_s=float(tracking_raw.get("max_age_s") or tracking_raw.get("max_age") or 15),
            min_hits=int(tracking_raw.get("min_hits", 2)),
            iou_match=float(tracking_raw.get("iou_match", 0.3)),
            dedup_seconds=float(tracking_raw.get("dedup_seconds", 8)),
        ),
        fusion=FusionConfig(
            enabled=bool(fusion_raw.get("enabled", True)),
            badge_window_seconds=float(fusion_raw.get("badge_window_seconds", 60)),
            fixture=str(fusion_raw.get("fixture") or "data/fusion/badges.jsonl"),
        ),
        mqtt=MqttSettings(
            enabled=bool(mqtt_raw.get("enabled", False)),
            host=str(mqtt_raw.get("host", "127.0.0.1")),
            port=int(mqtt_raw.get("port", 1883)),
            prefix=str(mqtt_raw.get("prefix") or "tundranvr"),
        ),
        embed=EmbedConfig(
            enabled=bool(embed_raw.get("enabled", True)),
            knn=int(embed_raw.get("knn", 5)),
            gate_alerts=bool(embed_raw.get("gate_alerts", False)),
        ),
        escalation=EscalationConfig(
            mode=_escalation_mode(escalation_raw.get("mode")),
            pol_score_min=float(escalation_raw.get("pol_score_min", 0.7)),
        ),
        targets=TargetModels(
            edge=str(targets_raw.get("edge") or TargetModels.edge),
            node=str(targets_raw.get("node") or TargetModels.node),
            hub=str(targets_raw.get("hub") or TargetModels.hub),
        ),
        zones=_zones(raw.get("zones") or []),
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
    (cfg.data_dir / "fusion").mkdir(parents=True, exist_ok=True)
    (cfg.data_dir / "eval").mkdir(parents=True, exist_ok=True)
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


def parse_settings_update(source: str) -> str | int:
    source_text = (source or "").strip()
    if not source_text:
        raise SettingsError("source is required")
    if "***" in source_text:
        raise SettingsError("redacted source is not usable; paste the full URL")
    parsed = _as_source(source_text)
    if isinstance(parsed, str) and "://" in parsed:
        scheme = source_scheme(parsed)
        if scheme not in ALLOWED_SOURCE_SCHEMES:
            raise SettingsError(f"unsupported source scheme: {scheme or 'empty'}")
    if isinstance(parsed, str) and "://" not in parsed:
        path = Path(parsed)
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise SettingsError(f"source file not found: {resolved}")
    return parsed


def public_settings(cfg: AppConfig) -> dict[str, Any]:
    from app.vision import effective_provider

    return {
        "source": redact_source(cfg.camera.source),
        "model": cfg.detection.model,
        "vision": effective_provider(cfg.vision) if cfg.vision.enabled else "off",
        "allow_cloud": bool(cfg.vision.allow_cloud),
        "auth_required": bool(cfg.server.api_token),
        "escalation": cfg.escalation.mode,
        "targets": {
            "edge": cfg.targets.edge,
            "node": cfg.targets.node,
            "hub": cfg.targets.hub,
        },
    }
