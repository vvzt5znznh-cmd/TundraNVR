from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import cv2
import numpy as np

from app.anomaly import DRONE_ALIASES, normalize_class
from app.config import ROOT

log = logging.getLogger(__name__)
ALERT_OVERLAY = frozenset({"drone", "airplane"})


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple[int, int, int, int]


def _resolve_model(name: str) -> str:
    text = (name or "").strip()
    if not text:
        return text
    path = Path(text)
    if path.is_file():
        return str(path)
    rooted = ROOT / path
    if rooted.is_file():
        return str(rooted)
    return text


class ObjectDetector:
    """YOLO detector plus an optional second model for drones."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.4,
        classes: list[str] | None = None,
        device: str = "cpu",
        drone_model: str = "",
        drone_conf: float = 0.55,
    ) -> None:
        self.model_name = model
        self.conf = conf
        self.allowed = {normalize_class(name) for name in (classes or ["person", "car", "dog", "cat"])}
        self.allowed.add("drone")
        self.device = device
        self.drone_model_name = (drone_model or "").strip()
        self.drone_conf = drone_conf
        self._model = None
        self._drone_model = None
        self._names: dict[int, str] = {}
        self._drone_names: dict[int, str] = {}
        self._class_ids: list[int] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        primary = _resolve_model(self.model_name)
        log.info("Loading YOLO model %s on %s", primary, self.device)
        self._model = YOLO(primary)
        raw_names = self._model.names
        self._names = {int(k): str(v) for k, v in raw_names.items()}
        self._class_ids = [
            idx
            for idx, name in self._names.items()
            if normalize_class(name) in self.allowed
        ]
        if not self._class_ids:
            log.warning("No YOLO classes matched allowlist %s", sorted(self.allowed))
            self._class_ids = None
        log.info("YOLO ready; filtering classes %s", self._class_ids)

        if not self.drone_model_name:
            return
        drone_path = _resolve_model(self.drone_model_name)
        if not Path(drone_path).is_file():
            log.warning("Drone model not found at %s; skipping drone detector", drone_path)
            return
        log.info("Loading drone model %s", drone_path)
        self._drone_model = YOLO(drone_path)
        raw_drone = self._drone_model.names
        self._drone_names = {int(k): str(v) for k, v in raw_drone.items()}
        log.info("Drone model classes %s", list(self._drone_names.values()))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.load()
        detections = self._predict(
            self._model,
            frame,
            self.conf,
            self._names,
            class_ids=self._class_ids,
        )
        if self._drone_model is not None:
            drones = self._predict(
                self._drone_model,
                frame,
                self.drone_conf,
                self._drone_names,
                class_ids=None,
                force_drone=True,
            )
            detections.extend(self._filter_drone_overlaps(detections, drones))
        return detections

    @staticmethod
    def _filter_drone_overlaps(
        detections: list[Detection], drones: list[Detection]
    ) -> list[Detection]:
        """Drop drone boxes that sit on a person — common indoor false positives."""
        people = [d for d in detections if d.cls == "person"]
        if not people:
            return drones
        kept: list[Detection] = []
        for drone in drones:
            if any(_iou(drone.xyxy, person.xyxy) >= 0.25 for person in people):
                continue
            kept.append(drone)
        return kept

    def _predict(
        self,
        model,
        frame: np.ndarray,
        conf: float,
        names: dict[int, str],
        class_ids: list[int] | None,
        force_drone: bool = False,
    ) -> list[Detection]:
        kwargs: dict = {
            "source": frame,
            "conf": conf,
            "device": self.device,
            "verbose": False,
        }
        if class_ids:
            kwargs["classes"] = class_ids
        results = model.predict(**kwargs)
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls_id = int(box.cls[0])
            raw = names.get(cls_id, str(cls_id)).lower()
            name = "drone" if force_drone or raw in DRONE_ALIASES else normalize_class(raw)
            if name not in self.allowed:
                continue
            xyxy = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    cls=name,
                    conf=float(box.conf[0]),
                    xyxy=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                )
            )
        return detections


def draw_overlay(
    frame: np.ndarray,
    detections: list[Detection],
    motion_area: int | None = None,
    has_motion: bool = False,
) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = (0, 60, 255) if det.cls in ALERT_OVERLAY else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls} {det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
        cv2.putText(
            vis,
            label,
            (x1 + 3, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    status = f"motion={'yes' if has_motion else 'no'}"
    if motion_area is not None:
        status += f" area={motion_area}"
    cv2.putText(
        vis,
        status,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return vis
