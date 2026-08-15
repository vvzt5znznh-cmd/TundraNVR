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


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None


def boxes_payload(
    detections: list[Detection],
    frame_wh: tuple[int, int],
) -> list[dict]:
    """Normalized boxes so Events can mark where Detect spotted something."""
    w, h = int(frame_wh[0]), int(frame_wh[1])
    w = max(w, 1)
    h = max(h, 1)
    out: list[dict] = []
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        out.append(
            {
                "cls": det.cls,
                "conf": round(float(det.conf), 3),
                "track_id": det.track_id,
                "xyxy": [x1, y1, x2, y2],
                "xn": round(x1 / w, 4),
                "yn": round(y1 / h, 4),
                "wn": round(max(x2 - x1, 0) / w, 4),
                "hn": round(max(y2 - y1, 0) / h, 4),
            }
        )
    return out


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
    """YOLO namer. Call only on Edge trips — never as a motion sensor."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.4,
        classes: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model
        self.conf = conf
        self.allowed = {normalize_class(name) for name in (classes or ["person", "car", "dog", "cat"])}
        self.device = device
        self._model = None
        self._names: dict[int, str] = {}
        self._class_ids: list[int] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        primary = _resolve_model(self.model_name)
        log.info("Loading YOLO namer %s on %s", primary, self.device)
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
        log.info("YOLO namer ready; filtering classes %s", self._class_ids)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.load()
        return self._predict(self._model, frame, self.conf, self._names, self._class_ids)

    def _predict(
        self,
        model,
        frame: np.ndarray,
        conf: float,
        names: dict[int, str],
        class_ids: list[int] | None,
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
            name = "drone" if raw in DRONE_ALIASES else normalize_class(raw)
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


def draw_overlay(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
        color = (0, 60, 255) if det.cls in ALERT_OVERLAY else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.cls} {det.conf:.2f}"
        if det.track_id is not None:
            label = f"#{det.track_id} {label}"
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
    return vis
