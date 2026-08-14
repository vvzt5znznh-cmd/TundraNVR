from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Detection:
    cls: str
    conf: float
    xyxy: tuple[int, int, int, int]


class ObjectDetector:
    """Ultralytics YOLO nano with a small class allowlist."""

    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.4,
        classes: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model
        self.conf = conf
        self.allowed = {name.lower() for name in (classes or ["person", "car", "dog", "cat"])}
        self.device = device
        self._model = None
        self._names: dict[int, str] = {}
        self._class_ids: list[int] | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        log.info("Loading YOLO model %s on %s", self.model_name, self.device)
        self._model = YOLO(self.model_name)
        raw_names = self._model.names
        self._names = {int(k): str(v) for k, v in raw_names.items()}
        self._class_ids = [
            idx for idx, name in self._names.items() if name.lower() in self.allowed
        ]
        if not self._class_ids:
            log.warning("No YOLO classes matched allowlist %s", sorted(self.allowed))
            self._class_ids = None
        log.info("YOLO ready; filtering classes %s", self._class_ids)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self.load()
        kwargs: dict = {
            "source": frame,
            "conf": self.conf,
            "device": self.device,
            "verbose": False,
        }
        if self._class_ids:
            kwargs["classes"] = self._class_ids
        results = self._model.predict(**kwargs)
        detections: list[Detection] = []
        if not results:
            return detections
        result = results[0]
        if result.boxes is None:
            return detections
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = self._names.get(cls_id, str(cls_id)).lower()
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
    color = (0, 200, 255)
    for det in detections:
        x1, y1, x2, y2 = det.xyxy
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
