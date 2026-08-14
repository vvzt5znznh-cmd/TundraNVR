from __future__ import annotations

import cv2
import numpy as np


class MotionDetector:
    """Frame-differencing motion gate with a minimum contour-area threshold."""

    def __init__(self, min_area: int = 1500, threshold: int = 25, blur_ksize: int = 21) -> None:
        self.min_area = min_area
        self.threshold = threshold
        k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
        self.blur_ksize = max(3, k)
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def measure(self, frame: np.ndarray) -> tuple[bool, int, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)
        empty = np.zeros((8, 8), dtype=np.float32)
        if self._prev is None:
            self._prev = gray
            return False, 0, empty

        delta = cv2.absdiff(self._prev, gray)
        self._prev = gray
        _, thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = int(sum(cv2.contourArea(contour) for contour in contours))
        grid = cv2.resize(thresh, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return area >= self.min_area, area, grid
