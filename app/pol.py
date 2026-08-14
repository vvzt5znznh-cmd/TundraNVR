from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

VIS_W = 32
VIS_H = 18
GRID = 8
LEARN_GRID_N = 40  # confident = this many motion ticks on the occupancy grid
SAVE_EVERY = 8


def source_key(source: str | int) -> str:
    text = str(source).strip() or "unknown"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    name = text if "://" in text else Path(text).name
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")[:40] or "cam"
    return f"{safe}-{digest}"


def downsample(frame: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (VIS_W, VIS_H), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


@dataclass
class PolScore:
    unusual: bool
    score: float
    reason: str
    state: str
    samples: int
    visual_delta: float
    occupancy_novelty: float
    motion_spike: float
    grid: list[list[float]] = field(default_factory=list)
    usual_grid: list[list[float]] = field(default_factory=list)
    confident: bool = False
    why: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "unusual": self.unusual,
            "score": round(self.score, 3),
            "reason": self.reason,
            "why": self.why,
            "state": self.state,
            "samples": self.samples,
            "confident": self.confident,
            "visual_delta": round(self.visual_delta, 3),
            "occupancy_novelty": round(self.occupancy_novelty, 3),
            "motion_spike": round(self.motion_spike, 3),
            "grid": self.grid,
            "usual_grid": self.usual_grid,
        }


class PatternOfLife:
    """Per-camera baseline: time, occupancy grid, slow visual mean."""

    def __init__(self, directory: Path, source: str | int) -> None:
        self.directory = directory
        self.source = str(source)
        self.key = source_key(source)
        self.path = directory / f"{self.key}.json"
        self._lock = threading.Lock()
        self.samples = 0
        self.grid_n = 0.0
        self.grid_counts = np.zeros((GRID, GRID), dtype=np.float32)
        self.visual_mean = np.zeros((VIS_H, VIS_W), dtype=np.float32)
        self.visual_ready = False
        self.hour_sum = np.zeros(168, dtype=np.float64)
        self.hour_n = np.zeros(168, dtype=np.int32)
        self.motion_sum = 0.0
        self.motion_n = 0
        self._dirty = 0
        self.directory.mkdir(parents=True, exist_ok=True)
        self._load()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            ticks = int(self.grid_n)
            return {
                "source": self.source,
                "key": self.key,
                "samples": ticks,
                "learn_samples": LEARN_GRID_N,
                "progress": round(min(1.0, ticks / max(LEARN_GRID_N, 1)), 3),
                "state": "confident" if ticks >= LEARN_GRID_N else "learning",
                "confident": ticks >= LEARN_GRID_N,
                "grid_n": ticks,
                "frame_samples": self.samples,
            }

    def observe(
        self,
        frame: np.ndarray,
        grid: np.ndarray,
        motion_area: int,
        has_motion: bool,
    ) -> tuple[PolScore, dict[str, Any]]:
        visual = downsample(frame)
        hour = _week_hour()
        with self._lock:
            vis_delta = self._visual_delta(visual)
            occ = self._occupancy_novelty(grid) if has_motion else 0.0
            spike = self._motion_spike(motion_area) if has_motion else 0.0
            score = max(vis_delta, occ, spike)
            learning = self.grid_n < LEARN_GRID_N
            if learning and has_motion:
                unusual = True
                score = max(score, 0.45)
                reason = "learning this camera"
                why = (
                    "Edge is still mapping where this camera is usually busy. "
                    "Until that baseline is ready, every motion is sent to Detect."
                )
            elif not has_motion:
                unusual = False
                reason = "no motion"
                why = "No pixel change on this tick."
            else:
                unusual = score >= 0.48
                parts = []
                why_parts = []
                if occ >= 0.48:
                    parts.append("motion where this camera is usually still")
                    why_parts.append(
                        "Motion is in cells this camera usually leaves empty "
                        f"(place {occ:.2f})."
                    )
                if vis_delta >= 0.48:
                    parts.append("frame does not match this camera")
                    why_parts.append(
                        f"The frame no longer matches this camera’s usual look (look {vis_delta:.2f})."
                    )
                if spike >= 0.48:
                    parts.append("more motion than usual")
                    why_parts.append(
                        f"There is more motion than this camera usually sees (amount {spike:.2f})."
                    )
                reason = ", ".join(parts) if parts else "looks like this camera"
                why = (
                    " ".join(why_parts)
                    if why_parts
                    else "Motion sits on this camera’s usual footprint."
                )
            usual = self._usual_freq().tolist()
            features = {
                "grid": np.round(grid, 3).tolist(),
                "visual": np.round(visual, 4).tolist(),
                "motion_area": int(motion_area),
                "hour": hour,
            }
            result = PolScore(
                unusual=unusual,
                score=float(min(1.0, score)),
                reason=reason,
                state="learning" if learning else "confident",
                samples=int(self.grid_n),
                visual_delta=float(vis_delta),
                occupancy_novelty=float(occ),
                motion_spike=float(spike),
                grid=np.round(grid, 3).tolist(),
                usual_grid=np.round(np.array(usual), 3).tolist(),
                confident=not learning,
                why=why,
            )
            self._update_locked(
                visual,
                grid,
                motion_area,
                hour,
                unusual=unusual,
                force=False,
                has_motion=has_motion,
            )
        return result, features

    def absorb(self, features: dict[str, Any]) -> None:
        """Operator dismissed: this observation is ordinary for the camera."""
        grid = np.array(features.get("grid") or np.zeros((GRID, GRID)), dtype=np.float32)
        if grid.shape != (GRID, GRID):
            grid = np.zeros((GRID, GRID), dtype=np.float32)
        visual_raw = features.get("visual")
        if visual_raw is None:
            visual = None
        else:
            visual = np.array(visual_raw, dtype=np.float32)
            if visual.ndim == 1:
                if visual.size != VIS_H * VIS_W:
                    visual = None
                else:
                    visual = visual.reshape(VIS_H, VIS_W)
            elif visual.shape != (VIS_H, VIS_W):
                visual = None
        area = int(features.get("motion_area") or 0)
        hour = int(features.get("hour") or _week_hour())
        with self._lock:
            if visual is None:
                visual = self.visual_mean.copy()
            self._update_locked(
                visual,
                grid,
                area,
                hour,
                unusual=False,
                force=True,
                has_motion=True,
            )
            self._save_locked()
        log.info("PoL absorbed dismissal for %s", self.key)

    def close(self) -> None:
        with self._lock:
            self._save_locked()

    def _visual_delta(self, visual: np.ndarray) -> float:
        if not self.visual_ready:
            return 0.0
        delta = float(np.mean(np.abs(visual - self.visual_mean)))
        return float(np.clip(delta / 0.16, 0.0, 1.0))

    def _usual_freq(self) -> np.ndarray:
        if self.grid_n <= 0:
            return np.zeros((GRID, GRID), dtype=np.float32)
        return self.grid_counts / float(self.grid_n)

    def _occupancy_novelty(self, grid: np.ndarray) -> float:
        """How much of *this* frame's motion sits outside this camera's usual footprint."""
        motion = np.clip(grid, 0.0, 1.0)
        total = float(motion.sum())
        if total < 1e-6:
            return 0.0
        if self.grid_n < 20:
            return 0.35
        freq = self._usual_freq()
        peak = float(freq.max())
        if peak < 0.005:
            return 0.35
        # Cells this camera has actually used, relative to its own hottest cell.
        hot = freq >= max(0.2 * peak, 1e-4)
        on_hot = float((motion * hot).sum() / total)
        return float(np.clip(1.0 - on_hot, 0.0, 1.0))

    def _motion_spike(self, area: int) -> float:
        typical = self.motion_sum / max(self.motion_n, 1)
        if self.motion_n < 12:
            return 0.0
        ratio = area / max(typical, 1.0)
        return float(np.clip((ratio - 1.4) / 3.0, 0.0, 1.0))

    def _update_locked(
        self,
        visual: np.ndarray,
        grid: np.ndarray,
        area: int,
        hour: int,
        *,
        unusual: bool,
        force: bool,
        has_motion: bool,
    ) -> None:
        alpha = 0.22 if force else 0.02
        if not self.visual_ready:
            self.visual_mean = visual.copy()
            self.visual_ready = True
        else:
            self.visual_mean = (1.0 - alpha) * self.visual_mean + alpha * visual
        hour = max(0, min(167, hour))
        if has_motion:
            self.hour_sum[hour] += area
            self.hour_n[hour] += 1
            self.motion_sum += area
            self.motion_n += 1
            cell = np.clip(grid, 0.0, 1.0)
            learning = self.grid_n < LEARN_GRID_N
            if force or learning or not unusual:
                self.grid_counts += cell
                self.grid_n += 1.0
            else:
                # Repeating "unusual" motion must still be able to become usual.
                self.grid_counts += 0.12 * cell
                self.grid_n += 0.12
        self.samples += 1
        self._dirty += 1
        if force or self._dirty >= SAVE_EVERY:
            self._save_locked()
            self._dirty = 0

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("Could not read PoL profile %s", self.path)
            return
        self.samples = int(data.get("samples") or 0)
        self.grid_n = float(data.get("grid_n") or 0)
        self.grid_counts = _as_grid(data.get("grid_counts"))
        mean = data.get("visual_mean")
        if mean is not None:
            arr = np.array(mean, dtype=np.float32)
            if arr.size == VIS_H * VIS_W:
                self.visual_mean = arr.reshape(VIS_H, VIS_W)
                self.visual_ready = True
        self.hour_sum = np.array(data.get("hour_sum") or [0.0] * 168, dtype=np.float64)
        if self.hour_sum.size != 168:
            self.hour_sum = np.zeros(168, dtype=np.float64)
        self.hour_n = np.array(data.get("hour_n") or [0] * 168, dtype=np.int32)
        if self.hour_n.size != 168:
            self.hour_n = np.zeros(168, dtype=np.int32)
        self.motion_sum = float(data.get("motion_sum") or 0.0)
        self.motion_n = int(data.get("motion_n") or 0)
        log.info("Loaded PoL %s samples=%s", self.key, self.samples)

    def _save_locked(self) -> None:
        payload = {
            "source": self.source,
            "key": self.key,
            "samples": self.samples,
            "grid_n": self.grid_n,
            "grid_counts": np.round(self.grid_counts, 4).tolist(),
            "visual_mean": np.round(self.visual_mean, 4).tolist() if self.visual_ready else None,
            "hour_sum": np.round(self.hour_sum, 2).tolist(),
            "hour_n": self.hour_n.tolist(),
            "motion_sum": round(self.motion_sum, 1),
            "motion_n": self.motion_n,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def absorb_into_file(directory: Path, source: str, features: dict[str, Any]) -> None:
    profile = PatternOfLife(directory, source)
    profile.absorb(features)
    profile.close()


def _week_hour() -> int:
    now = datetime.now(timezone.utc)
    return now.weekday() * 24 + now.hour


def _as_grid(value: Any) -> np.ndarray:
    arr = np.array(value if value is not None else 0, dtype=np.float32)
    if arr.shape == (GRID, GRID):
        return arr
    return np.zeros((GRID, GRID), dtype=np.float32)
