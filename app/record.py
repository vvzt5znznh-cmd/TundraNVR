from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from app.db import EventStore

log = logging.getLogger(__name__)


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


class ClipWriter:
    """Write BGR frames to an H.264 MP4 via ffmpeg, with OpenCV fallback."""

    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        self.path = path
        self.width = max(2, _even(width))
        self.height = max(2, _even(height))
        self.fps = max(1.0, float(fps))
        self._proc: subprocess.Popen | None = None
        self._cv: cv2.VideoWriter | None = None
        self._use_ffmpeg = shutil.which("ffmpeg") is not None
        if self._use_ffmpeg:
            self._start_ffmpeg()
        else:
            self._start_cv()

    def _start_ffmpeg(self) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin not available")

    def _start_cv(self) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._cv = cv2.VideoWriter(str(self.path), fourcc, self.fps, (self.width, self.height))
        if not self._cv.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {self.path}")

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            return cv2.resize(frame, (self.width, self.height))
        return frame

    def write(self, frame: np.ndarray) -> None:
        frame = self._resize(frame)
        if self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
                return
            except BrokenPipeError:
                log.warning("ffmpeg pipe broke; remaining frames dropped for %s", self.path)
                return
        if self._cv is not None:
            self._cv.write(frame)

    def close(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
                self._proc.wait(timeout=30)
            except Exception:
                self._proc.kill()
            finally:
                self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None


def save_thumb(frame: np.ndarray, path: Path, quality: int = 80) -> None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode thumbnail JPEG")
    path.write_bytes(buf.tobytes())


def cleanup_old_events(store: EventStore, data_dir: Path, retention_days: int) -> int:
    if retention_days <= 0:
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
    cutoff_iso = cutoff.isoformat()
    removed = 0
    for event in store.older_than(cutoff_iso):
        for rel in (event.get("thumb_path"), event.get("clip_path")):
            if not rel:
                continue
            path = data_dir / rel
            try:
                if path.exists():
                    path.unlink()
            except OSError as exc:
                log.warning("Could not delete %s: %s", path, exc)
        store.delete(int(event["id"]))
        removed += 1
    if removed:
        log.info("Retention removed %s event(s) older than %s days", removed, retention_days)
    return removed
