from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.anomaly import classify_anomaly
from app.config import AppConfig
from app.db import EventStore, utc_now
from app.detect import Detection, ObjectDetector, draw_overlay
from app.motion import MotionDetector
from app.record import ClipWriter, cleanup_old_events, save_thumb
from app.vision import describe_event, fallback_summary

log = logging.getLogger(__name__)
_SNAPSHOT_UA = "TundraNVR/0.5"


def _encode_jpeg(frame: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


@dataclass
class PipelineStatus:
    running: bool = False
    opened: bool = False
    source: str = ""
    ingest_fps: float = 0.0
    last_motion: bool = False
    motion_area: int = 0
    last_detections: list[dict] = field(default_factory=list)
    last_scene: str = ""
    last_anomaly: str = ""
    last_error: str | None = None
    reconnects: int = 0
    started_at: float = 0.0


@dataclass
class _ActiveEvent:
    started_at: float
    last_match_at: float
    classes: set[str]
    score: float
    writer: ClipWriter
    clip_name: str
    thumb_name: str
    ts_start: str


class Pipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.motion = MotionDetector(
            min_area=cfg.motion.min_area,
            threshold=cfg.motion.threshold,
            blur_ksize=cfg.motion.blur_ksize,
        )
        self.detector = ObjectDetector(
            model=cfg.detection.model,
            conf=cfg.detection.conf,
            classes=cfg.detection.classes,
            device=cfg.detection.device,
            drone_model=cfg.detection.drone_model,
            drone_conf=cfg.detection.drone_conf,
        )
        self.store = EventStore(cfg.db_path)
        self.status = PipelineStatus(source=str(cfg.resolved_source()))
        self._lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._latest_jpeg = b""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active: _ActiveEvent | None = None
        self._cooldown_until = 0.0
        overlay_ttl = 1.0 / max(cfg.pipeline.detect_fps, 1.0)
        self._overlay_until = 0.0
        self._last_dets: list[Detection] = []
        self._has_motion = False
        self._motion_area = 0
        clip_fps = max(1.0, cfg.events.clip_fps)
        ring_len = max(1, int(cfg.events.pre_seconds * clip_fps) + 1)
        self._ring: deque[np.ndarray] = deque(maxlen=ring_len)
        self._overlay_ttl = max(0.4, overlay_ttl)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.status.running = True
        self.status.started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="nvr-pipeline", daemon=True)
        self._thread.start()
        log.info("Pipeline thread started")

    def stop(self) -> None:
        self._stop.set()
        self.status.running = False
        if self._thread:
            self._thread.join(timeout=8)
        with self._event_lock:
            self._finalize_event()
        log.info("Pipeline stopped")

    def latest_jpeg(self) -> bytes:
        with self._lock:
            return self._latest_jpeg

    def health(self) -> dict:
        with self._lock:
            uptime = time.monotonic() - self.status.started_at if self.status.started_at else 0.0
            return {
                "status": "ok" if self.status.running else "stopped",
                "source": self.status.source,
                "opened": self.status.opened,
                "fps": round(self.status.ingest_fps, 2),
                "last_motion": self.status.last_motion,
                "motion_area": self.status.motion_area,
                "last_detections": list(self.status.last_detections),
                "last_scene": self.status.last_scene,
                "last_anomaly": self.status.last_anomaly,
                "last_error": self.status.last_error,
                "reconnects": self.status.reconnects,
                "uptime_s": round(uptime, 1),
            }

    def _run(self) -> None:
        try:
            self.detector.load()
        except Exception as exc:
            log.exception("Failed to load detector: %s", exc)
            with self._lock:
                self.status.last_error = f"detector: {exc}"

        cleanup_old_events(self.store, self.cfg.data_dir, self.cfg.events.retention_days)
        backoff = 1.0
        while not self._stop.is_set():
            snapshot = self._is_http_snapshot()
            cap = None if snapshot else self._open_capture()
            if not snapshot and (cap is None or not cap.isOpened()):
                with self._lock:
                    self.status.opened = False
                    self.status.last_error = "capture not opened"
                    self.status.reconnects += 1
                log.warning("Capture failed; retrying in %.1fs", backoff)
                self._stop.wait(backoff)
                backoff = min(15.0, backoff * 2)
                continue
            backoff = 1.0
            with self._lock:
                self.status.opened = True
                self.status.last_error = None
                self.status.source = str(self.cfg.resolved_source())
            log.info("Capture opened: %s", self.status.source)
            self.motion.reset()
            try:
                self._ingest_loop(cap)
            finally:
                if cap is not None:
                    cap.release()
                with self._lock:
                    self.status.opened = False
            if not self._stop.is_set():
                with self._lock:
                    self.status.reconnects += 1
                log.warning("Capture disconnected; watchdog restarting")
                self._stop.wait(2.0)

    def _open_capture(self) -> cv2.VideoCapture | None:
        source = self.cfg.resolved_source()
        with self._lock:
            self.status.source = str(source)
        cap = cv2.VideoCapture(source)
        if self.cfg.camera.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
        if self.cfg.camera.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
        return cap

    def _ingest_loop(self, cap: cv2.VideoCapture | None) -> None:
        detect_interval = 1.0 / max(self.cfg.pipeline.detect_fps, 0.1)
        live_interval = 1.0 / max(self.cfg.pipeline.live_fps, 1.0)
        clip_interval = 1.0 / max(self.cfg.events.clip_fps, 1.0)
        last_detect = 0.0
        last_live = 0.0
        last_clip = 0.0
        last_fps_t = time.monotonic()
        frames = 0
        consecutive_fail = 0
        loop_fails = 0
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) if cap is not None else 0.0
        pace_file = self._is_file_source() and native_fps > 1
        frame_interval = 1.0 / native_fps if pace_file else 0.0
        next_frame_at = time.monotonic()
        snapshot = self._is_http_snapshot()

        while not self._stop.is_set():
            if snapshot:
                frame = self._read_http_snapshot()
                ok = frame is not None
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_fail += 1
                if snapshot:
                    if consecutive_fail >= 8:
                        log.warning("Snapshot fetch failed %s times", consecutive_fail)
                        return
                    if self._stop.wait(1.5):
                        return
                    continue
                if self._is_file_source() and self.cfg.camera.loop_file:
                    loop_fails += 1
                    log.debug("End of file; looping source")
                    if loop_fails > 3 or not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        return
                    self.motion.reset()
                    consecutive_fail = 0
                    continue
                if self._is_file_source():
                    log.warning("File source ended")
                    return
                if consecutive_fail >= 8:
                    log.warning("Capture read failed %s times", consecutive_fail)
                    return
                time.sleep(0.05)
                continue
            consecutive_fail = 0
            loop_fails = 0
            frame = self._resize(frame)
            now = time.monotonic()
            frames += 1
            if now - last_fps_t >= 1.0:
                with self._lock:
                    self.status.ingest_fps = frames / (now - last_fps_t)
                frames = 0
                last_fps_t = now

            if now - last_live >= live_interval:
                last_live = now
                self._publish_live(frame, now)

            if now - last_clip >= clip_interval:
                last_clip = now
                self._ring.append(frame.copy())
                with self._event_lock:
                    if self._active is not None:
                        self._active.writer.write(frame)
                        if now - self._active.last_match_at >= self.cfg.events.post_seconds:
                            self._finalize_event()

            if now - last_detect >= detect_interval:
                last_detect = now
                self._process_detect(frame, now)

            if snapshot:
                if self._stop.wait(1.5):
                    break
            elif frame_interval:
                next_frame_at += frame_interval
                delay = next_frame_at - time.monotonic()
                if delay > 0:
                    if self._stop.wait(delay):
                        break
                else:
                    next_frame_at = time.monotonic()

    def _is_file_source(self) -> bool:
        source = self.cfg.resolved_source()
        if isinstance(source, int):
            return False
        if "://" in str(source):
            return False
        return Path(source).is_file()

    def _is_http_snapshot(self) -> bool:
        source = self.cfg.resolved_source()
        if isinstance(source, int):
            return False
        text = str(source).lower()
        if "://" not in text:
            return False
        return text.endswith((".jpg", ".jpeg", ".png")) or text.rstrip("/").endswith("/image")

    def _read_http_snapshot(self) -> np.ndarray | None:
        source = self.cfg.resolved_source()
        url = str(source)
        req = urllib.request.Request(url, headers={"User-Agent": _SNAPSHOT_UA})
        try:
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            log.debug("Snapshot fetch failed: %s", exc)
            with self._lock:
                self.status.last_error = f"snapshot: {exc}"
            return None
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            with self._lock:
                self.status.last_error = "snapshot: could not decode image"
            return None
        return frame

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        target_w, target_h = self.cfg.camera.width, self.cfg.camera.height
        if not target_w or not target_h:
            return frame
        if w == target_w and h == target_h:
            return frame
        return cv2.resize(frame, (target_w, target_h))

    def _publish_live(self, frame: np.ndarray, now: float) -> None:
        dets = self._last_dets if now <= self._overlay_until else []
        vis = draw_overlay(frame, dets, self._motion_area, self._has_motion)
        jpeg = _encode_jpeg(vis, self.cfg.pipeline.jpeg_quality)
        with self._lock:
            self._latest_jpeg = jpeg
            self.status.last_motion = self._has_motion
            self.status.motion_area = self._motion_area

    def _process_detect(self, frame: np.ndarray, now: float) -> None:
        has_motion, area = self.motion.measure(frame)
        self._has_motion = has_motion
        self._motion_area = area
        detections: list[Detection] = []
        if has_motion:
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                log.exception("Detection failed: %s", exc)
                with self._lock:
                    self.status.last_error = f"detect: {exc}"
        if detections:
            self._last_dets = detections
            self._overlay_until = now + self._overlay_ttl
            payload = [{"cls": d.cls, "conf": round(d.conf, 3)} for d in detections]
            classes = [d.cls for d in detections]
            anomaly = classify_anomaly(
                classes,
                expected=self.cfg.monitoring.expected_classes,
                alert=self.cfg.monitoring.alert_classes,
                unattended_bags=self.cfg.monitoring.unattended_bags,
            )
            scene = fallback_summary(sorted(set(classes)), max(d.conf for d in detections), anomaly.reason)
            with self._lock:
                self.status.last_detections = payload
                self.status.last_scene = scene
                self.status.last_anomaly = anomaly.reason
            self._on_match(frame, detections, now)
        else:
            with self._lock:
                if now > self._overlay_until:
                    self.status.last_detections = []

    def _on_match(self, frame: np.ndarray, detections: list[Detection], now: float) -> None:
        classes = {d.cls for d in detections}
        score = max(d.conf for d in detections)
        with self._event_lock:
            if self._active is not None:
                self._active.last_match_at = now
                self._active.classes.update(classes)
                self._active.score = max(self._active.score, score)
                return
            if now < self._cooldown_until:
                return
            self._start_event(frame, detections, now, classes, score)

    def _start_event(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        now: float,
        classes: set[str],
        score: float,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        clip_name = f"{stamp}.mp4"
        thumb_name = f"{stamp}.jpg"
        clip_path = self.cfg.clips_dir / clip_name
        thumb_path = self.cfg.thumbs_dir / thumb_name
        vis = draw_overlay(frame, detections, self._motion_area, True)
        try:
            save_thumb(vis, thumb_path, self.cfg.pipeline.jpeg_quality)
            writer = ClipWriter(
                clip_path,
                self.cfg.camera.width,
                self.cfg.camera.height,
                self.cfg.events.clip_fps,
            )
        except Exception:
            log.exception("Failed to start clip/thumb for event")
            return
        for buffered in list(self._ring):
            writer.write(buffered)
        writer.write(frame)
        self._active = _ActiveEvent(
            started_at=now,
            last_match_at=now,
            classes=set(classes),
            score=score,
            writer=writer,
            clip_name=clip_name,
            thumb_name=thumb_name,
            ts_start=utc_now(),
        )
        log.info("Event started classes=%s score=%.2f", sorted(classes), score)

    def _finalize_event(self) -> None:
        active = self._active
        if active is None:
            return
        self._active = None
        try:
            active.writer.close()
        except Exception:
            log.exception("Error closing clip writer")
        event_id = None
        classes = sorted(active.classes)
        score = round(active.score, 3)
        anomaly = classify_anomaly(
            classes,
            expected=self.cfg.monitoring.expected_classes,
            alert=self.cfg.monitoring.alert_classes,
            unattended_bags=self.cfg.monitoring.unattended_bags,
        )
        try:
            event_id = self.store.insert(
                ts_start=active.ts_start,
                ts_end=utc_now(),
                classes=classes,
                score=score,
                thumb_path=f"thumbs/{active.thumb_name}",
                clip_path=f"clips/{active.clip_name}",
                anomaly=anomaly.flagged,
                anomaly_reason=anomaly.reason,
            )
        except Exception:
            log.exception("Failed to persist event")
        self._cooldown_until = time.monotonic() + self.cfg.events.cooldown_seconds
        log.info(
            "Event saved %s anomaly=%s %s",
            active.clip_name,
            anomaly.flagged,
            anomaly.reason,
        )
        if event_id:
            thumb = self.cfg.thumbs_dir / active.thumb_name
            threading.Thread(
                target=self._describe_event,
                args=(event_id, thumb, classes, score, anomaly.reason),
                name=f"nvr-vision-{event_id}",
                daemon=True,
            ).start()
        try:
            cleanup_old_events(self.store, self.cfg.data_dir, self.cfg.events.retention_days)
        except Exception:
            log.exception("Retention cleanup failed")

    def _describe_event(
        self,
        event_id: int,
        thumb: Path,
        classes: list[str],
        score: float,
        anomaly_reason: str,
    ) -> None:
        try:
            summary, provider = describe_event(
                self.cfg.vision, thumb, classes, score, anomaly_reason
            )
            self.store.update_summary(event_id, summary)
            log.info("Event %s described via %s", event_id, provider)
        except Exception:
            log.exception("Vision summary failed for event %s", event_id)
