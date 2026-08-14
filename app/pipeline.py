from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig
from app.db import EventStore, utc_now
from app.detect import Detection, ObjectDetector, draw_edge_overlay, draw_overlay
from app.embed import EmbeddingIndex, thumb_hist
from app.escalate import decide_hub
from app.fusion import FusionBus, clock_context
from app.motion import MotionDetector
from app.mqtt_bus import MqttBus, MqttConfig
from app.pol import PatternOfLife, absorb_into_file
from app.record import ClipWriter, NullWriter, cleanup_old_events, save_thumb
from app.security import redact_source
from app.tiers import models_payload
from app.track import ByteTracker, Track, unattended_bags
from app.verify import verify_event
from app.vision import effective_provider, fallback_summary
from app.zones import Zone, ZoneMap

log = logging.getLogger(__name__)


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
    last_handoff: dict = field(default_factory=dict)
    last_error: str | None = None
    reconnects: int = 0
    started_at: float = 0.0
    yolo_ran: bool = False
    verifier_provider: str = ""


@dataclass
class _ActiveEvent:
    started_at: float
    last_match_at: float
    classes: set[str]
    score: float
    writer: ClipWriter | NullWriter
    clip_name: str
    thumb_name: str
    ts_start: str
    hub_needed: bool = False
    page_operator: bool = False
    anomaly_reason: str = ""
    pol_score: float = 0.0
    stopped_at: str = "node"
    handoff: dict = field(default_factory=dict)
    features: dict = field(default_factory=dict)
    source: str = ""
    track_id: int | None = None
    last_frame: np.ndarray | None = None
    last_dets: list[Detection] = field(default_factory=list)
    fusion: dict = field(default_factory=dict)
    t0_mono: float = 0.0


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
        )
        self.tracker = ByteTracker(
            max_age=cfg.tracking.max_age,
            min_hits=cfg.tracking.min_hits,
            iou_match=cfg.tracking.iou_match,
        )
        self.zones = ZoneMap(
            [Zone(z.name, [(p[0], p[1]) for p in z.polygon]) for z in cfg.zones],
            cfg.camera.width,
            cfg.camera.height,
        )
        fixture = Path(cfg.fusion.fixture)
        if not fixture.is_absolute():
            fixture = cfg.root / fixture
        self.fusion = FusionBus(
            fixture if cfg.fusion.enabled else None,
            window_seconds=cfg.fusion.badge_window_seconds,
        )
        self.mqtt = MqttBus(
            MqttConfig(
                enabled=cfg.mqtt.enabled,
                host=cfg.mqtt.host,
                port=cfg.mqtt.port,
                prefix=cfg.mqtt.prefix,
                camera=cfg.camera.name,
            )
        )
        self.store = EventStore(cfg.db_path)
        self.embeddings = EmbeddingIndex(self.store) if cfg.embed.enabled else None
        self.pol = PatternOfLife(cfg.pol_dir, cfg.camera.source)
        self.status = PipelineStatus(
            source=str(cfg.resolved_source()),
            verifier_provider=effective_provider(cfg.vision),
        )
        self._lock = threading.Lock()
        self._event_lock = threading.Lock()
        self._latest_jpeg = b""
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._active_by_track: dict[int, _ActiveEvent] = {}
        self._unnamed: _ActiveEvent | None = None
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
        self._jpeg_edge = b""
        self._jpeg_node = b""
        self._jpeg_hub = b""
        self._motion_grid = np.zeros((8, 8), dtype=np.float32)
        self._usual_grid = np.zeros((8, 8), dtype=np.float32)
        self._last_pol_score = 0.0
        self._last_pol_unusual = False
        self._last_pol_reason = ""
        self._hub_banner = ""
        self._yolo_ran = False
        self._live_tracks: list[Track] = []
        self._seen_track_ids: set[int] = set()
        self.escalation_counts = {
            "raspberry_trips": 0,
            "node_proposals": 0,
            "hub_handoffs": 0,
            "hub_alerts": 0,
            "operator_confirms": 0,
        }
        self.verdict_latencies_ms: list[float] = []

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
            for tid in list(self._active_by_track):
                self._finalize_track(tid)
            self._finalize_unnamed()
        try:
            self.pol.close()
        except Exception:
            log.exception("Failed to save Pattern of Life profile")
        self.mqtt.close()
        log.info("Pipeline stopped")

    def latest_jpeg(self, seat: str = "node") -> bytes:
        with self._lock:
            key = (seat or "node").lower()
            if key in {"edge", "raspberry"}:
                return self._jpeg_edge or self._latest_jpeg
            if key == "hub":
                return self._jpeg_hub or self._latest_jpeg
            return self._jpeg_node or self._latest_jpeg

    def health(self) -> dict:
        with self._lock:
            uptime = time.monotonic() - self.status.started_at if self.status.started_at else 0.0
            tracks = [
                {
                    "id": t.track_id,
                    "cls": t.cls,
                    "dwell_s": round(t.dwell_s, 1),
                    "zone": t.zone_name,
                }
                for t in self._live_tracks
            ]
            rasp = self.escalation_counts["raspberry_trips"]
            node = self.escalation_counts["node_proposals"]
            hub_h = self.escalation_counts["hub_handoffs"]
            hub_a = self.escalation_counts["hub_alerts"]
            confirms = self.escalation_counts["operator_confirms"]
            hub_ran = bool((self.status.last_handoff.get("hub") or {}).get("ran"))
            return {
                "status": "ok" if self.status.running else "stopped",
                "source": redact_source(self.status.source),
                "opened": self.status.opened,
                "fps": round(self.status.ingest_fps, 2),
                "last_motion": self.status.last_motion,
                "motion_area": self.status.motion_area,
                "last_detections": list(self.status.last_detections),
                "last_scene": self.status.last_scene,
                "last_anomaly": self.status.last_anomaly,
                "handoff": dict(self.status.last_handoff),
                "pol": self.pol.snapshot(),
                "model": self.cfg.detection.model,
                "vision": self.status.verifier_provider,
                "allow_cloud": bool(self.cfg.vision.allow_cloud),
                "yolo_ran": self.status.yolo_ran,
                "tracks": tracks,
                "last_error": self.status.last_error,
                "reconnects": self.status.reconnects,
                "uptime_s": round(uptime, 1),
                "escalation": {
                    "mode": self.cfg.escalation.mode,
                    "raspberry_trips": rasp,
                    "node_proposals": node,
                    "hub_handoffs": hub_h,
                    "hub_alerts": hub_a,
                    "operator_confirms": confirms,
                    "node_per_raspberry": round(node / rasp, 3) if rasp else 0.0,
                    "hub_per_node": round(hub_h / node, 3) if node else 0.0,
                    "alerts_per_hub": round(hub_a / hub_h, 3) if hub_h else 0.0,
                },
                "models": models_payload(
                    self.cfg,
                    provider=self.status.verifier_provider or effective_provider(self.cfg.vision),
                    yolo_ran=self.status.yolo_ran,
                    hub_ran=hub_ran,
                    edge_active=self.status.last_motion,
                ),
                "auth_required": bool(self.cfg.server.api_token),
            }

    def ingest_frame(self, frame: np.ndarray, now: float | None = None) -> None:
        """Offline/eval entry: one resized frame through detect + track."""
        frame = self._resize(frame)
        now = time.monotonic() if now is None else now
        self._ring.append(frame.copy())
        self._process_detect(frame, now)
        with self._event_lock:
            for active in list(self._active_by_track.values()):
                active.writer.write(frame)
            if self._unnamed is not None:
                self._unnamed.writer.write(frame)
                if now - self._unnamed.last_match_at >= self.cfg.events.post_seconds:
                    self._finalize_unnamed()

    def flush(self) -> None:
        with self._event_lock:
            for tid in list(self._active_by_track):
                self._finalize_track(tid)
            self._finalize_unnamed()

    def review_event(self, event_id: int, action: str) -> dict | None:
        action = (action or "").strip().lower()
        if action not in {"confirm", "dismiss"}:
            raise ValueError("action must be confirm or dismiss")
        row = self.store.get(event_id)
        if not row:
            return None
        status = "confirmed" if action == "confirm" else "dismissed"
        updated = self.store.update_review(event_id, status)
        if action == "confirm":
            self.escalation_counts["operator_confirms"] += 1
        if action == "dismiss":
            features = row.get("features") or {}
            source = row.get("source") or str(self.cfg.camera.source)
            if features:
                try:
                    if source == str(self.cfg.camera.source):
                        self.pol.absorb(features)
                    else:
                        absorb_into_file(self.cfg.pol_dir, source, features)
                except Exception:
                    log.exception("Failed to fold dismissal into PoL")
        return updated

    def _run(self) -> None:
        cleanup_old_events(self.store, self.cfg.data_dir, self.cfg.events.retention_days)
        backoff = 1.0
        while not self._stop.is_set():
            cap = self._open_capture()
            if cap is None or not cap.isOpened():
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
            log.info("Capture opened: %s", redact_source(self.status.source))
            self.motion.reset()
            self.tracker.reset()
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

    def _ingest_loop(self, cap: cv2.VideoCapture) -> None:
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
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        pace_file = self._is_file_source() and native_fps > 1
        frame_interval = 1.0 / native_fps if pace_file else 0.0
        next_frame_at = time.monotonic()

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive_fail += 1
                if self._is_file_source() and self.cfg.camera.loop_file:
                    loop_fails += 1
                    if loop_fails > 3 or not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        return
                    self.motion.reset()
                    consecutive_fail = 0
                    continue
                if self._is_file_source():
                    log.warning("File source ended")
                    return
                if consecutive_fail >= 8:
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
                    for active in list(self._active_by_track.values()):
                        active.writer.write(frame)
                    if self._unnamed is not None:
                        self._unnamed.writer.write(frame)
                        if now - self._unnamed.last_match_at >= self.cfg.events.post_seconds:
                            self._finalize_unnamed()

            if now - last_detect >= detect_interval:
                last_detect = now
                self._process_detect(frame, now)

            if frame_interval:
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
        edge = draw_edge_overlay(
            frame,
            self._motion_grid,
            self._usual_grid,
            self._has_motion,
            self._last_pol_score,
            self._last_pol_unusual,
            self._last_pol_reason,
        )
        node_banner = "Node · idle"
        if self._live_tracks:
            bits = [
                f"#{t.track_id} {t.cls} {t.dwell_s:.0f}s"
                for t in self._live_tracks[:3]
            ]
            node_banner = "Node · " + ", ".join(bits)
        elif dets:
            node_banner = "Node · " + ", ".join(f"{d.cls} {d.conf:.2f}" for d in dets[:3])
        elif self._yolo_ran:
            node_banner = "Node · unnamed"
        elif self._has_motion:
            node_banner = "Node · Raspberry kept this locally"
        node = draw_overlay(frame, dets, self._motion_area, self._has_motion, banner=node_banner)
        hub_dets = dets if self._hub_banner else []
        hub_banner = self._hub_banner or "Hub idle — waiting for Node to escalate"
        hub = draw_overlay(frame, hub_dets, self._motion_area, self._has_motion, banner=hub_banner[:72])
        jpeg_edge = _encode_jpeg(edge, self.cfg.pipeline.jpeg_quality)
        jpeg_node = _encode_jpeg(node, self.cfg.pipeline.jpeg_quality)
        jpeg_hub = _encode_jpeg(hub, self.cfg.pipeline.jpeg_quality)
        with self._lock:
            self._jpeg_edge = jpeg_edge
            self._jpeg_node = jpeg_node
            self._jpeg_hub = jpeg_hub
            self._latest_jpeg = jpeg_node
            self.status.last_motion = self._has_motion
            self.status.motion_area = self._motion_area

    def _process_detect(self, frame: np.ndarray, now: float) -> None:
        has_motion, area, grid = self.motion.measure(frame)
        self._has_motion = has_motion
        self._motion_area = area
        self._motion_grid = grid
        pol, features = self.pol.observe(frame, grid, area, has_motion)
        ug = np.array(pol.usual_grid, dtype=np.float32)
        self._usual_grid = ug if ug.shape == (8, 8) else np.zeros((8, 8), dtype=np.float32)
        self._last_pol_score = pol.score
        self._last_pol_unusual = pol.unusual
        self._last_pol_reason = pol.reason
        self.fusion.note_start(now)
        fusion = self.fusion.snapshot(now) if self.cfg.fusion.enabled else None
        fusion_d = fusion.as_dict() if fusion else {}

        edge_upload = has_motion and (pol.unusual or not pol.confident)
        detections: list[Detection] = []
        yolo_ran = False
        tracks: list[Track] = []
        if edge_upload:
            yolo_ran = True
            try:
                detections = self.detector.detect(frame)
            except Exception as exc:
                log.exception("Detection failed: %s", exc)
                with self._lock:
                    self.status.last_error = f"detect: {exc}"
            h, w = frame.shape[:2]
            tracks = self.tracker.update(
                detections,
                now,
                (w, h),
                zone_of=lambda x, y: self.zones.name_at(x, y),
            )
            for tr in tracks:
                for d in detections:
                    if d.xyxy == tr.xyxy:
                        d.track_id = tr.track_id
                self._persist_track(tr, frame.shape[0])

        bags = (
            unattended_bags(
                tracks,
                dwell_seconds=self.cfg.monitoring.bag_dwell_seconds,
                person_radius=self.cfg.monitoring.bag_person_radius,
            )
            if self.cfg.monitoring.unattended_bags
            else []
        )
        bag_ids = {t.track_id for t in bags}
        node_received = edge_upload
        named = bool(tracks) or bool(detections)
        no_badge = bool(fusion and not fusion.badge_within_window)
        decision = decide_hub(
            mode=self.cfg.escalation.mode,
            node_received=node_received,
            named=named,
            bag=bool(bag_ids),
            pol_confident=pol.confident,
            pol_score=pol.score,
            pol_min=self.cfg.escalation.pol_score_min,
            no_badge=no_badge,
        )
        hub_needed = decision.hub_needed
        # Fail-open: page until Hub adjudicates. Verifier may suppress.
        page_operator = hub_needed
        anomaly_reason = decision.reason
        if edge_upload:
            self.escalation_counts["raspberry_trips"] += 1
        peak = max((d.conf for d in detections), default=0.0)
        labels = sorted({t.cls for t in tracks} | {d.cls for d in detections})
        hub_detail = fallback_summary(labels, peak, anomaly_reason) if hub_needed else ""
        overlay_dets = [t.as_detection() for t in tracks] or detections
        handoff = _handoff_payload(
            pol=pol,
            has_motion=has_motion,
            detections=overlay_dets,
            yolo_ran=yolo_ran,
            node_received=node_received,
            hub_needed=hub_needed,
            page_operator=page_operator,
            anomaly_reason=anomaly_reason,
            hub_detail=hub_detail,
            tracks=tracks,
        )
        stopped = handoff["stopped_at"]
        self._yolo_ran = yolo_ran
        self._live_tracks = tracks

        live_ids = {t.track_id for t in tracks if t.confirmed}
        with self._event_lock:
            for tid in list(self._active_by_track):
                if tid not in self.tracker.tracks:
                    self._finalize_track(tid)

        if node_received:
            self._last_dets = overlay_dets
            self._overlay_until = now + self._overlay_ttl
            payload = [
                {
                    "cls": t.cls,
                    "conf": round(t.conf, 3),
                    "track_id": t.track_id,
                    "dwell_s": round(t.dwell_s, 1),
                }
                for t in tracks
            ] or [{"cls": d.cls, "conf": round(d.conf, 3)} for d in detections]
            scene = hub_detail or fallback_summary(labels, peak, anomaly_reason)
            self._hub_banner = ("Hub · " + hub_detail) if hub_needed else ""
            with self._lock:
                self.status.yolo_ran = yolo_ran
                self.status.last_motion = has_motion
                self.status.motion_area = area
                self.status.last_detections = payload
                self.status.last_scene = scene
                self.status.last_anomaly = anomaly_reason if page_operator else ""
                self.status.last_handoff = handoff
            self.mqtt.publish("state", {"motion": has_motion, "tracks": payload, "fusion": fusion_d})
            feats = dict(features)
            feats["fusion"] = fusion_d
            with self._event_lock:
                for tr in tracks:
                    if not tr.confirmed:
                        continue
                    need = hub_needed or tr.track_id in bag_ids
                    self._touch_track_event(
                        frame,
                        tr,
                        overlay_dets,
                        now,
                        hub_needed=need,
                        page_operator=need,
                        anomaly_reason=anomaly_reason if tr.track_id in bag_ids else (
                            anomaly_reason if need else ""
                        ),
                        pol_score=pol.score,
                        stopped_at=stopped,
                        handoff=handoff,
                        features={**feats, **tr.features(frame.shape[0])},
                        fusion=fusion_d,
                    )
                if not tracks:
                    self._touch_unnamed(
                        frame,
                        detections,
                        now,
                        hub_needed=hub_needed,
                        page_operator=page_operator,
                        anomaly_reason=anomaly_reason,
                        pol_score=pol.score,
                        stopped_at=stopped,
                        handoff=handoff,
                        features=feats,
                        fusion=fusion_d,
                    )
        else:
            self._hub_banner = ""
            self._last_dets = []
            self._yolo_ran = False
            self._live_tracks = []
            with self._lock:
                self.status.yolo_ran = False
                self.status.last_motion = has_motion
                self.status.motion_area = area
                self.status.last_detections = []
                self.status.last_anomaly = ""
                self.status.last_scene = pol.reason if has_motion else "Quiet."
                self.status.last_handoff = handoff
        _ = live_ids

    def _persist_track(self, tr: Track, frame_h: int) -> None:
        feat = tr.features(frame_h)
        self.store.upsert_track(
            {
                "id": tr.track_id,
                "source": str(self.cfg.camera.source),
                "cls": tr.cls,
                "first_seen": utc_now(),
                "last_seen": utc_now(),
                "dwell_s": feat["dwell_s"],
                "path_length": feat["path_length"],
                "mean_speed": feat["mean_speed"],
                "peak_speed": feat["peak_speed"],
                "direction": feat["direction"],
                "vertical_extent": feat["vertical_extent"],
                "hover_score": feat["hover_score"],
                "turn_rate": feat["turn_rate"],
                "zone": feat["zone"],
                "class_hist": tr.class_hist,
                "trajectory": [(round(t, 2), round(x, 1), round(y, 1)) for t, x, y in tr.traj[-40:]],
                "event_id": tr.event_id,
            }
        )

    def _touch_track_event(
        self,
        frame: np.ndarray,
        tr: Track,
        detections: list[Detection],
        now: float,
        **kwargs,
    ) -> None:
        existing = self._active_by_track.get(tr.track_id)
        if existing is not None:
            existing.last_match_at = now
            existing.classes.add(tr.cls)
            existing.score = max(existing.score, tr.conf)
            existing.hub_needed = existing.hub_needed or kwargs["hub_needed"]
            existing.page_operator = existing.page_operator or kwargs["page_operator"]
            if kwargs["anomaly_reason"]:
                existing.anomaly_reason = kwargs["anomaly_reason"]
            existing.pol_score = max(existing.pol_score, kwargs["pol_score"])
            existing.handoff = kwargs["handoff"]
            existing.features = kwargs["features"]
            existing.last_frame = frame.copy()
            existing.last_dets = detections
            existing.fusion = kwargs.get("fusion") or {}
            return
        if self.store.event_for_track(tr.track_id, str(self.cfg.camera.source)):
            return
        self._start_event(
            frame,
            detections,
            now,
            {tr.cls},
            tr.conf,
            track_id=tr.track_id,
            **kwargs,
        )

    def _touch_unnamed(self, frame, detections, now, **kwargs) -> None:
        if self._unnamed is not None:
            self._unnamed.last_match_at = now
            self._unnamed.hub_needed = self._unnamed.hub_needed or kwargs["hub_needed"]
            self._unnamed.page_operator = self._unnamed.page_operator or kwargs["page_operator"]
            if kwargs["anomaly_reason"]:
                self._unnamed.anomaly_reason = kwargs["anomaly_reason"]
            self._unnamed.last_frame = frame.copy()
            return
        if now < self._cooldown_until:
            return
        self._start_event(frame, detections, now, set(), 0.0, track_id=None, **kwargs)

    def _start_event(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        now: float,
        classes: set[str],
        score: float,
        *,
        hub_needed: bool,
        page_operator: bool,
        anomaly_reason: str,
        pol_score: float,
        stopped_at: str,
        handoff: dict,
        features: dict,
        track_id: int | None,
        fusion: dict | None = None,
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        suffix = f"-t{track_id}" if track_id is not None else ""
        clip_name = f"{stamp}{suffix}.mp4"
        thumb_name = f"{stamp}{suffix}.jpg"
        clip_path = self.cfg.clips_dir / clip_name
        thumb_path = self.cfg.thumbs_dir / thumb_name
        vis = draw_overlay(frame, detections, self._motion_area, True)
        writer: ClipWriter | NullWriter
        try:
            if self.cfg.events.write_media:
                save_thumb(vis, thumb_path, self.cfg.pipeline.jpeg_quality)
                writer = ClipWriter(
                    clip_path,
                    self.cfg.camera.width,
                    self.cfg.camera.height,
                    self.cfg.events.clip_fps,
                )
            else:
                writer = NullWriter()
        except Exception:
            log.exception("Failed to start clip/thumb for event")
            return
        if self.cfg.events.write_media:
            for buffered in list(self._ring):
                writer.write(buffered)
            writer.write(frame)
        active = _ActiveEvent(
            started_at=now,
            last_match_at=now,
            classes=set(classes),
            score=score,
            writer=writer,
            clip_name=clip_name,
            thumb_name=thumb_name,
            ts_start=utc_now(),
            hub_needed=hub_needed,
            page_operator=page_operator,
            anomaly_reason=anomaly_reason,
            pol_score=pol_score,
            stopped_at=stopped_at,
            handoff=handoff,
            features=features,
            source=str(self.cfg.camera.source),
            track_id=track_id,
            last_frame=frame.copy(),
            last_dets=list(detections),
            fusion=fusion or {},
            t0_mono=time.monotonic(),
        )
        if track_id is None:
            self._unnamed = active
        else:
            self._active_by_track[track_id] = active
            if track_id in self.tracker.tracks:
                self.tracker.tracks[track_id].event_id = -1
        log.info("Event started track=%s classes=%s stop=%s", track_id, sorted(classes), stopped_at)
        self.escalation_counts["node_proposals"] += 1

    def _finalize_track(self, track_id: int | None) -> None:
        if track_id is None:
            return
        active = self._active_by_track.pop(track_id, None)
        if active:
            self._finalize_event(active)

    def _finalize_unnamed(self) -> None:
        active = self._unnamed
        self._unnamed = None
        if active:
            self._finalize_event(active)

    def _finalize_event(self, active: _ActiveEvent) -> None:
        try:
            active.writer.close()
        except Exception:
            log.exception("Error closing clip writer")
        classes = sorted(active.classes)
        score = round(active.score, 3)
        rule_alert = bool(active.page_operator)
        anomaly_reason = active.anomaly_reason
        if not anomaly_reason and rule_alert:
            anomaly_reason = "escalated to operator"
        if active.page_operator:
            active.stopped_at = "operator"
        elif active.hub_needed:
            active.stopped_at = "hub"
        else:
            active.stopped_at = "node"
        node_note = fallback_summary(classes, score, anomaly_reason)
        if not active.hub_needed:
            labels = ", ".join(classes) if classes else "motion"
            node_note = f"Named {labels}. Node closed this without asking Hub."
        thumb_rel = f"thumbs/{active.thumb_name}" if self.cfg.events.write_media else None
        clip_rel = f"clips/{active.clip_name}" if self.cfg.events.write_media else None
        event_id = None
        try:
            event_id = self.store.insert(
                ts_start=active.ts_start,
                ts_end=utc_now(),
                classes=classes,
                score=score,
                thumb_path=thumb_rel,
                clip_path=clip_rel,
                anomaly=rule_alert,
                anomaly_reason=anomaly_reason,
                source=active.source,
                pol_score=round(active.pol_score, 3),
                stopped_at=active.stopped_at,
                handoff=active.handoff,
                features=active.features,
                operator_status="pending" if active.page_operator else "",
                track_id=active.track_id,
                verifier_provider="",
                verifier_status="",
            )
            self.store.update_summary(event_id, node_note)
            if active.track_id is not None:
                tr = self.tracker.tracks.get(active.track_id)
                if tr:
                    tr.event_id = event_id
                    self._persist_track(tr, self.cfg.camera.height)
        except Exception:
            log.exception("Failed to persist event")
        if active.track_id is None:
            self._cooldown_until = time.monotonic() + self.cfg.events.cooldown_seconds
        log.info(
            "Event saved %s track=%s stop=%s anomaly=%s",
            active.clip_name,
            active.track_id,
            active.stopped_at,
            rule_alert,
        )
        self.mqtt.publish(
            "event",
            {
                "id": event_id,
                "track_id": active.track_id,
                "classes": classes,
                "stopped_at": active.stopped_at,
            },
        )
        if event_id and active.hub_needed:
            self._run_verifier(event_id, active, classes, score, anomaly_reason, rule_alert)
        elif event_id:
            self.store.update_verdict(
                event_id,
                summary=node_note,
                anomaly=False,
                anomaly_reason="",
                verifier_provider="skipped",
                verifier_status="node-closed",
            )
            t0 = active.t0_mono or time.monotonic()
            self.verdict_latencies_ms.append((time.monotonic() - t0) * 1000.0)
        if event_id and self.embeddings and self.cfg.events.write_media:
            self._index_embedding(event_id, active, classes)
        try:
            cleanup_old_events(self.store, self.cfg.data_dir, self.cfg.events.retention_days)
        except Exception:
            log.exception("Retention cleanup failed")

    def _run_verifier(
        self,
        event_id: int,
        active: _ActiveEvent,
        classes: list[str],
        score: float,
        anomaly_reason: str,
        rule_alert: bool,
    ) -> None:
        clock = clock_context()
        pol_snap = self.pol.snapshot()
        context = {
            "camera": self.cfg.camera.name,
            "source": str(self.cfg.camera.source),
            "zone": (active.features or {}).get("zone"),
            "track": {k: active.features.get(k) for k in ("track_id", "dwell_s", "path_length", "hover_score")},
            "fusion": active.fusion,
            "pol": {"state": pol_snap.get("state"), "progress": pol_snap.get("progress")},
            "clock": clock,
        }
        try:
            verdict = verify_event(
                self.cfg.vision,
                active.last_frame,
                active.last_dets,
                classes=classes,
                score=score,
                anomaly_reason=anomaly_reason,
                rule_alert=rule_alert,
                context=context,
            )
        except Exception:
            log.exception("Verifier crashed; fail-open")
            from app.verify import rule_verdict

            verdict = rule_verdict(
                classes=classes,
                score=score,
                anomaly_reason=anomaly_reason,
                rule_alert=rule_alert,
                provider="error",
                status="unavailable",
            )
        self.store.update_verdict(
            event_id,
            summary=verdict.summary,
            anomaly=verdict.alert,
            anomaly_reason=verdict.reason,
            verifier_provider=verdict.provider,
            verifier_status=verdict.status,
            operator_status="pending" if verdict.alert else "",
            stopped_at="operator" if verdict.alert else "hub",
        )
        self.escalation_counts["hub_handoffs"] += 1
        if verdict.alert:
            self.escalation_counts["hub_alerts"] += 1
        t0 = active.t0_mono or time.monotonic()
        self.verdict_latencies_ms.append((time.monotonic() - t0) * 1000.0)
        self.mqtt.publish("verdict", {"id": event_id, **verdict.as_dict()})
        with self._lock:
            self.status.verifier_provider = verdict.provider
            self.status.last_scene = verdict.summary
            self.status.last_anomaly = verdict.reason if verdict.alert else ""
        log.info(
            "Event %s verdict alert=%s status=%s provider=%s",
            event_id,
            verdict.alert,
            verdict.status,
            verdict.provider,
        )

    def _index_embedding(self, event_id: int, active: _ActiveEvent, classes: list[str]) -> None:
        if not self.embeddings:
            return
        thumb = self.cfg.thumbs_dir / active.thumb_name
        if not thumb.is_file():
            return
        try:
            vec = thumb_hist(thumb)
            hour = datetime.now().hour
            self.embeddings.add(
                event_id,
                active.source,
                vec,
                hour,
                classes[0] if classes else "motion",
            )
            nov = self.embeddings.novelty(active.source, vec, k=self.cfg.embed.knn)
            self.store.update_verdict(
                event_id,
                summary=self.store.get(event_id).get("summary") or "",
                anomaly=bool(self.store.get(event_id).get("anomaly")),
                anomaly_reason=self.store.get(event_id).get("anomaly_reason") or "",
                verifier_provider=self.store.get(event_id).get("verifier_provider") or "",
                verifier_status=self.store.get(event_id).get("verifier_status") or "",
                novelty_score=nov.score,
            )
        except Exception:
            log.exception("Embedding index failed for event %s", event_id)


def _handoff_payload(
    *,
    pol,
    has_motion: bool,
    detections: list[Detection],
    yolo_ran: bool,
    node_received: bool,
    hub_needed: bool,
    page_operator: bool,
    anomaly_reason: str,
    hub_detail: str,
    tracks: list | None = None,
) -> dict:
    if not has_motion:
        edge_decision = "quiet"
        edge_detail = "no motion"
    elif pol.unusual or not pol.confident:
        edge_decision = "unusual"
        edge_detail = f"{pol.score:.2f} · {pol.reason}"
    else:
        edge_decision = "usual"
        edge_detail = f"{pol.score:.2f} · {pol.reason}"

    track_bits = []
    for t in tracks or []:
        track_bits.append(f"#{t.track_id} {t.cls} {t.dwell_s:.0f}s")
    labels = ", ".join(track_bits) or (
        ", ".join(f"{d.cls} {d.conf:.2f}" for d in detections[:4]) or "unnamed"
    )
    steps = [
        {
            "stage": "edge",
            "label": "Raspberry",
            "decision": edge_decision,
            "detail": edge_detail,
        }
    ]
    if not node_received:
        steps.append(
            {
                "stage": "node",
                "label": "Node",
                "decision": "skipped",
                "detail": "Raspberry kept this locally · detector idle",
            }
        )
        steps.append({"stage": "hub", "label": "Hub", "decision": "skipped", "detail": "Node did not escalate"})
        steps.append({"stage": "operator", "label": "Operator", "decision": "skipped", "detail": ""})
        stopped = "edge"
    else:
        if hub_needed:
            node_detail = f"{labels} · send to Hub"
            node_decision = "escalate"
        else:
            node_detail = f"{labels} · named and closed"
            node_decision = "closed"
        steps.append({"stage": "node", "label": "Node", "decision": node_decision, "detail": node_detail})
        if hub_needed:
            hub_decision = "alert" if page_operator else "log"
            steps.append(
                {
                    "stage": "hub",
                    "label": "Hub",
                    "decision": hub_decision,
                    "detail": (hub_detail or anomaly_reason or labels)[:160],
                }
            )
            if page_operator:
                steps.append(
                    {
                        "stage": "operator",
                        "label": "Operator",
                        "decision": "needed",
                        "detail": "Incident or Normal",
                    }
                )
                stopped = "operator"
            else:
                steps.append(
                    {
                        "stage": "operator",
                        "label": "Operator",
                        "decision": "skipped",
                        "detail": "Hub logged without paging",
                    }
                )
                stopped = "hub"
        else:
            steps.append({"stage": "hub", "label": "Hub", "decision": "skipped", "detail": "Node closed this"})
            steps.append({"stage": "operator", "label": "Operator", "decision": "skipped", "detail": ""})
            stopped = "node"

    edge_info = pol.as_dict()
    edge_info["upload"] = bool(edge_decision == "unusual")
    return {
        "stopped_at": stopped,
        "steps": steps,
        "edge": edge_info,
        "node": {
            "received": node_received,
            "ran": yolo_ran,
            "closed": node_received and not hub_needed,
            "classes": [
                {
                    "cls": d.cls,
                    "conf": round(d.conf, 3),
                    "track_id": d.track_id,
                }
                for d in detections
            ],
            "tracks": track_bits,
        },
        "hub": {
            "ran": hub_needed,
            "page_operator": page_operator,
            "detail": hub_detail,
        },
    }
