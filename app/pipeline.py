from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

from app.config import AppConfig
from app.db import EventStore, utc_now
from app.detect import Detection, ObjectDetector, boxes_payload, draw_overlay
from app.embed import EmbeddingIndex, thumb_hist
from app.escalate import decide_hub, effective_mode
from app.fusion import FusionBus, clock_context
from app.motion import MotionDetector
from app.mqtt_bus import MqttBus, MqttConfig
from app.page import choose_paged_because
from app.pol import PatternOfLife, absorb_into_file, source_key
from app.record import ClipWriter, NullWriter, cleanup_old_events, save_thumb
from app.security import redact_source
from app.situation import situation_lines
from app.tiers import SEAT_LABELS, models_payload
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


def _placeholder_jpeg(width: int, height: int, quality: int) -> bytes:
    w, h = max(width or 1280, 320), max(height or 720, 180)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (18, 16, 14)
    cv2.putText(
        frame,
        "No signal",
        (24, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )
    return _encode_jpeg(frame, quality)


def bundled_sample(root: Path) -> Path | None:
    names = ("street.mp4", "indoor.mp4", "package.mp4", "drone.mp4", "sample.mp4", "entrance.mp4")
    dirs = (root / "data" / "samples", root / "data")
    for folder in dirs:
        for name in names:
            path = folder / name
            if path.is_file() and path.stat().st_size > 10_000:
                return path
    return None


DEMO_CLIPS = (
    ("street", "Street", "street.mp4"),
    ("indoor", "Indoor", "indoor.mp4"),
    ("package", "Left bag", "package.mp4"),
)


def demo_clips(root: Path) -> list[dict]:
    out: list[dict] = []
    for cid, label, name in DEMO_CLIPS:
        path = root / "data" / "samples" / name
        out.append(
            {
                "id": cid,
                "label": label,
                "path": f"data/samples/{name}",
                "present": path.is_file() and path.stat().st_size > 10_000,
            }
        )
    return out


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
    provenance: str = "live"
    can_page: bool = True
    bag: bool = False
    named: bool = False
    learning: bool = False
    unusual: bool = False
    paged_because: str = ""
    mode_effective: str = "recall"


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
            max_age=cfg.tracking.max_age_s,
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
        self._active_source: str | int = cfg.resolved_source()
        self._file_loop = bool(cfg.camera.loop_file)
        self._fallback = False
        self._ingest_kind = "live"
        self._last_idle_detect = time.monotonic()
        self._verify_ok_until = 0.0
        self._verify_offline_since: float | None = time.time()
        self.audit_shown = 0
        self.audit_confirmed = 0
        self._paged_counts: dict[str, int] = {}
        placeholder = _placeholder_jpeg(
            cfg.camera.width, cfg.camera.height, cfg.pipeline.jpeg_quality
        )
        self._latest_jpeg = placeholder
        self._jpeg_edge = placeholder
        self._jpeg_node = placeholder
        self._jpeg_hub = placeholder
        self._motion_grid = np.zeros((8, 8), dtype=np.float32)
        self._usual_grid = np.zeros((8, 8), dtype=np.float32)
        self._last_pol_score = 0.0
        self._last_pol_unusual = False
        self._last_pol_reason = ""
        self._hub_banner = ""
        self._yolo_ran = False
        self._live_tracks: list[Track] = []
        self._situation: list[str] = []
        self._seen_track_ids: set[int] = set()
        self.escalation_counts = {
            "edge_trips": 0,
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
        self._last_idle_detect = time.monotonic()
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
            if key == "edge":
                return self._jpeg_edge or self._latest_jpeg
            if key == "hub":
                return self._jpeg_hub or self._latest_jpeg
            return self._jpeg_node or self._latest_jpeg

    def health(self) -> dict:
        with self._lock:
            uptime = time.monotonic() - self.status.started_at if self.status.started_at else 0.0
            now = time.monotonic()
            tracks = [
                {
                    "id": t.track_id,
                    "cls": t.cls,
                    "dwell_s": round(t.dwell_at(now), 1),
                    "zone": t.zone_name,
                }
                for t in self._live_tracks
            ]
            edge = self.escalation_counts["edge_trips"]
            node = self.escalation_counts["node_proposals"]
            hub_h = self.escalation_counts["hub_handoffs"]
            hub_a = self.escalation_counts["hub_alerts"]
            confirms = self.escalation_counts["operator_confirms"]
            hub_ran = bool((self.status.last_handoff.get("hub") or {}).get("ran"))
            lat = list(self.verdict_latencies_ms)
            mode_cfg = self.cfg.escalation.mode
            mode_eff = effective_mode(mode_cfg, verify_healthy=self._verify_healthy())
            offline = None
            if self._verify_offline_since is not None:
                offline = datetime.fromtimestamp(self._verify_offline_since).strftime("%H:%M")
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
                "situation": list(self._situation),
                "demo_clips": demo_clips(self.cfg.root),
                "last_error": self.status.last_error,
                "reconnects": self.status.reconnects,
                "uptime_s": round(uptime, 1),
                "escalation": {
                    "mode": mode_cfg,
                    "mode_effective": mode_eff,
                    "edge_trips": edge,
                    "node_proposals": node,
                    "hub_handoffs": hub_h,
                    "hub_alerts": hub_a,
                    "operator_confirms": confirms,
                    "node_per_edge": round(node / edge, 3) if edge else 0.0,
                    "hub_per_node": round(hub_h / node, 3) if node else 0.0,
                    "alerts_per_hub": round(hub_a / hub_h, 3) if hub_h else 0.0,
                    "paged_because": dict(self._paged_counts),
                    "audit_shown": self.audit_shown,
                    "audit_confirmed": self.audit_confirmed,
                },
                "latency_ms": {
                    "count": len(lat),
                    "p50": round(_percentile(lat, 50), 1) if lat else None,
                    "p95": round(_percentile(lat, 95), 1) if lat else None,
                },
                "clip_drops": int(ClipWriter.drops),
                "verify_offline_since": offline,
                "models": models_payload(
                    self.cfg,
                    provider=self.status.verifier_provider or effective_provider(self.cfg.vision),
                    yolo_ran=self.status.yolo_ran,
                    hub_ran=hub_ran,
                    edge_active=self.status.last_motion,
                ),
                "auth_required": bool(self.cfg.server.api_token),
                "fallback": self._fallback,
            }

    def _verify_healthy(self) -> bool:
        return time.monotonic() < self._verify_ok_until

    def _source_stored(self) -> str:
        return redact_source(self.cfg.camera.source)

    def _provenance(self) -> str:
        if self._ingest_kind == "fixture":
            return "fixture"
        if self._fallback:
            return "sample"
        return "live"

    def ingest_frame(self, frame: np.ndarray, now: float | None = None) -> None:
        """Offline/eval entry: one resized frame through detect + track."""
        self._ingest_kind = "fixture"
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
            if (row.get("paged_because") or "") == "audit":
                self.audit_confirmed += 1
        if action == "dismiss":
            features = row.get("features") or {}
            provenance = row.get("provenance") or "live"
            source = row.get("source") or self._source_stored()
            if provenance != "live":
                log.info("Skip PoL absorb for %s event %s", provenance, event_id)
            elif features:
                try:
                    current = {self._source_stored(), str(self.cfg.camera.source)}
                    if source in current or not source:
                        result = self.pol.absorb(features)
                    else:
                        absorb_into_file(self.cfg.pol_dir, source, features)
                        result = {"force": True, "delta": 0.0}
                    self.store.insert_absorb(
                        event_id=event_id,
                        source=source,
                        action="dismiss",
                        force=bool(result.get("force")),
                        delta=float(result.get("delta") or 0),
                        detail=str(result),
                    )
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
                self.status.source = str(self._active_source)
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
        cap = self._try_open(source)
        if cap is not None:
            self._active_source = source
            self._file_loop = self._source_is_file(source) and bool(self.cfg.camera.loop_file)
            self._fallback = False
            with self._lock:
                self.status.source = str(source)
            self._bind_pol(source)
            return cap
        sample = bundled_sample(self.cfg.root)
        if sample is not None:
            cap = self._try_open(str(sample))
            if cap is not None:
                log.warning(
                    "Camera %s unavailable; looping sample %s",
                    redact_source(source),
                    sample.name,
                )
                self._active_source = str(sample)
                self._file_loop = True
                self._fallback = True
                with self._lock:
                    self.status.source = str(sample)
                self._bind_pol(str(sample))
                return cap
        with self._lock:
            self.status.source = str(source)
        return None

    def _bind_pol(self, source: str | int) -> None:
        key = source_key(source)
        if self.pol.key == key:
            return
        try:
            self.pol.close()
        except Exception:
            log.exception("Failed to save Pattern of Life before rebind")
        self.pol = PatternOfLife(self.cfg.pol_dir, source)

    def _try_open(self, source: str | int) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(source)
        if self.cfg.camera.width and isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.camera.width)
        if self.cfg.camera.height and isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.camera.height)
        if not cap.isOpened():
            cap.release()
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return None
        if self._source_is_file(source):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap

    def _source_is_file(self, source: str | int) -> bool:
        if isinstance(source, int):
            return False
        if "://" in str(source):
            return False
        return Path(source).is_file()

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
                if self._is_file_source() and self._file_loop:
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
        return self._source_is_file(self._active_source)

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
        jpeg_edge = _encode_jpeg(frame, self.cfg.pipeline.jpeg_quality)
        node = draw_overlay(frame, dets)
        hub_dets = dets if self._hub_banner else []
        hub = draw_overlay(frame, hub_dets)
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

        edge_trip = has_motion and (pol.unusual or not pol.confident or self._fallback)
        idle_s = max(1.0, float(self.cfg.pipeline.idle_detect_seconds))
        idle_due = (now - self._last_idle_detect) >= idle_s
        run_detect = bool(edge_trip or idle_due)
        if idle_due:
            self._last_idle_detect = now
        provenance = self._provenance()
        can_page = provenance == "live" and bool(pol.confident)
        detections: list[Detection] = []
        yolo_ran = False
        tracks: list[Track] = []
        if run_detect:
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
                self._persist_track(tr, frame.shape[0], now)

        bags = (
            unattended_bags(
                tracks,
                now=now,
                dwell_seconds=self.cfg.monitoring.bag_dwell_seconds,
                person_radius=self.cfg.monitoring.bag_person_radius,
            )
            if self.cfg.monitoring.unattended_bags
            else []
        )
        bag_ids = {t.track_id for t in bags}
        if run_detect:
            self._situation = situation_lines(
                tracks,
                now=now,
                bags=bags,
                bag_radius=float(self.cfg.monitoring.bag_person_radius),
            )
        node_received = run_detect
        named = bool(tracks) or bool(detections)
        decision = decide_hub(
            mode=self.cfg.escalation.mode,
            node_received=node_received,
            named=named,
            bag=bool(bag_ids),
            pol_confident=pol.confident,
            pol_score=pol.score,
            pol_min=self.cfg.escalation.pol_score_min,
            verify_healthy=self._verify_healthy(),
        )
        hub_needed = decision.hub_needed and can_page
        # Fail-open into Verify only when we are allowed to page. Sample/learning never page.
        page_operator = hub_needed and can_page
        anomaly_reason = decision.reason if page_operator else ""
        if run_detect and edge_trip:
            self.escalation_counts["edge_trips"] += 1
        peak = max((d.conf for d in detections), default=0.0)
        labels = sorted({t.cls for t in tracks} | {d.cls for d in detections})
        hub_detail = fallback_summary(labels, peak, anomaly_reason) if hub_needed else ""
        overlay_dets = [t.as_detection() for t in tracks] or detections
        mode_eff = decision.mode
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
            now=now,
            mode_effective=mode_eff,
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
                    "dwell_s": round(t.dwell_at(now), 1),
                    "xyxy": [int(v) for v in t.xyxy],
                    "zone": t.zone_name or "",
                }
                for t in tracks
            ] or [
                {
                    "cls": d.cls,
                    "conf": round(d.conf, 3),
                    "track_id": d.track_id,
                    "xyxy": [int(v) for v in d.xyxy],
                }
                for d in detections
            ]
            scene = hub_detail or fallback_summary(labels, peak, anomaly_reason)
            self._hub_banner = ("Verify · " + hub_detail) if hub_needed else ""
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
            self._attach_spot(feats, frame, overlay_dets, pol)
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
                        page_operator=need and can_page,
                        anomaly_reason=anomaly_reason if tr.track_id in bag_ids else (
                            anomaly_reason if need else ""
                        ),
                        pol_score=pol.score,
                        stopped_at=stopped,
                        handoff=handoff,
                        features={**feats, **tr.features(frame.shape[0], now)},
                        fusion=fusion_d,
                        provenance=provenance,
                        can_page=can_page,
                        bag=tr.track_id in bag_ids,
                        named=named,
                        learning=not pol.confident,
                        unusual=bool(pol.unusual),
                        mode_effective=mode_eff,
                    )
                if not tracks and (detections or edge_trip):
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
                        provenance=provenance,
                        can_page=can_page,
                        bag=False,
                        named=named,
                        learning=not pol.confident,
                        unusual=bool(pol.unusual),
                        mode_effective=mode_eff,
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

    def _persist_track(self, tr: Track, frame_h: int, now: float | None = None) -> None:
        feat = tr.features(frame_h, now)
        self.store.upsert_track(
            {
                "id": tr.track_id,
                "source": self._source_stored(),
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
            existing.can_page = existing.can_page or kwargs.get("can_page", False)
            existing.bag = existing.bag or kwargs.get("bag", False)
            return
        if self._is_dup(tr.cls, detections, now):
            return
        if self.store.event_for_track(tr.track_id, self._source_stored()):
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
        provenance: str = "live",
        can_page: bool = True,
        bag: bool = False,
        named: bool = False,
        learning: bool = False,
        unusual: bool = False,
        mode_effective: str = "recall",
    ) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        suffix = f"-t{track_id}" if track_id is not None else ""
        clip_name = f"{stamp}{suffix}.mp4"
        thumb_name = f"{stamp}{suffix}.jpg"
        clip_path = self.cfg.clips_dir / clip_name
        thumb_path = self.cfg.thumbs_dir / thumb_name
        vis = draw_overlay(frame, detections)
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
            source=self._source_stored(),
            track_id=track_id,
            last_frame=frame.copy(),
            last_dets=list(detections),
            fusion=fusion or {},
            t0_mono=time.monotonic(),
            provenance=provenance,
            can_page=can_page,
            bag=bag,
            named=named,
            learning=learning,
            unusual=unusual,
            mode_effective=mode_effective,
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
        rule_alert = bool(active.page_operator and active.can_page)
        anomaly_reason = active.anomaly_reason
        if not anomaly_reason and rule_alert:
            anomaly_reason = "escalated to operator"
        paged_because = choose_paged_because(
            provenance=active.provenance,
            learning=active.learning,
            bag=active.bag,
            unusual=active.unusual,
            named=active.named,
        )
        active.features = dict(active.features or {})
        active.features["paged_because"] = paged_because
        if active.page_operator and active.can_page:
            active.stopped_at = "operator"
        elif active.hub_needed:
            active.stopped_at = "hub"
        else:
            active.stopped_at = "node"
        node_note = fallback_summary(classes, score, anomaly_reason)
        if not active.hub_needed:
            labels = ", ".join(classes) if classes else "motion"
            node_note = f"Named {labels}. Detect closed this without asking Verify."
        thumb_rel = f"thumbs/{active.thumb_name}" if self.cfg.events.write_media else None
        clip_rel = f"clips/{active.clip_name}" if self.cfg.events.write_media else None
        op_status = ""
        if active.can_page and active.page_operator:
            op_status = "pending"
        elif not active.can_page:
            op_status = ""
        event_id = None
        try:
            self._refresh_event_thumb(active)
            event_id = self.store.insert(
                ts_start=active.ts_start,
                ts_end=utc_now(),
                classes=classes,
                score=score,
                thumb_path=thumb_rel,
                clip_path=clip_rel,
                anomaly=rule_alert,
                anomaly_reason=anomaly_reason,
                source=self._source_stored(),
                pol_score=round(active.pol_score, 3),
                stopped_at=active.stopped_at,
                handoff=active.handoff,
                features=active.features,
                operator_status=op_status,
                track_id=active.track_id,
                verifier_provider="",
                verifier_status="",
                paged_because=paged_because,
                provenance=active.provenance,
            )
            self.store.update_summary(event_id, node_note)
            if active.track_id is not None:
                tr = self.tracker.tracks.get(active.track_id)
                if tr:
                    tr.event_id = event_id
                    self._persist_track(tr, self.cfg.camera.height, time.monotonic())
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
        if event_id and active.hub_needed and active.can_page:
            self._run_verifier(event_id, active, classes, score, anomaly_reason, rule_alert)
        elif event_id:
            self._paged_counts[paged_because] = self._paged_counts.get(paged_because, 0) + 1
            self.store.update_verdict(
                event_id,
                summary=node_note,
                anomaly=False,
                anomaly_reason="",
                verifier_provider="skipped",
                verifier_status="node-closed",
                paged_because=paged_because,
            )
            t0 = active.t0_mono or time.monotonic()
            self.verdict_latencies_ms.append((time.monotonic() - t0) * 1000.0)
        if event_id and self.embeddings and self.cfg.events.write_media and active.provenance == "live":
            self._index_embedding(event_id, active, classes)
        try:
            cleanup_old_events(self.store, self.cfg.data_dir, self.cfg.events.retention_days)
        except Exception:
            log.exception("Retention cleanup failed")

    def _is_dup(self, cls: str, detections: list[Detection], now: float) -> bool:
        window = max(1.0, float(self.cfg.tracking.dedup_seconds))
        since = (datetime.now(timezone.utc) - timedelta(seconds=window)).isoformat()
        recent = self.store.recent_for_dedup(self._source_stored(), since)
        w, h = self.cfg.camera.width, self.cfg.camera.height
        boxes = boxes_payload(detections, (w, h))
        for row in recent:
            classes = set(row.get("classes") or [])
            if classes and cls not in classes:
                continue
            other = (row.get("features") or {}).get("boxes") or []
            if _boxes_overlap(boxes, other):
                return True
        return False

    def _attach_spot(self, feats: dict, frame: np.ndarray, detections: list[Detection], pol) -> None:
        h, w = frame.shape[:2]
        feats["frame"] = {"w": int(w), "h": int(h)}
        feats["boxes"] = boxes_payload(detections, (w, h))
        detail = getattr(pol, "why", "") or pol.reason
        if self._fallback:
            extra = (
                "This host is looping a short demo file. The 16-cell motion sketch "
                "fills in seconds; that is not a Pattern of Life. Review is not paged."
            )
            if extra not in detail:
                detail = f"{detail} {extra}".strip()
        feats["why"] = {
            "reason": pol.reason,
            "detail": detail,
            "score": round(float(pol.score), 3),
            "occupancy_novelty": round(float(pol.occupancy_novelty), 3),
            "visual_delta": round(float(pol.visual_delta), 3),
            "motion_spike": round(float(pol.motion_spike), 3),
            "learning": not bool(pol.confident),
            "fallback": bool(self._fallback),
            "samples": int(pol.samples),
        }

    def _refresh_event_thumb(self, active: _ActiveEvent) -> None:
        if not self.cfg.events.write_media or active.last_frame is None:
            return
        dets = active.last_dets or []
        vis = draw_overlay(active.last_frame, dets)
        try:
            save_thumb(
                vis,
                self.cfg.thumbs_dir / active.thumb_name,
                self.cfg.pipeline.jpeg_quality,
            )
        except Exception:
            log.exception("Failed to refresh marked thumb for %s", active.thumb_name)
            return
        if dets:
            h, w = active.last_frame.shape[:2]
            active.features = dict(active.features or {})
            active.features["boxes"] = boxes_payload(dets, (w, h))
            active.features["frame"] = {"w": int(w), "h": int(h)}

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
            "source": self._source_stored(),
            "zone": (active.features or {}).get("zone"),
            "track": {k: active.features.get(k) for k in ("track_id", "dwell_s", "path_length", "hover_score")},
            "fusion": active.fusion,
            "pol": {"state": pol_snap.get("state"), "progress": pol_snap.get("progress")},
            "clock": clock,
            "situation": list(self._situation),
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
        prior = self.store.get(event_id)
        prior_status = (prior or {}).get("operator_status") or ""
        audit = False
        if (
            not verdict.alert
            and verdict.status == "ok"
            and active.can_page
            and random.random() < max(0.0, min(1.0, self.cfg.vision.audit_rate))
        ):
            audit = True
            self.audit_shown += 1
        if verdict.status == "ok":
            self._verify_ok_until = time.monotonic() + max(5.0, self.cfg.vision.verify_fresh_seconds)
            self._verify_offline_since = None
        else:
            if self._verify_offline_since is None:
                self._verify_offline_since = time.time()
        paged_because = choose_paged_because(
            provenance=active.provenance,
            learning=active.learning,
            bag=active.bag,
            unusual=active.unusual,
            named=active.named,
            verify_status=verdict.status,
            alert=verdict.alert,
            audit=audit,
        )
        if prior_status in {"confirmed", "dismissed"}:
            self.store.insert_disagreement(
                event_id=event_id,
                source=self._source_stored(),
                detail=f"operator={prior_status} verdict.alert={verdict.alert}",
            )
            op_status = None
            stopped = None
            anomaly = bool((prior or {}).get("anomaly"))
        elif audit:
            op_status = "pending"
            stopped = "operator"
            anomaly = True
        elif verdict.status != "ok" and active.can_page:
            op_status = "unverified"
            stopped = "operator"
            anomaly = True
        elif verdict.alert:
            op_status = "pending"
            stopped = "operator"
            anomaly = True
        else:
            op_status = ""
            stopped = "hub"
            anomaly = False
        self.store.update_verdict(
            event_id,
            summary=verdict.summary,
            anomaly=anomaly,
            anomaly_reason=verdict.reason,
            verifier_provider=verdict.provider,
            verifier_status=verdict.status,
            operator_status=op_status,
            stopped_at=stopped,
            paged_because=paged_because,
        )
        self._paged_counts[paged_because] = self._paged_counts.get(paged_because, 0) + 1
        self.escalation_counts["hub_handoffs"] += 1
        if anomaly and op_status == "pending" and not audit:
            self.escalation_counts["hub_alerts"] += 1
        t0 = active.t0_mono or time.monotonic()
        self.verdict_latencies_ms.append((time.monotonic() - t0) * 1000.0)
        self.mqtt.publish("verdict", {"id": event_id, **verdict.as_dict()})
        with self._lock:
            self.status.verifier_provider = verdict.provider
            self.status.last_scene = verdict.summary
            self.status.last_anomaly = verdict.reason if anomaly else ""
        log.info(
            "Event %s verdict alert=%s status=%s provider=%s page=%s",
            event_id,
            verdict.alert,
            verdict.status,
            verdict.provider,
            op_status,
        )

    def _index_embedding(self, event_id: int, active: _ActiveEvent, classes: list[str]) -> None:
        if not self.embeddings or active.provenance != "live":
            return
        thumb = self.cfg.thumbs_dir / active.thumb_name
        if not thumb.is_file():
            return
        try:
            vec = thumb_hist(thumb)
            hour = datetime.now().hour
            self.embeddings.add(
                event_id,
                self._source_stored(),
                vec,
                hour,
                classes[0] if classes else "motion",
            )
            nov = self.embeddings.novelty(self._source_stored(), vec, k=self.cfg.embed.knn)
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
    now: float | None = None,
    mode_effective: str = "recall",
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
        dwell = t.dwell_at(now) if now is not None else t.dwell_s
        track_bits.append(f"#{t.track_id} {t.cls} {dwell:.0f}s")
    labels = ", ".join(track_bits) or (
        ", ".join(f"{d.cls} {d.conf:.2f}" for d in detections[:4]) or "unnamed"
    )
    steps = [
        {
            "stage": "edge",
            "label": SEAT_LABELS["edge"],
            "decision": edge_decision,
            "detail": edge_detail,
        }
    ]
    if not node_received:
        steps.append(
            {
                "stage": "node",
                "label": SEAT_LABELS["node"],
                "decision": "skipped",
                "detail": "Edge kept this locally · detector idle",
            }
        )
        steps.append(
            {
                "stage": "hub",
                "label": SEAT_LABELS["hub"],
                "decision": "skipped",
                "detail": "Detect did not escalate",
            }
        )
        steps.append(
            {
                "stage": "operator",
                "label": SEAT_LABELS["operator"],
                "decision": "skipped",
                "detail": "",
            }
        )
        stopped = "edge"
    else:
        if hub_needed:
            if mode_effective == "recall":
                node_detail = f"{labels} · namer (not a filter)"
                node_decision = "named"
            else:
                node_detail = f"{labels} · send to Verify"
                node_decision = "escalate"
        else:
            node_detail = f"{labels} · named and closed"
            node_decision = "closed"
        steps.append(
            {
                "stage": "node",
                "label": SEAT_LABELS["node"],
                "decision": node_decision,
                "detail": node_detail,
            }
        )
        if hub_needed:
            hub_decision = "alert" if page_operator else "log"
            steps.append(
                {
                    "stage": "hub",
                    "label": SEAT_LABELS["hub"],
                    "decision": hub_decision,
                    "detail": (hub_detail or anomaly_reason or labels)[:160],
                }
            )
            if page_operator:
                steps.append(
                    {
                        "stage": "operator",
                        "label": SEAT_LABELS["operator"],
                        "decision": "needed",
                        "detail": "Incident or Normal",
                    }
                )
                stopped = "operator"
            else:
                steps.append(
                    {
                        "stage": "operator",
                        "label": SEAT_LABELS["operator"],
                        "decision": "skipped",
                        "detail": "Verify logged without paging",
                    }
                )
                stopped = "hub"
        else:
            steps.append(
                {
                    "stage": "hub",
                    "label": SEAT_LABELS["hub"],
                    "decision": "skipped",
                    "detail": "Detect closed this",
                }
            )
            steps.append(
                {
                    "stage": "operator",
                    "label": SEAT_LABELS["operator"],
                    "decision": "skipped",
                    "detail": "",
                }
            )
            stopped = "node"

    edge_info = pol.as_dict()
    edge_info["upload"] = bool(node_received)
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
                    "xyxy": [int(v) for v in d.xyxy],
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
        "mode_effective": mode_effective,
    }


def _boxes_overlap(a: list[dict], b: list[dict], min_iou: float = 0.45) -> bool:
    if not a or not b:
        return False
    for left in a:
        lx = left.get("xyxy") or []
        if len(lx) != 4:
            continue
        for right in b:
            rx = right.get("xyxy") or []
            if len(rx) != 4:
                continue
            if _iou_xyxy(tuple(int(x) for x in lx), tuple(int(x) for x in rx)) >= min_iou:
                return True
    return False


def _iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
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


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])
