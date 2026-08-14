from __future__ import annotations

from dataclasses import dataclass, field
import math

from app.anomaly import BAG_CLASSES, normalize_class
from app.detect import Detection

HIGH_IOU = 0.3
MAX_TRAJ = 120


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


def _center(xyxy: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class Track:
    track_id: int
    cls: str
    conf: float
    xyxy: tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    hits: int = 1
    time_since_update: int = 0
    class_hist: dict[str, int] = field(default_factory=dict)
    traj: list[tuple[float, float, float]] = field(default_factory=list)
    zone_name: str = ""
    zone_entries: list[str] = field(default_factory=list)
    zone_exits: list[str] = field(default_factory=list)
    event_id: int | None = None
    confirmed: bool = False

    def as_detection(self) -> Detection:
        return Detection(cls=self.cls, conf=self.conf, xyxy=self.xyxy, track_id=self.track_id)

    @property
    def dwell_s(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    @property
    def path_length(self) -> float:
        if len(self.traj) < 2:
            return 0.0
        return sum(_dist(self.traj[i][1:], self.traj[i + 1][1:]) for i in range(len(self.traj) - 1))

    def speeds(self) -> tuple[float, float]:
        if len(self.traj) < 2:
            return 0.0, 0.0
        vals: list[float] = []
        for i in range(1, len(self.traj)):
            dt = self.traj[i][0] - self.traj[i - 1][0]
            if dt <= 1e-3:
                continue
            vals.append(_dist(self.traj[i][1:], self.traj[i - 1][1:]) / dt)
        if not vals:
            return 0.0, 0.0
        return float(sum(vals) / len(vals)), float(max(vals))

    def direction(self) -> float:
        if len(self.traj) < 2:
            return 0.0
        x0, y0 = self.traj[0][1:]
        x1, y1 = self.traj[-1][1:]
        return math.degrees(math.atan2(y1 - y0, x1 - x0))

    def vertical_extent(self, frame_h: int) -> float:
        h = max(1, self.xyxy[3] - self.xyxy[1])
        return h / max(frame_h, 1)

    def hover_score(self) -> float:
        mean_s, _ = self.speeds()
        dwell = self.dwell_s
        if dwell < 1.0:
            return 0.0
        return float(min(1.0, (dwell / 12.0) * (1.0 / (1.0 + mean_s / 40.0))))

    def turn_rate(self) -> float:
        if len(self.traj) < 3:
            return 0.0
        headings: list[float] = []
        for i in range(1, len(self.traj)):
            x0, y0 = self.traj[i - 1][1:]
            x1, y1 = self.traj[i][1:]
            headings.append(math.atan2(y1 - y0, x1 - x0))
        if len(headings) < 2:
            return 0.0
        deltas = []
        for i in range(1, len(headings)):
            d = headings[i] - headings[i - 1]
            while d > math.pi:
                d -= 2 * math.pi
            while d < -math.pi:
                d += 2 * math.pi
            dt = self.traj[i + 1][0] - self.traj[i][0]
            if dt > 1e-3:
                deltas.append(abs(d) / dt)
        return float(sum(deltas) / len(deltas)) if deltas else 0.0

    def features(self, frame_h: int = 720) -> dict:
        mean_s, peak_s = self.speeds()
        return {
            "track_id": self.track_id,
            "cls": self.cls,
            "dwell_s": round(self.dwell_s, 2),
            "path_length": round(self.path_length, 1),
            "mean_speed": round(mean_s, 2),
            "peak_speed": round(peak_s, 2),
            "direction": round(self.direction(), 1),
            "vertical_extent": round(self.vertical_extent(frame_h), 3),
            "hover_score": round(self.hover_score(), 3),
            "turn_rate": round(self.turn_rate(), 3),
            "zone": self.zone_name,
            "zone_entries": list(self.zone_entries),
            "zone_exits": list(self.zone_exits),
            "hits": self.hits,
        }


class ByteTracker:
    """Greedy IoU tracker (ByteTrack-style high-score association)."""

    def __init__(self, max_age: int = 15, min_hits: int = 2, iou_match: float = HIGH_IOU) -> None:
        self.max_age = max(1, max_age)
        self.min_hits = max(1, min_hits)
        self.iou_match = iou_match
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    def update(
        self,
        detections: list[Detection],
        now: float,
        frame_wh: tuple[int, int] | None = None,
        zone_of=None,
    ) -> list[Track]:
        dets = [d for d in detections if d.xyxy[2] > d.xyxy[0] and d.xyxy[3] > d.xyxy[1]]
        unmatched_tracks = set(self.tracks)
        unmatched_dets = set(range(len(dets)))
        pairs: list[tuple[float, int, int]] = []
        for tid, tr in self.tracks.items():
            for di, det in enumerate(dets):
                if normalize_class(tr.cls) != normalize_class(det.cls):
                    continue
                score = _iou(tr.xyxy, det.xyxy)
                if score >= self.iou_match:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True)
        used_t: set[int] = set()
        used_d: set[int] = set()
        for _, tid, di in pairs:
            if tid in used_t or di in used_d:
                continue
            used_t.add(tid)
            used_d.add(di)
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)
            self._observe(self.tracks[tid], dets[di], now, frame_wh, zone_of)

        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]
            tr.time_since_update += 1
            if tr.time_since_update > self.max_age:
                del self.tracks[tid]

        for di in unmatched_dets:
            det = dets[di]
            tid = self._next_id
            self._next_id += 1
            cx, cy = _center(det.xyxy)
            label = normalize_class(det.cls)
            zone = zone_of(cx, cy) if zone_of else ""
            self.tracks[tid] = Track(
                track_id=tid,
                cls=label,
                conf=det.conf,
                xyxy=det.xyxy,
                first_seen=now,
                last_seen=now,
                class_hist={label: 1},
                traj=[(now, cx, cy)],
                zone_name=zone or "",
                zone_entries=[zone] if zone else [],
            )

        live = []
        for tr in self.tracks.values():
            if tr.hits >= self.min_hits:
                tr.confirmed = True
            if tr.confirmed and tr.time_since_update == 0:
                live.append(tr)
            elif tr.confirmed:
                live.append(tr)
        return sorted(live, key=lambda t: t.track_id)

    def lost(self) -> list[Track]:
        return []

    def _observe(self, tr: Track, det: Detection, now: float, frame_wh, zone_of) -> None:
        tr.xyxy = det.xyxy
        tr.conf = max(tr.conf, det.conf)
        label = normalize_class(det.cls)
        tr.cls = label
        tr.class_hist[label] = tr.class_hist.get(label, 0) + 1
        tr.hits += 1
        tr.time_since_update = 0
        tr.last_seen = now
        cx, cy = _center(det.xyxy)
        tr.traj.append((now, cx, cy))
        if len(tr.traj) > MAX_TRAJ:
            tr.traj = tr.traj[-MAX_TRAJ:]
        if zone_of:
            zone = zone_of(cx, cy) or ""
            if zone != tr.zone_name:
                if tr.zone_name:
                    tr.zone_exits.append(tr.zone_name)
                if zone:
                    tr.zone_entries.append(zone)
                tr.zone_name = zone


def unattended_bags(
    tracks: list[Track],
    *,
    dwell_seconds: float = 8.0,
    person_radius: float = 120.0,
) -> list[Track]:
    """Bag track persists > T seconds with no person track within R pixels."""
    people = [t for t in tracks if t.cls == "person" and t.time_since_update <= 2]
    flagged: list[Track] = []
    for bag in tracks:
        if bag.cls not in BAG_CLASSES:
            continue
        if bag.dwell_s < dwell_seconds:
            continue
        bc = _center(bag.xyxy)
        near = any(_dist(bc, _center(p.xyxy)) <= person_radius for p in people)
        if not near:
            flagged.append(bag)
    return flagged
