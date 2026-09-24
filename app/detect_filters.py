"""Precision filters for YOLO boxes before tracking / overlays.

Reject empty-pavement, shadow, and frame-edge ghosts that YOLOv8n often
names as car/person/bicycle at high confidence on parking-lot CCTV.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.anomaly import BAG_CLASSES, normalize_class
from app.detect import Detection

# Classes that idle sweeps may still seed as new tracks (dwell / bags).
IDLE_SEED_CLASSES = frozenset({"person"} | set(BAG_CLASSES))

# Vehicles / bikes: common pavement-line FPs; require stricter conf + aspect.
VEHICLE_CLASSES = frozenset(
    {"car", "truck", "bus", "motorcycle", "bicycle", "airplane", "drone"}
)

# Default per-class minimum confidence (overrides global floor when higher).
DEFAULT_CLASS_CONF: dict[str, float] = {
    "car": 0.55,
    "truck": 0.55,
    "bus": 0.55,
    "motorcycle": 0.5,
    "bicycle": 0.5,
    "person": 0.45,
    "airplane": 0.55,
    "drone": 0.55,
    "backpack": 0.4,
    "handbag": 0.4,
    "suitcase": 0.4,
}


@dataclass(frozen=True)
class BoxFilterConfig:
    """Geometric / confidence gates applied after YOLO predict."""

    conf: float = 0.45
    class_conf: dict[str, float] | None = None
    min_box_area_frac: float = 0.0012
    min_side_px: int = 24
    edge_margin_frac: float = 0.02
    # Reject boxes that hug the frame edge (compression ghosts).
    edge_reject: bool = True


def _class_floor(cls: str, cfg: BoxFilterConfig) -> float:
    overrides = cfg.class_conf if cfg.class_conf is not None else DEFAULT_CLASS_CONF
    return max(float(cfg.conf), float(overrides.get(cls, cfg.conf)))


def _aspect_ok(cls: str, w: int, h: int) -> bool:
    """Reject absurd aspect ratios that match parking-line ghosts."""
    if w <= 0 or h <= 0:
        return False
    ratio = w / float(h)
    if cls in {"car", "truck", "bus"}:
        # Overhead / rear CCTV cars are wider than tall or near-square.
        # Tall thin "cars" on empty pavement are almost always FPs.
        return 0.55 <= ratio <= 4.0
    if cls == "bicycle":
        return 0.35 <= ratio <= 3.5
    if cls == "motorcycle":
        return 0.35 <= ratio <= 3.0
    if cls == "person":
        # Standing people are usually taller than wide; allow crouched / cut-off
        # figures up to ~1.8. Reject wide flat pavement blobs.
        return 0.18 <= ratio <= 1.8
    if cls in BAG_CLASSES:
        return 0.35 <= ratio <= 2.5
    return 0.15 <= ratio <= 6.0


def _edge_ghost(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_w: int,
    frame_h: int,
    margin_frac: float,
) -> bool:
    """True when the box is a thin strip glued to a frame edge (encoder ghost)."""
    if frame_w <= 0 or frame_h <= 0:
        return False
    mx = max(1, int(round(frame_w * margin_frac)))
    my = max(1, int(round(frame_h * margin_frac)))
    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    if bw <= 0 or bh <= 0:
        return True
    touches_left = x1 <= mx
    touches_right = x2 >= frame_w - mx
    touches_top = y1 <= my
    touches_bottom = y2 >= frame_h - my
    # Only reject very thin edge strips — partial people/cars at the border stay.
    # Wider edge-glued pavement "cars" are handled by motion_gate_detections.
    thin_w = max(mx, int(frame_w * 0.03))
    thin_h = max(my, int(frame_h * 0.03))
    if (touches_left or touches_right) and bw <= thin_w:
        return True
    if (touches_top or touches_bottom) and bh <= thin_h:
        return True
    # Corner scrap: two edges and small area.
    corners = sum([touches_left, touches_right, touches_top, touches_bottom])
    area_frac = (bw * bh) / float(frame_w * frame_h)
    if corners >= 2 and area_frac < 0.015:
        return True
    return False


def keep_detection(
    det: Detection,
    frame_wh: tuple[int, int],
    cfg: BoxFilterConfig | None = None,
) -> bool:
    """Return True if this box is plausible for operator-facing labels."""
    cfg = cfg or BoxFilterConfig()
    frame_w, frame_h = int(frame_wh[0]), int(frame_wh[1])
    x1, y1, x2, y2 = (int(v) for v in det.xyxy)
    if x2 <= x1 or y2 <= y1:
        return False
    cls = normalize_class(det.cls)
    if float(det.conf) + 1e-6 < _class_floor(cls, cfg):
        return False
    bw = x2 - x1
    bh = y2 - y1
    if bw < cfg.min_side_px or bh < cfg.min_side_px:
        return False
    area = bw * bh
    frame_area = max(1, frame_w * frame_h)
    if area / frame_area < cfg.min_box_area_frac:
        return False
    if not _aspect_ok(cls, bw, bh):
        return False
    if cfg.edge_reject and _edge_ghost(x1, y1, x2, y2, frame_w, frame_h, cfg.edge_margin_frac):
        return False
    return True


def filter_detections(
    detections: list[Detection],
    frame_wh: tuple[int, int],
    cfg: BoxFilterConfig | None = None,
) -> list[Detection]:
    return [d for d in detections if keep_detection(d, frame_wh, cfg)]


def overlaps_motion(
    xyxy: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    grid,
    *,
    min_cell: float = 0.05,
) -> bool:
    """True when any 8x8 motion cell under the box is active.

    Used to drop pavement-line FPs that fire far from the Edge trip blob
    (e.g. left-edge \"car\" while the cyclist moves on the right).
    """
    if grid is None:
        return True
    try:
        import numpy as np

        arr = np.asarray(grid, dtype=float)
    except Exception:
        return True
    if arr.ndim != 2 or arr.size == 0:
        return True
    rows, cols = arr.shape
    frame_w, frame_h = max(int(frame_wh[0]), 1), max(int(frame_wh[1]), 1)
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    if x2 <= x1 or y2 <= y1:
        return False
    c0 = max(0, min(cols - 1, int(x1 * cols / frame_w)))
    c1 = max(0, min(cols - 1, int((x2 - 1) * cols / frame_w)))
    r0 = max(0, min(rows - 1, int(y1 * rows / frame_h)))
    r1 = max(0, min(rows - 1, int((y2 - 1) * rows / frame_h)))
    if c1 < c0:
        c0, c1 = c1, c0
    if r1 < r0:
        r0, r1 = r1, r0
    patch = arr[r0 : r1 + 1, c0 : c1 + 1]
    if patch.size == 0:
        return False
    return float(patch.max()) >= float(min_cell)


# Classes that must sit on the Edge motion blob (bags exempt — still objects).
MOTION_GATED_CLASSES = frozenset(VEHICLE_CLASSES | {"person"})


def motion_gate_detections(
    detections: list[Detection],
    frame_wh: tuple[int, int],
    grid,
    existing_xyxy: list[tuple[int, int, int, int]],
    *,
    min_cell: float = 0.05,
    iou_match: float = 0.3,
) -> list[Detection]:
    """On Edge trips, drop motion-gated classes whose box misses the motion grid.

    Existing tracks may still update (object can pause briefly). Bags skip the gate.
    """
    kept: list[Detection] = []
    for det in detections:
        cls = normalize_class(det.cls)
        if cls not in MOTION_GATED_CLASSES:
            kept.append(det)
            continue
        if any(_iou(det.xyxy, box) >= iou_match for box in existing_xyxy):
            kept.append(det)
            continue
        if overlaps_motion(det.xyxy, frame_wh, grid, min_cell=min_cell):
            kept.append(det)
    return kept


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


def idle_seed_detections(
    detections: list[Detection],
    existing_xyxy: list[tuple[int, int, int, int]],
    *,
    iou_match: float = 0.3,
) -> list[Detection]:
    """On idle sweeps, only seed bag/person tracks; keep dets that match live tracks.

    Still bags/people need idle YOLO. Vehicle pavement FPs must not spawn new
    tracks when Edge did not trip.
    """
    kept: list[Detection] = []
    for det in detections:
        cls = normalize_class(det.cls)
        if cls in IDLE_SEED_CLASSES:
            kept.append(det)
            continue
        if any(_iou(det.xyxy, box) >= iou_match for box in existing_xyxy):
            kept.append(det)
    return kept


def focus_detections(
    detections: list[Detection],
    track_id: int | None,
    *,
    also: set[int] | None = None,
) -> list[Detection]:
    """Prefer the trip/primary track for Review thumbs and event box lists."""
    if track_id is None and not also:
        return list(detections)
    want = set(also or ())
    if track_id is not None:
        want.add(int(track_id))
    focused = [d for d in detections if d.track_id is not None and int(d.track_id) in want]
    return focused if focused else list(detections)
