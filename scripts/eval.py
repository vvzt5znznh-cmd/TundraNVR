#!/usr/bin/env python3
"""Replay labelled footage through the pipeline and emit an ablation table.

Headline PoC numbers must come from a labelled hold-out of *our* building camera.
Sample clips, CAVIAR, and this --smoke path are fixtures only — never headline.
IEC 62676-4 mapping: NAR ≈ operator alerts per camera-day; Pd ≈ event-level
recall; FAR proxy ≈ 1 − precision. Confidence Level is not estimated here.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.detect import Detection
from app.pipeline import Pipeline


STAGES = ("motion", "detect", "track", "verifier", "fusion", "novelty")
FIXTURE_BANNER = (
    "eval fixture — not headline numbers. Do not report NAR/Pd/FAR from "
    "synthetic frames, CAVIAR, or bundled sample clips."
)


class FakeDetector:
    """Puts a person box on the moving blob so tracking can run without YOLO."""

    def __init__(self) -> None:
        self._model = "fake"

    def load(self) -> None:
        return None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        if w * h < 80:
            return []
        return [Detection("person", 0.9, (x, y, x + w, y + h))]


def synthetic_frames(n: int = 24, w: int = 320, h: int = 180) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        x = 20 + i * 8
        cv2.rectangle(img, (x, 60), (x + 40, 140), (200, 200, 200), -1)
        frames.append(img)
    return frames


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(0.95 * (len(ordered) - 1))]


def run_stage(stage: str, frames: list[np.ndarray], labels: dict) -> dict:
    import tempfile

    cfg = load_config()
    cfg.root = Path(tempfile.mkdtemp(prefix="tundra-eval-"))
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.clips_dir.mkdir(parents=True, exist_ok=True)
    cfg.thumbs_dir.mkdir(parents=True, exist_ok=True)
    cfg.pol_dir.mkdir(parents=True, exist_ok=True)
    cfg.camera.width = frames[0].shape[1]
    cfg.camera.height = frames[0].shape[0]
    cfg.events.write_media = False
    cfg.escalation.mode = "recall"
    cfg.vision.enabled = stage in {"verifier", "fusion", "novelty"}
    cfg.fusion.enabled = stage in {"fusion", "novelty"}
    cfg.embed.enabled = stage == "novelty"
    cfg.embed.gate_alerts = False
    pipe = Pipeline(cfg)
    if stage == "motion":
        pipe.detector.detect = lambda frame: []  # type: ignore[method-assign]
    else:
        pipe.detector = FakeDetector()  # type: ignore[assignment]
    t0 = time.perf_counter()
    ingest_ms = []
    now = 0.0
    for frame in frames:
        s = time.perf_counter()
        pipe.ingest_frame(frame, now)
        ingest_ms.append((time.perf_counter() - s) * 1000.0)
        now += 0.2
    pipe.flush()
    elapsed = time.perf_counter() - t0
    events = pipe.store.list_events(200)
    alerts = [e for e in events if e.get("anomaly")]
    labelled = labels.get("events") or []
    true_alerts = [x for x in labelled if x.get("alert")]
    # Event-level, not frame-level. Count match only — no IoU on this fixture.
    tp = min(len(alerts), len(true_alerts)) if true_alerts else 0
    recall = (tp / len(true_alerts)) if true_alerts else (1.0 if not alerts else 1.0)
    precision = (tp / len(alerts)) if alerts else (1.0 if not true_alerts else 0.0)
    far_proxy = round(1.0 - precision, 3)
    verdict_ms = list(pipe.verdict_latencies_ms) or ingest_ms
    p95 = _p95(verdict_ms)
    duration_s = max(0.2 * len(frames), 1e-6)
    cam_days = duration_s / 86400.0
    nar = len(alerts) / cam_days if cam_days else 0.0
    tracks = {e.get("track_id") for e in events if e.get("track_id") is not None}
    drone_events = [e for e in events if "drone" in (e.get("classes") or [])]
    esc = pipe.escalation_counts
    rasp = esc["raspberry_trips"]
    node = esc["node_proposals"]
    hub_h = esc["hub_handoffs"]
    hub_a = esc["hub_alerts"]
    confirms = sum(1 for e in events if (e.get("operator_status") or "") == "confirmed")
    pipe.store.close()
    return {
        "stage": stage,
        "alerts": len(alerts),
        "events": len(events),
        "tracks": len(tracks),
        "nar_per_cam_day": round(nar, 1),
        "alerts_per_cam_day": round(nar, 1),
        "pd_recall": round(recall, 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "far_proxy": far_proxy,
        "p95_ms": round(p95, 1),
        "elapsed_s": round(elapsed, 3),
        "drone_events": len(drone_events),
        "raspberry_trips": rasp,
        "node_proposals": node,
        "hub_handoffs": hub_h,
        "hub_alerts": hub_a,
        "operator_confirms": confirms,
        "node_per_raspberry": round(node / rasp, 3) if rasp else 0.0,
        "hub_per_node": round(hub_h / node, 3) if node else 0.0,
        "alerts_per_hub": round(hub_a / hub_h, 3) if hub_h else 0.0,
        "verifier_unavailable": sum(
            1 for e in events if (e.get("verifier_status") or "") == "unavailable"
        ),
    }


def write_table(rows: list[dict], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stage",
        "nar_per_cam_day",
        "pd_recall",
        "far_proxy",
        "precision",
        "recall",
        "p95_ms",
        "events",
        "tracks",
        "raspberry_trips",
        "node_proposals",
        "hub_handoffs",
        "hub_alerts",
        "operator_confirms",
        "node_per_raspberry",
        "hub_per_node",
        "alerts_per_hub",
        "drone_events",
    ]
    with dest.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Ablation (fixture — not headline numbers)",
        "",
        FIXTURE_BANNER,
        "",
        "IEC 62676-4 mapping: **NAR** ≈ operator alerts/camera-day; **Pd** ≈ event-level recall; "
        "**FAR proxy** ≈ 1 − precision. Confidence Level is not estimated on this fixture.",
        "",
        "| Stage | NAR/cam-day | Pd (recall) | FAR proxy | Precision | P95 ms | Raspberry | Node | Hub handoffs | Hub alerts | Confirms | Node/Rasp | Hub/Node |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['nar_per_cam_day']} | {row['pd_recall']} | "
            f"{row['far_proxy']} | {row['precision']} | {row['p95_ms']} | "
            f"{row['raspberry_trips']} | {row['node_proposals']} | {row['hub_handoffs']} | "
            f"{row['hub_alerts']} | {row['operator_confirms']} | "
            f"{row['node_per_raspberry']} | {row['hub_per_node']} |"
        )
    lines.append("")
    lines.append(
        "Drone metrics are a separate column (`drone_events`) and must not be folded into the headline sentence."
    )
    dest.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Synthetic frames; CI fixture only")
    parser.add_argument("--labels", type=Path, default=ROOT / "data/eval/labels.example.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data/eval/ablation")
    parser.add_argument("--video", type=Path, default=None)
    args = parser.parse_args()
    labels = {}
    if args.labels.is_file():
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
    if args.smoke or not args.video:
        frames = synthetic_frames()
        print(FIXTURE_BANNER, file=sys.stderr)
    else:
        cap = cv2.VideoCapture(str(args.video))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if len(frames) >= 400:
                break
        cap.release()
        if not frames:
            print("no frames", file=sys.stderr)
            return 1
        print(
            "eval on provided video — still not a site hold-out unless labels are from that camera",
            file=sys.stderr,
        )
    rows = [run_stage(stage, frames, labels) for stage in STAGES]
    write_table(rows, args.out)
    print(args.out.with_suffix(".md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
