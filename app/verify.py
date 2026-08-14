from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re

import cv2
import numpy as np

from app.detect import Detection
from app.vision import complete_vision, fallback_summary
from app.config import VisionConfig

log = logging.getLogger(__name__)

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Verdict:
    alert: bool
    category: str
    confidence: float
    reason: str
    evidence_marks: list[int] = field(default_factory=list)
    provider: str = "off"
    status: str = "unavailable"
    summary: str = ""

    def as_dict(self) -> dict:
        return {
            "alert": self.alert,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "evidence_marks": list(self.evidence_marks),
            "provider": self.provider,
            "status": self.status,
            "summary": self.summary,
        }


def som_jpeg(
    frame: np.ndarray,
    detections: list[Detection],
    quality: int = 80,
) -> bytes:
    vis = frame.copy()
    for i, det in enumerate(detections[:12], start=1):
        x1, y1, x2, y2 = det.xyxy
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
        mark = f"{i}:{det.cls}"
        cv2.putText(
            vis,
            mark,
            (x1 + 4, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def build_prompt(*, context: dict, policy: str, classes: list[str]) -> str:
    ctx = json.dumps(context, default=str)
    return (
        "You are the Hub verifier for a building CCTV camera. "
        "The detector proposed the numbered boxes (Set-of-Mark). You adjudicate. "
        "Reply with JSON only, no markdown:\n"
        '{"alert": true or false, "category": "normal|unattended_object|after_hours|intrusion|other",'
        ' "confidence": 0.0-1.0, "reason": "one sentence", "evidence_marks": [1],'
        ' "caption": "search sentence"}\n'
        f"Policy:\n{policy.strip() or 'Alert on unattended bags and after-hours people without a badge.'}\n"
        f"Named classes: {', '.join(classes) or 'none'}\n"
        f"Context: {ctx}\n"
    )


def parse_verdict(text: str) -> dict | None:
    if not text:
        return None
    raw = text.strip()
    match = JSON_RE.search(raw)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def rule_verdict(
    *,
    classes: list[str],
    score: float,
    anomaly_reason: str,
    rule_alert: bool,
    provider: str,
    status: str,
) -> Verdict:
    summary = fallback_summary(classes, score, anomaly_reason)
    return Verdict(
        alert=rule_alert,
        category="other" if rule_alert else "normal",
        confidence=score if classes else 0.4,
        reason=anomaly_reason or summary,
        evidence_marks=[],
        provider=provider,
        status=status,
        summary=summary,
    )


def verify_event(
    cfg: VisionConfig,
    frame: np.ndarray | None,
    detections: list[Detection],
    *,
    classes: list[str],
    score: float,
    anomaly_reason: str,
    rule_alert: bool,
    context: dict,
) -> Verdict:
    """Detector proposes. VLM adjudicates. Fail-open to the rule check."""
    jpeg = som_jpeg(frame, detections) if frame is not None else b""
    prompt = build_prompt(context=context, policy=cfg.policy, classes=classes)
    text, provider, status = complete_vision(cfg, jpeg, prompt)
    if status != "ok" or not text:
        verdict = rule_verdict(
            classes=classes,
            score=score,
            anomaly_reason=anomaly_reason,
            rule_alert=rule_alert,
            provider=provider,
            status="unavailable",
        )
        log.info("Verifier unavailable (%s); fail-open rule_alert=%s", provider, rule_alert)
        return verdict
    data = parse_verdict(text)
    if not data:
        verdict = rule_verdict(
            classes=classes,
            score=score,
            anomaly_reason=anomaly_reason,
            rule_alert=rule_alert,
            provider=provider,
            status="malformed",
        )
        log.info("Verifier malformed JSON; fail-open rule_alert=%s", rule_alert)
        return verdict
    alert = bool(data.get("alert"))
    caption = str(data.get("caption") or data.get("reason") or "")
    marks = data.get("evidence_marks") or []
    if not isinstance(marks, list):
        marks = []
    return Verdict(
        alert=alert,
        category=str(data.get("category") or "other"),
        confidence=float(data.get("confidence") or score or 0.5),
        reason=str(data.get("reason") or anomaly_reason),
        evidence_marks=[int(m) for m in marks if str(m).isdigit() or isinstance(m, int)],
        provider=provider,
        status="ok",
        summary=caption or fallback_summary(classes, score, anomaly_reason),
    )
