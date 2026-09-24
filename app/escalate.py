"""Recall-oriented Detect → Verify escalation.

Small VLMs are conservatively biased (high precision, recall collapse).
Do not gate Verify on YOLO softmax or VLM confidence. Edge over-sends;
Verify suppresses. `pol_score` is the previous PoL ≥ threshold behaviour.
`auto` uses recall while Verify is healthy, else pol_score.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("auto", "recall", "pol_score")


@dataclass(frozen=True)
class EscalationDecision:
    hub_needed: bool
    reason: str
    mode: str


def effective_mode(mode: str, *, verify_healthy: bool) -> str:
    raw = (mode or "auto").strip().lower()
    if raw not in MODES:
        raw = "auto"
    if raw == "auto":
        return "recall" if verify_healthy else "pol_score"
    return raw


def decide_hub(
    *,
    mode: str,
    node_received: bool,
    named: bool,
    bag: bool,
    pol_confident: bool,
    pol_score: float,
    pol_min: float,
    verify_healthy: bool = True,
) -> EscalationDecision:
    """Return whether Detect should hand this trip to Verify.

    `named` is any detector/track label. Confidence of that label is ignored.
    """
    resolved = effective_mode(mode, verify_healthy=verify_healthy)
    if not node_received:
        return EscalationDecision(False, "", resolved)
    if bag:
        return EscalationDecision(True, "unattended bag (track)", resolved)
    unsure = not named
    if resolved == "recall":
        if unsure:
            return EscalationDecision(True, "Detect could not name this", resolved)
        return EscalationDecision(True, "Detect named this — asking Verify", resolved)
    still_high = bool(named and pol_confident and pol_score >= pol_min)
    if unsure:
        return EscalationDecision(True, "Detect could not name this", resolved)
    if still_high:
        return EscalationDecision(True, "still unusual after naming", resolved)
    return EscalationDecision(False, "", resolved)
