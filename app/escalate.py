"""Recall-oriented Node → Hub escalation.

Small VLMs are conservatively biased (high precision, recall collapse).
Do not gate Hub on YOLO softmax or VLM confidence. Edge over-sends;
Hub suppresses. `pol_score` is the previous PoL ≥ threshold behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EscalationDecision:
    hub_needed: bool
    reason: str
    mode: str


def decide_hub(
    *,
    mode: str,
    node_received: bool,
    named: bool,
    bag: bool,
    pol_confident: bool,
    pol_score: float,
    pol_min: float,
    no_badge: bool,
) -> EscalationDecision:
    """Return whether Node should hand this trip to Hub.

    `named` is any detector/track label. Confidence of that label is ignored.
    """
    mode = (mode or "recall").strip().lower()
    if mode not in {"recall", "pol_score"}:
        mode = "recall"
    if not node_received:
        return EscalationDecision(False, "", mode)
    if bag:
        return EscalationDecision(True, "unattended bag (track)", mode)
    unsure = not named
    if mode == "recall":
        if unsure:
            return EscalationDecision(True, "Node could not name this", mode)
        return EscalationDecision(True, "recall-oriented proposal", mode)
    still_high = bool(named and pol_confident and pol_score >= pol_min)
    if unsure:
        return EscalationDecision(True, "Node could not name this", mode)
    if still_high:
        reason = "still unusual after naming"
        if no_badge:
            reason = "no badge within window"
        return EscalationDecision(True, reason, mode)
    return EscalationDecision(False, "", mode)
