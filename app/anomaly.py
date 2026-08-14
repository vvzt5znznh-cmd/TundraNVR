from __future__ import annotations

from dataclasses import dataclass

BAG_CLASSES = frozenset({"backpack", "handbag", "suitcase"})
DRONE_ALIASES = frozenset(
    {"drone", "quadcopter", "fixed-wing", "fixed_wing", "uav", "uav_drone"}
)


@dataclass(frozen=True)
class AnomalyResult:
    flagged: bool
    reason: str


def normalize_class(name: str) -> str:
    label = (name or "").strip().lower().replace(" ", "-")
    if label in DRONE_ALIASES:
        return "drone"
    return label


def classify_anomaly(
    classes: list[str] | set[str],
    *,
    expected: list[str],
    alert: list[str],
    unattended_bags: bool = True,
) -> AnomalyResult:
    """Rule-based fail-open check. Unattended bags are a track rule in app.track, not this."""
    seen = {normalize_class(c) for c in classes if c}
    alert_set = {normalize_class(c) for c in alert}
    expected_set = {normalize_class(c) for c in expected} | alert_set

    hits = sorted(seen & alert_set)
    if hits:
        return AnomalyResult(True, "alert: " + ", ".join(hits))
    if unattended_bags and (seen & BAG_CLASSES) and "person" not in seen:
        bags = ", ".join(sorted(seen & BAG_CLASSES))
        return AnomalyResult(True, f"unattended bag ({bags})")
    unexpected = sorted(seen - expected_set)
    if unexpected:
        return AnomalyResult(True, "unexpected: " + ", ".join(unexpected))
    return AnomalyResult(False, "")
