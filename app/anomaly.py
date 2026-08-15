from __future__ import annotations

BAG_CLASSES = frozenset({"backpack", "handbag", "suitcase"})
DRONE_ALIASES = frozenset(
    {"drone", "quadcopter", "fixed-wing", "fixed_wing", "uav", "uav_drone"}
)


def normalize_class(name: str) -> str:
    label = (name or "").strip().lower().replace(" ", "-")
    if label in DRONE_ALIASES:
        return "drone"
    return label
