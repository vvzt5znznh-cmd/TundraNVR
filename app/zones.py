from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: list[tuple[float, float]]


def point_in_poly(x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
    if len(polygon) < 3:
        return False
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


class ZoneMap:
    def __init__(self, zones: list[Zone], frame_w: int = 1280, frame_h: int = 720) -> None:
        self.zones = zones
        self.frame_w = max(1, frame_w)
        self.frame_h = max(1, frame_h)

    def name_at(self, x: float, y: float) -> str:
        nx = x / self.frame_w
        ny = y / self.frame_h
        for zone in self.zones:
            if point_in_poly(nx, ny, zone.polygon):
                return zone.name
        return ""
