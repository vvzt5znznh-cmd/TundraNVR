"""Template situation lines from tracks. Facts only — not a narrator."""

from __future__ import annotations

from app.track import Track, _center, _dist, unattended_bags

VEHICLE = frozenset({"car", "bus", "truck", "motorcycle"})
NEAR_VEHICLE_PX = 160
MAX_LINES = 6


def situation_lines(
    tracks: list[Track],
    *,
    now: float,
    bags: list[Track] | None = None,
    bag_radius: float = 120.0,
    near_radius: float = NEAR_VEHICLE_PX,
    max_lines: int = MAX_LINES,
) -> list[str]:
    """Short operator lines. Never says entered/returned; track ids are not identity."""
    lines: list[str] = []
    seen: set[str] = set()

    def add(line: str) -> None:
        if line and line not in seen and len(lines) < max_lines:
            seen.add(line)
            lines.append(line)

    flagged = bags if bags is not None else unattended_bags(tracks, now=now)
    bag_ids = {t.track_id for t in flagged}
    for bag in flagged:
        dwell = int(round(bag.dwell_at(now)))
        add(
            f"#{bag.track_id} {bag.cls} {dwell}s, no person within {int(bag_radius)}px"
        )

    people = [t for t in tracks if t.cls == "person"]
    vehicles = [t for t in tracks if t.cls in VEHICLE]
    for vehicle in vehicles:
        vc = _center(vehicle.xyxy)
        for person in people:
            if _dist(vc, _center(person.xyxy)) <= near_radius:
                add(f"#{vehicle.track_id} {vehicle.cls} · #{person.track_id} person nearby")

    for person in people:
        if person.track_id in bag_ids:
            continue
        dwell = int(round(person.dwell_at(now)))
        if person.zone_name:
            add(f"#{person.track_id} person {dwell}s in {person.zone_name}")
        elif dwell >= 8:
            add(f"#{person.track_id} person {dwell}s")

    return lines
