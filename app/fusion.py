from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass
class FusionState:
    badge_within_window: bool = False
    door_open: bool = False
    source: str = "fixture"

    def as_dict(self) -> dict:
        return {
            "badge_within_window": self.badge_within_window,
            "door_open": self.door_open,
            "source": self.source,
        }


class FusionBus:
    """Decision-level sensor context. Fixture file or in-memory badge pings."""

    def __init__(self, fixture: Path | None = None, window_seconds: float = 60.0) -> None:
        self.window = max(1.0, window_seconds)
        self._badges: list[float] = []
        self._door_open = False
        self._t0: float | None = None
        if fixture and fixture.is_file():
            self._load(fixture)

    def _load(self, path: Path) -> None:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                row = json.loads(line)
                t = row.get("t")
                if t is not None:
                    self._badges.append(float(t))
                if row.get("door_open"):
                    self._door_open = True
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Fusion fixture unreadable: %s", exc)

    def note_start(self, now: float) -> None:
        if self._t0 is None:
            self._t0 = now

    def ingest_badge(self, now: float) -> None:
        self._badges.append(self._rel(now))

    def snapshot(self, now: float) -> FusionState:
        self.note_start(now)
        rel = self._rel(now)
        hit = any(abs(rel - t) <= self.window for t in self._badges)
        return FusionState(badge_within_window=hit, door_open=self._door_open)

    def _rel(self, now: float) -> float:
        return now - (self._t0 or now)


def clock_context() -> dict:
    now = datetime.now().astimezone()
    utc = datetime.now(timezone.utc)
    return {
        "local_time": now.strftime("%H:%M"),
        "weekday": now.strftime("%A"),
        "iso": utc.isoformat(),
        "hour": now.hour,
        "week_hour": now.weekday() * 24 + now.hour,
    }
