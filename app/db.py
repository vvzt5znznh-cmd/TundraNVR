from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_start TEXT NOT NULL,
                    ts_end TEXT,
                    classes TEXT NOT NULL,
                    score REAL,
                    thumb_path TEXT,
                    clip_path TEXT,
                    summary TEXT,
                    anomaly INTEGER DEFAULT 0,
                    anomaly_reason TEXT,
                    source TEXT,
                    pol_score REAL,
                    stopped_at TEXT,
                    handoff TEXT,
                    features TEXT,
                    operator_status TEXT
                )
                """
            )
            cols = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(events)").fetchall()
            }
            for name, ddl in (
                ("summary", "ALTER TABLE events ADD COLUMN summary TEXT"),
                ("anomaly", "ALTER TABLE events ADD COLUMN anomaly INTEGER DEFAULT 0"),
                ("anomaly_reason", "ALTER TABLE events ADD COLUMN anomaly_reason TEXT"),
                ("source", "ALTER TABLE events ADD COLUMN source TEXT"),
                ("pol_score", "ALTER TABLE events ADD COLUMN pol_score REAL"),
                ("stopped_at", "ALTER TABLE events ADD COLUMN stopped_at TEXT"),
                ("handoff", "ALTER TABLE events ADD COLUMN handoff TEXT"),
                ("features", "ALTER TABLE events ADD COLUMN features TEXT"),
                ("operator_status", "ALTER TABLE events ADD COLUMN operator_status TEXT"),
            ):
                if name not in cols:
                    self._conn.execute(ddl)
            self._conn.commit()

    def insert(
        self,
        ts_start: str,
        ts_end: str | None,
        classes: list[str],
        score: float,
        thumb_path: str | None,
        clip_path: str | None,
        anomaly: bool = False,
        anomaly_reason: str = "",
        source: str = "",
        pol_score: float | None = None,
        stopped_at: str = "",
        handoff: dict[str, Any] | None = None,
        features: dict[str, Any] | None = None,
        operator_status: str = "",
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO events (
                    ts_start, ts_end, classes, score, thumb_path, clip_path,
                    anomaly, anomaly_reason, source, pol_score, stopped_at,
                    handoff, features, operator_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_start,
                    ts_end,
                    json.dumps(classes),
                    score,
                    thumb_path,
                    clip_path,
                    1 if anomaly else 0,
                    anomaly_reason or None,
                    source or None,
                    pol_score,
                    stopped_at or None,
                    json.dumps(handoff) if handoff is not None else None,
                    json.dumps(features) if features is not None else None,
                    operator_status or None,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def update_summary(self, event_id: int, summary: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET summary = ? WHERE id = ?",
                (summary, event_id),
            )
            self._conn.commit()

    def update_review(self, event_id: int, operator_status: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
            if row is None:
                return None
            self._conn.execute(
                "UPDATE events SET operator_status = ? WHERE id = ?",
                (operator_status, event_id),
            )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list_events(self, limit: int = 50, alerts_only: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            if alerts_only:
                rows = self._conn.execute(
                    "SELECT * FROM events WHERE anomaly = 1 ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM events ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get(self, event_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def older_than(self, cutoff_iso: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE ts_start < ?",
                (cutoff_iso,),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def delete(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["classes"] = json.loads(data["classes"])
        except (TypeError, json.JSONDecodeError):
            data["classes"] = []
        data["anomaly"] = bool(data.get("anomaly"))
        data["anomaly_reason"] = data.get("anomaly_reason") or ""
        data["source"] = data.get("source") or ""
        data["stopped_at"] = data.get("stopped_at") or ""
        data["operator_status"] = data.get("operator_status") or ""
        data["handoff"] = _load_json(data.get("handoff"), {})
        data["features"] = _load_json(data.get("features"), {})
        return data


def _load_json(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def utc_now() -> str:
    return _utc_now()
