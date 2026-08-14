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
                    operator_status TEXT,
                    track_id INTEGER,
                    verifier_provider TEXT,
                    verifier_status TEXT,
                    novelty_score REAL
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
                ("track_id", "ALTER TABLE events ADD COLUMN track_id INTEGER"),
                ("verifier_provider", "ALTER TABLE events ADD COLUMN verifier_provider TEXT"),
                ("verifier_status", "ALTER TABLE events ADD COLUMN verifier_status TEXT"),
                ("novelty_score", "ALTER TABLE events ADD COLUMN novelty_score REAL"),
            ):
                if name not in cols:
                    self._conn.execute(ddl)
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    cls TEXT,
                    first_seen TEXT,
                    last_seen TEXT,
                    dwell_s REAL,
                    path_length REAL,
                    mean_speed REAL,
                    peak_speed REAL,
                    direction REAL,
                    vertical_extent REAL,
                    hover_score REAL,
                    turn_rate REAL,
                    zone TEXT,
                    class_hist TEXT,
                    trajectory TEXT,
                    event_id INTEGER,
                    PRIMARY KEY (id, source)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    event_id INTEGER PRIMARY KEY,
                    source TEXT,
                    vector BLOB,
                    hour INTEGER,
                    cls TEXT
                )
                """
            )
            self._conn.commit()

    def enable_sqlite_vec(self) -> bool:
        try:
            import sqlite_vec
        except ImportError:
            return False
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            return True
        except Exception:
            return False

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
        track_id: int | None = None,
        verifier_provider: str = "",
        verifier_status: str = "",
        novelty_score: float | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO events (
                    ts_start, ts_end, classes, score, thumb_path, clip_path,
                    anomaly, anomaly_reason, source, pol_score, stopped_at,
                    handoff, features, operator_status, track_id,
                    verifier_provider, verifier_status, novelty_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    track_id,
                    verifier_provider or None,
                    verifier_status or None,
                    novelty_score,
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

    def update_verdict(
        self,
        event_id: int,
        *,
        summary: str,
        anomaly: bool,
        anomaly_reason: str,
        verifier_provider: str,
        verifier_status: str,
        novelty_score: float | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE events SET summary = ?, anomaly = ?, anomaly_reason = ?,
                    verifier_provider = ?, verifier_status = ?, novelty_score = ?
                WHERE id = ?
                """,
                (
                    summary,
                    1 if anomaly else 0,
                    anomaly_reason,
                    verifier_provider,
                    verifier_status,
                    novelty_score,
                    event_id,
                ),
            )
            self._conn.commit()

    def event_for_track(self, track_id: int, source: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE track_id = ? AND source = ? ORDER BY id DESC LIMIT 1",
                (track_id, source),
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def upsert_track(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tracks (
                    id, source, cls, first_seen, last_seen, dwell_s, path_length,
                    mean_speed, peak_speed, direction, vertical_extent, hover_score,
                    turn_rate, zone, class_hist, trajectory, event_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id, source) DO UPDATE SET
                    cls=excluded.cls, last_seen=excluded.last_seen, dwell_s=excluded.dwell_s,
                    path_length=excluded.path_length, mean_speed=excluded.mean_speed,
                    peak_speed=excluded.peak_speed, direction=excluded.direction,
                    vertical_extent=excluded.vertical_extent, hover_score=excluded.hover_score,
                    turn_rate=excluded.turn_rate, zone=excluded.zone,
                    class_hist=excluded.class_hist, trajectory=excluded.trajectory,
                    event_id=COALESCE(excluded.event_id, tracks.event_id)
                """,
                (
                    int(payload["id"]),
                    payload.get("source"),
                    payload.get("cls"),
                    payload.get("first_seen"),
                    payload.get("last_seen"),
                    payload.get("dwell_s"),
                    payload.get("path_length"),
                    payload.get("mean_speed"),
                    payload.get("peak_speed"),
                    payload.get("direction"),
                    payload.get("vertical_extent"),
                    payload.get("hover_score"),
                    payload.get("turn_rate"),
                    payload.get("zone"),
                    json.dumps(payload.get("class_hist") or {}),
                    json.dumps(payload.get("trajectory") or []),
                    payload.get("event_id"),
                ),
            )
            self._conn.commit()

    def get_track(self, track_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tracks WHERE id = ?",
                (track_id,),
            ).fetchone()
        if not row:
            return None
        data = dict(row)
        data["class_hist"] = _load_json(data.get("class_hist"), {})
        data["trajectory"] = _load_json(data.get("trajectory"), [])
        return data

    def insert_embedding(
        self, event_id: int, source: str, vector: bytes, hour: int, cls: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (event_id, source, vector, hour, cls) VALUES (?, ?, ?, ?, ?)",
                (event_id, source, vector, hour, cls),
            )
            self._conn.commit()

    def embeddings_for(self, source: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM embeddings WHERE source = ?",
                (source,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_embedding(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM embeddings WHERE event_id = ?", (event_id,))
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
            self._conn.execute("DELETE FROM embeddings WHERE event_id = ?", (event_id,))
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
        data["track_id"] = data.get("track_id")
        data["verifier_provider"] = data.get("verifier_provider") or ""
        data["verifier_status"] = data.get("verifier_status") or ""
        data["novelty_score"] = data.get("novelty_score")
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
