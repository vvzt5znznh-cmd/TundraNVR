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
                    clip_path TEXT
                )
                """
            )
            self._conn.commit()

    def insert(
        self,
        ts_start: str,
        ts_end: str | None,
        classes: list[str],
        score: float,
        thumb_path: str | None,
        clip_path: str | None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO events (ts_start, ts_end, classes, score, thumb_path, clip_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts_start, ts_end, json.dumps(classes), score, thumb_path, clip_path),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_events(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
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
        return data


def utc_now() -> str:
    return _utc_now()
