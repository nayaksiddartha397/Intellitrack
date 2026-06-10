"""
event_logger.py — Persist zone/wire events + detections to SQLite.

Two tables:
  detections — one row per detected object per frame (sampled)
  events     — zone entry/exit and tripwire crossing events

Provides export_csv() so the user can download all data via /api/export.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import threading
from datetime import datetime

DB_PATH = "intellitrack_events.db"


class EventLogger:
    def __init__(self, db_path: str = DB_PATH) -> None:
        self._db = db_path
        self._lock = threading.Lock()
        self._init_db()

    # ── Setup ─────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._conn() as cx:
            cx.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT    NOT NULL,
                    event_type TEXT,
                    object_id  INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    zone_name  TEXT
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT    NOT NULL,
                    object_id  INTEGER,
                    class_name TEXT,
                    confidence REAL,
                    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db, check_same_thread=False)

    # ── Write ─────────────────────────────────────────────────────

    def log_event(
        self,
        event_type: str,
        object_id: int,
        class_name: str,
        zone_name: str,
        confidence: float = 0.0,
    ) -> None:
        with self._lock, self._conn() as cx:
            cx.execute(
                """INSERT INTO events
                   (timestamp, event_type, object_id, class_name, confidence, zone_name)
                   VALUES (?,?,?,?,?,?)""",
                (
                    datetime.now().isoformat(timespec="milliseconds"),
                    event_type, object_id, class_name, confidence, zone_name,
                ),
            )

    def log_detection_batch(self, detections: list[dict]) -> None:
        """Bulk-insert a frame's detections (sampled; not every frame)."""
        ts = datetime.now().isoformat(timespec="milliseconds")
        rows = [
            (
                ts,
                d["id"], d["class"], d["conf"],
                *d["box"],
            )
            for d in detections
        ]
        with self._lock, self._conn() as cx:
            cx.executemany(
                """INSERT INTO detections
                   (timestamp, object_id, class_name, confidence, x1,y1,x2,y2)
                   VALUES (?,?,?,?,?,?,?,?)""",
                rows,
            )

    # ── Read ──────────────────────────────────────────────────────

    def get_recent_events(self, n: int = 30) -> list[dict]:
        with self._lock, self._conn() as cx:
            rows = cx.execute(
                """SELECT timestamp, event_type, object_id,
                          class_name, confidence, zone_name
                   FROM events
                   ORDER BY id DESC LIMIT ?""",
                (n,),
            ).fetchall()
        keys = ["time", "type", "id", "class", "conf", "zone"]
        return [dict(zip(keys, r)) for r in rows]

    def event_summary(self) -> dict:
        """Return counts grouped by event_type."""
        with self._lock, self._conn() as cx:
            rows = cx.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

    # ── Export ────────────────────────────────────────────────────

    def export_csv(self) -> str:
        """Return all events + detections as a single CSV string."""
        buf = io.StringIO()
        w   = csv.writer(buf)

        with self._lock, self._conn() as cx:
            # events
            w.writerow(["--- ZONE / WIRE EVENTS ---"])
            w.writerow(["timestamp", "event_type", "object_id",
                        "class_name", "confidence", "zone_name"])
            for row in cx.execute("SELECT timestamp,event_type,object_id,"
                                  "class_name,confidence,zone_name "
                                  "FROM events ORDER BY id").fetchall():
                w.writerow(row)

            # gap
            w.writerow([])
            w.writerow(["--- DETECTION LOG ---"])
            w.writerow(["timestamp", "object_id", "class_name",
                        "confidence", "x1", "y1", "x2", "y2"])
            for row in cx.execute("SELECT timestamp,object_id,class_name,"
                                  "confidence,x1,y1,x2,y2 "
                                  "FROM detections ORDER BY id").fetchall():
                w.writerow(row)

        return buf.getvalue()

    # ── Maintenance ───────────────────────────────────────────────

    def clear_all(self) -> None:
        with self._lock, self._conn() as cx:
            cx.executescript("DELETE FROM events; DELETE FROM detections;")
