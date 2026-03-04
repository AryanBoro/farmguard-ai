"""
FarmGuard AI — Crop History Service
SQLite-backed persistence for scan history and trend analysis.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Optional
from contextlib import contextmanager

DB_PATH = "/tmp/farmguard_history.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT NOT NULL,
                crop_type   TEXT,
                crop_age    INTEGER,
                class_name  TEXT NOT NULL,
                common_name TEXT NOT NULL,
                confidence  REAL NOT NULL,
                is_healthy  INTEGER NOT NULL,
                severity    TEXT,
                location    TEXT,
                weather_json TEXT,
                notes       TEXT
            );

            CREATE TABLE IF NOT EXISTS fields (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                crop_type   TEXT,
                location    TEXT,
                created_at  TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_scans_crop ON scans(crop_type);
            CREATE INDEX IF NOT EXISTS idx_scans_created ON scans(created_at);
            CREATE INDEX IF NOT EXISTS idx_scans_healthy ON scans(is_healthy);
        """)


def record_scan(
    class_name: str,
    common_name: str,
    confidence: float,
    is_healthy: bool,
    severity: str,
    crop_type: Optional[str] = None,
    crop_age: Optional[int] = None,
    location: Optional[str] = None,
    weather: Optional[dict] = None,
    notes: Optional[str] = None
) -> int:
    """Save a scan result and return the new scan ID."""
    with get_conn() as conn:
        cursor = conn.execute("""
            INSERT INTO scans
              (created_at, crop_type, crop_age, class_name, common_name,
               confidence, is_healthy, severity, location, weather_json, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            crop_type,
            crop_age,
            class_name,
            common_name,
            confidence,
            int(is_healthy),
            severity,
            location,
            json.dumps(weather) if weather else None,
            notes
        ))
        return cursor.lastrowid


def get_scan_history(
    limit: int = 50,
    crop_type: Optional[str] = None,
    only_diseases: bool = False
) -> List[dict]:
    """Retrieve recent scan history with optional filters."""
    query = "SELECT * FROM scans WHERE 1=1"
    params = []

    if crop_type:
        query += " AND crop_type = ?"
        params.append(crop_type)

    if only_diseases:
        query += " AND is_healthy = 0"

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_disease_trend(days: int = 30, crop_type: Optional[str] = None) -> dict:
    """
    Returns disease detection trends over the past N days.
    Useful for dashboard charts.
    """
    query = """
        SELECT 
            date(created_at) as day,
            COUNT(*) as total_scans,
            SUM(CASE WHEN is_healthy = 0 THEN 1 ELSE 0 END) as disease_count,
            AVG(confidence) as avg_confidence
        FROM scans
        WHERE created_at >= datetime('now', ?)
    """
    params = [f"-{days} days"]

    if crop_type:
        query += " AND crop_type = ?"
        params.append(crop_type)

    query += " GROUP BY date(created_at) ORDER BY day ASC"

    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return {
            "days": days,
            "crop_filter": crop_type,
            "daily_data": [dict(r) for r in rows]
        }


def get_summary_stats() -> dict:
    """High-level stats for the dashboard."""
    with get_conn() as conn:
        stats = conn.execute("""
            SELECT
                COUNT(*) as total_scans,
                SUM(CASE WHEN is_healthy = 0 THEN 1 ELSE 0 END) as disease_detections,
                SUM(CASE WHEN is_healthy = 1 THEN 1 ELSE 0 END) as healthy_detections,
                AVG(confidence) as avg_confidence,
                MAX(created_at) as last_scan
            FROM scans
        """).fetchone()

        top_diseases = conn.execute("""
            SELECT class_name, common_name, COUNT(*) as count
            FROM scans
            WHERE is_healthy = 0
            GROUP BY class_name
            ORDER BY count DESC
            LIMIT 5
        """).fetchall()

        return {
            **dict(stats),
            "top_diseases": [dict(r) for r in top_diseases]
        }


def delete_scan(scan_id: int) -> bool:
    """Delete a scan record by ID."""
    with get_conn() as conn:
        cursor = conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        return cursor.rowcount > 0
