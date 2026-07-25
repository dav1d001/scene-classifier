"""
api/database.py
----------------
Very small SQLite layer used purely for monitoring/UI purposes:
  - logs every prediction (class, confidence, latency) so the UI can show
    live traffic + latency charts
  - logs every retrain job's before/after metrics so the UI can show a
    retraining history table
"""

import sqlite3
import json
import os
from datetime import datetime
from contextlib import contextmanager

DB_PATH = "data/monitoring.db"


@contextmanager
def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                predicted_class TEXT,
                confidence REAL,
                latency_ms REAL,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS retrain_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_json TEXT,
                created_at TEXT
            )
        """)


def log_prediction(predicted_class: str, confidence: float, latency_ms: float):
    init_db()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO predictions (predicted_class, confidence, latency_ms, created_at) VALUES (?, ?, ?, ?)",
            (predicted_class, confidence, latency_ms, datetime.now().isoformat()),
        )


def log_retrain_job(report: dict):
    init_db()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO retrain_jobs (report_json, created_at) VALUES (?, ?)",
            (json.dumps(report), datetime.now().isoformat()),
        )


def get_recent_predictions(limit=50):
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT predicted_class, confidence, latency_ms, created_at FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"predicted_class": r[0], "confidence": r[1], "latency_ms": r[2], "created_at": r[3]}
        for r in rows
    ]


def get_retrain_history(limit=20):
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT report_json, created_at FROM retrain_jobs ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [{"report": json.loads(r[0]), "created_at": r[1]} for r in rows]
