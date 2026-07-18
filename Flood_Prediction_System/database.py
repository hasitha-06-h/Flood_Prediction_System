"""
database.py

Thin SQLite layer - no ORM, just sqlite3 with row_factory set to dict-like
rows. Three tables:

    users               - placeholder auth table (kept simple, no login
                           flow was in scope for this build, but the table
                           exists so the ER diagram / future login page
                           has something real to attach to)
    weather_data        - every raw reading a visitor submits on the
                           predict page
    prediction_history  - the model's verdict for each of those readings,
                           foreign-keyed back to weather_data

Kept in a single module so app.py only ever imports `database` rather than
scattering raw SQL through the route handlers.
"""

import sqlite3
import os
from datetime import datetime

from config import Config


def get_connection():
    os.makedirs(Config.DATABASE_DIR, exist_ok=True)
    conn = sqlite3.connect(Config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Creates all tables if they don't already exist. Safe to call on
    every app startup."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            annual_rainfall REAL NOT NULL,
            cloud_cover REAL NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            seasonal_rainfall REAL NOT NULL,
            submitted_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            weather_data_id INTEGER NOT NULL,
            model_used TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            probability REAL,
            risk_level TEXT NOT NULL,
            predicted_at TEXT NOT NULL,
            FOREIGN KEY (weather_data_id) REFERENCES weather_data (id)
        )
    """)

    conn.commit()
    conn.close()


def save_reading_and_prediction(reading, model_used, prediction, probability, risk_level):
    """
    Persists one full prediction cycle: the raw inputs (weather_data) and
    the resulting verdict (prediction_history), linked by foreign key.
    Returns the new prediction_history row id.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat(timespec="seconds")

    cursor.execute("""
        INSERT INTO weather_data
            (annual_rainfall, cloud_cover, temperature, humidity, seasonal_rainfall, submitted_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        reading["ANNUAL"], reading["Cloud Cover"], reading["Temp"],
        reading["Humidity"], reading["Jun-Sep"], now,
    ))
    weather_data_id = cursor.lastrowid

    cursor.execute("""
        INSERT INTO prediction_history
            (weather_data_id, model_used, prediction, probability, risk_level, predicted_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (weather_data_id, model_used, int(prediction), probability, risk_level, now))
    prediction_id = cursor.lastrowid

    conn.commit()
    conn.close()
    return prediction_id


def fetch_history(limit=50):
    """Returns the most recent predictions joined with their weather readings,
    newest first - used by the History page."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            ph.id, ph.model_used, ph.prediction, ph.probability,
            ph.risk_level, ph.predicted_at,
            wd.annual_rainfall, wd.cloud_cover, wd.temperature,
            wd.humidity, wd.seasonal_rainfall
        FROM prediction_history ph
        JOIN weather_data wd ON wd.id = ph.weather_data_id
        ORDER BY ph.id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def count_predictions_by_risk():
    """Small aggregate used on the History page header (e.g. '12 flood
    alerts out of 40 total readings')."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM prediction_history")
    total = cursor.fetchone()["total"]
    cursor.execute("SELECT COUNT(*) as flood_count FROM prediction_history WHERE prediction = 1")
    flood_count = cursor.fetchone()["flood_count"]
    conn.close()
    return {"total": total, "flood_count": flood_count, "safe_count": total - flood_count}
