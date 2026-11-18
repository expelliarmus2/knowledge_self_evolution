"""SQLite 打开 / 建表 / 种子载入。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().with_name("schema.sql")


def connect(db_path: Path | str) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def insert_seed(conn: sqlite3.Connection, data: dict) -> None:
    conn.executemany(
        "INSERT INTO patients(id, name, birth_date, sex) VALUES (:id, :name, :birth_date, :sex)",
        data["patients"],
    )
    conn.executemany(
        "INSERT INTO diagnoses(patient_id, disease, disease_std, status, diagnosed_at) "
        "VALUES (:patient_id, :disease, :disease_std, :status, :diagnosed_at)",
        data["diagnoses"],
    )
    conn.executemany(
        "INSERT INTO labs(patient_id, item, value, unit, sampled_at) "
        "VALUES (:patient_id, :item, :value, :unit, :sampled_at)",
        data["labs"],
    )
    conn.executemany(
        "INSERT INTO medications(patient_id, drug, status, started_at, ended_at) "
        "VALUES (:patient_id, :drug, :status, :started_at, :ended_at)",
        data["medications"],
    )
    conn.executemany(
        "INSERT INTO events(patient_id, event_type, event_name, event_at) "
        "VALUES (:patient_id, :event_type, :event_name, :event_at)",
        data["events"],
    )
