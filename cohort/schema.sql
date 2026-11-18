-- 科研入组 demo：患者结构化事实（模拟 LLM 抽取、细筛后的产物）
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS patients (
    id          INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    birth_date  TEXT    NOT NULL,          -- YYYY-MM-DD
    sex         TEXT
);

CREATE TABLE IF NOT EXISTS diagnoses (
    id           INTEGER PRIMARY KEY,
    patient_id   INTEGER NOT NULL REFERENCES patients(id),
    disease      TEXT    NOT NULL,         -- 原始病名（抽取结果）
    disease_std  TEXT    NOT NULL,         -- 标准名（mock 细筛结果）
    status       TEXT    NOT NULL,         -- 确诊 / 病理确诊 / 疑似 / 否定 / 待排
    diagnosed_at TEXT                      -- 可空 → UNKNOWN（时间窗无法确认）
);

CREATE TABLE IF NOT EXISTS labs (
    id         INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    item       TEXT    NOT NULL,
    value      REAL,                       -- 可空 → UNKNOWN（检查值缺失）
    unit       TEXT,
    sampled_at TEXT
);

CREATE TABLE IF NOT EXISTS medications (
    id         INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    drug       TEXT    NOT NULL,
    status     TEXT,                       -- 已用 / 计划 / 停用 / 否认；可空 → UNKNOWN
    started_at TEXT,
    ended_at   TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY,
    patient_id INTEGER NOT NULL REFERENCES patients(id),
    event_type TEXT    NOT NULL,
    event_name TEXT    NOT NULL,
    event_at   TEXT                        -- 可空 → UNKNOWN（时间窗无法确认）
);
