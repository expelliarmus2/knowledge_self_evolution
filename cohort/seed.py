"""合成"已抽取好"的结构化患者数据（模拟 LLM 抽取 + 细筛后的产物，不跑 openie）。

设计覆盖 6 条路径（相对 demo IR：年龄 18-75、肝细胞癌确诊（2020-2025 窗）、AFP>400、
排除 索拉非尼已用）：

- P001 合格（全条件命中 → PASS）
- P002 年龄不符（1940 年生，>75 → FAIL）
- P003 诊断不符（肝硬化 → FAIL）
- P004 排除用药（索拉非尼 已用 → FAIL）
- P005 缺检查值（AFP 行 value 为空 → UNKNOWN）
- P006 缺诊断时间（肝细胞癌确诊但 diagnosed_at 为空 → UNKNOWN）

其余为确定性随机的填充患者，保证 demo 数据不单调。
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from pathlib import Path
from typing import Any

# 演示用入组 IR（与 unified/demo.py 保持一致）
DEMO_IR: dict[str, Any] = {
    "reference_date": "2026-11-18",
    "include": [
        {"type": "age", "min": 18, "max": 75},
        {"type": "diagnosis", "disease_std": "肝细胞癌", "status": "确诊",
         "window_start": "2020-01-01", "window_end": "2025-12-31"},
        {"type": "lab", "item": "AFP", "op": ">", "value": 400},
    ],
    "exclude": [
        {"type": "medication", "drug": "索拉非尼", "status": "已用"},
    ],
}

_ARCHETYPES: list[dict] = [
    dict(id=1, name="P001-合格", birth_date="1970-05-01", sex="M",
         diagnoses=[("肝细胞癌", "肝细胞癌", "确诊", "2023-03-01")],
         labs=[("AFP", 1200.0, "ng/mL", "2023-03-01")],
         meds=[], events=[]),
    dict(id=2, name="P002-年龄不符", birth_date="1940-01-01", sex="M",
         diagnoses=[("肝细胞癌", "肝细胞癌", "确诊", "2023-03-01")],
         labs=[("AFP", 900.0, "ng/mL", "2023-03-01")],
         meds=[], events=[]),
    dict(id=3, name="P003-诊断不符", birth_date="1980-07-15", sex="F",
         diagnoses=[("肝硬化", "肝硬化", "确诊", "2023-01-01")],
         labs=[("AFP", 950.0, "ng/mL", "2023-01-01")],
         meds=[], events=[]),
    dict(id=4, name="P004-排除用药", birth_date="1975-02-20", sex="M",
         diagnoses=[("肝细胞癌", "肝细胞癌", "确诊", "2023-01-01")],
         labs=[("AFP", 1100.0, "ng/mL", "2023-01-01")],
         meds=[("索拉非尼", "已用", "2023-06-01", None)],
         events=[]),
    dict(id=5, name="P005-缺检查值", birth_date="1968-09-09", sex="F",
         diagnoses=[("肝细胞癌", "肝细胞癌", "确诊", "2023-01-01")],
         labs=[("AFP", None, None, "2023-01-01")],
         meds=[], events=[]),
    dict(id=6, name="P006-缺诊断时间", birth_date="1972-11-11", sex="M",
         diagnoses=[("肝细胞癌", "肝细胞癌", "确诊", None)],
         labs=[("AFP", 700.0, "ng/mL", "2023-01-01")],
         meds=[], events=[]),
]


def build_seed(count: int = 30, rng_seed: int = 20261118) -> dict:
    patients: list[dict] = []
    diagnoses: list[dict] = []
    labs: list[dict] = []
    medications: list[dict] = []
    events: list[dict] = []

    def add(p: dict) -> None:
        patients.append({"id": p["id"], "name": p["name"],
                         "birth_date": p["birth_date"], "sex": p["sex"]})
        for (disease, std, status, at) in p["diagnoses"]:
            diagnoses.append({"patient_id": p["id"], "disease": disease, "disease_std": std,
                              "status": status, "diagnosed_at": at})
        for (item, value, unit, at) in p["labs"]:
            labs.append({"patient_id": p["id"], "item": item, "value": value,
                         "unit": unit, "sampled_at": at})
        for (drug, status, start, end) in p["meds"]:
            medications.append({"patient_id": p["id"], "drug": drug, "status": status,
                                "started_at": start, "ended_at": end})
        for (etype, ename, at) in p["events"]:
            events.append({"patient_id": p["id"], "event_type": etype,
                           "event_name": ename, "event_at": at})

    for p in _ARCHETYPES:
        add(p)

    rng = random.Random(rng_seed)
    diseases = ["肝细胞癌", "肝硬化", "乙型肝炎"]
    for i in range(len(_ARCHETYPES) + 1, count + 1):
        birth_year = 1960 + (i * 7) % 40          # 1960-1999
        disease = diseases[i % 3]
        add(dict(
            id=i,
            name=f"P{i:03d}-填充",
            birth_date=f"{birth_year}-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            sex="M" if i % 2 else "F",
            diagnoses=[(disease, disease, "确诊", "2022-01-01")],
            labs=[("AFP", None if i % 5 == 0 else float((i * 137) % 2000), "ng/mL", "2022-01-01")],
            meds=[("索拉非尼", "已用", "2022-06-01", None)] if i % 7 == 0 else [],
            events=[],
        ))

    return {"patients": patients, "diagnoses": diagnoses, "labs": labs,
            "medications": medications, "events": events}


def seed_database(db_path: Path | str, count: int = 30, rng_seed: int = 20261118) -> Path:
    """重建一个含种子数据的 SQLite 库，返回 db 路径。"""
    from cohort.db import connect, init_db, insert_seed

    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    try:
        init_db(conn)
        insert_seed(conn, build_seed(count=count, rng_seed=rng_seed))
        conn.commit()
    finally:
        conn.close()
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(description="生成科研入组 demo SQLite 库")
    parser.add_argument("--db", default=None, help="输出 db 路径（默认取 config cohort.sqlite_path）")
    parser.add_argument("--count", type=int, default=30)
    args = parser.parse_args()

    from utils.config import Config

    cfg = Config.load()
    db = Path(args.db) if args.db else cfg.path("cohort.sqlite_path", "data/cohort/demo.db")
    seed_database(db, count=args.count)
    print(f"已生成 demo 数据库: {db}")


if __name__ == "__main__":
    main()
