"""CohortEngine：科研入组筛查 demo 引擎（SQLite + IR→SQL 编译）。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cohort.compiler import CompiledQuery, compile_ir
from cohort.query_ir import QueryIR

SCHEMA_VERSION = "cohort-sqlite-v1"


@dataclass
class CohortResult:
    query: Any
    reference_date: str
    sql: str
    params: dict
    decisions: list[dict] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @property
    def summary(self) -> dict:
        counts = {"PASS": 0, "FAIL": 0, "UNKNOWN": 0}
        for d in self.decisions:
            counts[d["eligibility"]] = counts.get(d["eligibility"], 0) + 1
        return {"total": len(self.decisions), **counts}

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "reference_date": self.reference_date,
            "summary": self.summary,
            "compiled_sql": self.sql,
            "params": self.params,
            "decisions": self.decisions,
            "schema_version": self.schema_version,
        }


class CohortEngine:
    """统一门面的第二个引擎：结构化入组条件 → 逐患者 PASS/FAIL/UNKNOWN。"""

    name = "cohort"

    def __init__(self, db_path: Path | str, *, seed_if_missing: bool = True, seed_count: int = 30):
        self.db_path = Path(db_path)
        if seed_if_missing and not self.db_path.exists():
            from cohort.seed import seed_database

            seed_database(self.db_path, count=seed_count)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "query_types": ["cohort"],
            "index_types": ["sqlite"],
            "tables": ["patients", "diagnoses", "labs", "medications", "events"],
            "schema_version": SCHEMA_VERSION,
        }

    def search(self, query: Any, query_ir: Any = None, reference_date: str | None = None,
               **params: Any) -> CohortResult:
        raw = query_ir if query_ir is not None else query
        if isinstance(raw, QueryIR):
            ir = raw
        elif isinstance(raw, str):
            try:
                ir = QueryIR.from_dict(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "cohort 引擎需要结构化入组 IR（dict 或 JSON 字符串）；"
                    f"无法解析: {exc}"
                ) from exc
        else:
            ir = QueryIR.from_dict(raw)

        compiled: CompiledQuery = compile_ir(ir, reference_date)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(compiled.sql, compiled.params).fetchall()
        finally:
            conn.close()

        decisions = [compiled.decide(row) for row in rows]
        return CohortResult(
            query=ir.to_dict(),
            reference_date=compiled.reference_date,
            sql=compiled.sql,
            params=compiled.params,
            decisions=decisions,
        )
