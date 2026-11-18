"""IR → SQL 编译与逐患者决策。

语义：
- include 条件之间为 AND（各自 EXISTS 子查询，最终在 decide() 中做交集判定）；
- exclude 为否决（TRIGGERED → 直接 FAIL）；
- UNKNOWN：关键字段缺失（lab.value 为空、诊断/事件时间缺失、用药状态无法确认）时
  不得自动纳入，患者标记 UNKNOWN 并给出缺失原因。

生成的 SQL 形如（每条件两个布尔列 *_pass / *_unknown）：:

    SELECT p.id AS patient_id, p.name AS patient_name,
           CAST((julianday(:ref)-julianday(p.birth_date))/365.25 AS REAL) AS age,
           CASE WHEN (EXISTS (...)) THEN 1 ELSE 0 END AS in0_pass, ...
    FROM patients p ORDER BY p.id
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Mapping

from cohort.query_ir import Condition, QueryIR


@dataclass
class CompiledQuery:
    sql: str
    params: dict[str, Any]
    include: list[Condition] = field(default_factory=list)
    exclude: list[Condition] = field(default_factory=list)
    reference_date: str = ""

    # ------------------------------------------------------------------
    # 决策
    # ------------------------------------------------------------------
    def decide(self, row: Mapping) -> dict:
        details: list[dict] = []
        include_status: list[str] = []
        for i, cond in enumerate(self.include):
            passed = bool(row[f"in{i}_pass"])
            unknown = bool(row[f"in{i}_unknown"])
            status = "PASS" if passed else ("UNKNOWN" if unknown else "FAIL")
            include_status.append(status)
            details.append({"group": "include", "index": i, "type": cond.type, "status": status})

        exclusion_status: list[str] = []
        for i, cond in enumerate(self.exclude):
            triggered = bool(row[f"ex{i}_trigger"])
            unknown = bool(row[f"ex{i}_unknown"])
            status = "TRIGGERED" if triggered else ("UNKNOWN" if unknown else "NOT_TRIGGERED")
            exclusion_status.append(status)
            details.append({"group": "exclude", "index": i, "type": cond.type, "status": status})

        # 优先级：exclude 触发（否决）> include FAIL > include UNKNOWN > exclude UNKNOWN
        if "TRIGGERED" in exclusion_status:
            overall = "FAIL"
        elif any(s == "FAIL" for s in include_status):
            overall = "FAIL"
        elif any(s == "UNKNOWN" for s in include_status):
            overall = "UNKNOWN"
        elif "UNKNOWN" in exclusion_status:
            overall = "UNKNOWN"
        else:
            overall = "PASS"

        missing = [
            f"include[{d['index']}].{d['type']}"
            for d in details if d["group"] == "include" and d["status"] == "UNKNOWN"
        ] + [
            f"exclude[{d['index']}].{d['type']}"
            for d in details if d["group"] == "exclude" and d["status"] == "UNKNOWN"
        ]

        return {
            "patient_id": row["patient_id"],
            "name": row["patient_name"],
            "age": round(row["age"], 2) if row["age"] is not None else None,
            "eligibility": overall,
            "conditions": details,
            "missing": missing,
        }


# ----------------------------------------------------------------------
# 编译
# ----------------------------------------------------------------------
def compile_ir(ir: QueryIR, reference_date: str | None = None) -> CompiledQuery:
    ref = reference_date or ir.reference_date or date.today().isoformat()
    params: dict[str, Any] = {"ref": ref}

    age_expr = "CAST((julianday(:ref) - julianday(p.birth_date)) / 365.25 AS REAL)"

    select = [
        "p.id AS patient_id",
        "p.name AS patient_name",
        f"{age_expr} AS age",
    ]

    for i, cond in enumerate(ir.include):
        select.append(
            f"CASE WHEN {_pass_expr(cond, f'in{i}', params)} THEN 1 ELSE 0 END AS in{i}_pass"
        )
        select.append(
            f"CASE WHEN {_unknown_expr(cond, f'in{i}', params)} THEN 1 ELSE 0 END AS in{i}_unknown"
        )
    for i, cond in enumerate(ir.exclude):
        select.append(
            f"CASE WHEN {_pass_expr(cond, f'ex{i}', params)} THEN 1 ELSE 0 END AS ex{i}_trigger"
        )
        select.append(
            f"CASE WHEN {_unknown_expr(cond, f'ex{i}', params)} THEN 1 ELSE 0 END AS ex{i}_unknown"
        )

    sql = "SELECT\n  " + ",\n  ".join(select) + "\nFROM patients p\nORDER BY p.id"
    return CompiledQuery(sql=sql, params=params, include=ir.include,
                         exclude=ir.exclude, reference_date=ref)


def _pass_expr(cond: Condition, prefix: str, params: dict) -> str:
    """该条件"满足"的 SQL 谓词（EXISTS / 范围比较）。"""
    ctype = cond.type
    if ctype == "age":
        parts = ["p.birth_date IS NOT NULL"]
        if cond.get("min") is not None:
            key = f"{prefix}_min"
            params[key] = cond.get("min")
            parts.append(f"{_age() } >= :{key}")
        if cond.get("max") is not None:
            key = f"{prefix}_max"
            params[key] = cond.get("max")
            parts.append(f"{_age()} <= :{key}")
        return " AND ".join(parts)

    if ctype == "diagnosis":
        key_d = f"{prefix}_disease"
        params[key_d] = cond.get("disease_std")
        parts = ["d.patient_id = p.id", f"d.disease_std = :{key_d}"]
        if cond.get("status") is not None:
            key_s = f"{prefix}_status"
            params[key_s] = cond.get("status")
            parts.append(f"d.status = :{key_s}")
        _window(cond, parts, "d.diagnosed_at", prefix, params)
        return f"EXISTS (SELECT 1 FROM diagnoses d WHERE {' AND '.join(parts)})"

    if ctype == "lab":
        key_i = f"{prefix}_item"
        params[key_i] = cond.get("item")
        key_v = f"{prefix}_value"
        params[key_v] = cond.get("value")
        parts = [
            "l.patient_id = p.id",
            f"l.item = :{key_i}",
            "l.value IS NOT NULL",
            f"l.value {cond.get('op')} :{key_v}",
        ]
        _window(cond, parts, "l.sampled_at", prefix, params)
        return f"EXISTS (SELECT 1 FROM labs l WHERE {' AND '.join(parts)})"

    if ctype == "medication":
        key_d = f"{prefix}_drug"
        params[key_d] = cond.get("drug")
        parts = ["m.patient_id = p.id", f"m.drug = :{key_d}"]
        if cond.get("status") is not None:
            key_s = f"{prefix}_status"
            params[key_s] = cond.get("status")
            parts.append(f"m.status = :{key_s}")
        _window(cond, parts, "m.started_at", prefix, params)
        return f"EXISTS (SELECT 1 FROM medications m WHERE {' AND '.join(parts)})"

    if ctype == "event":
        key_n = f"{prefix}_name"
        params[key_n] = cond.get("event_name")
        parts = ["e.patient_id = p.id", f"e.event_name = :{key_n}"]
        if cond.get("event_type") is not None:
            key_t = f"{prefix}_type"
            params[key_t] = cond.get("event_type")
            parts.append(f"e.event_type = :{key_t}")
        _window(cond, parts, "e.event_at", prefix, params)
        return f"EXISTS (SELECT 1 FROM events e WHERE {' AND '.join(parts)})"

    raise ValueError(f"不支持的条件类型: {ctype}")


def _unknown_expr(cond: Condition, prefix: str, params: dict) -> str:
    """该条件"无法确认（关键字段缺失）"的 SQL 谓词。"""
    ctype = cond.type
    if ctype == "age":
        return "p.birth_date IS NULL"
    if ctype == "diagnosis":
        # 只有请求时间窗时，缺失诊断时间才导致 UNKNOWN
        if cond.get("window_start") is None and cond.get("window_end") is None:
            return "0"
        params[f"{prefix}_disease"] = cond.get("disease_std")
        return (f"EXISTS (SELECT 1 FROM diagnoses d WHERE d.patient_id = p.id "
                f"AND d.disease_std = :{prefix}_disease AND d.diagnosed_at IS NULL)")
    if ctype == "lab":
        params[f"{prefix}_item"] = cond.get("item")
        return (f"EXISTS (SELECT 1 FROM labs l WHERE l.patient_id = p.id "
                f"AND l.item = :{prefix}_item AND l.value IS NULL)")
    if ctype == "medication":
        params[f"{prefix}_drug"] = cond.get("drug")
        return (f"EXISTS (SELECT 1 FROM medications m WHERE m.patient_id = p.id "
                f"AND m.drug = :{prefix}_drug AND m.status IS NULL)")
    if ctype == "event":
        if cond.get("window_start") is None and cond.get("window_end") is None:
            return "0"
        params[f"{prefix}_name"] = cond.get("event_name")
        return (f"EXISTS (SELECT 1 FROM events e WHERE e.patient_id = p.id "
                f"AND e.event_name = :{prefix}_name AND e.event_at IS NULL)")
    raise ValueError(f"不支持的条件类型: {ctype}")


def _age() -> str:
    return "CAST((julianday(:ref) - julianday(p.birth_date)) / 365.25 AS REAL)"


def _window(cond: Condition, parts: list[str], column: str, prefix: str, params: dict) -> None:
    start = cond.get("window_start")
    end = cond.get("window_end")
    if start is not None:
        key = f"{prefix}_ws"
        params[key] = start
        parts.append(f"{column} IS NOT NULL AND {column} >= :{key}")
    if end is not None:
        key = f"{prefix}_we"
        params[key] = end
        parts.append(f"{column} IS NOT NULL AND {column} <= :{key}")
