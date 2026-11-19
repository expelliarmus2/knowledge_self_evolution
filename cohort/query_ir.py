"""入组条件 IR：结构化描述 include / exclude 条件并做基础校验。

IR 形状::

    {
      "reference_date": "2026-11-18",          # 可选，计算年龄的基准日
      "include": [
        {"type": "age",       "min": 18, "max": 75},
        {"type": "diagnosis", "disease_std": "肝细胞癌", "status": "确诊"},
        {"type": "lab",       "item": "AFP", "op": ">", "value": 400},
        {"type": "event",     "event_type": "surgery", "event_name": "肝切除",
         "window_start": "2024-01-01", "window_end": "2024-12-31"}
      ],
      "exclude": [
        {"type": "medication", "drug": "索拉非尼", "status": "已用"}
      ]
    }
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

CONDITION_TYPES = {"age", "diagnosis", "lab", "medication", "event"}
LAB_OPS = {">", ">=", "<", "<=", "=", "!="}
DIAG_STATUS = {"确诊", "病理确诊", "疑似", "否定", "待排"}
MED_STATUS = {"已用", "计划", "停用", "否认"}


@dataclass
class Condition:
    type: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Any) -> "Condition":
        if isinstance(data, Condition):
            return data
        return cls(data.get("type"), {k: v for k, v in data.items() if k != "type"})

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def to_dict(self) -> dict:
        return {"type": self.type, **self.params}


def _require(cond: Condition, keys: tuple[str, ...]) -> None:
    missing = [k for k in keys if cond.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{cond.type} 条件缺少必填字段: {missing}")


def _validate_condition(cond: Condition) -> None:
    if cond.type not in CONDITION_TYPES:
        raise ValueError(f"不支持的条件类型: {cond.type}")
    if cond.type == "age":
        if cond.get("min") is None and cond.get("max") is None:
            raise ValueError("age 条件至少需要 min 或 max 之一")
    elif cond.type == "diagnosis":
        _require(cond, ("disease_std",))
        if cond.get("status") is not None and cond.get("status") not in DIAG_STATUS:
            raise ValueError(f"diagnosis.status 非法: {cond.get('status')}")
    elif cond.type == "lab":
        _require(cond, ("item", "op", "value"))
        if cond.get("op") not in LAB_OPS:
            raise ValueError(f"lab.op 非法: {cond.get('op')}")
    elif cond.type == "medication":
        _require(cond, ("drug",))
        if cond.get("status") is not None and cond.get("status") not in MED_STATUS:
            raise ValueError(f"medication.status 非法: {cond.get('status')}")
    elif cond.type == "event":
        _require(cond, ("event_name",))
    # 时间窗字段统一做 ISO 日期校验
    for key in ("window_start", "window_end"):
        value = cond.get(key)
        if value is not None:
            date.fromisoformat(str(value))


@dataclass
class QueryIR:
    include: list[Condition] = field(default_factory=list)
    exclude: list[Condition] = field(default_factory=list)
    reference_date: str | None = None

    def __post_init__(self) -> None:
        """容错：允许直接传 dict 条件（等价于 from_dict 的子集）。"""
        self.include = [c if isinstance(c, Condition) else Condition.from_dict(c)
                        for c in self.include]
        self.exclude = [c if isinstance(c, Condition) else Condition.from_dict(c)
                        for c in self.exclude]

    @classmethod
    def from_dict(cls, data: Any) -> "QueryIR":
        if not isinstance(data, dict):
            raise ValueError("入组 IR 必须是 dict")
        include = [Condition.from_dict(c) for c in data.get("include", [])]
        exclude = [Condition.from_dict(c) for c in data.get("exclude", [])]
        ir = cls(include=include, exclude=exclude, reference_date=data.get("reference_date"))
        ir.validate()
        return ir

    def validate(self) -> None:
        if not self.include:
            raise ValueError("include 至少需要一个条件")
        for cond in [*self.include, *self.exclude]:
            _validate_condition(cond)
        if self.reference_date is not None:
            date.fromisoformat(str(self.reference_date))

    def to_dict(self) -> dict:
        data: dict = {
            "include": [c.to_dict() for c in self.include],
            "exclude": [c.to_dict() for c in self.exclude],
        }
        if self.reference_date:
            data["reference_date"] = self.reference_date
        return data
