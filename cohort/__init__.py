"""科研入组筛查 demo 引擎（SQLite 关系库）。"""

from .engine import CohortEngine, CohortResult
from .query_ir import Condition, QueryIR

__all__ = ["CohortEngine", "CohortResult", "QueryIR", "Condition"]
