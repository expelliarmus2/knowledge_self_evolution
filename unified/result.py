"""统一检索结果信封。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedResult:
    query: Any
    engine: str
    payload: Any
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "query": self.query if isinstance(self.query, str) else repr(self.query),
            "engine": self.engine,
            "payload": self.payload,
            "meta": self.meta,
        }
