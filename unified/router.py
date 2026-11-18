"""查询路由：query_type 显式优先；否则按关键词 / 输入形态选引擎。"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_COHORT_KEYWORDS = (
    "入组", "筛查", "符合", "年龄", "患者", "用药", "排除", "纳入", "队列", "受试者",
)


class Router:
    def __init__(self, registry: Any, cohort_keywords: Optional[list[str]] = None,
                 default_engine: str = "kb"):
        self.registry = registry
        self.cohort_keywords = tuple(cohort_keywords or DEFAULT_COHORT_KEYWORDS)
        self.default_engine = default_engine

    def route(self, query: Any, query_type: Optional[str] = None) -> str:
        names = self.registry.names()
        if query_type:
            if query_type not in names:
                raise KeyError(f"未注册的 query_type: {query_type}；已注册: {names}")
            return query_type

        # 结构化 IR（dict）几乎只可能来自入组场景
        if isinstance(query, dict) and "cohort" in names:
            return "cohort"

        text = str(query)
        if any(keyword in text for keyword in self.cohort_keywords) and "cohort" in names:
            return "cohort"

        if self.default_engine in names:
            return self.default_engine
        if not names:
            raise KeyError("引擎注册表为空")
        return names[0]
