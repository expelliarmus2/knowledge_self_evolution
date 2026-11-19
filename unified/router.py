"""查询路由：query_type 显式优先；否则按输入形态（QueryIR/逻辑表达式/dict IR/关键词）选引擎。

输入形态约定：

- ``QueryIR`` 实例 / 入组 IR dict → cohort（科研入组）
- ``str`` 自然语言 → 含入组关键词时 cohort，否则默认 kb（RAG）
"""

from __future__ import annotations

from typing import Any, Optional

from cohort.query_ir import QueryIR

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

        # 结构化 IR（QueryIR 实例或入组 dict）→ 科研入组
        if isinstance(query, (QueryIR, dict)) and "cohort" in names:
            return "cohort"

        text = str(query)
        if any(keyword in text for keyword in self.cohort_keywords) and "cohort" in names:
            return "cohort"

        if self.default_engine in names:
            return self.default_engine
        if not names:
            raise KeyError("引擎注册表为空")
        return names[0]
