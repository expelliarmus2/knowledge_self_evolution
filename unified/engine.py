"""统一门面：引擎注册表、路由、统一查询入口。

物理索引保持双轨（KB = NetworkX + FAISS；cohort = SQLite），
本层只统一：入口协议、结果信封、注册表、路由。
"""

from __future__ import annotations

import time
from typing import Any, Optional, Protocol, runtime_checkable

from unified.result import UnifiedResult


@runtime_checkable
class BaseEngine(Protocol):
    name: str

    def search(self, query: Any, **params: Any) -> Any: ...

    def describe(self) -> dict: ...


class EngineRegistry:
    def __init__(self) -> None:
        self._engines: dict[str, BaseEngine] = {}

    def register(self, engine: BaseEngine) -> None:
        if not getattr(engine, "name", None):
            raise ValueError("engine 必须有 name")
        self._engines[engine.name] = engine

    def get(self, name: str) -> BaseEngine:
        if name not in self._engines:
            raise KeyError(f"未注册的引擎: {name}；已注册: {self.names()}")
        return self._engines[name]

    def names(self) -> list[str]:
        return list(self._engines)

    def describe(self) -> dict:
        return {name: engine.describe() for name, engine in self._engines.items()}


class KBEngineAdapter:
    """把现有 retrieval.Retriever 适配为统一引擎（payload 透传 SearchResult）。"""

    name = "kb"

    def __init__(self, retriever: Any, name: str = "kb"):
        self.retriever = retriever
        self.name = name

    def describe(self) -> dict:
        return {
            "name": self.name,
            "query_types": ["kb"],
            "index_types": ["networkx", "faiss"],
        }

    def search(self, query: Any, top_k: Optional[int] = None, hops: Optional[int] = None,
               subgraph_mode: Optional[str] = None, **params: Any) -> Any:
        kwargs: dict[str, Any] = {}
        if top_k is not None:
            kwargs["top_k"] = top_k
        if hops is not None:
            kwargs["hops"] = hops
        if subgraph_mode is not None:
            kwargs["subgraph_mode"] = subgraph_mode
        return self.retriever.search(query, **kwargs)


class UnifiedEngine:
    """统一查询入口：route → engine.search → UnifiedResult。"""

    def __init__(self, registry: EngineRegistry, router: Optional["Router"] = None):
        self.registry = registry
        from unified.router import Router

        self.router = router or Router(registry)

    def search(self, query: Any, query_type: Optional[str] = None, **params: Any) -> UnifiedResult:
        name = self.router.route(query, query_type)
        engine = self.registry.get(name)
        started = time.time()
        payload = engine.search(query, **params)
        latency_ms = round((time.time() - started) * 1000, 3)
        return UnifiedResult(
            query=query,
            engine=name,
            payload=payload.to_dict() if hasattr(payload, "to_dict") else payload,
            meta={
                "latency_ms": latency_ms,
                "engine": engine.describe(),
            },
        )
