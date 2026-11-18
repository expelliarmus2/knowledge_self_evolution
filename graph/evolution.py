"""演进接口（EvoGraph-R1 / RSI 对接预留）。

设计原则：后续接入 EvoGraph-R1 时无需重构核心模块——
1. GraphOperation    : 图增删改操作的统一数据结构（add/remove/update × node/edge）；
2. EvolutionStrategy : 演进策略抽象，通过 on_event 接收流水线/图事件，返回待应用的图操作；
3. EvolutionManager  : 策略注册表 + 事件分发（pipeline 的演进策略注入点）；
4. GraphEvolutionAPI : 图演进的门面 API，EvoGraph-R1 直接调用即可完成
                       增删改节点与边 + 增量向量库同步 + 版本通知。
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kb")

# 支持的图操作（EvoGraph-R1 增删改节点与边）
VALID_OPS = {
    "add_entity", "remove_entity", "update_entity",
    "add_triple", "remove_triple", "update_triple",
    "add_chunk", "remove_chunk",
}

# 流水线事件注入点
PIPELINE_EVENTS = [
    "pipeline_start",
    "before_chunking", "after_chunking",
    "before_extract", "after_extract",
    "on_extract_batch",
    "before_graph_build", "after_graph_build",
    "before_vectorize", "after_vectorize",
    "pipeline_complete",
    "on_error",
]


@dataclass
class GraphOperation:
    """一次图结构演进操作。

    payload 字段：
    - add_entity    : {name, entity_type?, meta?}
    - remove_entity : {name}
    - update_entity : {name, attrs}
    - add_triple    : {head, relation, tail, chunk_id?, weight?}
    - remove_triple : {head, relation, tail}
    - update_triple : {head, relation, tail, weight?, meta?}
    - add_chunk     : {chunk_id, text, doc_id?, order?, meta?}
    - remove_chunk  : {chunk_id}
    """

    op: str
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""  # 演进依据（EvoGraph-R1 决策的可追溯记录）

    def validate(self) -> None:
        if self.op not in VALID_OPS:
            raise ValueError(f"未知图操作: {self.op}（支持: {sorted(VALID_OPS)}）")
        p = self.payload
        if self.op in ("add_entity", "remove_entity", "update_entity") and not p.get("name"):
            raise ValueError(f"{self.op} 缺少 name")
        if self.op in ("add_triple", "remove_triple", "update_triple"):
            for k in ("head", "relation", "tail"):
                if not p.get(k):
                    raise ValueError(f"{self.op} 缺少 {k}")
        if self.op in ("add_chunk", "remove_chunk") and not p.get("chunk_id"):
            raise ValueError(f"{self.op} 缺少 chunk_id")

    def to_dict(self) -> dict:
        return {"op": self.op, "payload": self.payload, "reason": self.reason}


class EvolutionContext:
    """演进策略可见的上下文（依赖注入，避免策略触碰流水线内部）。"""

    def __init__(self, config=None, graph=None, vector_manager=None, embedder=None, llm=None, artifacts: Optional[dict] = None):
        self.config = config
        self.graph = graph
        self.vector_manager = vector_manager
        self.embedder = embedder
        self.llm = llm
        self.artifacts: Dict[str, Any] = artifacts or {}
        # 流水线过程数据（如当前批抽取结果），事件回调里可读
        self.event_data: Dict[str, Any] = {}


class EvolutionStrategy:
    """演进策略基类。EvoGraph-R1 的策略应继承此类并注册到流水线。"""

    name: str = "base_strategy"

    def setup(self, ctx: EvolutionContext) -> None:  # noqa: B027
        """流水线启动时调用一次。"""

    def on_event(self, event: str, ctx: EvolutionContext) -> List[GraphOperation]:
        """接收流水线/图事件；可返回待应用的图操作列表（在安全点应用）。"""
        return []

    def teardown(self, ctx: EvolutionContext) -> None:  # noqa: B027
        """流水线结束时调用。"""


class NoopStrategy(EvolutionStrategy):
    """空策略（默认）：不做任何演进，仅记录事件。"""

    name = "noop"

    def __init__(self):
        self.seen_events: List[str] = []

    def on_event(self, event: str, ctx: EvolutionContext) -> List[GraphOperation]:
        self.seen_events.append(event)
        return []


class EvolutionManager:
    """策略注册表 + 事件分发。pipeline 的演进策略注入点。"""

    def __init__(self, ctx: Optional[EvolutionContext] = None):
        self.strategies: List[EvolutionStrategy] = []
        self.ctx = ctx or EvolutionContext()

    def register(self, strategy: EvolutionStrategy) -> None:
        self.strategies.append(strategy)
        strategy.setup(self.ctx)
        logger.info("已注册演进策略: %s", strategy.name)

    def load_from_config(self, cfg) -> None:
        """从 config evolution.strategies 加载策略模块（"pkg.module:ClassName"）。"""
        for spec in cfg.get("evolution.strategies", []) or []:
            try:
                mod_path, cls_name = str(spec).rsplit(":", 1)
                mod = importlib.import_module(mod_path)
                cls = getattr(mod, cls_name)
                self.register(cls())
            except Exception as err:  # noqa: BLE001
                logger.error("演进策略加载失败 %s: %s", spec, err)

    def emit(self, event: str, **data) -> List[GraphOperation]:
        """广播事件，汇总各策略返回的图操作（在安全点由调用方应用）。"""
        ops: List[GraphOperation] = []
        if not self.strategies:
            return ops
        self.ctx.event_data = data
        for strategy in self.strategies:
            try:
                ops.extend(strategy.on_event(event, self.ctx) or [])
            except Exception as err:  # noqa: BLE001
                logger.error("演进策略 %s 处理事件 %s 失败: %s", strategy.name, event, err)
        return [op for op in ops if isinstance(op, GraphOperation)]


class GraphEvolutionAPI:
    """图演进门面 API（EvoGraph-R1 / RSI 侧的稳定调用入口）。

    - apply()       : 应用一批图操作（增删改节点与边），自动同步向量库；
    - add_triple/remove_entity/... : 便捷方法；
    - 每次变更 graph.version 递增，检索侧 Retriever 依据 version 感知变化。
    """

    def __init__(self, graph, vector_manager=None, auto_sync_vectorstore: bool = True):
        self.graph = graph
        self.vector_manager = vector_manager
        self.auto_sync_vectorstore = auto_sync_vectorstore

    # ---- 应用操作 ----
    def apply(self, ops: List[GraphOperation]) -> Dict[str, Any]:
        applied, failed = [], []
        for op in ops:
            try:
                op.validate()
                self.graph.apply_operation(op)
                applied.append(op.to_dict())
            except Exception as err:  # noqa: BLE001
                failed.append({**op.to_dict(), "error": str(err)})
                logger.error("图操作失败 %s: %s", op.op, err)
        if self.auto_sync_vectorstore and self.vector_manager is not None and applied:
            try:
                self.vector_manager.sync_from_graph(self.graph, incremental=True)
            except Exception as err:  # noqa: BLE001
                logger.error("演进后向量库同步失败: %s", err)
        return {
            "applied": applied,
            "failed": failed,
            "version": self.graph.version,
            "stats": self.graph.stats(),
            "ts": time.time(),
        }

    # ---- 便捷方法（EvoGraph-R1 常用）----
    def add_triple(self, head: str, relation: str, tail: str, weight: float = 1.0, reason: str = "") -> Dict[str, Any]:
        return self.apply([GraphOperation("add_triple", {"head": head, "relation": relation, "tail": tail, "weight": weight}, reason)])

    def remove_triple(self, head: str, relation: str, tail: str, reason: str = "") -> Dict[str, Any]:
        return self.apply([GraphOperation("remove_triple", {"head": head, "relation": relation, "tail": tail}, reason)])

    def update_triple_weight(self, head: str, relation: str, tail: str, weight: float, reason: str = "") -> Dict[str, Any]:
        return self.apply([GraphOperation("update_triple", {"head": head, "relation": relation, "tail": tail, "weight": weight}, reason)])

    def remove_entity(self, name: str, reason: str = "") -> Dict[str, Any]:
        return self.apply([GraphOperation("remove_entity", {"name": name}, reason)])

    def update_entity(self, name: str, attrs: dict, reason: str = "") -> Dict[str, Any]:
        return self.apply([GraphOperation("update_entity", {"name": name, "attrs": attrs}, reason)])

    def add_entity(self, name: str, entity_type: Optional[str] = None, reason: str = "") -> Dict[str, Any]:
        return self.apply([GraphOperation("add_entity", {"name": name, "entity_type": entity_type}, reason)])
