"""知识图谱模块：三元组建图（实体为节点、三元组为边）+ 演进接口。

- knowledge_graph : NetworkX 封装，支持增量更新 / 图操作日志 / 版本号
- graph_builder   : 流式并行构图（吸收 huaxi_poc 组织方式）
- evolution       : EvoGraph-R1 / RSI 演进接口（策略注入、图操作 API）
"""
from graph.knowledge_graph import KnowledgeGraph, NODE_ENTITY, NODE_CHUNK, REL_TEXT_CONCLUDE
from graph.graph_builder import GraphBuilder
from graph.evolution import (
    GraphOperation,
    EvolutionStrategy,
    EvolutionManager,
    EvolutionContext,
    GraphEvolutionAPI,
    NoopStrategy,
)

__all__ = [
    "KnowledgeGraph", "NODE_ENTITY", "NODE_CHUNK", "REL_TEXT_CONCLUDE",
    "GraphBuilder",
    "GraphOperation", "EvolutionStrategy", "EvolutionManager",
    "EvolutionContext", "GraphEvolutionAPI", "NoopStrategy",
]
