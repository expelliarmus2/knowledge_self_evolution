"""CHUNK_HOP 多跳扩展（吸收 Green-ChunkHop 的 chunk 组织与多跳检索思路）。

- chunk 之间通过共享实体形成邻接（text_conclude 反向索引两两配对）；
- BFS 逐跳扩散，每跳受 max_hop_width 剪枝、min_shared_entities / min_hop_weight 过滤；
- 边权 = 共享实体数 × (1 + 与查询实体的交集加成)，查询感知；
- 邻接缓存绑定 graph.version —— 图演进（EvoGraph-R1 增删改）后自动重建，
  检索侧由此感知图结构变化。
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from graph.knowledge_graph import KnowledgeGraph

logger = logging.getLogger("kb")


class HopExpander:
    def __init__(self, graph: KnowledgeGraph, max_hop_width: int = 50):
        self.graph = graph
        self.max_hop_width = max_hop_width
        self._cached_version: Optional[int] = None
        self._adj: Dict[str, Dict[str, dict]] = {}

    # ------------------------------------------------------------------
    def _build_adjacency(self) -> Dict[str, Dict[str, dict]]:
        """chunk 共享实体邻接：{chunk: {other: {shared: [...], count: int}}}"""
        entity_to_chunks: Dict[str, List[str]] = defaultdict(list)
        for u, v, data in self.graph.g.edges(data=True):
            if data.get("relation") == "text_conclude":
                entity_to_chunks[u].append(v)
        adj: Dict[str, Dict[str, dict]] = defaultdict(dict)
        for _entity, chunks in entity_to_chunks.items():
            if len(chunks) < 2:
                continue
            uniq = sorted(set(chunks))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    a, b = uniq[i], uniq[j]
                    slot = adj[a].setdefault(b, {"shared": [], "count": 0})
                    slot["count"] += 1
                    slot["shared"].append(_entity)
                    adj[b].setdefault(a, {"shared": [], "count": 0})
                    adj[b][a]["count"] += 1
                    adj[b][a]["shared"].append(_entity)
        return dict(adj)

    def adjacency(self) -> Dict[str, Dict[str, dict]]:
        """版本绑定的邻接缓存：graph.version 变化即重建（演进感知）。"""
        if self._cached_version != self.graph.version or self._adj is None:
            self._adj = self._build_adjacency()
            self._cached_version = self.graph.version
        return self._adj

    def invalidate(self) -> None:
        self._cached_version = None
        self._adj = {}

    # ------------------------------------------------------------------
    def expand(
        self,
        seed_chunks: List[str],
        query_entities: Optional[Set[str]] = None,
        hops: int = 2,
        min_shared_entities: int = 1,
        min_hop_weight: float = 0.05,
        min_hop_intersection: int = 0,
        max_hop_width: Optional[int] = None,
    ) -> Dict[str, dict]:
        """从 seed chunk 出发逐跳扩散。

        返回 {chunk_id: {hop, weight, shared_entities, via}}，不含 seed 本身。
        权重公式（Green-ChunkHop 风格）：
            weight = shared_count × (1 + query_entity_intersection)
        """
        max_hop_width = self.max_hop_width if max_hop_width is None else max_hop_width
        query_entities = query_entities or set()
        adj = self.adjacency()
        visited: Set[str] = set(seed_chunks)
        frontier: Set[str] = set(seed_chunks)
        found: Dict[str, dict] = {}

        for hop in range(1, max(1, hops) + 1):
            candidates: List[Tuple[float, str, dict]] = []
            for src in frontier:
                for dst, info in adj.get(src, {}).items():
                    if dst in visited:
                        continue
                    shared = set(info.get("shared", []))
                    if len(shared) < min_shared_entities:
                        continue
                    inter = len(shared & query_entities)
                    if min_hop_intersection > 0 and inter < min_hop_intersection:
                        continue
                    weight = float(info.get("count", len(shared))) * (1.0 + inter)
                    if weight < min_hop_weight:
                        continue
                    candidates.append((weight, dst, {"hop": hop, "shared": sorted(shared), "via": src}))
            if not candidates:
                break
            candidates.sort(key=lambda x: -x[0])
            added = 0
            for weight, dst, info in candidates:
                if dst in visited or added >= max_hop_width:
                    continue
                visited.add(dst)
                found[dst] = {
                    "hop": info["hop"],
                    "weight": weight,
                    "shared_entities": info["shared"],
                    "via": info["via"],
                }
                added += 1
            frontier = {dst for dst in found if found[dst]["hop"] == hop}
            if not frontier:
                break
        return found
