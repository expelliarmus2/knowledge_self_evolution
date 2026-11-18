"""工具层：把 `Retriever.search()` 与图查询封装成 agent 可调用工具。"""
from __future__ import annotations

from typing import Dict, List, Optional

from retrieval.retriever import Retriever, SearchResult


class Tools:
    def __init__(self, retriever: Retriever):
        self.r = retriever

    # ------------------------------------------------------------------
    def search(self, query: str, k: Optional[int] = None) -> dict:
        """检索工具：返回 chunk 列表 + 命中三元组 + 重要实体（dual_seed 时含 selected）。"""
        res: SearchResult = self.r.search(query, top_k=k)
        return {
            "chunks": res.chunks,
            "triples": list(res.triples[:60]),
            "entities": list(res.entities),
            "selected_entities": list(getattr(res, "_selected_entities", [])),
            "timing": dict(res.timing),
        }

    def neighbor_chunks(self, entity: str, limit: int = 8) -> List[str]:
        """实体的一跳 chunk（图内 text_conclude 邻居）。"""
        g = self.r.graph
        ent = g.canonical(entity)
        if not g.has_entity(ent):
            return []
        return g.get_entity_chunks(ent)[:limit]

    def entity_triples(self, entity: str, limit: int = 12) -> List[dict]:
        """实体为中心的原图三元组（头尾任意一侧）。"""
        g = self.r.graph
        ent = g.canonical(entity)
        out: List[dict] = []
        for u, v, rel, data in g.entity_neighbor_triples(ent):
            other = v if u == ent else u
            out.append({
                "head": u,
                "relation": rel,
                "tail": v,
                "score": float(data.get("weight", 1.0)),
            })
            if len(out) >= limit:
                break
        return out
