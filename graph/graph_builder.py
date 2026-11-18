"""流式并行构图：OpenIE 结果 → 知识图谱。

吸收 huaxi_poc 的组织方式：
- 逐批（batch）消费抽取结果，边抽取边构图（流式，不等全部完成）；
- 实体去重（大小写可配置归一）；
- 可选按出现频次剪枝低频实体（min_entity_freq）。
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional

from graph.knowledge_graph import KnowledgeGraph, NODE_ENTITY, REL_TEXT_CONCLUDE
from openie.extractor import OpenIEResult

logger = logging.getLogger("kb")


class GraphBuilder:
    def __init__(self, cfg=None, graph: Optional[KnowledgeGraph] = None):
        cfg = cfg or {}
        self.graph = graph or KnowledgeGraph(
            case_insensitive_entities=bool(cfg.get("graph.case_insensitive_entities", False)),
            max_changelog=int(cfg.get("graph.max_changelog", 1000)),
        )
        self.min_entity_freq = int(cfg.get("graph.min_entity_freq", 1))

    # ------------------------------------------------------------------
    def apply_result(self, result: OpenIEResult, chunk_text: str = "") -> None:
        """将单个 chunk 的抽取结果写入图（增量，可反复调用）。"""
        g = self.graph
        # chunk 节点
        g.add_chunk(result.chunk_id, chunk_text, doc_id=result.doc_id)
        # 实体 + text_conclude
        for ent in result.entities:
            g.add_entity(ent)
            g.link_entity_to_chunk(ent, result.chunk_id)
        # 三元组边
        for tri in result.triples:
            g.add_triple(tri.head, tri.relation, tri.tail, chunk_id=result.chunk_id)

    def apply_batch(self, results: Iterable[OpenIEResult], chunk_texts: Dict[str, str]) -> int:
        """消费一批抽取结果（流式构建的最小单元）。返回本批写入的三元组数。"""
        n = 0
        for res in results:
            if res.error:
                logger.warning("跳过失败结果 chunk=%s: %s", res.chunk_id, res.error)
                continue
            self.apply_result(res, chunk_text=chunk_texts.get(res.chunk_id, ""))
            n += len(res.triples)
        return n

    # ------------------------------------------------------------------
    def build(
        self,
        results: Iterable[Iterable[OpenIEResult]],
        chunk_texts: Dict[str, str],
        on_batch_done=None,
    ) -> KnowledgeGraph:
        """流式构建：results 为「批」的迭代器（每批是抽取结果列表）。

        on_batch_done(batch_idx, batch_results, graph) 为每批完成后的回调，
        供 pipeline 发射演进事件（如 EvoGraph-R1 增量注入）。
        """
        total_triples = 0
        for idx, batch in enumerate(results):
            total_triples += self.apply_batch(batch, chunk_texts)
            if on_batch_done is not None:
                on_batch_done(idx, batch, self.graph)
        # 低频实体剪枝
        if self.min_entity_freq > 1:
            self._prune_low_freq(self.min_entity_freq)
        logger.info(
            "构图完成: entities=%d chunks=%d triples=%d",
            len(self.graph.entities()), len(self.graph.chunks()), total_triples,
        )
        return self.graph

    def _prune_low_freq(self, min_freq: int) -> None:
        """剪枝：删除出现 chunk 数 < min_freq 的实体（演进式清洗的内置示例）。"""
        g = self.graph
        victims = [
            name for name in g.entities()
            if len(g.get_entity_chunks(name)) < min_freq
        ]
        for name in victims:
            g.remove_entity(name)
        if victims:
            logger.info("低频实体剪枝: 移除 %d 个", len(victims))

    # ------------------------------------------------------------------
    def export_artifacts(self, out_dir) -> Dict[str, int]:
        """导出中间产物：entities.json / triples.jsonl / chunks.jsonl。"""
        import json
        from pathlib import Path

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        g = self.graph
        n_ent = n_tri = n_chunk = 0
        with open(out / "entities.json", "w", encoding="utf-8") as f:
            payload = {}
            for name in g.entities():
                d = g.g.nodes[name]
                payload[name] = {
                    "display": d.get("display", name),
                    "entity_type": d.get("entity_type"),
                    "freq": d.get("freq", 0),
                    "chunk_ids": d.get("chunk_ids", []),
                }
                n_ent += 1
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with open(out / "triples.jsonl", "w", encoding="utf-8") as f:
            for u, v, rel, data in g.get_triples():
                f.write(json.dumps({
                    "head": u, "relation": rel, "tail": v,
                    "weight": data.get("weight", 1.0),
                    "chunk_ids": data.get("chunk_ids", []),
                }, ensure_ascii=False) + "\n")
                n_tri += 1
        with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
            for cid in g.chunks():
                d = g.g.nodes[cid]
                f.write(json.dumps({
                    "chunk_id": cid, "doc_id": d.get("doc_id", ""),
                    "order": d.get("order", 0), "text": d.get("text", ""),
                }, ensure_ascii=False) + "\n")
                n_chunk += 1
        return {"entities": n_ent, "triples": n_tri, "chunks": n_chunk}
