"""向量库管理器：与知识图谱联动的增量同步。

三个库：
- entities : key = 实体规范名，meta={display, entity_type, freq, chunk_ids}
- triples  : key = "head\trelation\ttail"，meta={head, relation, tail, weight}
- chunks   : key = chunk_id，meta={doc_id, order}

增量同步（embedding 调度优化的核心）：
- 只对新增/变更项调用 embedding（增量构建、图演进都不重复计算）；
- 图中被删除的项自动从向量库移除（图删 → 库删）；
- 批量 + 有界并发由 Embedder.embed_batch 承担。

可选：基于实体向量相似度补 synonym 边（huaxi_poc 思路，graph.synonym_enable）。
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from graph.knowledge_graph import KnowledgeGraph, REL_TEXT_CONCLUDE
from utils.config import Config
from vectorstore.faiss_store import FaissStore

logger = logging.getLogger("kb")


class VectorStoreManager:
    def __init__(self, cfg: Config, embedder, work_dir: Path):
        self.cfg = cfg
        self.embedder = embedder
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(cfg.require("embedding.dim"))
        self.triple_text_template: str = cfg.get(
            "vectorstore.triple_text_template", "{head}，{relation}，{tail}"
        )
        self.stores: Dict[str, FaissStore] = {}
        for name in cfg.get("vectorstore.stores", ["entities", "triples", "chunks"]) or []:
            self.stores[name] = FaissStore(name, self.dim, self.work_dir / f"vdb_{name}.faiss")

    # ------------------------------------------------------------------
    # 文本构造
    # ------------------------------------------------------------------
    def triple_key(self, head: str, relation: str, tail: str) -> str:
        return f"{head}\t{relation}\t{tail}"

    def triple_text(self, head: str, relation: str, tail: str) -> str:
        return self.triple_text_template.format(head=head, relation=relation, tail=tail)

    def parse_triple_key(self, key: str) -> Tuple[str, str, str]:
        parts = key.split("\t")
        return (parts[0], parts[1], parts[2]) if len(parts) == 3 else ("", "", "")

    # ------------------------------------------------------------------
    # 增量同步
    # ------------------------------------------------------------------
    def sync_from_graph(self, graph: KnowledgeGraph, incremental: bool = True, progress: bool = False) -> dict:
        """图 → 向量库同步（增量：只嵌入新增/变更项；移除图中已删项）。"""
        report: Dict[str, dict] = {}
        for name in self.stores:
            report[name] = {"added": 0, "removed": 0, "total": 0}

        # ---- 实体 ----
        if "entities" in self.stores:
            store = self.stores["entities"]
            items = []
            for ent in graph.entities():
                d = graph.g.nodes[ent]
                meta = {
                    "display": d.get("display", ent),
                    "entity_type": d.get("entity_type"),
                    "freq": d.get("freq", 0),
                    "chunk_ids": d.get("chunk_ids", []),
                }
                if incremental and store.has(ent) and store.get_meta(ent) == meta:
                    continue
                items.append({"key": ent, "text": d.get("display", ent), "meta": meta})
            self._embed_add(store, items, progress=progress)
            removed = self._prune_missing(store, set(graph.entities()))
            report["entities"] = {"added": len(items), "removed": removed, "total": store.count()}

        # ---- 三元组 ----
        if "triples" in self.stores:
            store = self.stores["triples"]
            items = []
            graph_triples = {}
            for u, v, rel, data in graph.get_triples():
                key = self.triple_key(u, rel, v)
                meta = {"head": u, "relation": rel, "tail": v, "weight": data.get("weight", 1.0)}
                graph_triples[key] = meta
                if incremental and store.has(key):
                    continue
                items.append({"key": key, "text": self.triple_text(u, rel, v), "meta": meta})
            self._embed_add(store, items, progress=progress)
            removed = self._prune_missing(store, set(graph_triples.keys()))
            report["triples"] = {"added": len(items), "removed": removed, "total": store.count()}

        # ---- chunk ----
        if "chunks" in self.stores:
            store = self.stores["chunks"]
            items = []
            for cid in graph.chunks():
                d = graph.g.nodes[cid]
                meta = {"doc_id": d.get("doc_id", ""), "order": d.get("order", 0)}
                if incremental and store.has(cid) and store.get_meta(cid) == meta:
                    continue
                items.append({"key": cid, "text": d.get("text", ""), "meta": meta})
            self._embed_add(store, items, progress=progress)
            removed = self._prune_missing(store, set(graph.chunks()))
            report["chunks"] = {"added": len(items), "removed": removed, "total": store.count()}

        # ---- 可选：同义词边（huaxi_poc 思路）----
        if self.cfg.get("graph.synonym_enable", False) and "entities" in self.stores:
            self._link_synonyms(graph)

        logger.info("向量库同步: %s", report)
        return report

    def _embed_add(self, store: FaissStore, items: List[dict], progress: bool = False) -> None:
        if not items:
            return
        vecs = self.embedder.embed_batch([it["text"] for it in items], progress=progress)
        store.add([
            {"key": it["key"], "vector": vecs[i], "meta": it.get("meta", {})}
            for i, it in enumerate(items)
        ])

    def _prune_missing(self, store: FaissStore, live_keys: set) -> int:
        dead = [k for k in store.keys() if k not in live_keys]
        if dead:
            store.remove(dead)
        return len(dead)

    def _link_synonyms(self, graph: KnowledgeGraph) -> None:
        """实体 KNN 相似度超过阈值 → synonym 边。"""
        store = self.stores["entities"]
        ents = graph.entities()
        if len(ents) < 2:
            return
        threshold = float(self.cfg.get("graph.synonym_threshold", 0.85))
        vecs = np.array([store.index.reconstruct(store.key2id[e]) for e in ents if e in store.key2id])
        names = [e for e in ents if e in store.key2id]
        if len(names) < 2:
            return
        k = min(10, len(names))
        sims, idxs = store.index.search(vecs, k)
        n_linked = 0
        for i, name in enumerate(names):
            for sim, j in zip(sims[i], idxs[i]):
                if j < 0 or int(j) == store.key2id[name]:
                    continue
                other = store.id2key.get(int(j))
                if other is None or other == name:
                    continue
                if sim >= threshold:
                    graph.add_synonym_edge(name, other, float(sim))
                    n_linked += 1
        logger.info("同义词边: 新增 %d 条", n_linked)

    # ------------------------------------------------------------------
    # 检索接口
    # ------------------------------------------------------------------
    def search(self, store_name: str, query_vec, top_k: int, min_sim: float = 0.0):
        store = self.stores.get(store_name)
        if store is None:
            raise KeyError(f"未配置向量库: {store_name}")
        results = store.search(query_vec, top_k)
        return [(k, s, m) for k, s, m in results if s >= min_sim]

    # ------------------------------------------------------------------
    def save(self) -> None:
        for store in self.stores.values():
            store.save()

    def stats(self) -> dict:
        return {name: store.stats() for name, store in self.stores.items()}
