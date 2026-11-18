"""知识图谱：NetworkX MultiDiGraph 封装。

数据模型（吸收 huaxi_poc / Green-ChunkHop 的 chunk 组织方式）：
- 节点
  - entity 节点: {type: "entity", display, freq, chunk_ids, meta}
  - chunk  节点: {type: "chunk", text, doc_id, order, meta}
- 边
  - 三元组边: entity --(relation, weight, count, chunk_ids)--> entity
  - text_conclude 边: entity --(text_conclude)--> chunk（实体出现在该 chunk）
  - synonym 边（可选）: entity --(synonym, weight=相似度)--> entity

演进能力（EvoGraph-R1 对接预留）：
- 所有增删改通过 apply_operation(GraphOperation) 走统一入口，记录 changelog；
- version 单调递增，检索侧据此感知图结构变化；
- save/load pickle + export_json（web 可视化数据源）。
"""
from __future__ import annotations

import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from utils.text import normalize_name

logger = logging.getLogger("kb")

NODE_ENTITY = "entity"
NODE_CHUNK = "chunk"
REL_TEXT_CONCLUDE = "text_conclude"
REL_SYNONYM = "synonym"


class KnowledgeGraph:
    def __init__(self, case_insensitive_entities: bool = False, max_changelog: int = 1000):
        self.g = nx.MultiDiGraph()
        self.case_insensitive_entities = case_insensitive_entities
        self.max_changelog = max_changelog
        self.version: int = 0
        self.changelog: List[dict] = []
        self._name2canonical: Dict[str, str] = {}  # 归一化名 -> 规范名
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # 实体名规范化
    # ------------------------------------------------------------------
    def _norm(self, name: str) -> str:
        return normalize_name(name, case_insensitive=self.case_insensitive_entities)

    def canonical(self, name: str) -> str:
        """返回已存在实体的规范名；不存在则原样返回（调用方决定是否新建）。"""
        return self._name2canonical.get(self._norm(name), normalize_name(name))

    # ------------------------------------------------------------------
    # 节点 / 边 增量维护（内部原语，均会提升 version）
    # ------------------------------------------------------------------
    def _bump(self, op: str, **payload) -> None:
        self.version += 1
        self.changelog.append({"version": self.version, "op": op, "ts": time.time(), **payload})
        if len(self.changelog) > self.max_changelog:
            del self.changelog[: len(self.changelog) - self.max_changelog]

    def add_entity(self, name: str, entity_type: Optional[str] = None, meta: Optional[dict] = None) -> str:
        """新增/获取实体节点，返回规范名。"""
        name = normalize_name(name)
        if not name:
            raise ValueError("实体名不能为空")
        key = self._norm(name)
        with self._lock:
            if key in self._name2canonical:
                canonical = self._name2canonical[key]
                self.g.nodes[canonical]["freq"] = self.g.nodes[canonical].get("freq", 0) + 1
                return canonical
            self._name2canonical[key] = name
            self.g.add_node(
                name,
                type=NODE_ENTITY,
                display=name,
                entity_type=entity_type or "unknown",
                freq=1,
                chunk_ids=[],
                meta=meta or {},
            )
            self._bump("add_entity", name=name)
            return name

    def add_chunk(self, chunk_id: str, text: str, doc_id: str = "", order: int = 0, meta: Optional[dict] = None) -> str:
        with self._lock:
            if self.g.has_node(chunk_id):
                self.g.nodes[chunk_id]["text"] = text
                return chunk_id
            self.g.add_node(
                chunk_id,
                type=NODE_CHUNK,
                text=text,
                doc_id=doc_id,
                order=order,
                meta=meta or {},
            )
            self._bump("add_chunk", chunk_id=chunk_id)
            return chunk_id

    def add_triple(
        self,
        head: str,
        relation: str,
        tail: str,
        chunk_id: Optional[str] = None,
        weight: float = 1.0,
        meta: Optional[dict] = None,
    ) -> Tuple[str, str]:
        """新增三元组边（实体不存在则自动创建）。相同 (head, tail, relation) 累积权重。"""
        head = self.add_entity(head)
        tail = self.add_entity(tail)
        relation = normalize_name(relation)
        with self._lock:
            key = (head, tail, relation)
            if self.g.has_edge(*key):
                data = self.g.edges[key]
                data["weight"] = float(data.get("weight", 1.0)) + float(weight)
                data["count"] = int(data.get("count", 1)) + 1
                if chunk_id and chunk_id not in data.get("chunk_ids", []):
                    data.setdefault("chunk_ids", []).append(chunk_id)
            else:
                self.g.add_edge(
                    head, tail, key=relation, relation=relation,
                    weight=float(weight), count=1,
                    chunk_ids=[chunk_id] if chunk_id else [],
                    meta=meta or {},
                )
            if chunk_id:
                self.link_entity_to_chunk(head, chunk_id)
                self.link_entity_to_chunk(tail, chunk_id)
            self._bump("add_triple", head=head, relation=relation, tail=tail, chunk_id=chunk_id)
            return head, tail

    def link_entity_to_chunk(self, entity: str, chunk_id: str) -> None:
        """text_conclude 边：实体出现在 chunk。"""
        entity = self.canonical(entity)
        with self._lock:
            if not self.g.has_node(entity):
                return
            key = (entity, chunk_id, REL_TEXT_CONCLUDE)
            if not self.g.has_edge(*key):
                self.g.add_edge(entity, chunk_id, key=REL_TEXT_CONCLUDE, relation=REL_TEXT_CONCLUDE, count=1)
            else:
                self.g.edges[key]["count"] += 1
            node = self.g.nodes[entity]
            if chunk_id not in node.get("chunk_ids", []):
                node.setdefault("chunk_ids", []).append(chunk_id)
        # 不 bump：随 add_triple 等外层操作一起计数

    def add_synonym_edge(self, a: str, b: str, score: float) -> None:
        a, b = self.canonical(a), self.canonical(b)
        if a == b or not self.g.has_node(a) or not self.g.has_node(b):
            return
        with self._lock:
            if self.g.has_edge(a, b, REL_SYNONYM) or self.g.has_edge(b, a, REL_SYNONYM):
                return
            self.g.add_edge(a, b, key=REL_SYNONYM, relation=REL_SYNONYM, weight=float(score))
            self._bump("add_synonym", a=a, b=b, score=score)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def has_entity(self, name: str) -> bool:
        return self._norm(self.canonical(name)) in self._name2canonical

    def get_entity(self, name: str) -> Optional[dict]:
        canonical = self._name2canonical.get(self._norm(name))
        return dict(self.g.nodes[canonical]) if canonical else None

    def entities(self) -> List[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == NODE_ENTITY]

    def chunks(self) -> List[str]:
        return [n for n, d in self.g.nodes(data=True) if d.get("type") == NODE_CHUNK]

    def chunk_text(self, chunk_id: str) -> str:
        node = self.g.nodes.get(chunk_id)
        return node.get("text", "") if node else ""

    def get_entity_chunks(self, entity: str) -> List[str]:
        node = self.get_entity(entity)
        return list(node.get("chunk_ids", [])) if node else []

    def get_chunk_entities(self, chunk_id: str) -> List[str]:
        if not self.g.has_node(chunk_id):
            return []
        return [
            u for u, _, data in self.g.in_edges(chunk_id, data=True)
            if data.get("relation") == REL_TEXT_CONCLUDE
        ]

    def get_triples(self) -> List[Tuple[str, str, str, dict]]:
        """所有三元组边 (head, tail, relation, data)。"""
        out = []
        for u, v, data in self.g.edges(data=True):
            rel = data.get("relation")
            if rel not in (REL_TEXT_CONCLUDE, REL_SYNONYM):
                out.append((u, v, rel, data))
        return out

    def entity_neighbor_triples(self, entity: str) -> List[Tuple[str, str, str, dict]]:
        """实体的一跳三元组（出边 + 入边）。"""
        entity = self.canonical(entity)
        out = []
        for u, v, key, data in self.g.out_edges(entity, keys=True, data=True):
            if data.get("relation") not in (REL_TEXT_CONCLUDE, REL_SYNONYM):
                out.append((u, v, data.get("relation"), data))
        for u, v, key, data in self.g.in_edges(entity, keys=True, data=True):
            if data.get("relation") not in (REL_TEXT_CONCLUDE, REL_SYNONYM):
                out.append((u, v, data.get("relation"), data))
        return out

    def has_triple(self, head: str, relation: str, tail: str) -> bool:
        h, t = self.canonical(head), self.canonical(tail)
        return self.g.has_edge(h, t, normalize_name(relation))

    def stats(self) -> dict:
        triples = self.get_triples()
        return {
            "version": self.version,
            "entities": len(self.entities()),
            "chunks": len(self.chunks()),
            "triples": len(triples),
            "text_conclude_edges": sum(
                1 for _, _, d in self.g.edges(data=True) if d.get("relation") == REL_TEXT_CONCLUDE
            ),
            "changelog_len": len(self.changelog),
        }

    # ------------------------------------------------------------------
    # 演进原语：删 / 改（供 GraphOperation 与 EvoGraph-R1 调用）
    # ------------------------------------------------------------------
    def remove_entity(self, name: str) -> bool:
        canonical = self._name2canonical.get(self._norm(name))
        if not canonical:
            return False
        with self._lock:
            self.g.remove_node(canonical)
            self._name2canonical.pop(self._norm(name), None)
            self._bump("remove_entity", name=canonical)
        return True

    def remove_chunk(self, chunk_id: str) -> bool:
        if not self.g.has_node(chunk_id):
            return False
        with self._lock:
            self.g.remove_node(chunk_id)
            for node, data in self.g.nodes(data=True):
                if node == chunk_id:
                    continue
                ids = data.get("chunk_ids")
                if isinstance(ids, list) and chunk_id in ids:
                    ids.remove(chunk_id)
            self._bump("remove_chunk", chunk_id=chunk_id)
        return True

    def update_entity(self, name: str, attrs: dict) -> bool:
        canonical = self._name2canonical.get(self._norm(name))
        if not canonical:
            return False
        with self._lock:
            self.g.nodes[canonical].update(attrs)
            self._bump("update_entity", name=canonical, attrs=list(attrs.keys()))
        return True

    def remove_triple(self, head: str, relation: str, tail: str) -> bool:
        h, t = self.canonical(head), self.canonical(tail)
        key = normalize_name(relation)
        if not self.g.has_edge(h, t, key):
            return False
        with self._lock:
            self.g.remove_edge(h, t, key)
            self._bump("remove_triple", head=h, relation=key, tail=t)
        return True

    def update_triple(self, head: str, relation: str, tail: str, weight: Optional[float] = None, meta: Optional[dict] = None) -> bool:
        h, t = self.canonical(head), self.canonical(tail)
        key = normalize_name(relation)
        if not self.g.has_edge(h, t, key):
            return False
        with self._lock:
            data = self.g.edges[(h, t, key)]
            if weight is not None:
                data["weight"] = float(weight)
            if meta:
                data.setdefault("meta", {}).update(meta)
            self._bump("update_triple", head=h, relation=key, tail=t)
        return True

    # ------------------------------------------------------------------
    # 统一演进操作入口
    # ------------------------------------------------------------------
    def apply_operation(self, op: "GraphOperation") -> Any:  # noqa: F821
        """按 op.op 分发到对应原语（演进接口的图侧实现）。"""
        p = op.payload
        if op.op == "add_entity":
            return self.add_entity(p.get("name", ""), p.get("entity_type"), p.get("meta"))
        if op.op == "remove_entity":
            return self.remove_entity(p.get("name", ""))
        if op.op == "update_entity":
            return self.update_entity(p.get("name", ""), p.get("attrs") or {})
        if op.op == "add_triple":
            return self.add_triple(
                p.get("head", ""), p.get("relation", ""), p.get("tail", ""),
                chunk_id=p.get("chunk_id"), weight=float(p.get("weight", 1.0)),
            )
        if op.op == "remove_triple":
            return self.remove_triple(p.get("head", ""), p.get("relation", ""), p.get("tail", ""))
        if op.op == "update_triple":
            return self.update_triple(
                p.get("head", ""), p.get("relation", ""), p.get("tail", ""),
                weight=p.get("weight"), meta=p.get("meta"),
            )
        if op.op == "add_chunk":
            return self.add_chunk(
                p.get("chunk_id", ""), p.get("text", ""), p.get("doc_id", ""),
                int(p.get("order", 0)), p.get("meta"),
            )
        if op.op == "remove_chunk":
            return self.remove_chunk(p.get("chunk_id", ""))
        raise ValueError(f"未知图操作: {op.op}")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "g": self.g,
                        "version": self.version,
                        "changelog": self.changelog,
                        "case_insensitive_entities": self.case_insensitive_entities,
                        "max_changelog": self.max_changelog,
                    },
                    f,
                )

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        kg = cls(
            case_insensitive_entities=payload.get("case_insensitive_entities", False),
            max_changelog=payload.get("max_changelog", 1000),
        )
        kg.g = payload["g"]
        kg.version = payload.get("version", 0)
        kg.changelog = payload.get("changelog", [])
        kg._name2canonical = {normalize_name(n, kg.case_insensitive_entities): n for n in kg.entities()}
        return kg

    def export_json(self, path: Path, node_limit: Optional[int] = None, include_chunks: bool = True) -> dict:
        """导出 web 可视化 JSON。node_limit 超限时按度数保留核心节点。"""
        nodes_out: List[dict] = []
        edges_out: List[dict] = []
        with self._lock:
            entity_nodes = [(n, d) for n, d in self.g.nodes(data=True) if d.get("type") == NODE_ENTITY]
            chunk_nodes = [(n, d) for n, d in self.g.nodes(data=True) if d.get("type") == NODE_CHUNK]
            if node_limit is not None and len(entity_nodes) > node_limit:
                entity_nodes.sort(key=lambda x: -self.g.degree(x[0]))
                entity_nodes = entity_nodes[:node_limit]
                keep = {n for n, _ in entity_nodes}
            else:
                keep = {n for n, _ in entity_nodes}
            for n, d in entity_nodes:
                nodes_out.append({
                    "id": n, "label": d.get("display", n), "type": NODE_ENTITY,
                    "freq": d.get("freq", 0), "entity_type": d.get("entity_type"),
                })
            if include_chunks:
                for n, d in chunk_nodes:
                    nodes_out.append({
                        "id": n, "label": f"[{d.get('doc_id','')}/{d.get('order',0)}]",
                        "type": NODE_CHUNK, "doc_id": d.get("doc_id"), "order": d.get("order"),
                        "text": d.get("text", ""),
                    })
            chunk_ids = {n for n, _ in chunk_nodes}
            for u, v, key, data in self.g.edges(keys=True, data=True):
                rel = data.get("relation")
                if rel == REL_TEXT_CONCLUDE:
                    if not include_chunks or v not in chunk_ids:
                        continue
                    edges_out.append({"source": u, "target": v, "relation": rel, "type": "text_conclude"})
                elif rel == REL_SYNONYM:
                    if u in keep or v in keep:
                        edges_out.append({"source": u, "target": v, "relation": rel, "type": "synonym", "weight": data.get("weight")})
                else:
                    if u in keep or v in keep:
                        edges_out.append({
                            "source": u, "target": v, "relation": rel, "type": "triple",
                            "weight": data.get("weight", 1.0),
                            "chunk_ids": data.get("chunk_ids", []),
                        })
        payload = {"version": self.version, "nodes": nodes_out, "edges": edges_out, "stats": self.stats()}
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        return payload

    def merge(self, other: "KnowledgeGraph") -> None:
        """将另一张图并入（增量构建 / 演进合并场景）。"""
        for n, d in other.g.nodes(data=True):
            if d.get("type") == NODE_CHUNK:
                self.add_chunk(n, d.get("text", ""), d.get("doc_id", ""), d.get("order", 0), d.get("meta"))
        for u, v, key, data in other.g.edges(data=True):
            rel = data.get("relation")
            if rel == REL_TEXT_CONCLUDE:
                self.link_entity_to_chunk(u, v)
            elif rel == REL_SYNONYM:
                self.add_synonym_edge(u, v, data.get("weight", 1.0))
            else:
                for cid in data.get("chunk_ids", [])[:1]:
                    self.add_triple(u, rel, v, chunk_id=cid, weight=data.get("weight", 1.0))
