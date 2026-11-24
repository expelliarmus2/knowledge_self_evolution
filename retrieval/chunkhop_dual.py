"""ChunkHop dual-seed 多跳检索（对齐 Green-ChunkHop main 分支的 dual_seed 算法）。

与旧流程（`hop_expander` 离线全局邻接 + personal 打分）的区别：

1. **双路入图**：chunk 种子（向量路 + 三元组/实体定位路）与实体种子（三元组端点 +
   实体路）同时进入 PPR 图，作为 personalization 向量。
2. **重要实体选取**：`entity_score = w×种子分 + (1-w)×chunk 传导分`，取 Top-K。
   chunk 传导分 = Σ_{含该实体的种子 chunk} seed_score(c)/deg(c)（与参考实现 rank_entities 同构）。
3. **在线 CHUNK_HOP 边**：只经由**已选重要实体**在两个 chunk 之间建边（边权 = 共享重要
   实体数 × 查询交集加成 × 逐跳衰减）。这是抑制年份/国家类枢纽实体的关键 ——
   枢纽实体进不了 Top-K，就不会产生跨全域的噪声边。
4. **真 PPR**：种子分作 personalization、只保留重要实体的 text_conclude 边与实体间三元组边，
   alpha 默认 0.6（参考实现 config.yaml：ppr_alpha=0.6，hops=4，entity_top_k=30）。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

logger = logging.getLogger("kb")


@dataclass
class DualSeedResult:
    scores: Dict[str, float] = field(default_factory=dict)          # chunk -> ppr score
    selected_entities: List[dict] = field(default_factory=list)     # [{entity, seed_score, chunk_score, score}]
    hop_chunks: Dict[str, dict] = field(default_factory=dict)       # cid -> {hop, weight, via, shared}
    hop_edges: List[Tuple[str, str, float, str]] = field(default_factory=list)  # (a,b,w,entity)
    graph_stats: dict = field(default_factory=dict)


class DualSeedChunkHop:
    """双路种子 + 重要实体 + 在线 chunkhop 边 + PPR。"""

    def __init__(self, graph, cfg):
        self.graph = graph
        self.entity_top_k = int(cfg.get("retrieval.entity_top_k", 30))
        self.seed_weight = float(cfg.get("retrieval.entity_seed_weight", 0.5))
        self.hops = int(cfg.get("retrieval.dual_hops", 2))
        self.width = int(cfg.get("retrieval.dual_hop_width", 50))
        self.alpha = float(cfg.get("retrieval.ppr_alpha", 0.6))
        self.evolved_boost = float(cfg.get("retrieval.evolved_boost", 1.0))
        self.decay = float(cfg.get("retrieval.hop_decay", 0.5))
        self.hop_boost = float(cfg.get("retrieval.hop_chunk_boost", 1.0))
        self.min_seed = float(cfg.get("retrieval.dual_min_seed", 0.0))
        # 预建边（作者图格式对齐）：图内已预建 chunk_hop / projected / bridge 边时直接消费
        self.prebuild_hop = bool(cfg.get("retrieval.prebuild_hop", True))
        self.use_prebuilt_edges = bool(cfg.get("retrieval.use_prebuilt_edges", True))
        self.prebuild_rel = str(cfg.get("retrieval.prebuild_rel", "chunk_hop"))
        self.prebuilt_edge_rels = set(cfg.get("retrieval.prebuilt_edge_rels", [
            "chunk_hop", "entity_projected_chunk",
            "bridge_entity_to_chunk", "bridge_chunk_to_entity",
            "bridge_entity_to_entity",
        ]))

    # ------------------------------------------------------------------
    def _prebuilt_chunkhop(self, src: str):
        """src 的图内预建 chunk_hop 邻居 [(dst, weight)]（MultiDiGraph 兼容）。"""
        g = self.graph
        if not g.g.has_node(src):
            return []
        out = []
        for dst, ed in g.g[src].items():
            for _key, data in ed.items() if g.g.is_multigraph() else [(None, ed)]:
                if data.get("relation") == self.prebuild_rel:
                    out.append((dst, float(data.get("weight", 0.0))))
        return out

    def _prebuilt_edges_between(self, nodes: Set[str]):
        """节点集合内的预建边 [(a, b, weight)]（含 projected/bridge/chunk_hop）。"""
        g = self.graph
        out = []
        multigraph = g.g.is_multigraph()
        for n in nodes:
            if not g.g.has_node(n):
                continue
            for m, ed in g.g[n].items():
                if m not in nodes:
                    continue
                for _key, data in (ed.items() if multigraph else [(None, ed)]):
                    rel = data.get("relation")
                    if rel in self.prebuilt_edge_rels:
                        out.append((n, m, float(data.get("weight", 1.0))))
        return out

    # ------------------------------------------------------------------
    def _seed_entity_scores(self, seed_entities, triple_hits) -> Dict[str, float]:
        """实体种子分：实体路相似度 + 三元组端点分（与定位/PPR 口径一致）。"""
        g = self.graph
        scores: Dict[str, float] = {}
        for e in seed_entities or ():
            name = g.canonical(e.get("name", e) if isinstance(e, dict) else e)
            if not g.has_entity(name):
                continue
            s = float(e.get("score", 0.0)) if isinstance(e, dict) else float(scores.get(name, 1.0))
            scores[name] = max(scores.get(name, 0.0), s)
        for t in triple_hits or ():
            for side in ("head", "tail"):
                name = g.canonical(t[side])
                if not g.has_entity(name):
                    continue
                scores[name] = max(scores.get(name, 0.0), float(t.get("score", 0.0)) * 0.5)
        return scores

    # ------------------------------------------------------------------
    def select_entities(
        self,
        seed_chunk_scores: Dict[str, float],
        entity_seed_scores: Dict[str, float],
        query_entities: Optional[Set[str]] = None,
    ) -> List[dict]:
        """重要实体选取：种子信号 + chunk 传导信号（参考 rank_entities）。

        hub 惩罚：传导分除以 ln(1+度) —— England/artist 这类泛化枢纽
        即使在很多种子 chunk 里出现，也因度巨大而被压下去；
        查询实体加成：出现在查询里的实体更可能是题眼。
        """
        g = self.graph
        query_entities = query_entities or set()
        cand: Set[str] = set(entity_seed_scores)
        for cid in seed_chunk_scores:
            cand.update(e for e in g.get_chunk_entities(cid) if g.has_entity(e))

        prop: Dict[str, float] = {}
        deg: Dict[str, int] = {}
        for e in cand:
            chunks = g.get_entity_chunks(e)
            if not chunks:
                continue
            deg[e] = len(chunks)
            hub = math.log1p(len(chunks))
            total = 0.0
            for cid in chunks:
                base = seed_chunk_scores.get(cid)
                if base is None or base <= 0:
                    continue
                total += base / max(1, len(g.get_chunk_entities(cid)))
            prop[e] = total / hub   # hub 惩罚

        max_seed = max((entity_seed_scores.get(e, 0.0) for e in cand), default=0.0) or 1.0
        max_prop = max(prop.values(), default=0.0) or 1.0
        rows = []
        for e in cand:
            ts = entity_seed_scores.get(e, 0.0) / max_seed
            cs = prop.get(e, 0.0) / max_prop
            s = self.seed_weight * ts + (1.0 - self.seed_weight) * cs
            if e in query_entities:
                s *= 2.5                      # 查询题眼加成
            rows.append({
                "entity": e,
                "seed_score": round(ts, 6),
                "chunk_score": round(cs, 6),
                "score": round(s, 6),
                "degree": deg.get(e, 0),
            })
        rows.sort(key=lambda r: (-r["score"], r["entity"]))
        return rows[: self.entity_top_k]

    # ------------------------------------------------------------------
    def expand_online(
        self,
        seed_chunk_scores: Dict[str, float],
        selected: List[dict],
        query_entities: Optional[Set[str]] = None,
        hub_degree_cap: int = 600,
    ) -> Tuple[Dict[str, dict], List[Tuple[str, str, float, str]]]:
        """建 CHUNK_HOP 边并逐跳扩散。

        prebuild_hop=True 且图内已有预建 chunk_hop 边：直接消费预建边（作者图格式对齐），
        权重 = 预建 weight × 查询交集加成 × 逐跳衰减；
        否则回退在线构建（只经由重要实体，抑制枢纽）。
        """
        g = self.graph
        query_entities = query_entities or set()
        selected_names = {
            r["entity"] for r in selected
            if int(r.get("degree", 0)) <= hub_degree_cap       # 跳过超大规模枢纽实体
        }
        # 重要实体 -> 其所在 chunk 集合（在线构建，只覆盖已选实体）
        ent_chunks: Dict[str, Set[str]] = {e: set(g.get_entity_chunks(e)) for e in selected_names}

        visited: Set[str] = set(seed_chunk_scores)
        frontier: Dict[str, float] = dict(seed_chunk_scores)
        hop_chunks: Dict[str, dict] = {}
        hop_edges: List[Tuple[str, str, float, str]] = []

        for hop in range(1, self.hops + 1):
            cand: Dict[str, float] = {}
            via: Dict[str, Tuple[str, str]] = {}
            for src, src_w in frontier.items():
                if self.prebuild_hop and g.g.has_node(src):
                    pre = self._prebuilt_chunkhop(src)
                    if pre:
                        for dst, w0 in pre:
                            if dst == src or dst in visited:
                                continue
                            # 预建边权重已含共享实体数信息，不再叠加查询加成
                            w = self.hop_boost * w0 * (self.decay ** (hop - 1))
                            cand[dst] = cand.get(dst, 0.0) + w
                            if dst not in via or w > via[dst][1]:
                                via[dst] = (src, w)
                        continue
                # 在线回退：只经由重要实体
                src_ents = set(g.get_chunk_entities(src)) & selected_names
                if not src_ents:
                    continue
                for e in src_ents:
                    shared_chunks = ent_chunks.get(e, ())
                    if len(shared_chunks) < 2:
                        continue  # 只在本轮候选集内有参照才有边意义
                    inter = 1.0 + (0.5 if e in query_entities else 0.0)
                    for dst in shared_chunks:
                        if dst == src or dst in visited:
                            continue
                        # 共享重要实体数越多，边越可信
                        w = self.hop_boost * inter * (self.decay ** (hop - 1))
                        cand[dst] = cand.get(dst, 0.0) + w
                        if dst not in via or w > via[dst][1]:
                            via[dst] = (src, w)
            if not cand:
                break
            ranked = sorted(cand.items(), key=lambda kv: -kv[1])[: self.width]
            for dst, w in ranked:
                visited.add(dst)
                src, ew = via[dst]
                hop_chunks[dst] = {"hop": hop, "weight": round(w, 6), "via": src}
                hop_edges.append((src, dst, round(w, 6), ""))
            frontier = {dst: w for dst, w in ranked}
        return hop_chunks, hop_edges

    # ------------------------------------------------------------------
    def run(
        self,
        seed_chunks: Dict[str, dict],
        seed_entities,
        triple_hits: Optional[List[dict]] = None,
        query_entities: Optional[Set[str]] = None,
    ) -> DualSeedResult:
        g = self.graph
        seed_chunk_scores = {
            cid: float(info.get("score", 0.0))
            for cid, info in (seed_chunks or {}).items()
            if info.get("score", 0.0) > self.min_seed
        }
        if not seed_chunk_scores:
            return DualSeedResult()
        entity_scores = self._seed_entity_scores(seed_entities, triple_hits)

        # 1) 重要实体
        selected = self.select_entities(seed_chunk_scores, entity_scores, query_entities)
        selected_names = {r["entity"] for r in selected}

        # 2) 在线 chunkhop 边 + 扩散
        hop_chunks, hop_edges = self.expand_online(seed_chunk_scores, selected, query_entities)

        # 3) PPR 图（只挂重要实体的边，抑制枢纽）
        pg = nx.Graph()
        chunk_nodes = set(seed_chunk_scores) | set(hop_chunks)
        ents = {e for e in selected_names if g.has_entity(e)}
        for n in chunk_nodes:
            if g.g.has_node(n):
                pg.add_node(n)
        for e in ents:
            pg.add_node(e)

        # text_conclude：重要实体 <-> 候选 chunk
        for e in ents:
            for cid in g.get_entity_chunks(e):
                if cid in chunk_nodes and pg.has_node(cid):
                    pg.add_edge(e, cid, weight=1.0)

        # 实体间三元组边（重要实体之间，权取种子分或兜底）
        max_ts = max((entity_scores.get(e, 0.0) for e in ents), default=0.0) or 1.0
        for e in ents:
            for u, v, rel, data in g.entity_neighbor_triples(e):
                other = v if u == e else u
                if other in ents and pg.has_node(other):
                    w = 0.5 + 0.5 * (entity_scores.get(other, 0.0) / max_ts)
                    # 自进化边加成：LLM 写回的知识应在 PPR 传播中更强
                    if (data.get("meta") or {}).get("evolved"):
                        w *= self.evolved_boost
                    pg.add_edge(e, other, weight=w)

        # 在线 chunkhop 边
        for a, b, w, _ent in hop_edges:
            if pg.has_node(a) and pg.has_node(b):
                pg.add_edge(a, b, weight=w)

        # 预建边（projected / bridge / chunk_hop）：作者图格式对齐，覆盖/补足在线边
        if self.use_prebuilt_edges:
            nodes_in_pg = set(pg.nodes)
            for a, b, w in self._prebuilt_edges_between(nodes_in_pg):
                if a == b:
                    continue
                if pg.has_edge(a, b):
                    if w > pg[a][b].get("weight", 0.0):
                        pg[a][b]["weight"] = w
                else:
                    pg.add_edge(a, b, weight=w)

        # personalization：种子 chunk + 已选重要实体（未选中的孤立实体不注入质量，
        # 否则其 teleport 份额≈自身分数直接霸榜，无关 chunk 混入 Top-K）
        personal: Dict[str, float] = {}
        sum_chunk = sum(seed_chunk_scores.values()) or 1.0
        for cid, sc in seed_chunk_scores.items():
            if pg.has_node(cid) and pg.degree(cid) > 0:   # 孤立 chunk（无证据边）不注入质量
                personal[cid] = personal.get(cid, 0.0) + (1.0 - self.seed_weight) * sc / sum_chunk
        sum_ent = sum(entity_scores.get(e, 0.0) for e in ents) or 1.0
        for e in ents:
            if pg.has_node(e):
                personal[e] = personal.get(e, 0.0) + self.seed_weight * entity_scores.get(e, 0.0) / sum_ent
        if not personal:
            personal = {n: 1.0 for n in pg.nodes}

        stats = {
            "seed_chunks": len(seed_chunk_scores),
            "selected_entities": len(selected),
            "hop_chunks": len(hop_chunks),
            "hop_edges": len(hop_edges),
            "ppr_nodes": pg.number_of_nodes(),
            "ppr_edges": pg.number_of_edges(),
        }
        try:
            scores = nx.pagerank(
                pg, alpha=self.alpha, personalization=personal,
                weight="weight", max_iter=100, tol=1e-8,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("dual_seed PPR 失败，退化为 personalization: %s", err)
            scores = dict(personal)

        return DualSeedResult(
            scores={k: float(v) for k, v in scores.items()},
            selected_entities=selected,
            hop_chunks=hop_chunks,
            hop_edges=hop_edges,
            graph_stats=stats,
        )
