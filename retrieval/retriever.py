"""核心检索器：双路召回 → CHUNK_HOP 扩散 → PPR 排序 → LLM 三元组过滤。

流程（吸收 Green-ChunkHop 的检索组织）：
1. 查询 embedding；
2. 三路并行种子召回（三元组库 / 实体库 / chunk 向量库，线程并行）；
3. 实体→chunk 定位（text_conclude 边）得到三元组路 seed；
4. CHUNK_HOP 多跳扩散（共享实体 chunk 邻接，查询感知权重）；
5. （可选）三元组 n 跳扩展；
6. 子图 PPR（networkx pagerank + personalized 向量）得到 chunk 排序；
7. LLM 对候选三元组做相关性过滤（HippoRAG2 filter）；
8. 组装可视化子图（实体节点 + 三元组边 + chunk 节点 + 定位/hop 边）。

演进感知（EvoGraph-R1 对接预留）：
- _ensure_fresh(): 图文件 mtime / graph.version 变化时自动重建 hop 邻接缓存；
- 三元组/实体召回结果与当前图做存在性校验——图被演进删除的元素自动失效。
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from graph.knowledge_graph import KnowledgeGraph, REL_TEXT_CONCLUDE
from openie.embedder import Embedder
from retrieval.filterer import TripleFilterer
from retrieval.hop_expander import HopExpander
from utils.config import Config

logger = logging.getLogger("kb")


@dataclass
class SearchResult:
    query: str
    graph_version: int
    triples: List[dict] = field(default_factory=list)          # 命中三元组 [{head,relation,tail,score}]
    entities: List[dict] = field(default_factory=list)         # 命中实体 [{name,score,display}]
    chunks: List[dict] = field(default_factory=list)           # PPR 排序 chunk [{chunk_id,text,score,rank,source}]
    filtered_triples: List[list] = field(default_factory=list) # LLM 过滤后三元组 [[s,p,o]]
    filter_info: dict = field(default_factory=dict)
    subgraph: dict = field(default_factory=dict)               # 可视化子图 {nodes, edges, mode}
    timing: dict = field(default_factory=dict)
    # --- 诊断字段（失败归因用，不入 to_dict）---
    _full_rank: List[Tuple[str, float]] = field(default_factory=list, repr=False)
    _seed_chunks: Dict[str, str] = field(default_factory=dict, repr=False)
    _hop_chunks: Dict[str, dict] = field(default_factory=dict, repr=False)
    _dual_stats: dict = field(default_factory=dict, repr=False)
    _selected_entities: List[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "graph_version": self.graph_version,
            "triples": self.triples[:60],               # 候选三元组仅返回前 60 条（降传输/渲染开销）
            "triples_total": len(self.triples),
            "entities": self.entities,
            "chunks": self.chunks,
            "filtered_triples": self.filtered_triples,
            "filter_info": self.filter_info,
            "subgraph": self.subgraph,
            "timing": self.timing,
        }


class Retriever:
    def __init__(
        self,
        cfg: Config,
        graph: KnowledgeGraph,
        vector_manager,
        embedder: Embedder,
        llm=None,
        prompts=None,
        graph_path: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.graph = graph
        self.graph_path = Path(graph_path) if graph_path else None
        self._graph_mtime: Optional[float] = None
        self.vdb = vector_manager
        self.embedder = embedder
        self.filterer = TripleFilterer.from_config(cfg, llm, prompts) if (llm and prompts) else None
        self.hop_expander = HopExpander(graph, max_hop_width=int(cfg.get("retrieval.max_hop_width", 50)))
        from retrieval.chunkhop_dual import DualSeedChunkHop
        self.dual_hopper = DualSeedChunkHop(graph, cfg)

    # ------------------------------------------------------------------
    # 演进感知：图变更 → 重建缓存
    # ------------------------------------------------------------------
    def _ensure_fresh(self) -> None:
        """图文件更新（外部进程演进）时重载；版本变化时失效 hop 缓存。"""
        if self.graph_path is not None and self.graph_path.exists():
            mtime = self.graph_path.stat().st_mtime
            if self._graph_mtime is not None and mtime != self._graph_mtime:
                logger.info("检测到图文件更新，重新加载 (mtime=%s)", mtime)
                self.graph = KnowledgeGraph.load(self.graph_path)
                self.hop_expander = HopExpander(self.graph, max_hop_width=self.hop_expander.max_hop_width)
            self._graph_mtime = mtime
        if self.hop_expander._cached_version != self.graph.version:
            self.hop_expander.invalidate()

    # ------------------------------------------------------------------
    # 三路并行种子召回
    # ------------------------------------------------------------------
    def _recall(self, qvec) -> Tuple[List[dict], List[dict], List[dict]]:
        """返回 (triples, entities, vector_chunks)，线程并行（Green-ChunkHop 双路思路）。"""
        out: Dict[str, List[dict]] = {}
        errors: List[Exception] = []
        min_sim = float(self.cfg.get("retrieval.min_similarity", 0.0))

        def _do_triples() -> None:
            hits = self.vdb.search(
                "triples", qvec, int(self.cfg.get("retrieval.triple_seed", 10)), min_sim=min_sim
            )
            out["triples"] = [
                {
                    "head": m.get("head", k.split("\t")[0]),
                    "relation": m.get("relation", k.split("\t")[1] if k.count("\t") >= 2 else ""),
                    "tail": m.get("tail", k.split("\t")[2] if k.count("\t") >= 2 else ""),
                    "score": s,
                }
                for k, s, m in hits
            ]

        def _do_entities() -> None:
            hits = self.vdb.search(
                "entities", qvec, int(self.cfg.get("retrieval.entity_seed", 10)), min_sim=min_sim
            )
            out["entities"] = [
                {"name": k, "score": s, "display": m.get("display", k), "entity_type": m.get("entity_type")}
                for k, s, m in hits
            ]

        def _do_chunks() -> None:
            hits = self.vdb.search(
                "chunks", qvec, int(self.cfg.get("retrieval.vector_seed", 20)), min_sim=min_sim * 0.5
            )
            out["chunks"] = [{"chunk_id": k, "score": s, "meta": m} for k, s, m in hits]

        threads = []
        for target in (_do_triples, _do_entities, _do_chunks):
            def runner(t=target):
                try:
                    t()
                except Exception as err:  # noqa: BLE001
                    errors.append(err)
            th = threading.Thread(target=runner)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        for err in errors:
            logger.error("种子召回失败: %s", err)

        return out.get("triples", []), out.get("entities", []), out.get("chunks", [])

    # ------------------------------------------------------------------
    # 演进校验：召回结果与当前图核对
    # ------------------------------------------------------------------
    def _validate_triples(self, triples: List[dict]) -> List[dict]:
        keep = []
        for t in triples:
            if self.graph.has_triple(t["head"], t["relation"], t["tail"]):
                keep.append(t)
        return keep

    def _validate_entities(self, entities: List[dict]) -> List[dict]:
        return [e for e in entities if self.graph.has_entity(e["name"])]

    # ------------------------------------------------------------------
    # 实体 → chunk 定位（三元组路 seed）
    # ------------------------------------------------------------------
    def _locate_chunks(self, triples: List[dict], entities: List[dict]) -> Dict[str, dict]:
        """text_conclude 定位：chunk 分数 = Σ(锚定三元组分) / 度（huaxi/Green 思路）。"""
        entity_score: Dict[str, float] = {e["name"]: e["score"] for e in entities}
        for t in triples:
            for side in ("head", "tail"):
                name = self.graph.canonical(t[side])
                entity_score[name] = max(entity_score.get(name, 0.0), t["score"] * 0.5)
        located: Dict[str, dict] = {}
        for name, score in entity_score.items():
            if score <= 0:
                continue
            for cid in self.graph.get_entity_chunks(name):
                degree = max(1, len(self.graph.get_chunk_entities(cid)))
                prev = located.setdefault(cid, {"score": 0.0, "via_entities": [], "source": "triple"})
                prev["score"] += score / degree
                if name not in prev["via_entities"]:
                    prev["via_entities"].append(name)
        # 泛化实体（the city/explorer 等）会链接大量 chunk，冲淡种子权重：按分排序截断
        max_located = int(self.cfg.get("retrieval.locate_max_chunks", 80))
        if len(located) > max_located:
            ranked = sorted(located.items(), key=lambda kv: -kv[1]["score"])[:max_located]
            located = dict(ranked)
        return located

    # ------------------------------------------------------------------
    # 三元组 n 跳扩展
    # ------------------------------------------------------------------
    def _expand_triples(
        self, seed_entities: Set[str], hops: int, decay: float
    ) -> List[dict]:
        """BFS 沿三元组边扩展，收集新三元组（分数逐跳衰减）。"""
        frontier = {self.graph.canonical(e) for e in seed_entities if self.graph.has_entity(e)}
        visited = set(frontier)
        collected: List[dict] = []
        seen: Set[tuple] = set()
        score = 1.0
        for hop in range(1, max(1, hops) + 1):
            nxt: Set[str] = set()
            for ent in frontier:
                for u, v, rel, data in self.graph.entity_neighbor_triples(ent):
                    other = v if u == ent else u
                    key = (u, rel, v)
                    if key in seen:
                        continue
                    seen.add(key)
                    collected.append({
                        "head": u, "relation": rel, "tail": v,
                        "score": float(data.get("weight", 1.0)) * score,
                        "expanded_hop": hop,
                    })
                    if other not in visited:
                        nxt.add(other)
            visited |= nxt
            frontier = nxt
            score *= decay
            if not frontier:
                break
        return collected

    # ------------------------------------------------------------------
    # 子图 PPR
    # ------------------------------------------------------------------
    def _run_ppr(
        self,
        triple_hits: List[dict],
        entity_hits: List[dict],
        vector_chunks: List[dict],
        located_chunks: Dict[str, dict],
        hop_chunks: Dict[str, dict],
    ) -> Tuple[Dict[str, float], nx.Graph]:
        g = nx.Graph()  # 无向传播
        edge_weights: Dict[Tuple[str, str], float] = {}
        node_set: Set[str] = set()

        seed_entities: Set[str] = set()
        for e in entity_hits:
            seed_entities.add(e["name"])
        for t in triple_hits:
            seed_entities.add(self.graph.canonical(t["head"]))
            seed_entities.add(self.graph.canonical(t["tail"]))
        seed_entities = {e for e in seed_entities if self.graph.has_entity(e)}
        node_set |= seed_entities

        chunk_nodes: Set[str] = set()
        for c in vector_chunks:
            chunk_nodes.add(c["chunk_id"])
        chunk_nodes |= set(located_chunks.keys())
        chunk_nodes |= set(hop_chunks.keys())
        node_set |= {c for c in chunk_nodes if self.graph.g.has_node(c)}

        # 三元组边（实体-实体）
        for t in triple_hits:
            h = self.graph.canonical(t["head"])
            tl = self.graph.canonical(t["tail"])
            if h in node_set and tl in node_set and h != tl:
                key = (min(h, tl), max(h, tl))
                edge_weights[key] = edge_weights.get(key, 0.0) + max(0.05, t.get("score", 0.0))

        # text_conclude 边（实体-chunk）
        for cid in [c for c in chunk_nodes if self.graph.g.has_node(c)]:
            for ent in self.graph.get_chunk_entities(cid):
                if ent in node_set:
                    key = (min(ent, cid), max(ent, cid))
                    edge_weights[key] = edge_weights.get(key, 0.0) + 1.0

        # chunk_hop 边（chunk-chunk，共享实体）—— 按 hop 衰减，避免深层噪声枢纽边把 PPR 质量抽干
        hop_decay = float(self.cfg.get("retrieval.hop_decay", 0.5))
        for cid, info in hop_chunks.items():
            via = info.get("via")
            if via in chunk_nodes and self.graph.g.has_node(cid):
                key = (min(via, cid), max(via, cid))
                w = info.get("weight", 1.0) * (hop_decay ** max(0, info.get("hop", 1) - 1))
                edge_weights[key] = max(edge_weights.get(key, 0.0), w)

        for (u, v), w in edge_weights.items():
            g.add_edge(u, v, weight=w)
        for n in node_set:
            if not g.has_node(n):
                g.add_node(n)

        # personalization
        personal: Dict[str, float] = {}
        for e in entity_hits:
            if e["name"] in g:
                personal[e["name"]] = personal.get(e["name"], 0.0) + max(1e-6, e["score"])
        for t in triple_hits:
            for side in ("head", "tail"):
                name = self.graph.canonical(t[side])
                if name in g:
                    personal[name] = personal.get(name, 0.0) + max(1e-6, t.get("score", 0.0)) * 0.5
        if not personal:
            for e in seed_entities:
                personal[e] = 1.0
        for c in vector_chunks:
            if c["chunk_id"] in g:
                personal[c["chunk_id"]] = personal.get(c["chunk_id"], 0.0) + max(1e-6, c["score"])
        for cid, info in located_chunks.items():
            if cid in g:
                personal[cid] = personal.get(cid, 0.0) + max(1e-6, info.get("score", 0.0))
        for cid, info in hop_chunks.items():
            if cid in g:
                # hop chunk 权重归一（相对最大 hop 权重）后按系数计入，并按跳数衰减
                decay = hop_decay ** max(0, info.get("hop", 1) - 1)
                boost = float(self.cfg.get("retrieval.hop_chunk_boost", 1e-3))
                personal[cid] = personal.get(cid, 0.0) + 1e-6 + info.get("weight", 1.0) * boost * decay

        if g.number_of_nodes() == 0:
            return {}, g
        ppr_mode = str(self.cfg.get("retrieval.ppr_mode", "pagerank"))
        if ppr_mode == "personal":
            # 不扩散：直接用个性化向量（种子分）排序，避免高连通枢纽 chunk 吸走 PPR 质量
            return {k: v for k, v in personal.items()}, g
        try:
            scores = nx.pagerank(
                g,
                alpha=float(self.cfg.get("retrieval.ppr_alpha", 0.85)),
                personalization=personal or None,
                weight="weight",
                max_iter=int(self.cfg.get("retrieval.ppr_max_iter", 50)),
                tol=1e-6,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("PPR 失败，退化为 personalization 排序: %s", err)
            scores = {k: v for k, v in personal.items() if k in g}
        return scores, g

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: Optional[int] = None, hops: Optional[int] = None,
               subgraph_mode: Optional[str] = None) -> SearchResult:
        t_start = time.time()
        self._ensure_fresh()
        result = SearchResult(query=query, graph_version=self.graph.version)

        # 1. 查询向量
        t0 = time.time()
        qvec = self.embedder.embed(query)
        result.timing["embed"] = round(time.time() - t0, 4)

        # 2. 三路并行召回
        t0 = time.time()
        triple_hits, entity_hits, vector_chunks = self._recall(qvec)
        triple_hits = self._validate_triples(triple_hits)   # 演进感知：剔除图中已删除的三元组
        entity_hits = self._validate_entities(entity_hits)
        result.timing["recall"] = round(time.time() - t0, 4)
        result.triples = triple_hits
        result.entities = entity_hits

        # 3. 实体→chunk 定位（三元组路 seed）
        t0 = time.time()
        located = self._locate_chunks(triple_hits, entity_hits)
        seed_chunks: Dict[str, dict] = {}
        for c in vector_chunks:
            slot = seed_chunks.setdefault(c["chunk_id"], {"score": 0.0, "via_entities": [], "source": "vector"})
            slot["score"] += c["score"]
        for cid, info in located.items():
            slot = seed_chunks.setdefault(cid, {"score": 0.0, "via_entities": [], "source": "triple"})
            slot["score"] += info["score"]
            slot["source"] = "both" if slot["source"] == "vector" else slot["source"]
            slot["via_entities"].extend(info["via_entities"])
        result.timing["locate"] = round(time.time() - t0, 4)

        # 4. CHUNK_HOP 扩散（legacy）/ 重要实体+在线hop边（dual_seed）
        t0 = time.time()
        _raw_skip = self.cfg.get("retrieval.skip_hop", "auto")
        chunkhop_mode = str(self.cfg.get("retrieval.chunkhop_mode", "legacy")).lower()
        query_entities: Set[str] = {e["name"] for e in entity_hits}
        query_entities |= {self.graph.canonical(t[side]) for t in triple_hits for side in ("head", "tail")}
        dual_result = None
        if chunkhop_mode == "dual_seed":
            # 双路入图 + 重要实体选取 + 在线 CHUNK_HOP 边 + 真 PPR（chunkhop_dual.py）
            dual_result = self.dual_hopper.run(seed_chunks, entity_hits, triple_hits, query_entities)
            hop_chunks = {
                cid: {"hop": int(info.get("hop", 1)), "weight": info.get("weight", 1.0),
                       "via": info.get("via")}
                for cid, info in dual_result.hop_chunks.items()
            }
            result.timing["hop"] = round(time.time() - t0, 4)
            skip_hop = True
        else:
            if isinstance(_raw_skip, bool):
                skip_hop = _raw_skip          # YAML 布尔直接使用（str(False)=="false" 是真值，勿字符串化）
            elif str(_raw_skip).lower() in ("true", "1", "yes", "on"):
                skip_hop = True
            elif str(_raw_skip).lower() in ("false", "0", "no", "off"):
                skip_hop = False
            else:  # auto：personal 模式下跳过 CHUNK_HOP
                skip_hop = str(self.cfg.get("retrieval.ppr_mode", "pagerank")) == "personal"
            if skip_hop:
                hop_chunks = {}
            else:
                hop_chunks = self.hop_expander.expand(
                    seed_chunks=list(seed_chunks.keys()),
                    query_entities=query_entities,
                hops=hops if hops is not None else int(self.cfg.get("retrieval.hops", 2)),
                min_shared_entities=int(self.cfg.get("retrieval.min_shared_entities", 1)),
                min_hop_weight=float(self.cfg.get("retrieval.min_hop_weight", 0.05)),
                min_hop_intersection=int(self.cfg.get("retrieval.min_hop_intersection", 0)),
            )
        result.timing["hop"] = result.timing.get("hop", round(time.time() - t0, 4))

        # 5. 三元组 n 跳扩展（可选）
        t0 = time.time()
        if bool(self.cfg.get("retrieval.use_triple_expand", True)):
            seed_entities = {e for e in query_entities if self.graph.has_entity(e)}
            if seed_entities:
                expanded = self._expand_triples(
                    seed_entities,
                    hops=int(self.cfg.get("retrieval.triple_expand_hops", 2)),
                    decay=float(self.cfg.get("retrieval.triple_expand_decay", 0.5)),
                )
                known_keys = {(t["head"], t["relation"], t["tail"]) for t in triple_hits}
                for t in expanded:
                    if (t["head"], t["relation"], t["tail"]) not in known_keys:
                        triple_hits.append(t)
        result.timing["expand"] = round(time.time() - t0, 4)

        # 6. PPR 排序（dual_seed 已在 run() 内完成；legacy 走 personal/pagerank）
        t0 = time.time()
        if dual_result is not None:
            ppr_scores, ppr_graph = dual_result.scores, None
            # 混合：PPR 分数（覆盖 hop/桥接）+ 种子分（精度）。纯 PPR 会把质量摊到
            # hop chunk 上，Top-5 排序被稀释；blend=1 退化为纯 PPR，blend=0 退化为种子直排
            blend = float(self.cfg.get("retrieval.dual_score_blend", 0.5))
            max_p = max(ppr_scores.values(), default=0.0) or 1.0
            max_s = max((v.get("score", 0.0) for v in seed_chunks.values()), default=0.0) or 1.0
            combined: Dict[str, float] = {}
            for cid in set(ppr_scores) | set(seed_chunks) | set(hop_chunks):
                p = ppr_scores.get(cid, 0.0) / max_p
                s = seed_chunks.get(cid, {}).get("score", 0.0) / max_s
                combined[cid] = blend * p + (1.0 - blend) * s
            ppr_scores = combined
        else:
            ppr_scores, ppr_graph = self._run_ppr(triple_hits, entity_hits, vector_chunks, located, hop_chunks)
        k = top_k or int(self.cfg.get("retrieval.final_top_k", 10))
        chunk_rank = [
            (cid, sc) for cid, sc in ppr_scores.items()
            if self.graph.g.has_node(cid) and self.graph.g.nodes[cid].get("type") == "chunk"
        ]
        chunk_rank.sort(key=lambda x: -x[1])
        # 文档级去重：同一 doc 的多 chunk 只保留得分最高者（MuSiQue 金标为段落/文档级，
        # 且 RAG 上下文不应重复段落；对照实验 @5 0.565→0.599）
        seen_docs: Set[str] = set()
        top_chunks: List[Tuple[str, float]] = []
        for cid, sc in chunk_rank:
            d = self.graph.g.nodes[cid].get("doc_id", "")
            if d in seen_docs:
                continue
            seen_docs.add(d)
            top_chunks.append((cid, sc))
            if len(top_chunks) >= k:
                break
        for rank, (cid, sc) in enumerate(top_chunks, start=1):
            node = self.graph.g.nodes[cid]
            source = seed_chunks.get(cid, {}).get("source", "hop")
            result.chunks.append({
                "chunk_id": cid,
                "doc_id": node.get("doc_id", ""),
                "text": node.get("text", ""),
                "score": round(float(sc), 6),
                "rank": rank,
                "source": source,
                "hop": hop_chunks.get(cid, {}).get("hop", 0),
            })
        result.timing["ppr"] = round(time.time() - t0, 4)

        # 7. LLM 三元组过滤
        t0 = time.time()
        if self.filterer is not None:
            facts = [(t["head"], t["relation"], t["tail"]) for t in triple_hits]
            scores = [t.get("score", 0.0) for t in triple_hits]
            kept, info = self.filterer.filter(query, facts, scores)
            result.filtered_triples = [list(f) for f in kept]
            result.filter_info = info
        else:
            result.filtered_triples = [[t["head"], t["relation"], t["tail"]] for t in triple_hits[:k]]
            result.filter_info = {"mode": "disabled"}
        result.timing["filter"] = round(time.time() - t0, 4)

        # 8. 可视化子图（trace = 溯源子图，chunkhop M3+ 方案；ppr = 原始展开全图）
        mode = subgraph_mode or str(self.cfg.get("retrieval.subgraph_mode", "trace"))
        if mode == "trace":
            result.subgraph = self._trace_subgraph(
                result, triple_hits, hop_chunks,
                max_entities=int(self.cfg.get("retrieval.trace_max_entities", 30)),
            )
        else:
            result.subgraph = self._build_subgraph(result, triple_hits, seed_chunks, hop_chunks, ppr_scores)
        # 诊断：完整候选排序 + 种子/hop 来源（失败归因）
        result._full_rank = [(cid, round(float(sc), 4)) for cid, sc in chunk_rank]
        result._seed_chunks = {cid: info["source"] for cid, info in seed_chunks.items()}
        result._hop_chunks = {cid: {"hop": int(info.get("hop", 0)), "via": info.get("via")}
                               for cid, info in hop_chunks.items()}
        if dual_result is not None:
            result._dual_stats = dual_result.graph_stats
            result._selected_entities = dual_result.selected_entities
        result.timing["total"] = round(time.time() - t_start, 4)
        return result

    # ------------------------------------------------------------------
    def _trace_subgraph(
        self,
        result: SearchResult,
        triple_hits: List[dict],
        hop_chunks: Dict[str, dict],
        max_entities: int = 30,
    ) -> dict:
        """溯源子图（参考 chunkhop entity_subgraph / M3+ 方案）：

        规模完全可控 —— 只有三类节点、三类边：
          - chunk 节点：最终 Top-K（result.chunks）
          - 实体节点：候选 = (seed 三元组实体 ∪ Top-K chunk 关联实体)，
            按 seed_weight×归一化三元组分 + (1-seed_weight)×归一化 chunk 聚合分 排序取 Top-N
          - triple 边：选中实体对之间在原图上的完整关系（seed/filtered 高亮）
          - text_conclude 边：选中实体 → Top-K chunk
          - chunk_hop 边：两端均在 Top-K 内的跳转
        """
        top = result.chunks
        top_ids = [c["chunk_id"] for c in top]
        top_set = set(top_ids)
        chunk_score = {c["chunk_id"]: c["score"] for c in top}

        # 1) 候选实体：seed 三元组实体（三元组分）+ Top-K chunk 关联实体（chunk 分）
        seed_scores: Dict[str, float] = {}
        for t in triple_hits:
            for side in ("head", "tail"):
                eid = self.graph.canonical(t[side])
                if self.graph.has_entity(eid):
                    seed_scores[eid] = seed_scores.get(eid, 0.0) + float(t.get("score", 0.0))
        links: Dict[str, set] = {}
        for cid in top_ids:
            for ent in self.graph.get_chunk_entities(cid):
                links.setdefault(ent, set()).add(cid)

        # 2) 实体打分排序（对齐 chunkhop rank_entities：各自按最大值归一后加权）
        max_seed = max(seed_scores.values(), default=0.0)
        rows = []
        for eid in set(seed_scores) | set(links):
            t_norm = (seed_scores.get(eid, 0.0) / max_seed) if max_seed else 0.0
            c_sum = sum(chunk_score.get(cid, 0.0) for cid in links.get(eid, ()))
            rows.append((eid, t_norm, c_sum))
        max_chunk = max((r[2] for r in rows), default=0.0)
        seed_w = float(self.cfg.get("retrieval.trace_seed_weight", 0.5))
        rows.sort(key=lambda r: (-(seed_w * r[1] + (1 - seed_w) * (r[2] / max_chunk if max_chunk else 0.0)), r[0]))
        selected = [r[0] for r in rows[:max_entities]]
        sel_set = set(selected)

        nodes: List[dict] = []
        edges: List[dict] = []

        def _disp(eid: str) -> str:
            return self.graph.g.nodes[eid].get("display", eid)

        # 3) chunk 节点（仅 Top-K；text 截断为预览，全文在 result.chunks）
        for c in top:
            nodes.append({
                "id": c["chunk_id"], "label": f"[{c['doc_id']}/{c['chunk_id'].rsplit('chunk', 1)[-1]}]",
                "type": "chunk", "seed": c["source"] in ("vector", "triple", "both"),
                "hop": hop_chunks.get(c["chunk_id"], {}).get("hop", 0), "rank": c["rank"],
                "doc_id": c["doc_id"], "text": c["text"][:160],
            })

        # 4) 实体节点
        for eid in selected:
            d = self.graph.g.nodes[eid]
            nodes.append({
                "id": eid, "label": d.get("display", eid), "type": "entity",
                "matched": eid in seed_scores, "score": round(seed_scores.get(eid, 0.0), 4),
            })

        # 5a) triple 边：选中实体间的完整关系（原图邻接枚举，同事实去重）
        filtered_set = {tuple(t) for t in result.filtered_triples}
        seen: set = set()
        for eid in selected:
            for _, tgt, rel, data in self.graph.g.out_edges(eid, keys=True, data=True):
                if tgt not in sel_set or (eid, tgt, rel) in seen:
                    continue
                seen.add((eid, tgt, rel))
                is_filtered = (_disp(eid), rel, _disp(tgt)) in filtered_set
                edges.append({
                    "source": eid, "target": tgt, "relation": rel, "type": "triple",
                    "score": round(float(data.get("weight", 0.0)), 4),
                    "filtered": is_filtered, "expanded_hop": 0,
                })

        # 5b) text_conclude 边：选中实体 → Top-K chunk
        for eid in selected:
            for cid in sorted(links.get(eid, ())):
                edges.append({"source": eid, "target": cid, "relation": "text_conclude", "type": "text_conclude"})

        # 5c) chunk_hop 边：两端均在 Top-K 内
        for cid, info in hop_chunks.items():
            via = info.get("via")
            if cid in top_set and via in top_set:
                edges.append({
                    "source": via, "target": cid, "relation": "chunk_hop", "type": "chunk_hop",
                    "weight": round(info.get("weight", 1.0), 4),
                    "shared_entities": info.get("shared_entities", [])[:8], "hop": info.get("hop", 0),
                })

        return {"nodes": nodes, "edges": edges, "mode": "trace", "version": self.graph.version,
                "entity_selection": {"max_entities": max_entities, "selected": len(selected),
                                       "candidates": len(rows), "seed_weight": seed_w}}

    # ------------------------------------------------------------------
    def _build_subgraph(
        self,
        result: SearchResult,
        triple_hits: List[dict],
        seed_chunks: Dict[str, dict],
        hop_chunks: Dict[str, dict],
        ppr_scores: Dict[str, float],
    ) -> dict:
        nodes: List[dict] = []
        edges: List[dict] = []
        node_ids: Set[str] = set()

        def add_entity_node(name: str, score: float = 0.0, matched: bool = False) -> None:
            canonical = self.graph.canonical(name)
            if canonical not in node_ids and self.graph.has_entity(canonical):
                d = self.graph.g.nodes[canonical]
                nodes.append({
                    "id": canonical,
                    "label": d.get("display", canonical),
                    "type": "entity",
                    "matched": matched,
                    "score": round(float(score), 4),
                })
                node_ids.add(canonical)

        # 过滤后的三元组 → 实体节点 + 高亮边
        filtered_set = {tuple(t) for t in result.filtered_triples}
        for t in triple_hits:
            key = (t["head"], t["relation"], t["tail"])
            matched = key in filtered_set
            add_entity_node(t["head"], t.get("score", 0.0), True)
            add_entity_node(t["tail"], t.get("score", 0.0), True)
            edges.append({
                "source": self.graph.canonical(t["head"]),
                "target": self.graph.canonical(t["tail"]),
                "relation": t["relation"],
                "type": "triple",
                "score": round(t.get("score", 0.0), 4),
                "filtered": matched,
                "expanded_hop": t.get("expanded_hop", 0),
            })
        for e in result.entities:
            add_entity_node(e["name"], e.get("score", 0.0), True)

        # chunk 节点（seed + hop + top）
        top_ids = {c["chunk_id"] for c in result.chunks}
        chunk_ids: Set[str] = set(top_ids)
        for cid in list(seed_chunks.keys()) + list(hop_chunks.keys()):
            if self.graph.g.has_node(cid):
                chunk_ids.add(cid)
        for cid in chunk_ids:
            node = self.graph.g.nodes.get(cid)
            if node is None:
                continue
            is_seed = cid in seed_chunks
            is_hop = cid in hop_chunks
            nodes.append({
                "id": cid,
                "label": f"[{node.get('doc_id', '')}/{node.get('order', 0)}]",
                "type": "chunk",
                "seed": is_seed,
                "hop": hop_chunks.get(cid, {}).get("hop", 0) if is_hop else 0,
                "rank": next((c["rank"] for c in result.chunks if c["chunk_id"] == cid), None),
                "doc_id": node.get("doc_id", ""),
                "text": (node.get("text", "") or "")[:160],   # 预览截断，全文在 chunks 列表
            })
            node_ids.add(cid)

        # text_conclude 边（实体→chunk 定位）
        for cid in chunk_ids:
            for ent in self.graph.get_chunk_entities(cid):
                if ent in node_ids:
                    edges.append({"source": ent, "target": cid, "relation": "text_conclude", "type": "text_conclude"})

        # chunk_hop 边
        for cid, info in hop_chunks.items():
            via = info.get("via")
            if via in node_ids and cid in node_ids:
                edges.append({
                    "source": via,
                    "target": cid,
                    "relation": "chunk_hop",
                    "type": "chunk_hop",
                    "weight": round(info.get("weight", 1.0), 4),
                    "shared_entities": info.get("shared_entities", [])[:8],
                    "hop": info.get("hop", 0),
                })

        # 孤立实体节点补齐（保持信息完整）
        for e in result.entities:
            add_entity_node(e["name"], e.get("score", 0.0), True)

        return {"nodes": nodes, "edges": edges, "mode": "ppr", "version": self.graph.version}
