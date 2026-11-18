"""失败归因探针：对 1000 题做检索结果分解。

对每题检索 top_k=30 一次，把每篇金标文档分类为：
  A. top30      —— 进入最终 Top-30（命中）
  B. pool       —— 进入候选池（种子/hop 可达，PPR 有分）但排名在 30 之后（排序问题）
  C. not_reach  —— 从未进入候选池（连通性/抽取问题）

并对 not_reach 做二级分解：
  C1. uncapped BFS 可达：2 跳内共享实体可达（不受 locate/hop 宽度截断影响）→ 截断/上限问题
  C2. 完全不可达 → 抽取/种子问题（金标 chunk 与查询种子在图上无 2 跳内联系）

用法：
  python -m evaluation.probe_decomp --processed data/processed_musique_onestep --limit 1000
"""
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.knowledge_graph import KnowledgeGraph, REL_TEXT_CONCLUDE  # noqa: E402
from openie.embedder import build_embedder  # noqa: E402
from openie.llm_client import build_llm  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.prompts import PromptManager  # noqa: E402
from vectorstore.manager import VectorStoreManager  # noqa: E402


def build_retriever(processed_dir: Path, cfg: Config):
    graph = KnowledgeGraph.load(processed_dir / "graph.pkl")
    vdb = VectorStoreManager(cfg, build_embedder(cfg), processed_dir)
    llm = build_llm(cfg)
    prompts = PromptManager.from_config(cfg)
    return Retriever(cfg, graph, vdb, build_embedder(cfg), llm, prompts, graph_path=processed_dir / "graph.pkl"), graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data/processed_musique_onestep")
    ap.add_argument("--test", type=str, default="data/test/musique_dev.json")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = Config.load()
    retriever, graph = build_retriever(Path(args.processed), cfg)
    questions = json.load(open(args.test, encoding="utf-8"))[: args.limit]

    # doc_id → chunk ids
    doc2chunks: dict = {}
    for nid, nd in graph.g.nodes(data=True):
        if nd.get("type") == "chunk":
            doc2chunks.setdefault(nd.get("doc_id", ""), []).append(nid)
    # chunk_hop 邻接（uncapped BFS 用）：共享实体 ≥1 即连边；预计算 实体→chunk 索引加速
    chunk_adj: dict = {}
    ent2chunks: dict = {}
    for nid, nd in graph.g.nodes(data=True):
        if nd.get("type") == "chunk":
            ents = set(graph.get_chunk_entities(nid))
            chunk_adj[nid] = ents
            for e in ents:
                ent2chunks.setdefault(e, set()).add(nid)

    def neighbors(cid):
        out = set()
        for e in chunk_adj.get(cid, ()):
            out |= ent2chunks.get(e, ())
        return out - {cid}

    def bfs_reachable(seed_chunk_ids, max_depth=2):
        """uncapped BFS：chunk 间共享 ≥1 实体即连边（与 hop_expander 同规则，但不截断宽度）"""
        reached = set(seed_chunk_ids)
        frontier = set(seed_chunk_ids)
        for _ in range(max_depth):
            nxt = set()
            for cid in frontier:
                nxt |= neighbors(cid)
            nxt -= reached
            reached |= nxt
            frontier = nxt
            if not frontier:
                break
        return reached

    stats = Counter()
    gold_dist = Counter()  # 金标文档 按分类分布
    below_ranks = []
    unreach_samples = []
    n_q = 0

    def search_one(q):
        return q, retriever.search(q["question"], top_k=30)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(search_one, q): q for q in questions}
        for fut in as_completed(futs):
            q, res = fut.result()
            n_q += 1
            top_ids = {c["chunk_id"] for c in res.chunks}
            top_docs = {c["doc_id"] for c in res.chunks}
            pool_rank = dict(res._full_rank)
            pool_docs = {graph.g.nodes[cid].get("doc_id", "") for cid, sc in res._full_rank if graph.g.has_node(cid)}
            gold = {p["title"] for p in q["paragraphs"] if p.get("is_supporting")}
            if not gold:
                gold = {p["title"] for p in q["paragraphs"]}
            gold_in_graph = [g for g in gold if doc2chunks.get(g)]
            n_reach = 0
            for g in gold_in_graph:
                gc_ids = doc2chunks[g]
                if g in top_docs:
                    gold_dist["top30"] += 1
                    continue
                in_pool = any(cid in pool_rank for cid in gc_ids)
                if in_pool:
                    gold_dist["pool_below30"] += 1
                    r = min(pool_rank[cid] for cid in gc_ids if cid in pool_rank)
                    below_ranks.append(r)
                    continue
                gold_dist["not_reach"] += 1
                n_reach += 1
            if n_reach:
                unreach_samples.append((q.get("idx", q["id"]), q["question"], [g for g in gold_in_graph
                                                                 if not any(cid in pool_rank for cid in doc2chunks[g])][:3],
                                        res._seed_chunks, res._hop_chunks))
            stats["queries"] += 1
            if n_q % 100 == 0:
                sys.stdout.write(f"\r{n_q}/{len(questions)} ...")
                sys.stdout.flush()
    print(f"\r分解完成 {n_q} 题, 耗时 {time.time() - t0:.1f}s")

    total = sum(gold_dist.values())
    print("\n=== 金标文档分类（按文档计数）===")
    for k in ("top30", "pool_below30", "not_reach"):
        c = gold_dist[k]
        print(f"  {k:14s}: {c:6d} ({c / max(1, total) * 100:.1f}%)")
    print(f"  total gold docs in graph: {total}")
    if below_ranks:
        below_ranks.sort()
        import statistics
        print(f"\npool_below30 的最低排名(排在最前的金标chunk): median={statistics.median(below_ranks):.0f} "
              f"p10={below_ranks[len(below_ranks)//10]:.0f} p25={below_ranks[len(below_ranks)//4]:.0f} "
              f"p75={below_ranks[3*len(below_ranks)//4]:.0f} max={below_ranks[-1]}")

    # not_reach 二级分解（抽样 40 题，uncapped BFS）
    print("\n=== not_reach 二级分解（抽样 40 题）===")
    sub = Counter(); cap_limited = 0; n_checked = 0
    for idx, question, golds, seed_chunks, hop_chunks in unreach_samples[:40]:
        seed_ids = set(seed_chunks.keys())
        if not seed_ids:
            sub["no_seed_chunks"] += 1
            continue
        reached = bfs_reachable(seed_ids, max_depth=2)
        for g in golds:
            gc_ids = doc2chunks[g]
            if any(c in reached for c in gc_ids):
                sub["uncapped_reachable(截断问题)"] += 1
            else:
                sub["genuine_disconnect(抽取/种子问题)"] += 1
            n_checked += 1
    for k, v in sub.items():
        print(f"  {k:42s}: {v}  ({v / max(1, n_checked) * 100:.0f}%)")

    print("\n=== not_reach 样例（3 条）===")
    for idx, question, golds, seed_chunks, hop_chunks in unreach_samples[:3]:
        print(f"  Q[{idx}]: {question[:90]}")
        print(f"     未达金标: {golds[:3]}")
        print(f"     种子chunk({len(seed_chunks)}): {list(seed_chunks.items())[:4]}")
        print(f"     hop chunks({len(hop_chunks)}): {list(hop_chunks.items())[:3]}")


if __name__ == "__main__":
    main()
