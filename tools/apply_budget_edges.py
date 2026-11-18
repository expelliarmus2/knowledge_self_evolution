#!/usr/bin/env python
"""把 doclevel 图的 chunk_hop 边替换为 Coverage10000 选边结果（v2，顶点顺序已修正），
生成 budget 版项目图。

顶点映射（重排后 igraph 顺序）:
  idx < 11656  ->  chunk "doc:{idx}"（= corpus 顺序）
  idx >= 11656 ->  entity 顶点（igraph text = 规范名 = 项目图节点名）

边类型:
  PP -> chunk_hop
  PE -> entity_projected_chunk
  EE -> bridge_entity_to_entity

产物: data/processed_musique_doclevel_b10k/graph.pkl
用法（chunkhop env）:
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/apply_budget_edges.py
"""
from __future__ import annotations

import pickle
from pathlib import Path

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
SRC = PROJ / "data" / "processed_musique_doclevel"
DST = PROJ / "data" / "processed_musique_doclevel_b10k"
N_PASSAGES = 11656


def main():
    import networkx as nx

    DST.mkdir(exist_ok=True)
    print("load graph.pkl ...")
    p = pickle.load(open(SRC / "graph.pkl", "rb"))
    g = p["g"]
    n0, e0 = g.number_of_nodes(), g.number_of_edges()
    print(f"loaded: {n0} nodes, {e0} edges")

    # 1) 删除全部 chunk_hop 边（425万）
    drop = [(u, v, k) for u, v, k, d in g.edges(keys=True, data=True)
            if d.get("relation") == "chunk_hop"]
    for u, v, k in drop:
        g.remove_edge(u, v, k)
    print(f"removed chunk_hop: {len(drop)}  (edges now {g.number_of_edges()})")

    # 2) igraph entity 顶点 text（11656..）
    ig = pickle.load(open(SRC / "graph_export_with_chunk_hop.pkl", "rb"))
    assert ig.vcount() == 93292

    # 3) 从 budget v2 邻接加边（10000 对）
    adj = pickle.load(open(SRC / "_overlay/budget10000v2.adjacency.pkl", "rb"))
    added = {"chunk_hop": 0, "entity_projected_chunk": 0, "bridge_entity_to_entity": 0}
    skipped = 0
    for u, v, w, shared_count, entities in adj:
        def node(idx):
            if idx < N_PASSAGES:
                return f"doc:{idx}", "chunk"
            name = ig.vs[idx]["text"]
            return name, "entity"

        nu, tu = node(u)
        nv, tv = node(v)
        if not g.has_node(nu) or not g.has_node(nv):
            skipped += 1
            continue
        if tu == "chunk" and tv == "chunk":
            rel = "chunk_hop"
        elif tu == "entity" and tv == "entity":
            rel = "bridge_entity_to_entity"
        else:
            rel = "entity_projected_chunk"
        data = {"relation": rel, "weight": float(w),
                "shared_entity_count": float(shared_count),
                "shared_entities": entities}
        g.add_edge(nu, nv, **data)
        g.add_edge(nv, nu, **data)
        added[rel] += 2
    print(f"added: {added}, skipped: {skipped}")

    # 4) 保存
    p["g"] = g
    out = DST / "graph.pkl"
    with open(out, "wb") as f:
        pickle.dump(p, f)
    print(f"saved {out} ({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)")

    # 5) 软链向量库（复用 doclevel 三套项目格式向量库）
    for name in ("vdb_chunks.faiss", "vdb_chunks.meta.json", "vdb_triples.faiss",
                 "vdb_triples.meta.json", "vdb_entities.faiss", "vdb_entities.meta.json",
                 "entity_vecs.npy", "entity_names.txt"):
        tgt = DST / name
        if not tgt.exists():
            tgt.symlink_to(SRC / name)
    print("symlinked vdb files")


if __name__ == "__main__":
    main()
