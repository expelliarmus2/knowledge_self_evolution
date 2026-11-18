#!/usr/bin/env python
"""给 h2prompt 图构建 Coverage10000 的 chunk_hop 预建边（复用 ablation 的 coverage 选边）。

流程:
  1. 从 h2prompt graph.pkl 的 text_conclude 边统计实体→chunk 归属
  2. 倒排计数共享实体 >= MIN_SHARED 的 chunk 对（候选）
  3. 复用 build_budget_chunkhop.select_budget 做 coverage 选边 10000
  4. 把选中对写回图（relation=chunk_hop，双向），另存 graph_budget.pkl

用法（chunkhop env）:
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/build_h2_budget_edges.py
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/mnt/e/yf/worktrees/ablation/scripts")
from build_budget_chunkhop import select_budget  # noqa: E402

MIN_SHARED = int(__import__("os").environ.get("MIN_SHARED", "5"))
BUDGET = 10000

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
SRC = PROJ / "data" / "processed_musique_h2prompt"
DST = PROJ / "data" / f"processed_musique_h2prompt_b10k_ms{MIN_SHARED}"


def main():
    DST.mkdir(exist_ok=True)
    p = pickle.load(open(SRC / "graph.pkl", "rb"))
    g = p["g"]
    print(f"loaded {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # 1) 实体 → chunk 倒排（text_conclude 边）
    ent_to_chunks: dict[str, set] = defaultdict(set)
    for u, v, d in g.edges(data=True):
        if d.get("relation") != "text_conclude":
            continue
        if g.nodes[u].get("type") == "entity":
            ent, chk = u, v
        else:
            ent, chk = v, u
        if g.nodes[chk].get("type") == "chunk":
            ent_to_chunks[ent].add(chk)

    # 2) 候选对（共享实体 >= 5）
    chunk_to_ents: dict[str, set] = defaultdict(set)
    for ent, chunks in ent_to_chunks.items():
        for c in chunks:
            chunk_to_ents[c].add(ent)
    chunks = sorted(chunk_to_ents.keys())
    pair_cnt: dict[tuple, int] = defaultdict(int)
    for ent, cs in ent_to_chunks.items():
        cs = list(cs)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                a, b = (cs[i], cs[j]) if cs[i] < cs[j] else (cs[j], cs[i])
                pair_cnt[(a, b)] += 1
    print(f"候选对(shared>=1): {len(pair_cnt)}")

    records = []
    id2name = {c: i for i, c in enumerate(chunks)}
    for (a, b), cnt in pair_cnt.items():
        if cnt < MIN_SHARED:
            continue
        shared = sorted(chunk_to_ents[a] & chunk_to_ents[b])
        w = len(shared) / 100.0
        ia, ib = id2name[a], id2name[b]
        records.append((ia, ib, w, float(len(shared)), json.dumps(shared[:100])))
    print(f"候选对(shared>={MIN_SHARED}): {len(records)}")

    # 3) coverage 选边（int 索引）→ 映射回 chunk 名
    name2id = {c: i for i, c in enumerate(chunks)}
    id2name_rt = {i: c for c, i in name2id.items()}
    chosen = select_budget(records, BUDGET, "coverage")
    print(f"选中: {len(chosen)} 对")
    uniq = set()
    for u, v, *_ in chosen:
        uniq.add(u); uniq.add(v)
    print(f"覆盖 chunk: {len(uniq)} / {len(chunks)}")
    chosen_named = [(id2name_rt[u], id2name_rt[v], w, sc, ents)
                    for u, v, w, sc, ents in chosen]

    # 4) 写回图
    for u, v, w, sc, ents in chosen_named:
        data = {"relation": "chunk_hop", "weight": float(w),
                "shared_entity_count": float(sc), "shared_entities": ents}
        g.add_edge(u, v, **data)
        g.add_edge(v, u, **data)
    print(f"写回后 edges: {g.number_of_edges()}")

    p["g"] = g
    with open(DST / "graph.pkl", "wb") as f:
        pickle.dump(p, f)
    print(f"saved {DST/'graph.pkl'}")

    # 5) 软链向量库 + query cache
    for name in ("vdb_chunks.faiss", "vdb_chunks.meta.json", "vdb_triples.faiss",
                 "vdb_triples.meta.json", "vdb_entities.faiss", "vdb_entities.meta.json",
                 "query_cache.npz"):
        tgt = DST / name
        if not tgt.exists():
            tgt.symlink_to(SRC / name)
    print("symlinked")


if __name__ == "__main__":
    main()
