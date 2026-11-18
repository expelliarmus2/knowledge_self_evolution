#!/usr/bin/env python
"""重排 doclevel igraph 顶点：OriginalText 按 id 数值序（=corpus 顺序）在前，Entity 在后。

背景: build_igraph 用 sorted(nodes.keys()) 字符串排序，十进制 id 被打乱；
而 vdb(HNSW) 按 corpus 数组顺序嵌入，评测器用 passage_indices 把 vdb 索引映射到
图顶点 → 图乱序导致检索全部错位。

产物:
  graph_export_with_chunk_hop.pkl 覆盖为有序版（原文件备份 .sorted_orig.pkl）
  vertex_mapping.json: {old_index: new_index}

用法（chunkhop env）:
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/reorder_igraph.py
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from pathlib import Path

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
OUT = Path(os.environ.get("OUT", str(PROJ / "data" / "processed_musique_doclevel")))
PKL = OUT / "graph_export_with_chunk_hop.pkl"
BACKUP = OUT / "graph_export_with_chunk_hop.sorted_orig.pkl"


def main():
    import igraph as ig
    g = pickle.load(open(PKL, "rb"))
    print(f"loaded: {g.vcount()} nodes, {g.ecount()} edges")

    ids = [v["id"] for v in g.vs]
    types = [v["node_type"] for v in g.vs]

    # 验证当前是否已有序
    ot_ids = [int(ids[i]) for i, t in enumerate(types) if t == "OriginalText"]
    if ot_ids == sorted(ot_ids) and len(ot_ids) == len([t for t in types if t == "OriginalText"]):
        # 检查 OriginalText 是否都在最前
        first_ent = next((i for i, t in enumerate(types) if t == "Entity"), len(types))
        if all(t == "OriginalText" for t in types[:first_ent]):
            print("already ordered, skip")
            return

    # new_order[new] = old
    order = sorted(range(len(types)),
                   key=lambda i: (0 if types[i] == "OriginalText" else 1, int(ids[i])))
    old_to_new = {old: new for new, old in enumerate(order)}
    g2 = g.permute_vertices(order)

    shutil.copy2(PKL, BACKUP)
    with open(PKL, "wb") as f:
        pickle.dump(g2, f)
    (OUT / "vertex_mapping.json").write_text(
        json.dumps({"old_to_new": old_to_new}), encoding="utf-8")

    # 验证
    import json as _j
    corpus = _j.load(open(PROJ / "data/raw/corpus_musique.json", encoding="utf-8"))
    ok = sum(1 for i in range(200)
             if g2.vs[i]["doc_id"] == corpus[i]["title"])
    print(f"saved ordered pkl: {g2.vcount()} nodes, {g2.ecount()} edges; 前200顺序一致 {ok}/200")


if __name__ == "__main__":
    main()
