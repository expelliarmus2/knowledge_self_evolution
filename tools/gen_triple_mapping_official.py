#!/usr/bin/env python
"""从新图重生成 triple_mapping.json（head_id/tail_id 用新图顶点 id）。

官方 vector_builder 的 triple_mapping 字段:
  {"head": 实体名, "head_id": 图顶点 id, "head_type": node_type,
   "relation": 关系, "tail": ..., "tail_id": ..., "tail_type": ...}

用法:
  OUT=data/h2official /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/gen_triple_mapping_official.py
"""
import json
import os
from pathlib import Path

import igraph as ig

OUT = Path(os.environ.get("OUT", "data/h2official"))
GRAPH = os.environ.get("GRAPH",
                       str(OUT / "graph_export_with_chunk_hop_m1.pkl"))
SKIP_RELS = {"CHUNK_HOP", "text_conclude"}


def main():
    g = ig.Graph.Read_Pickle(GRAPH) if GRAPH.endswith(".pkl") else ig.Graph.Read_GraphML(GRAPH)
    def node_text(v):
        return v["text"] if ("text" in v.attributes() and v["text"]) else v["id"]

    rows = []
    for e in g.es:
        rel = e["relation"]
        if rel in SKIP_RELS:
            continue
        s = g.vs[e.source]
        t = g.vs[e.target]
        rows.append({
            "head": node_text(s),
            "head_id": s["id"],
            "head_type": s["node_type"],
            "relation": rel,
            "tail": node_text(t),
            "tail_id": t["id"],
            "tail_type": t["node_type"],
        })
    with open(OUT / "triple_mapping.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"triple_mapping.json: {len(rows)} triples (graph {g.vcount()}v/{g.ecount()}e)", flush=True)


if __name__ == "__main__":
    main()
