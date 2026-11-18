#!/usr/bin/env python
"""为 doclevel 图建 faiss 向量库 + mapping（作者 vdb 格式对齐）。

产物（data/processed_musique_doclevel/）:
  vdb_passages_hnsw.faiss + passage_mapping.json
  vdb_triples_hnsw.faiss   + triple_mapping.json

用法（chunkhop env，有 faiss）:
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/build_faiss_doclevel.py
"""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
OUT = Path(__import__("os").environ.get("OUT", str(PROJ / "data" / "processed_musique_doclevel")))
EMBED_DIM = 1024


def main():
    # passages
    pv = np.load(OUT / "passage_vecs.npy")
    corpus = json.load(open(PROJ / "data/raw/corpus_musique.json", encoding="utf-8"))
    assert len(pv) == len(corpus), (len(pv), len(corpus))
    pindex = faiss.IndexHNSWFlat(EMBED_DIM, 32, faiss.METRIC_INNER_PRODUCT)
    pindex.add(pv)
    faiss.write_index(pindex, str(OUT / "vdb_passages_hnsw.faiss"))
    with open(OUT / "passage_mapping.json", "w", encoding="utf-8") as f:
        json.dump([
            {"index": i, "text": corpus[i]["title"] + "\n" + corpus[i]["text"],
             "title": corpus[i]["title"], "node_id": str(i)}
            for i in range(len(corpus))
        ], f, ensure_ascii=False)
    print(f"passage index: {pindex.ntotal}")

    # triples
    tv = np.load(OUT / "triple_vecs.npy")
    mapping = json.load(open(OUT / "triple_mapping_raw.json", encoding="utf-8"))
    assert len(tv) == len(mapping), (len(tv), len(mapping))
    tindex = faiss.IndexHNSWFlat(EMBED_DIM, 32, faiss.METRIC_INNER_PRODUCT)
    tindex.add(tv)
    faiss.write_index(tindex, str(OUT / "vdb_triples_hnsw.faiss"))
    with open(OUT / "triple_mapping.json", "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)
    print(f"triple index: {tindex.ntotal}")


if __name__ == "__main__":
    main()
