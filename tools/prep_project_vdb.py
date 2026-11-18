#!/usr/bin/env python
"""为 doclevel 图生成项目 vectorstore 格式（IndexIDMap2 + meta.json）。

产物（data/processed_musique_doclevel/）:
  vdb_chunks.faiss/.meta.json  （key="doc:{pid}", meta={doc_id,title}）
  vdb_triples.faiss/.meta.json （key="h\\tR\\tt"）
  vdb_entities.faiss/.meta.json（key=实体名）

用法（chunkhop env，有 faiss）:
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/prep_project_vdb.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
OUT = PROJ / "data" / "processed_musique_doclevel"
DIM = 1024


def build_store(name, keys, metas, vecs):
    idx = faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))
    ids = np.arange(len(vecs), dtype="int64")
    idx.add_with_ids(np.asarray(vecs, dtype=np.float32), ids)
    faiss.write_index(idx, str(OUT / f"vdb_{name}.faiss"))
    state = {
        "key2id": {k: int(i) for i, k in enumerate(keys)},
        "id2key": {str(i): k for i, k in enumerate(keys)},
        "metas": metas,
        "next_id": len(keys),
        "dim": DIM,
    }
    (OUT / f"vdb_{name}.meta.json").write_text(
        json.dumps(state, ensure_ascii=False), encoding="utf-8")
    print(f"vdb_{name}: {len(keys)}")


def main():
    # chunks
    corpus = json.load(open(PROJ / "data/raw/corpus_musique.json", encoding="utf-8"))
    pv = np.load(OUT / "passage_vecs.npy")
    assert len(pv) == len(corpus)
    ckeys = [f"doc:{i}" for i in range(len(corpus))]
    cmetas = {f"doc:{i}": {"doc_id": corpus[i]["title"], "title": corpus[i]["title"],
                           "order": 0} for i in range(len(corpus))}
    build_store("chunks", ckeys, cmetas, pv)

    # triples
    tv = np.load(OUT / "triple_vecs.npy")
    mapping = json.load(open(OUT / "triple_mapping_raw.json", encoding="utf-8"))
    assert len(tv) == len(mapping)
    tkeys = [f"{m['head']}\t{m['relation']}\t{m['tail']}" for m in mapping]
    tmetas = {k: {"head": m["head"], "relation": m["relation"], "tail": m["tail"],
                  "weight": 1.0} for k, m in zip(tkeys, mapping)}
    build_store("triples", tkeys, tmetas, tv)

    # entities（等待 entity_vecs.npy 生成后运行）
    ev = np.load(OUT / "entity_vecs.npy")
    names = (OUT / "entity_names.txt").read_text(encoding="utf-8").splitlines()
    assert len(ev) == len(names), (len(ev), len(names))
    with open(OUT / "graph.pkl", "rb") as f:
        payload = pickle.load(f)
    g = payload["g"]
    emetas = {}
    for n in names:
        d = g.nodes[n]
        emetas[n] = {"display": d.get("display", n), "entity_type": d.get("entity_type", "unknown"),
                     "freq": d.get("freq", 0), "chunk_ids": d.get("chunk_ids", [])}
    build_store("entities", names, emetas, ev)


if __name__ == "__main__":
    main()
