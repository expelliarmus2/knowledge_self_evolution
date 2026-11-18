"""从 npy 向量建官方格式 faiss 索引（等价 vector_builder.build_passages/triples_vectors）。

用法:
  OUT=data/h2official /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/build_faiss_official.py
"""
import json
import os
from pathlib import Path

import faiss
import numpy as np

OUT = Path(os.environ.get("OUT", "data/h2official"))
DIM = 1024


def build(name, vecs, nvecs):
    flat = faiss.IndexFlatIP(DIM)
    flat.add(vecs)
    faiss.write_index(flat, str(OUT / f"vdb_{name}.faiss"))
    hnsw = faiss.IndexHNSWFlat(DIM, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(vecs)
    hnsw.hnsw.efSearch = 256
    faiss.write_index(hnsw, str(OUT / f"vdb_{name}_hnsw.faiss"))
    print(f"{name}: flat ntotal={flat.ntotal} / hnsw ntotal={hnsw.ntotal}")


def main():
    pv = np.load(OUT / "passage_vecs.npy")
    assert pv.shape == (len(json.load(open(OUT / "passage_mapping.json", encoding="utf-8"))), DIM)
    build("passages", pv, None)

    tv = np.load(OUT / "triple_vecs.npy")
    assert tv.shape == (len(json.load(open(OUT / "triple_mapping.json", encoding="utf-8"))), DIM)
    build("triples", tv, None)


if __name__ == "__main__":
    main()
