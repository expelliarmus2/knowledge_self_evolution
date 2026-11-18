"""按官方 vector_builder 的文本/顺序做进程内编码（绕过不稳定的 8800 HTTP 服务）。

产物等价：vdb_passages.faiss / vdb_passages_hnsw.faiss / vdb_triples.faiss / vdb_triples_hnsw.faiss
（FlatIP + HNSW M=32 efC=200 efS=256），mapping 沿用官方已生成的文件。

用法:
  OUT=data/h2official /mnt/e/yf/models/.venv-bge/bin/python tools/embed_official_vectors.py
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = "/mnt/e/yf/models/bge-m3"
OUT = Path(os.environ.get("OUT", "data/h2official"))
BS = 64


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device).eval()
    if device == "cuda":
        model = model.half()

    # passages
    pm = json.load(open(OUT / "passage_mapping.json", encoding="utf-8"))
    passages = [p["text"] for p in pm]
    print(f"passages: {len(passages)}", flush=True)

    # triples（重建官方 triple_text = "{head}  {relation}  {tail}"）
    tm = json.load(open(OUT / "triple_mapping.json", encoding="utf-8"))
    triples = [f"{t['head']}  {t['relation']}  {t['tail']}" for t in tm]
    print(f"triples: {len(triples)}", flush=True)

    def encode(texts, tag):
        vecs = []
        t0 = time.time()
        for i in range(0, len(texts), BS):
            batch = texts[i:i + BS]
            with torch.no_grad():
                inp = tok(batch, padding=True, truncation=True, max_length=int(os.environ.get("MAXL", "1024")),
                          return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda"):
                    out = model(**inp).last_hidden_state
                v = out[:, 0].float().cpu().numpy()
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                vecs.append(v)
            if (i // BS) % 200 == 0:
                print(f"  {tag}: {i}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
        return np.concatenate(vecs, axis=0).astype(np.float32)

    pv = encode(passages, "passages")
    np.save(OUT / "passage_vecs.npy", pv)
    print(f"passages: {pv.shape} -> passage_vecs.npy", flush=True)

    if os.environ.get("SKIP_TRIPLES"):
        return

    tv = encode(triples, "triples")
    np.save(OUT / "triple_vecs.npy", tv)
    print(f"triples: {tv.shape} -> triple_vecs.npy", flush=True)


if __name__ == "__main__":
    main()
