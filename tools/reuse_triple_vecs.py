#!/usr/bin/env python
"""按文本复用旧 triple 向量，只对新增 triple 重新编码（进程内 BGE）。

用法:
  OUT=data/h2official /mnt/e/yf/models/.venv-bge/bin/python tools/reuse_triple_vecs.py
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("OUT", "data/h2official"))
OLD_MAP = Path(os.environ.get("OLD_MAP", str(OUT / "triple_mapping_m1.json")))
NEW_MAP = Path(os.environ.get("NEW_MAP", str(OUT / "triple_mapping.json")))
OLD_VEC = Path(os.environ.get("OLD_VEC", str(OUT / "triple_vecs.npy")))
NEW_VEC = Path(os.environ.get("NEW_VEC", str(OUT / "triple_vecs_ng2.npy")))
MODEL = os.environ.get("MODEL", "/mnt/e/yf/models/bge-m3")
BS = int(os.environ.get("BS", "64"))


def ttext(t):
    return f"{t['head']}  {t['relation']}  {t['tail']}"


def main():
    old_map = json.load(open(OLD_MAP, encoding="utf-8"))
    old_vec = np.load(OLD_VEC)
    assert len(old_map) == old_vec.shape[0], (len(old_map), old_vec.shape)
    cache = {}
    for t, v in zip(old_map, old_vec):
        cache[ttext(t)] = v
    print(f"旧向量缓存: {len(cache)} 条, dim={old_vec.shape[1]}", flush=True)

    new_map = json.load(open(NEW_MAP, encoding="utf-8"))
    texts = [ttext(t) for t in new_map]
    print(f"新 mapping: {len(texts)} 条", flush=True)

    hit = 0
    miss_idx = []
    for i, tx in enumerate(texts):
        if tx in cache:
            hit += 1
        else:
            miss_idx.append(i)
    print(f"命中复用 {hit} / 需新编码 {len(miss_idx)}", flush=True)

    dim = old_vec.shape[1]
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, tx in enumerate(texts):
        v = cache.get(tx)
        if v is not None:
            out[i] = v

    if miss_idx:
        import torch
        from transformers import AutoModel, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
        print(f"模型加载: {MODEL} on {dev}", flush=True)

        def encode(batch):
            enc = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            with torch.no_grad():
                o = model(**enc).last_hidden_state[:, 0]
            return torch.nn.functional.normalize(o.float(), dim=-1).cpu().numpy()

        t0 = time.time()
        for s in range(0, len(miss_idx), BS):
            idx = miss_idx[s:s + BS]
            batch = [texts[i] for i in idx]
            out[idx] = encode(batch)
            if (s // BS) % 10 == 0:
                print(f"  new {s}/{len(miss_idx)} ({time.time()-t0:.0f}s)", flush=True)
        print(f"新编码完成 ({time.time()-t0:.0f}s)", flush=True)

    zero = int((np.abs(out).sum(axis=1) == 0).sum())
    print(f"triples: {out.shape} zero={zero}", flush=True)
    np.save(NEW_VEC, out)
    print(f"saved -> {NEW_VEC}", flush=True)


if __name__ == "__main__":
    main()
