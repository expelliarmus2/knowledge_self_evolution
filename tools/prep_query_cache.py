#!/usr/bin/env python
"""doclevel 评测查询向量 cache（bge-m3 进程内，.venv-bge 环境）。

产物: data/processed_musique_doclevel/query_cache.npz
用法:
  /mnt/e/yf/models/.venv-bge/bin/python tools/prep_query_cache.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
OUT = PROJ / "data" / "processed_musique_doclevel"
TEST = PROJ / "data" / "test" / "musique_dev.json"
MODEL_DIR = "/mnt/e/yf/models/bge-m3"
BS = 64


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    questions = [str(q["question"]) for q in json.load(open(TEST, encoding="utf-8"))]
    print(f"queries: {len(questions)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device).eval()
    if device == "cuda":
        model = model.half()
    print(f"model on {device}")

    vecs = []
    t0 = time.time()
    for i in range(0, len(questions), BS):
        batch = questions[i:i + BS]
        with torch.no_grad():
            inp = tok(batch, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda"):
                out = model(**inp).last_hidden_state
            v = out[:, 0].float().cpu().numpy()
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            vecs.append(v)
        if (i // BS) % 100 == 0:
            print(f"  {i}/{len(questions)} ({time.time()-t0:.0f}s)", flush=True)
    mat = np.concatenate(vecs, axis=0).astype(np.float32)
    np.savez(OUT / "query_cache.npz", queries=questions, vecs=mat)
    print(f"query_cache.npz {mat.shape} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
