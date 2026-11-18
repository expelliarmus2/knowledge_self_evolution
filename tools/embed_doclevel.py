#!/usr/bin/env python
"""嵌入 doclevel 图的 passage / triple（.venv-bge 进程内，绕 HTTP 劣化）。

产物（data/processed_musique_doclevel/）:
  passage_vecs.npy   (11656, 1024) L2 归一化 float32
  triple_vecs.npy    (99247, 1024) L2 归一化 float32

用法（.venv-bge env，torch + GPU）:
  /mnt/e/yf/models/.venv-bge/bin/python tools/embed_doclevel.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
OUT = Path(__import__("os").environ.get("OUT", str(PROJ / "data" / "processed_musique_doclevel")))
MODEL_DIR = "/mnt/e/yf/models/bge-m3"
BS = 64


def main():
    from transformers import AutoModel, AutoTokenizer
    import torch
    import numpy as np

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device).eval()
    if device == "cuda":
        model = model.half()
    print(f"model loaded on {device}")

    def embed_file(name, out_name, tag):
        texts = (OUT / name).read_text(encoding="utf-8").splitlines()
        vecs = []
        t0 = time.time()
        for i in range(0, len(texts), BS):
            batch = texts[i:i + BS]
            with torch.no_grad():
                inp = tok(batch, padding=True, truncation=True, max_length=1024,
                          return_tensors="pt").to(device)
                with torch.autocast(device_type="cuda"):
                    out = model(**inp).last_hidden_state
                # CLS pooling（bge-m3 无 pooler，用 CLS 向量）
                v = out[:, 0].float().cpu().numpy()
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
                vecs.append(v)
            if (i // BS) % 200 == 0:
                print(f"  {tag}: {i}/{len(texts)} ({time.time()-t0:.0f}s)", flush=True)
        mat = np.concatenate(vecs, axis=0).astype(np.float32)
        np.save(OUT / out_name, mat)
        print(f"{tag}: {mat.shape} -> {out_name} ({time.time()-t0:.0f}s)")

    embed_file("passage_texts.txt", "passage_vecs.npy", "passages")
    embed_file("triple_texts.txt", "triple_vecs.npy", "triples")


if __name__ == "__main__":
    main()
