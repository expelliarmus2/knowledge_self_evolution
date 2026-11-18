#!/usr/bin/env python
"""嵌入 doclevel 图的实体名（.venv-bge 进程内）。

产物: data/processed_musique_doclevel/entity_names.txt + entity_vecs.npy
用法（.venv-bge env）:
  /mnt/e/yf/models/.venv-bge/bin/python tools/embed_entities_doclevel.py
"""
from __future__ import annotations

import time
from pathlib import Path

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
OUT = PROJ / "data" / "processed_musique_doclevel"
MODEL_DIR = "/mnt/e/yf/models/bge-m3"
BS = 128


def main():
    import torch
    import numpy as np
    from transformers import AutoModel, AutoTokenizer

    # 从 graph.pkl 提取实体名（保证顺序稳定）
    import pickle
    with open(OUT / "graph.pkl", "rb") as f:
        payload = pickle.load(f)
    names = [n for n, d in payload["g"].nodes(data=True) if d.get("type") == "entity"]
    (OUT / "entity_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    print(f"entities: {len(names)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(device).eval()
    if device == "cuda":
        model = model.half()
    print(f"model on {device}")

    vecs = []
    t0 = time.time()
    for i in range(0, len(names), BS):
        batch = names[i:i + BS]
        with torch.no_grad():
            inp = tok(batch, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda"):
                out = model(**inp).last_hidden_state
            v = out[:, 0].float().cpu().numpy()
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            vecs.append(v)
        if (i // BS) % 200 == 0:
            print(f"  {i}/{len(names)} ({time.time()-t0:.0f}s)", flush=True)
    mat = np.concatenate(vecs, axis=0).astype(np.float32)
    np.save(OUT / "entity_vecs.npy", mat)
    print(f"entity_vecs.npy {mat.shape} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
