#!/usr/bin/env python
"""doclevel 图（整文档 passage + deepseek 三元组）在 Green-ChunkHop 评测器上的 recall 评测。

对齐作者图 + hpopt full 配置（0.6171/0.7287/0.7821/0.8191/0.8538）：
  --mode dual_seed / fuzzy 0.8 / vsc 50 / tsc 10 / α 0.8 / hops 3 /
  bridge 32/50/0.65 / projected 1.0 / diffusion-restart 20 /
  dense_final_weight 0.03 + doclevel idf_triple overlay

环境变量:
  LOCATE_MODE=triple_sim|degree   entity_locate_score_mode
  PPR_ALPHA=0.8                   ppr alpha
  DENSE_W=0.03                    dense_final_weight（后融合）
  NO_OVERLAY=1                    不用 idf_triple overlay
  DUMP=/path.jsonl                逐查询 dump
  NO_PREBUILT=1                   关闭预建边消费（回退在线 CHUNK_HOP）

用法（chunkhop env）:
  cd /mnt/e/yf/worktrees/ablation_main/Green-ChunkHop/exp_weight_ablation
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python \
      /mnt/e/yf/knowledge_self_evolution/tools/run_hpopt_doclevel.py
"""
import os
import sys

EXP = os.environ.get("REPO", "/mnt/e/yf/worktrees/ablation_main/Green-ChunkHop/exp_weight_ablation")
DOC = os.environ.get("DOC", "/mnt/e/yf/knowledge_self_evolution/data/processed_musique_doclevel")
for p in (os.path.join(EXP, "src", "search_graph"), os.path.join(EXP, "src")):
    sys.path.insert(0, p)

import numpy as np


def patch_query_cache(cache_path=os.path.join(DOC, "query_cache.npz")):
    z = np.load(cache_path, allow_pickle=True)
    cache = dict(zip([str(q) for q in z["queries"]], z["vecs"].astype(np.float32)))
    import chunk_hop_retriever

    def _embed_query_from_cache(self, query, _cache=cache):
        v = _cache.get(str(query))
        if v is None:
            raise KeyError(f"query not in cache: {query[:80]}")
        return v.reshape(1, -1).astype(np.float32)

    chunk_hop_retriever.ChunkHopRetriever.embed_query = _embed_query_from_cache


patch_query_cache()

import chunk_hop_retriever  # noqa: E402

_oinit = chunk_hop_retriever.ChunkHopRetriever.__init__


def _pinit(self, *a, **kw):
    m = os.environ.get("LOCATE_MODE")
    if m:
        kw["entity_locate_score_mode"] = m
    return _oinit(self, *a, **kw)


chunk_hop_retriever.ChunkHopRetriever.__init__ = _pinit

import evaluate_with_timing as E  # noqa: E402

assert E.ChunkHopRetriever is chunk_hop_retriever.ChunkHopRetriever

args = [
    "--config", os.path.join(EXP, "config.yaml"),
    "--graph-path", os.path.join(DOC, "graph_export_with_chunk_hop" + os.environ.get("GRAPH_SUFFIX", "")),
    "--test-file", "/mnt/e/yf/knowledge_self_evolution/data/test/musique_dev.json",
    "--dataset", os.environ.get("DATASET", "musique_doclevel"),
    "--max-samples", os.environ.get("N", "1000"),
    "--mode", "dual_seed",
    "--match-mode", "fuzzy", "--fuzzy-threshold", "0.8",
    "--top-k", "50", "--triple-seed-count", os.environ.get("TSC", "10"), "--vector-seed-count", os.environ.get("VSC", "50"),
    "--ppr-alpha", os.environ.get("PPR_ALPHA", "0.8"),
    "--min-similarity-threshold", "0.3",
    "--k-list", "5,10,15,20,30",
    "--hops", os.environ.get("HOP_DEPTH", "3"),
    "--embedding-model-url", "http://localhost:8800/v1",
    "--bridge-entity-count", "32", "--bridge-max-degree", "50", "--bridge-hop-share", "0.65",
    "--diffusion-semantic-temperature", "0.05",
    "--projected-hop-strength", "1.0",
    "--diffusion-vector-restart-count", "20",
    "--diffusion-preserve-hop-budget",
    "--hop-weight-no-plus-one",
    "--dense-final-weight", os.environ.get("DENSE_W", "0.03"),
    "--dump-results", os.environ.get("DUMP", os.path.join(DOC, "dump_hpopt.jsonl")),
]
if not os.environ.get("NO_OVERLAY"):
    _adj = os.environ.get("DENSE_HOP_ADJ",
                          os.path.join(DOC, "_overlay/idf_triple.adjacency.pkl"))
    args += ["--dense-hop-adj", _adj,
             "--dense-hop-k", os.environ.get("DENSE_HOP_K", "20"),
             "--dense-hop-sim-min", os.environ.get("DENSE_HOP_SIM_MIN", "0.3")]

sys.argv = ["evaluate_with_timing.py"] + args
E.main()
