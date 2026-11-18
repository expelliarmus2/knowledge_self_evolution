"""在 h2official(t) 图上重生成 idf_triple overlay 邻接（复用 ablation 脚本逻辑）。"""
import importlib.util
from pathlib import Path

SRC = "/mnt/e/yf/worktrees/ablation/scripts/build_rare_triple_edges.py"
GRAPH = "/mnt/e/yf/knowledge_self_evolution/data/h2official/graph_export_with_chunk_hop_t.pkl"
OUT = Path("/mnt/e/yf/knowledge_self_evolution/data/h2official/_overlay_t")
RUNTIME = Path("/mnt/e/yf/knowledge_self_evolution/data/h2official/_overlay_t_runtime")

spec = importlib.util.spec_from_file_location("brte", SRC)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUT = OUT
m.RUNTIME = RUNTIME
m.main(GRAPH)
print("done")
