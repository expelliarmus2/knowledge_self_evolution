"""用官方 ablation/main 的 VectorBuilder 建 vdb，但限制嵌入并发（8800 HTTP 服务 32 并发会崩）。

用法:
  cd /mnt/e/yf/worktrees/kg_ablation_main/Green-ChunkHop
  python /mnt/e/yf/knowledge_self_evolution/tools/run_vector_builder_official.py \
      --graph-path <graph_export_with_chunk_hop> --output-dir <dir> --max-workers 8
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--graph-path", required=True, help="不带 .pkl 后缀")
ap.add_argument("--output-dir", required=True)
ap.add_argument("--max-workers", type=int, default=8)
ap.add_argument("--batch-size", type=int, default=32)
ap.add_argument("--repo", default="/mnt/e/yf/worktrees/kg_ablation_main/Green-ChunkHop")
args = ap.parse_args()

sys.path.insert(0, os.path.join(args.repo, "src", "search_graph"))
sys.path.insert(0, os.path.join(args.repo, "src"))

from vector_builder import VectorBuilder          # noqa: E402
from config_loader import load_config, get_embedding_model  # noqa: E402

cfg = load_config(None)
model_name = cfg["embedding_model_name"]
model_url = os.environ.get("EMBEDDING_MODEL_URL", cfg["embedding_model_url"])
dim = cfg["embedding_dim"]
print(f"embedding: {model_name} @ {model_url}, dim={dim}", flush=True)

embedding_model, dim = get_embedding_model({**cfg, "embedding_model_url": model_url})

builder = VectorBuilder(
    graph_path=args.graph_path,
    embedding_model=embedding_model,
    embedding_dim=dim,
    output_dir=args.output_dir,
    batch_size=args.batch_size,
)

_orig_embed = builder._embed_texts


def _embed_texts(texts, max_workers=None):
    return _orig_embed(texts, max_workers=args.max_workers)


builder._embed_texts = _embed_texts
builder.build_all()
