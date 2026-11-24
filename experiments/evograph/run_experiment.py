"""EvoGraph 实验编排：baseline（无自进化）vs evolved（自进化）。

用法：
    python -m experiments.evograph.run_experiment --limit 8 --mode mock     # 冒烟
    python -m experiments.evograph.run_experiment --mode mock                # 全量 mock
    python -m experiments.evograph.run_experiment --mode real --processed data/processed_musique
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.config import Config  # noqa: E402

ART = Path("experiments/evograph/artifacts")
DEV = Path("data/test/musique_dev.json")


def _qkey(q: dict) -> str:
    return q["id"]


def _decomposition_text(q: dict) -> str:
    parts = []
    for s in q.get("question_decomposition") or []:
        parts.append(f"{s.get('question', '')} -> {s.get('answer', '')}")
    return " ; ".join(parts)


def _title_overlap(split, evolve_ids, eval_ids) -> dict:
    qmap = {q["id"]: q for q in split.questions}
    def titles(ids):
        return {p.get("title", "") for i in ids for p in qmap[i]["paragraphs"]}
    te, tv = titles(evolve_ids), titles(eval_ids)
    return {
        "evolve_titles": len(te), "eval_titles": len(tv),
        "shared_titles": len(te & tv),
        "eval_covered_ratio": round(len(te & tv) / max(1, len(tv)), 4),
    }


def _build_retriever(cfg: Config, processed: Path):
    from graph.knowledge_graph import KnowledgeGraph
    from openie.embedder import build_embedder
    from openie.llm_client import build_llm
    from retrieval.retriever import Retriever
    from utils.prompts import PromptManager
    from vectorstore.manager import VectorStoreManager

    graph_path = processed / "graph.pkl"
    graph = KnowledgeGraph.load(graph_path)
    embedder = build_embedder(cfg)
    llm = build_llm(cfg)
    vdb = VectorStoreManager(cfg, embedder, processed)
    prompts = PromptManager.from_config(cfg)
    retriever = Retriever(cfg, graph, vdb, embedder, llm, prompts, graph_path=graph_path)
    return graph, retriever, llm


def _run_eval(retriever, graph, questions: list, ev, top_k: int):
    """在给定题集上检索，产出评测 records。"""
    from experiments.evograph.evaluator import rank_chunks
    records = []
    for q in questions:
        try:
            res = retriever.search(q["question"], top_k=top_k)
        except Exception as exc:  # 单题失败不中断（记录原因）
            records.append({"qid": q["id"], "error": f"{type(exc).__name__}: {exc}",
                            "ranked_docs": [], "support_docs": ev.supporting_doc_ids(q)})
            continue
        records.append({
            "qid": q["id"],
            "ranked_docs": rank_chunks(res, ev.doc_of_chunk),
            "support_docs": ev.supporting_doc_ids(q),
            "graph_version": graph.version,
        })
    return records


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="EvoGraph: 自进化 vs 基线检索精度对比")
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    ap.add_argument("--limit", type=int, default=0, help="冒烟：各取前 N 题（0=全量 500/500）")
    ap.add_argument("--processed", default=None, help="real 模式：复用已有 KB 目录")
    ap.add_argument("--workspace", default=str(ART / "workspace"), help="实验工作目录")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--max-write", type=int, default=8, help="每题最多回写三元组数")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    from experiments.evograph.data import build_mock_kb, load_split
    from experiments.evograph.evaluator import compare, score, to_dict
    from experiments.evograph.evolver import Evolver

    # --limit 同时收缩语料（只取前 2N 题的段落），否则冒烟也要为 11656 段建库
    split = load_split(DEV, shuffle=args.shuffle, seed=args.seed,
                       max_questions=2 * args.limit if args.limit else None)
    ev_ids = split.evolve_ids[: args.limit] if args.limit else split.evolve_ids
    te_ids = split.eval_ids[: args.limit] if args.limit else split.eval_ids
    qmap = {q["id"]: q for q in split.questions}

    # 语料构建 / 复用
    if args.mode == "mock":
        processed = build_mock_kb(split, workspace, Config.load())
        cfg = Config.load(overrides={
            "embedding": {"provider": "mock", "dim": 256},
            "llm": {"provider": "mock"},
            "paths": {"processed_dir": str(processed)},
        })
        os.environ["KB_PATHS__PROCESSED_DIR"] = str(processed)
    else:
        processed = Path(args.processed or "data/processed_musique")
        cfg = Config.load(overrides={"paths": {"processed_dir": str(processed)}})
        os.environ["KB_PATHS__PROCESSED_DIR"] = str(processed)

    # chunk_id → doc_id 映射
    doc_of_chunk: dict = {}
    chunks_file = processed / "chunks.jsonl"
    if chunks_file.exists():
        for line in open(chunks_file, encoding="utf-8"):
            c = json.loads(line)
            doc_of_chunk[c["chunk_id"]] = c.get("doc_id") or c["chunk_id"].rsplit("::", 1)[0]
    split.doc_of_chunk = doc_of_chunk  # type: ignore[attr-defined]

    meta = {
        "mode": args.mode, "limit": args.limit, "top_k": args.top_k,
        "n_evolve": len(ev_ids), "n_eval": len(te_ids),
        "corpus_docs": len(split.doc_meta), "corpus_chunks": len(doc_of_chunk),
        "split_overlap": _title_overlap(split, split.evolve_ids, split.eval_ids),
        "shuffle": args.shuffle, "seed": args.seed,
    }

    baseline_graph, baseline_ret, _ = _build_retriever(cfg, processed)
    v0 = baseline_graph.version
    t0 = time.time()
    base_records = _run_eval(baseline_ret, baseline_graph, [qmap[i] for i in te_ids], split, args.top_k)
    base_time = time.time() - t0
    base = score(base_records)

    evolved_graph, evolved_ret, evo_llm = _build_retriever(cfg, processed)
    evolver = Evolver(evolved_graph, evo_llm, mode=args.mode, max_write_per_q=args.max_write)
    t0 = time.time()
    for q in (qmap[i] for i in ev_ids):
        res = evolved_ret.search(q["question"], top_k=args.top_k)
        # 进化信号 = 检索命中三元组（加权路径）∪ 从 chunk 文本抽取的新三元组（增量路径）
        candidates = [dict(t) for t in (res.triples or [])]
        candidates += evolver.extract(q["question"], res.chunks)
        useful = evolver.judge(q["question"], _decomposition_text(q), candidates)
        evolver.write_back(q["question"], q["id"], useful)
    evolve_time = time.time() - t0
    v1 = evolved_graph.version

    t0 = time.time()
    evo_records = _run_eval(evolved_ret, evolved_graph, [qmap[i] for i in te_ids], split, args.top_k)
    evo_time = time.time() - t0
    evo = score(evo_records)

    result = {
        "meta": meta,
        "graph": {"version_before": v0, "version_after": v1,
                  "triples_after": evolved_graph.stats().get("num_edges") if hasattr(evolved_graph, "stats") else None},
        "evolve_stats": evolver.stats.to_dict(),
        "timing": {"baseline_eval_s": round(base_time, 2), "evolve_s": round(evolve_time, 2),
                   "evolved_eval_s": round(evo_time, 2)},
        "baseline": to_dict(base),
        "evolved": to_dict(evo),
        "comparison": compare(to_dict(base), to_dict(evo)),
    }

    tag = f"{args.mode}_n{len(te_ids)}"
    (ART / f"split_{tag}.json").write_text(json.dumps({
        "evolve_ids": ev_ids, "eval_ids": te_ids,
        "overlap": meta["split_overlap"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (ART / f"evolution_log_{tag}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in evolver.stats.log), encoding="utf-8")
    (ART / f"result_{tag}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.mode == "mock":
        print("\n[mock] 仅验证流程：mock LLM/embedding 对英文语料无语义能力，"
              "精度结论须在 --mode real 下取得。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
