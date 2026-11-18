"""Agentic Search 精度评测：EM / F1 / 证据召回率。

用法（项目根目录）：
    python -m evaluation.eval_agentic --processed data/processed_musique_onestep \
        --test data/test/musique_dev.json --mode ircot --workers 8 --limit 200

指标：
  - em@1        : 答案归一化后完全匹配（gold answer 或其 alias）
  - f1@1        : token F1（短答案，与 MuSiQue/2Wiki 论文口径一致）
  - evidence_recall : 金标证据文档出现在证据池（最终检索到的文档集合）的比例
  - 平均步数 / LLM 调用数 / 单条耗时（成本代理）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentic import run_ircot, run_planner  # noqa: E402
from evaluation.eval_recall import load_test, norm_doc  # noqa: E402
from graph.knowledge_graph import KnowledgeGraph  # noqa: E402
from openie.embedder import build_embedder  # noqa: E402
from openie.llm_client import build_llm  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.prompts import PromptManager  # noqa: E402
from vectorstore.manager import VectorStoreManager  # noqa: E402

_ART = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[\.,;:'\"!?\(\)\[\]{}]+")
_WS = re.compile(r"\s+")


def norm_answer(s: str) -> str:
    """MuSiQue 短答案归一化：小写、去冠词/标点、压缩空白。"""
    s = _PUNCT.sub(" ", str(s or "").lower())
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()


def tokenize(s: str):
    return norm_answer(s).split()


def em_score(pred: str, gold: str) -> int:
    return int(norm_answer(pred) == norm_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    p_tok, g_tok = tokenize(pred), tokenize(gold)
    if not g_tok:
        return 1.0 if not p_tok else 0.0
    if not p_tok:
        return 0.0
    common = set(p_tok) & set(g_tok)
    if not common:
        return 0.0
    prec = len(common) / len(p_tok)
    rec = len(common) / len(g_tok)
    return 2 * prec * rec / (prec + rec)


def build_retriever(processed_dir: Path, cfg: Config):
    graph = KnowledgeGraph.load(processed_dir / "graph.pkl")
    embedder = build_embedder(cfg)
    prompts = PromptManager.from_config(cfg)
    vdb = VectorStoreManager(cfg, embedder, processed_dir)
    retriever = Retriever(cfg, graph, vdb, embedder, None, prompts,
                          graph_path=processed_dir / "graph.pkl")
    return retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", required=True, help="知识库产物目录")
    ap.add_argument("--test", default="data/test/musique_dev.json")
    ap.add_argument("--mode", choices=["ircot", "planner"], default="ircot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="结果目录名（默认 recall_agentic_<ts>）")
    args = ap.parse_args()

    cfg = Config.load(path=args.config)
    processed = Path(args.processed)
    test = load_test(Path(args.test), supporting_only=True)

    retriever = build_retriever(processed, cfg)
    llm = build_llm(cfg)
    prompts = PromptManager.from_config(cfg)
    from agentic.tools import Tools
    tools = Tools(retriever)

    if args.limit:
        test = test[: args.limit]
    print(f"评测 {args.mode} | {len(test)} 题 | LLM {cfg.get('llm.model')} | workers {args.workers}")

    def run_one(q: dict):
        t0 = time.time()
        try:
            if args.mode == "ircot":
                res = run_ircot(q["question"], tools, llm, prompts, cfg)
            else:
                res = run_planner(q["question"], tools, llm, prompts, cfg)
        except Exception as err:  # noqa: BLE001
            return {"idx": q["idx"], "error": str(err), "ms": (time.time() - t0) * 1000}
        golds = [q["answer"]] + q.get("answer_aliases", [])
        em = max(em_score(res.answer, g) for g in golds)
        f1 = max(f1_score(res.answer, g) for g in golds)
        gold_docs = {norm_doc(d) for d in q["gold"]}
        ev_docs = {norm_doc(d) for d in res.evidence_docs}
        ev_recall = len(gold_docs & ev_docs) / len(gold_docs) if gold_docs else 0.0
        return {
            "idx": q["idx"],
            "answer": res.answer,
            "gold": q["answer"],
            "em": em,
            "f1": round(f1, 4),
            "ev_recall": round(ev_recall, 4),
            "n_docs": len(res.evidence_docs),
            "steps": res.steps,
            "llm_calls": res.llm_calls,
            "stop": res.stop_reason,
            "ms": round((time.time() - t0) * 1000, 1),
        }

    rows: list = []
    t_start = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, q): q for q in test}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(test)} ...", flush=True)
    elapsed = time.time() - t_start

    ok = [r for r in rows if "error" not in r]
    n = len(ok)
    em = sum(r["em"] for r in ok) / n if n else 0.0
    f1 = sum(r["f1"] for r in ok) / n if n else 0.0
    ev_rec = sum(r["ev_recall"] for r in ok) / n if n else 0.0
    avg_docs = sum(r["n_docs"] for r in ok) / n if n else 0.0
    avg_calls = sum(r["llm_calls"] for r in ok) / n if n else 0.0
    avg_ms = sum(r["ms"] for r in ok) / n if n else 0.0

    print("=" * 72)
    print(f"  mode={args.mode}  n={n}  failures={len(rows) - n}")
    print(f"  EM        {em:.4f}")
    print(f"  F1        {f1:.4f}")
    print(f"  evidence_recall  {ev_rec:.4f}")
    print(f"  avg_docs  {avg_docs:.1f}  avg_llm_calls {avg_calls:.1f}  avg_ms {avg_ms:.0f}")
    print(f"  总耗时 {elapsed:.1f}s")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("evaluation/results") / (args.out or f"agentic_{args.mode}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": args.mode, "n": n, "failures": len(rows) - n,
        "em": em, "f1": f1, "evidence_recall": ev_rec,
        "avg_docs": avg_docs, "avg_llm_calls": avg_calls, "avg_ms": avg_ms,
        "elapsed_sec": round(elapsed, 1),
        "config": {k: cfg.get(f"agentic.{k}") for k in
                   ("max_steps", "top_k_per_step", "evidence_max_docs", "include_triples")},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(out_dir / "detail.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  结果: {out_dir}")


if __name__ == "__main__":
    main()
