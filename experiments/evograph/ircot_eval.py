"""IRCoT 检索能力评测：同一 eval 集上，多轮 IRCoT 轨迹检索 vs 单轮检索的证据召回与答案精度。

回答的问题："加入 IRCoT 之后，检索本身的精度提升多少？"
（自进化实验主指标测的是单轮 base retriever，IRCoT 的检索能力未被计入。）

指标：
- pool_recall@10  证据池前 10 个发现文档对 gold 支持段落的召回（对齐单轮 R@10 口径）
- pool_recall      证据池全量（≤32 doc）召回上限
- answer_em / answer_contains  最终答案精确匹配 / 包含匹配
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

ART = Path(__file__).resolve().parent / "artifacts"
DEV = Path("data/test/musique_dev.json")
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s or "")).strip().lower()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="IRCoT 多轮检索能力评测（vs 单轮 R@10=0.4817）")
    ap.add_argument("--limit", type=int, default=0, help="冒烟：前 N 题（0=全量 500）")
    ap.add_argument("--processed", default="data/processed_musique")
    ap.add_argument("--graph", default=None,
                    help="改进图 pickle（run_experiment 保存）；缺省=原图 processed/graph.pkl")
    ap.add_argument("--tag", default="ircot_eval")
    args = ap.parse_args(argv)

    from experiments.evograph.data import load_split_processed
    from experiments.evograph.run_experiment import _build_retriever
    from utils.config import Config

    processed = Path(args.processed)
    split = load_split_processed(
        DEV, Path("data/raw/corpus_musique.json"), processed / "chunks.jsonl",
        shuffle=False, seed=42,
        max_questions=2 * args.limit if args.limit else None)
    te_ids = split.eval_ids[: args.limit] if args.limit else split.eval_ids
    qmap = {q["id"]: q for q in split.questions}

    cfg = Config.load(overrides={"paths": {"processed_dir": str(processed)}})
    os.environ["KB_PATHS__PROCESSED_DIR"] = str(processed)
    graph, retriever, llm = _build_retriever(cfg, processed, graph_path=args.graph)
    graph_label = "evolved" if args.graph else "original"

    from agentic import run_ircot
    from agentic.tools import Tools
    from utils.prompts import PromptManager

    prompts = PromptManager.from_config(cfg)
    tools = Tools(retriever)

    t0 = time.time()
    recs = []
    n_done = 0
    for i, qid in enumerate(te_ids, 1):
        q = qmap[qid]
        gold = set(split.supporting_doc_ids(q))
        try:
            res = run_ircot(q["question"], tools, llm, prompts, cfg)
            docs = res.evidence_docs
            top10 = set(docs[:10])
            rec = {
                "qid": qid,
                "steps": len(res.steps),
                "llm_calls": res.llm_calls,
                "stop_reason": res.stop_reason,
                "pool_size": len(docs),
                "pool_recall@10": len(gold & top10) / len(gold) if gold else 0.0,
                "pool_recall": len(gold & set(docs)) / len(gold) if gold else 0.0,
                "answer": res.answer[:200],
                "gold_answer": q.get("answer", ""),
                "answer_em": int(_norm(res.answer) == _norm(q.get("answer", ""))),
                "answer_contains": int(_norm(q.get("answer", "")) in _norm(res.answer)
                                        and bool(_norm(q.get("answer", "")))),
            }
        except Exception as err:  # noqa: BLE001
            rec = {"qid": qid, "error": f"{type(err).__name__}: {err}"}
        recs.append(rec)
        n_done += 1
        if i % 25 == 0 or i == len(te_ids):
            ok = [r for r in recs if "error" not in r]
            print(f"[{i}/{len(te_ids)}] pool_recall@10="
                  f"{sum(r['pool_recall@10'] for r in ok) / max(1, len(ok)):.4f} "
                  f"em={sum(r['answer_em'] for r in ok) / max(1, len(ok)):.4f} "
                  f"err={n_done - len(ok)}", flush=True)

    ok = [r for r in recs if "error" not in r]
    out = {
        "meta": {"n_eval": len(te_ids), "n_ok": len(ok),
                 "graph": graph_label,
                 "doc_coverage": getattr(split, "doc_coverage", None),
                 "elapsed_s": round(time.time() - t0, 1)},
        "summary": {
            "pool_recall@10": round(sum(r["pool_recall@10"] for r in ok) / max(1, len(ok)), 4),
            "pool_recall": round(sum(r["pool_recall"] for r in ok) / max(1, len(ok)), 4),
            "answer_em": round(sum(r["answer_em"] for r in ok) / max(1, len(ok)), 4),
            "answer_contains": round(sum(r["answer_contains"] for r in ok) / max(1, len(ok)), 4),
            "avg_steps": round(sum(r["steps"] for r in ok) / max(1, len(ok)), 2),
            "avg_llm_calls": round(sum(r["llm_calls"] for r in ok) / max(1, len(ok)), 2),
        },
        "records": recs,
    }
    path = ART / f"{args.tag}_n{len(te_ids)}.json"
    if graph_label == "evolved":
        path = ART / f"{args.tag}_evolved_n{len(te_ids)}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False))
    print(f"saved → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
