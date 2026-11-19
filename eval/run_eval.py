"""评测 CLI：检索（ChunkHop）与筛查（cohort）两场景的离线/在线指标。

用法：
    python -m eval.run_eval retrieval --corpus main --mode mock --top-k 10
    python -m eval.run_eval retrieval --corpus multihop --mode real   # 需真实服务
    python -m eval.run_eval cohort
    python -m eval.run_eval both --mode mock --out eval/reports/daily.json

指标：
    检索  → Recall@k / MRR / 延迟 / 按 hop 分组（多跳样本=0 时明确标注不可测）
    筛查  → 三值混淆矩阵 / accuracy / macro-F1 / UNKNOWN 漏判率 / 假 UNKNOWN 率

gold 数据（eval/gold/）：
    retrieval_gold.jsonl  question→answer_docs(+hop,corpus)
    cohort_gold.jsonl     patient_id→expected(PASS/FAIL/UNKNOWN)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.metrics import aggregate_cohort, aggregate_retrieval  # noqa: E402
from utils.config import Config  # noqa: E402

GOLD_DIR = Path(__file__).resolve().parent / "gold"


# ---------------------------------------------------------------------------
# 检索评测
# ---------------------------------------------------------------------------
def _build_pipeline(cfg):
    from pipeline.build_pipeline import BuildPipeline
    from retrieval.retriever import Retriever

    pipe = BuildPipeline(cfg)
    pipe.run(incremental=False)
    retriever = Retriever(
        cfg, pipe.graph, pipe.vdb, pipe.embedder,
        pipe.llm, pipe.prompts, graph_path=pipe.graph_path,
    )
    return retriever


def _retrieval_cfg(mode: str, corpus: str) -> tuple[Config, Path]:
    """返回 (cfg, 报告的 corpus 标识)。mock 走临时目录 + mock 服务。"""
    if mode == "mock":
        tmp = Path(tempfile.mkdtemp(prefix=f"eval_{corpus}_"))
        (tmp / "raw").mkdir()
        src = GOLD_DIR / "multihop_docs.jsonl" if corpus == "multihop" \
            else PROJECT_ROOT / "data" / "raw" / "sample_docs.jsonl"
        shutil.copy(src, tmp / "raw" / src.name)
        cfg = Config.load(overrides={
            "paths": {"raw_dir": str(tmp / "raw"), "processed_dir": str(tmp / "processed"),
                      "logs_dir": str(tmp / "logs")},
            "embedding": {"provider": "mock", "dim": 256, "cache_enable": False},
            "llm": {"provider": "mock"},
        })
        return cfg, tmp
    # real：使用 config/default.yaml 与真实数据/服务
    return Config.load(), PROJECT_ROOT


def run_retrieval(args) -> dict:
    cfg, work = _retrieval_cfg(args.mode, args.corpus)
    retriever = _build_pipeline(cfg)

    gold = [json.loads(l) for l in (GOLD_DIR / "retrieval_gold.jsonl").read_text(encoding="utf-8").splitlines()]
    gold = [g for g in gold if g.get("corpus", "main") == args.corpus]
    if not gold:
        print(f"[warn] corpus={args.corpus} 无 gold 样本")
        return {}

    records = []
    for item in gold:
        q = item["question"]
        t0 = time.time()
        result = retriever.search(q, top_k=args.top_k)
        latency_ms = round((time.time() - t0) * 1000, 2)
        seen: set[str] = set()
        ranked_docs = [c["doc_id"] for c in result.chunks
                       if not (c["doc_id"] in seen or seen.add(c["doc_id"]))]
        records.append({
            "question": q, "hop": item.get("hop", 1),
            "gold_docs": item["answer_docs"], "ranked_docs": ranked_docs,
            "latency_ms": latency_ms,
            "graph_version": result.graph_version,
            "dual_stats": result._dual_stats,
        })
        first = next((i + 1 for i, d in enumerate(ranked_docs) if d in item["answer_docs"]), None)
        print(f"  [hop{item.get('hop', 1)}] {q[:24]:<26} 命中位次={first} "
              f"({latency_ms}ms, top1={ranked_docs[0] if ranked_docs else '-'})")

    report = aggregate_retrieval(records, ks=(1, 3, 5, args.top_k))
    report["corpus"] = args.corpus
    report["mode"] = args.mode
    report["graph_version"] = records[0]["graph_version"] if records else None
    _print_retrieval(report)
    return report


def _print_retrieval(report: dict) -> None:
    print(f"\n===== 检索评测 [{report.get('corpus')} / {report.get('mode')}] =====")
    ov = report["overall"]
    print(f"样本 {ov['n']}  graph_version={report.get('graph_version')}")
    for k in (1, 3, 5):
        if f"recall@{k}" in ov:
            print(f"  Recall@{k:<2} {ov[f'recall@{k}']:.3f}")
    print(f"  MRR      {ov['mrr']:.3f}   延迟 avg={ov['latency_ms_avg']}ms max={ov['latency_ms_max']}ms")
    for name, st in report["by_hop"].items():
        if st.get("n"):
            print(f"  [{name}] n={st['n']} Recall@1={st['recall@1']:.3f} MRR={st['mrr']:.3f}")
    print(f"  多跳覆盖: {report['multihop_coverage']}")
    if ov.get("misses"):
        print(f"  未命中: {ov['misses']}")


# ---------------------------------------------------------------------------
# 筛查评测
# ---------------------------------------------------------------------------
def run_cohort(args) -> dict:
    from cohort.engine import CohortEngine
    from cohort.seed import DEMO_IR

    tmp = Path(tempfile.mkdtemp(prefix="eval_cohort_"))
    db_path = tmp / "demo.db"
    engine = CohortEngine(db_path, seed_count=30)
    result = engine.search("科研入组筛查", DEMO_IR, reference_date=DEMO_IR.reference_date
                           if hasattr(DEMO_IR, "reference_date") else "2026-11-18")
    actual = {int(d["patient_id"]): d["eligibility"] for d in result.decisions}

    gold = [json.loads(l) for l in (GOLD_DIR / "cohort_gold.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs = [(g["expected"], actual[g["patient_id"]]) for g in gold if g["patient_id"] in actual]
    report = aggregate_cohort(pairs)
    report["reference_date"] = result.reference_date
    report["mode"] = "sqlite"
    _print_cohort(report, gold, actual)
    return report


def _print_cohort(report: dict, gold: list[dict], actual: dict[int, str]) -> None:
    print("\n===== 筛查评测 [SQLite] =====")
    print(f"样本 {report['n']}  accuracy={report['accuracy']:.3f}  macro-F1={report['macro_f1']:.3f}")
    for label, m in report["per_class"].items():
        print(f"  {label:<7} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print(f"  UNKNOWN 漏判率(缺失当证据)={report['unknown_miss_rate']}  "
          f"假 UNKNOWN 率={report['false_unknown_rate']}")
    mism = [g for g in gold if actual.get(g["patient_id"]) != g["expected"]]
    if mism:
        print(f"  不一致 {len(mism)} 例:")
        for g in mism:
            print(f"    P{g['patient_id']:03d} 期望={g['expected']} 实际={actual[g['patient_id']]} ({g['reason']})")


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=["retrieval", "cohort", "both"])
    parser.add_argument("--corpus", choices=["main", "multihop"], default="main",
                        help="检索语料：main=5 篇样例，multihop=跨文档多跳合成语料")
    parser.add_argument("--mode", choices=["mock", "real"], default="mock",
                        help="mock=离线确定性，real=config/default.yaml 真实服务")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="JSON 报告输出路径")
    args = parser.parse_args()

    report: dict = {}
    if args.task in ("retrieval", "both"):
        report["retrieval"] = run_retrieval(args)
    if args.task in ("cohort", "both"):
        report["cohort"] = run_cohort(args)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = PROJECT_ROOT / "eval" / "reports" / f"{ts}_{args.task}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
