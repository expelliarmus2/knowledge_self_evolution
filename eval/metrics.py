"""评测指标（纯函数，零外部依赖，便于单测与复用）。

检索侧（binary relevance，文档级）：
  Recall@k / MRR / Hit@k / 首命中位次，支持按 hop 分组聚合。
筛查侧（三值分类 PASS/FAIL/UNKNOWN）：
  混淆矩阵 / accuracy / macro-F1 / 每类 P-R-F1 /
  UNKNOWN 漏判率（gold=UNKNOWN 被判成 PASS|FAIL）/ 假 UNKNOWN 率。
"""
from __future__ import annotations

from typing import Iterable, Sequence

# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------
def first_hit_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> int | None:
    """第一个命中的 1-based 位次；未命中返回 None。"""
    gold = set(gold_ids)
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in gold:
            return i
    return None


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    """文档级 binary recall：Top-k 内出现任一 gold 文档即 1.0。"""
    rank = first_hit_rank(ranked_ids[:k], gold_ids)
    return 1.0 if rank is not None else 0.0


def hit_at_k(ranked_ids: Sequence[str], gold_ids: Iterable[str], k: int) -> float:
    return recall_at_k(ranked_ids, gold_ids, k)


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    rank = first_hit_rank(ranked_ids, gold_ids)
    return 1.0 / rank if rank else 0.0


def aggregate_retrieval(
    records: Sequence[dict],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict:
    """records: [{question, ranked_docs, gold_docs, hop, latency_ms}]。

    返回整体指标 + 按 hop 分组的指标（多跳子集样本为 0 时明确标注不可测）。
    """
    if not records:
        return {"n": 0, "note": "无样本"}

    def _stats(subset: Sequence[dict]) -> dict:
        if not subset:
            return {"n": 0}
        out: dict = {"n": len(subset)}
        for k in ks:
            out[f"recall@{k}"] = round(
                sum(recall_at_k(r["ranked_docs"], r["gold_docs"], k) for r in subset) / len(subset), 4)
        out["mrr"] = round(sum(reciprocal_rank(r["ranked_docs"], r["gold_docs"]) for r in subset) / len(subset), 4)
        lat = [r.get("latency_ms", 0.0) for r in subset]
        out["latency_ms_avg"] = round(sum(lat) / len(lat), 2)
        out["latency_ms_max"] = round(max(lat), 2)
        misses = [r["question"] for r in subset
                  if first_hit_rank(r["ranked_docs"], r["gold_docs"]) is None]
        out["misses"] = misses
        return out

    report = {"overall": _stats(records)}
    by_hop: dict[str, dict] = {}
    for hop in sorted({int(r.get("hop", 1)) for r in records}):
        by_hop[f"hop{hop}"] = _stats([r for r in records if int(r.get("hop", 1)) == hop])
    report["by_hop"] = by_hop
    multihop = [r for r in records if int(r.get("hop", 1)) >= 2]
    report["multihop_coverage"] = (
        f"{len(multihop)} 条 hop>=2 样本" if multihop
        else "0 条 hop>=2 样本：多跳指标不可测，需扩充跨文档共享实体的语料"
    )
    return report


# ---------------------------------------------------------------------------
# 筛查（三值）
# ---------------------------------------------------------------------------
LABELS = ("PASS", "FAIL", "UNKNOWN")


def confusion_matrix(pairs: Iterable[tuple[str, str]]) -> dict:
    """pairs: (expected, actual)。返回 {expected: {actual: count}}，缺格补 0。"""
    matrix = {exp: {act: 0 for act in LABELS} for exp in LABELS}
    for exp, act in pairs:
        matrix.setdefault(exp, {}).setdefault(act, 0)
        matrix[exp][act] = matrix[exp].get(act, 0) + 1
    return matrix


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def aggregate_cohort(pairs: Sequence[tuple[str, str]]) -> dict:
    """pairs: (expected, actual) 三元标签。

    重点指标：
      - unknown_miss_rate：gold=UNKNOWN 却给出 PASS/FAIL 的比例（数据缺失被当成证据）
      - false_unknown_rate：gold 有定论却判 UNKNOWN 的比例（过度保守）
    """
    if not pairs:
        return {"n": 0, "note": "无样本"}
    matrix = confusion_matrix(pairs)
    n = len(pairs)
    correct = sum(matrix[l][l] for l in LABELS)

    per_class: dict = {}
    macro_support = []
    for label in LABELS:
        tp = matrix[label][label]
        fn = sum(matrix[label][x] for x in LABELS if x != label)
        fp = sum(matrix[x][label] for x in LABELS if x != label)
        p, r, f = _prf(tp, fp, fn)
        support = tp + fn
        per_class[label] = {"precision": p, "recall": r, "f1": f, "support": support}
        if support:
            macro_support.append(f)

    gold_unknown = [a for e, a in pairs if e == "UNKNOWN"]
    gold_decided = [a for e, a in pairs if e != "UNKNOWN"]
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "macro_f1": round(sum(macro_support) / len(macro_support), 4) if macro_support else 0.0,
        "per_class": per_class,
        "confusion": matrix,
        "unknown_miss_rate": round(
            sum(1 for a in gold_unknown if a in ("PASS", "FAIL")) / len(gold_unknown), 4
        ) if gold_unknown else None,
        "false_unknown_rate": round(
            sum(1 for a in gold_decided if a == "UNKNOWN") / len(gold_decided), 4
        ) if gold_decided else None,
    }
