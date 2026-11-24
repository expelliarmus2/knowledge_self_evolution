"""检索精度指标：Recall@k / MRR@k / hit@k（按支持段落对应 doc_id 判定）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class RetrievalScore:
    recall: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit: Dict[int, float] = field(default_factory=dict)
    n: int = 0
    per_q: List[dict] = field(default_factory=list)


def rank_chunks(result, doc_of_chunk: Dict[str, str]) -> List[str]:
    """把 SearchResult.chunks 的 chunk_id 转成 doc_id 排名列表（保持顺序）。"""
    seen, out = set(), []
    for c in result.chunks:
        cid = c.get("chunk_id", "")
        did = doc_of_chunk.get(cid)
        if did and did not in seen:
            seen.add(did)
            out.append(did)
    return out


def score(records: List[dict], ks=(1, 5, 10)) -> RetrievalScore:
    """records: [{ranked_docs, support_docs, qid}]。"""
    s = RetrievalScore(n=len(records))
    recall = {k: 0.0 for k in ks}
    hit = {k: 0.0 for k in ks}
    mrr_sum = 0.0
    for r in records:
        ranked: List[str] = r["ranked_docs"]
        support = set(r["support_docs"])
        if not support:
            continue
        for k in ks:
            topk = set(ranked[:k])
            recall[k] += len(topk & support) / len(support)
            hit[k] += 1.0 if topk & support else 0.0
        for i, did in enumerate(ranked[:max(ks)], start=1):
            if did in support:
                mrr_sum += 1.0 / i
                break
    n = max(1, s.n)
    s.recall = {k: round(v / n, 4) for k, v in recall.items()}
    s.hit = {k: round(v / n, 4) for k, v in hit.items()}
    s.mrr = round(mrr_sum / n, 4)
    s.per_q = records
    return s


def to_dict(s: RetrievalScore) -> dict:
    return {"n": s.n, "recall": s.recall, "mrr": s.mrr, "hit": s.hit}


def compare(base: dict, evolved: dict) -> dict:
    """对比两臂指标，输出绝对/相对提升。"""
    out = {}
    for key, getter in (
        ("recall", lambda d: d["recall"]),
        ("hit", lambda d: d["hit"]),
    ):
        b, e = getter(base), getter(evolved)
        out[key] = {
            f"@{k}": {
                "baseline": b[k], "evolved": e[k],
                "delta": round(e[k] - b[k], 4),
                "rel": round((e[k] - b[k]) / b[k], 4) if b[k] else None,
            }
            for k in b
        }
    b, e = base["mrr"], evolved["mrr"]
    out["mrr"] = {"baseline": b, "evolved": e, "delta": round(e - b, 4),
                  "rel": round((e - b) / b, 4) if b else None}
    return out
