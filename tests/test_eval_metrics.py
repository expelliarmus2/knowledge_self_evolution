"""eval/metrics 纯函数单测 + run_eval 冒烟（cohort 确定性，~1s）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from eval.metrics import (  # noqa: E402
    aggregate_cohort,
    aggregate_retrieval,
    confusion_matrix,
    first_hit_rank,
    reciprocal_rank,
    recall_at_k,
)


# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------
def test_first_hit_rank():
    assert first_hit_rank(["a", "b", "c"], ["b"]) == 2
    assert first_hit_rank(["a", "b", "c"], ["c", "a"]) == 1   # gold 任一命中
    assert first_hit_rank(["a", "b"], ["z"]) is None


def test_recall_at_k_topk_cutoff():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, ["c"], 2) == 0.0
    assert recall_at_k(ranked, ["c"], 3) == 1.0
    assert recall_at_k(ranked, ["d"], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["a", "b"], ["b"]) == 0.5
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_aggregate_retrieval_by_hop():
    records = [
        {"question": "q1", "ranked_docs": ["x1", "x2"], "gold_docs": ["x1"], "hop": 1, "latency_ms": 10},
        {"question": "q2", "ranked_docs": ["y1", "y2"], "gold_docs": ["y3"], "hop": 2, "latency_ms": 20},
        {"question": "q3", "ranked_docs": ["y1", "y2"], "gold_docs": ["y2"], "hop": 2, "latency_ms": 30},
    ]
    rep = aggregate_retrieval(records, ks=(1, 2))
    # q1 命中(rank1)、q2 未命中、q3 命中(rank2，不计入 recall@1)
    assert rep["overall"]["recall@1"] == pytest.approx(1 / 3, abs=1e-4)
    assert rep["overall"]["mrr"] == pytest.approx((1.0 + 0.0 + 0.5) / 3)
    assert rep["by_hop"]["hop1"]["recall@1"] == 1.0
    assert rep["by_hop"]["hop2"]["recall@1"] == 0.0
    assert rep["by_hop"]["hop2"]["mrr"] == pytest.approx(0.25)
    assert rep["overall"]["latency_ms_avg"] == pytest.approx(20.0)
    assert rep["overall"]["misses"] == ["q2"]


def test_aggregate_retrieval_multihop_coverage_empty():
    rep = aggregate_retrieval([
        {"question": "q", "ranked_docs": ["a"], "gold_docs": ["a"], "hop": 1, "latency_ms": 1},
    ])
    assert "0 条 hop>=2 样本" in rep["multihop_coverage"]


def test_aggregate_retrieval_empty():
    rep = aggregate_retrieval([])
    assert rep["n"] == 0


# ---------------------------------------------------------------------------
# 筛查指标
# ---------------------------------------------------------------------------
def test_confusion_matrix_pads_missing_cells():
    m = confusion_matrix([("PASS", "PASS"), ("FAIL", "PASS")])
    assert m["PASS"]["PASS"] == 1 and m["PASS"]["FAIL"] == 0
    assert m["UNKNOWN"]["PASS"] == 0   # 缺格补 0


def test_aggregate_cohort_perfect():
    pairs = [("PASS", "PASS"), ("FAIL", "FAIL"), ("UNKNOWN", "UNKNOWN")] * 3
    rep = aggregate_cohort(pairs)
    assert rep["accuracy"] == 1.0 and rep["macro_f1"] == 1.0
    assert rep["unknown_miss_rate"] == 0.0 and rep["false_unknown_rate"] == 0.0


def test_aggregate_cohort_unknown_misjudged():
    """gold=UNKNOWN 被判成 PASS → 缺失被当成证据（最危险的误判）。"""
    pairs = [("PASS", "PASS"), ("UNKNOWN", "PASS"), ("UNKNOWN", "FAIL"), ("FAIL", "FAIL")]
    rep = aggregate_cohort(pairs)
    assert rep["unknown_miss_rate"] == 1.0
    assert rep["false_unknown_rate"] == 0.0
    assert rep["confusion"]["UNKNOWN"]["PASS"] == 1


def test_aggregate_cohort_false_unknown():
    """gold 有定论却判 UNKNOWN → 过度保守（1/2 例）。"""
    pairs = [("PASS", "UNKNOWN"), ("FAIL", "FAIL")]
    rep = aggregate_cohort(pairs)
    assert rep["false_unknown_rate"] == 0.5
    assert rep["unknown_miss_rate"] is None  # 没有 gold=UNKNOWN 样本


def test_aggregate_cohort_prf_edge_zero():
    rep = aggregate_cohort([("PASS", "FAIL")])
    assert rep["accuracy"] == 0.0
    assert rep["per_class"]["PASS"]["f1"] == 0.0  # tp=0 不除零


# ---------------------------------------------------------------------------
# run_eval 冒烟：cohort 30 例确定性 → accuracy 必须 1.0
# ---------------------------------------------------------------------------
def test_run_eval_cohort_smoke(tmp_path):
    import subprocess
    out = tmp_path / "r.json"
    r = subprocess.run(
        [sys.executable, "-m", "eval.run_eval", "cohort", "--out", str(out)],
        capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    rep = json.loads(out.read_text(encoding="utf-8"))["cohort"]
    assert rep["n"] == 30
    assert rep["accuracy"] == 1.0, f"seed 30 例与 gold 必须一致: {rep['confusion']}"
    assert rep["unknown_miss_rate"] == 0.0
