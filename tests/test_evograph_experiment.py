"""EvoGraph 实验子系统测试：切分、指标、自进化写入约束。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.evograph.evaluator import compare, rank_chunks, score  # noqa: E402
from experiments.evograph.evolver import Evolver  # noqa: E402

DEV = Path("data/test/musique_dev.json")


@pytest.mark.skipif(not DEV.exists(), reason="缺少 MuSiQue dev 数据")
def test_split_is_500_500_and_covers_all():
    from experiments.evograph.data import load_split

    split = load_split(DEV)
    assert len(split.evolve_ids) == 500
    assert len(split.eval_ids) == 500
    assert len(set(split.evolve_ids) & set(split.eval_ids)) == 0
    assert len(split.evolve_ids) + len(split.eval_ids) == len(split.questions) == 1000
    assert len(split.doc_meta) > 0


@pytest.mark.skipif(not DEV.exists(), reason="缺少 MuSiQue dev 数据")
def test_supporting_doc_ids_map_to_corpus_docs():
    from experiments.evograph.data import load_split

    split = load_split(DEV, max_questions=4)
    q = split.questions[0]
    sup = split.supporting_doc_ids(q)
    expected = sum(1 for p in q["paragraphs"] if p.get("is_supporting"))
    assert len(sup) == expected >= 2
    for did in sup:
        assert did in split.doc_meta


def test_evaluator_recall_and_mrr():
    records = [
        {"ranked_docs": ["a", "b", "c"], "support_docs": ["b"]},           # rank2
        {"ranked_docs": ["x", "y"], "support_docs": ["y", "z"]},            # hit@2 但 z 缺
        {"ranked_docs": ["m"], "support_docs": ["n"]},                      # 未命中
    ]
    s = score(records, ks=(1, 2, 3))
    assert s.n == 3
    # recall@1: (0 + 0 + 0)/3
    assert s.recall[1] == 0.0
    # recall@2: (1/1 + 1/2 + 0)/3 = 0.5
    assert s.recall[2] == pytest.approx(0.5, abs=1e-4)
    assert s.recall[3] == pytest.approx((1.0 + 0.5 + 0) / 3, abs=1e-4)
    # mrr: 1/2 + 1/2 + 0 → (1/3)
    assert s.mrr == pytest.approx(1 / 3, abs=1e-4)
    assert s.hit[2] == pytest.approx(2 / 3, abs=1e-4)


def test_rank_chunks_dedups_and_maps_docs():
    class R:
        chunks = [
            {"chunk_id": "d1::chunk0"},
            {"chunk_id": "d1::chunk1"},
            {"chunk_id": "d2::chunk0"},
            {"chunk_id": "unknown::chunk0"},
        ]

    doc_of_chunk = {"d1::chunk0": "d1", "d1::chunk1": "d1", "d2::chunk0": "d2"}
    assert rank_chunks(R(), doc_of_chunk) == ["d1", "d2"]


def test_compare_reports_delta_and_rel():
    base = {"n": 10, "recall": {1: 0.2, 5: 0.4, 10: 0.6}, "mrr": 0.3, "hit": {1: 0.2, 5: 0.4, 10: 0.6}}
    evo = {"n": 10, "recall": {1: 0.3, 5: 0.4, 10: 0.6}, "mrr": 0.45, "hit": {1: 0.3, 5: 0.4, 10: 0.6}}
    c = compare(base, evo)
    assert c["recall"]["@1"]["delta"] == pytest.approx(0.1)
    assert c["recall"]["@1"]["rel"] == pytest.approx(0.5)
    assert c["recall"]["@5"]["delta"] == 0.0
    assert c["mrr"]["delta"] == pytest.approx(0.15)


class _StubLLM:
    """返回固定 JSON 的 LLM stub。"""

    def __init__(self, payload: dict):
        self.payload = payload

    def generate(self, messages):
        return json.dumps(self.payload, ensure_ascii=False)


@pytest.fixture()
def tiny_graph():
    try:
        from graph.knowledge_graph import KnowledgeGraph
    except Exception:  # pragma: no cover
        pytest.skip("graph 模块不可用")
    g = KnowledgeGraph()
    g.add_triple("Alpha", "related_to", "Beta", chunk_id="d1::chunk0", meta={"origin": "build"})
    return g


def test_evolver_rejects_hallucinated_triples(tiny_graph):
    """LLM 只能在候选集合内选择，凭空造的三元组被丢弃。"""
    llm = _StubLLM({"useful": [
        {"head": "Alpha", "relation": "related_to", "tail": "Beta"},   # 候选内
        {"head": "Fake", "relation": "made_up", "tail": "Nonsense"},   # 候选外
    ]})
    ev = Evolver(tiny_graph, llm, mode="real")
    candidates = [{"head": "Alpha", "relation": "related_to", "tail": "Beta", "score": 1.0}]
    useful = ev.judge("question about Alpha", "", candidates)
    assert len(useful) == 1
    assert useful[0]["head"] == "Alpha"


def test_evolver_write_back_marks_evolved_and_prefers_new(tiny_graph):
    ev = Evolver(tiny_graph, _StubLLM({}), mode="mock", max_write_per_q=2)
    useful = [
        {"head": "Alpha", "relation": "related_to", "tail": "Beta", "confidence": 0.6},
        {"head": "Gamma", "relation": "born_in", "tail": "Paris", "confidence": 0.5, "is_new": True},
    ]
    ev.write_back("q?", "qid1", useful)
    assert ev.stats.written == 1          # 新边
    assert ev.stats.deduped == 1          # 已有边累权
    edge = tiny_graph.g.edges["Gamma", "Paris", "born_in"]
    assert edge["meta"]["evolved"] is True
    assert edge["meta"]["source_qid"] == "qid1"
    assert all(x["qid"] == "qid1" for x in ev.stats.log)


def test_evolver_extract_skips_existing_edges(tiny_graph):
    """抽取阶段：与图谱已有 (h,r,t) 相同的不进候选（走加权路径）。"""
    llm = _StubLLM({"triples": [["Alpha", "related_to", "Beta"],
                                ["Delta", "won", "Prize"]]})
    ev = Evolver(tiny_graph, llm, mode="real")
    new = ev.extract("q", [{"text": "Alpha ... Beta ... Delta won Prize"}])
    assert {(x["head"], x["relation"], x["tail"]) for x in new} == {("Delta", "won", "Prize")}
    assert all(x["is_new"] for x in new)
