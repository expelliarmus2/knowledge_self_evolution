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


# ------------------------- 方式A：IRCoT 轨迹级评价 -------------------------
def test_judge_trajectory_mock_heuristic_and_dedup():
    from experiments.evograph.trajectory import judge_trajectory, selected_triples

    steps = [
        {"sentence": "Find X.", "n_new": 3, "stop": "",
         "step_chunks": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.8, "text": "X lives in Y."},
                         {"chunk_id": "c2", "doc_id": "d2", "score": 0.5, "text": "Z unrelated."}],
         "step_triples": [{"head": "X", "relation": "lives_in", "tail": "Y", "score": 0.9}]},
        {"sentence": "No progress.", "n_new": 0, "stop": "",
         "step_chunks": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.4, "text": "dup."}],
         "step_triples": []},
    ]
    values, sel = judge_trajectory("Where does X live?", steps, llm=None, mode="mock")
    assert values[1] == "high" and values[2] == "low"
    # 去重：c1 只出现一次；低价值轮次不贡献
    ids = [c["chunk_id"] for c in sel]
    assert ids.count("c1") == 1 and "c2" in ids
    tris = selected_triples(steps, values)
    assert len(tris) == 1 and tris[0]["tail"] == "Y"


def test_judge_trajectory_llm_rating_and_fallback():
    from experiments.evograph.trajectory import judge_trajectory

    steps = [{"sentence": "Find X.", "n_new": 2, "stop": "",
              "step_chunks": [{"chunk_id": "c1", "doc_id": "d1", "score": 0.7, "text": "t"}]}]
    # LLM 给出 low → 但该轮有新增证据，不应回退覆盖（显式评级优先）
    llm = _StubLLM({"rated_steps": [{"step": 1, "value": "low", "reason": "noise"}]})
    values, sel = judge_trajectory("q", steps, llm, mode="real")
    assert values[1] == "low" and sel == []
    # LLM 漏评 → 回退 n_new 启发式
    llm2 = _StubLLM({"rated_steps": []})
    values2, sel2 = judge_trajectory("q", steps, llm2, mode="real")
    assert values2[1] == "high" and len(sel2) == 1
    # LLM 异常 → 回退且不抛
    class Boom:
        def generate(self, messages):
            raise RuntimeError("llm down")
    values3, sel3 = judge_trajectory("q", steps, Boom(), mode="real")
    assert values3[1] == "high" and len(sel3) == 1


def test_ircot_records_step_chunks_for_trajectory(tmp_path):
    """run_ircot 每轮记录 step_chunks/step_triples（方式A 的数据基础）。"""
    try:
        from agentic import run_ircot
        from agentic.memory import EvidencePool  # noqa: F401
        from agentic.tools import Tools
    except Exception:  # pragma: no cover
        pytest.skip("agentic 模块不可用")

    class _ChunkRes:
        """模拟 SearchResult（属性访问）。"""
        chunks = [{"chunk_id": "c1", "doc_id": "d1", "score": 0.9,
                   "source": "vector", "text": "Alpha lives in Beta city."}]
        triples = [{"head": "Alpha", "relation": "related_to",
                    "tail": "Beta", "score": 1.0}]
        entities = []
        timing = {}

    class _Ret:
        def search(self, query, top_k=None):
            return _ChunkRes()

    class _LLM:
        def __init__(self):
            self.n = 0

        def generate(self, messages):
            self.n += 1
            return "NEXT: Where is Alpha located?" if self.n == 1 else "FINAL ANSWER: Beta city"

    class _Prompts:
        def render(self, name, **kw):
            return "irrelevant"

    class _Cfg(dict):
        pass

    cfg = _Cfg({"agentic.max_steps": 6, "agentic.top_k_per_step": 5,
                "agentic.min_steps": 1, "agentic.evidence_max_docs": 32,
                "agentic.evidence_max_chars": 6000, "agentic.include_triples": True,
                "agentic.max_triples": 12})
    res = run_ircot("Where does Alpha live?", Tools(_Ret()), _LLM(), _Prompts(), cfg)
    # 至少 1 轮检索发生（FINAL 前的 NEXT 轮）
    assert res.steps, "IRCoT 应至少记录一轮"
    s = res.steps[0]
    assert "step_chunks" in s and s["step_chunks"][0]["chunk_id"] == "c1"
    assert s["step_triples"][0]["relation"] == "related_to"
