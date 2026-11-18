"""检索质量测试（离线 mock，确定性，无需任何服务）。

断言分四类：
1. **Top-1 准确率**：gold 问题集的答案 doc 必须是 Top-1 chunk（等价 Recall@1）
2. **排序结构**：rank 连续、分数单调不增、文档级去重、source 取值合法
3. **可复现性**：同一查询两次结果完全一致（MockEmbedder 是确定性哈希袋）
4. **链路诊断**：_full_rank/_seed_chunks/_dual_stats 与主输出自洽 + 子图结构 + 图版本

为什么只做 chunk 级断言：MockLLMClient 在中文语料上抽不出事实三元组
（257 条三元组全是英文噪声），图证据与多跳扩散无法在 mock 端到端里验证 ——
那部分见 `tests/test_retrieval_golden_graph.py`（手工构造黄金图，绕过抽取噪声）。
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from pipeline.build_pipeline import BuildPipeline  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from utils.config import Config  # noqa: E402

# gold 问题集：(问题, 答案 doc_id, 答案关键词)。答案 doc 均在 Top-1（已验证）。
GOLD = [
    ("星辰科技的创始人是谁？", "doc_star_001", "林远"),
    ("星辰科技获得哪家机构的投资？", "doc_star_001", "云峰资本"),
    ("星图平台的客户华西医院位于哪个城市？", "doc_star_002", "成都"),
    ("HippoRAG2 是什么的升级版本？", "doc_kg_001", "HippoRAG"),
    ("知识库自演进是指什么？", "doc_evo_001", "自演进"),
]


@pytest.fixture(scope="module")
def workdir():
    tmp = Path(tempfile.mkdtemp(prefix="kb_quality_"))
    (tmp / "raw").mkdir()
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "sample_docs.jsonl", tmp / "raw" / "sample_docs.jsonl")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def cfg(workdir):
    return Config.load(overrides={
        "paths": {"raw_dir": str(workdir / "raw"), "processed_dir": str(workdir / "processed"),
                  "logs_dir": str(workdir / "logs")},
        "embedding": {"provider": "mock", "dim": 256, "cache_enable": False},
        "llm": {"provider": "mock"},
    })


@pytest.fixture(scope="module")
def pipeline(cfg):
    pipe = BuildPipeline(cfg)
    pipe.run(incremental=False)
    return pipe


@pytest.fixture(scope="module")
def retriever(pipeline, cfg):
    return Retriever(
        cfg, pipeline.graph, pipeline.vdb, pipeline.embedder,
        pipeline.llm, pipeline.prompts, graph_path=pipeline.graph_path,
    )


# ---------------------------------------------------------------------------
# 1. 召回质量
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,doc_id,answer_term", GOLD, ids=[g[1] for g in GOLD])
def test_answer_doc_is_top1(retriever, question, doc_id, answer_term):
    result = retriever.search(question, top_k=5)
    assert result.chunks, f"{question} 未召回任何 chunk"
    assert result.chunks[0]["doc_id"] == doc_id, (
        f"{question} Top-1 期望 {doc_id}，实际 {[(c['rank'], c['doc_id']) for c in result.chunks]}"
    )
    assert answer_term in result.chunks[0]["text"], f"Top-1 chunk 应包含答案 {answer_term}"


def test_recall_at_5_covers_all_gold_docs(retriever):
    """5 篇文档语料下，每问的 Top-5 应覆盖答案 doc；整体 Recall@5 = 100%。"""
    hits = 0
    for question, doc_id, _ in GOLD:
        docs = {c["doc_id"] for c in retriever.search(question, top_k=5).chunks}
        hits += doc_id in docs
    assert hits == len(GOLD), f"Recall@5 = {hits}/{len(GOLD)}"


# ---------------------------------------------------------------------------
# 2. 排序结构
# ---------------------------------------------------------------------------
def test_ranking_structure(retriever):
    result = retriever.search("星辰科技的创始人是谁？", top_k=5)
    ranks = [c["rank"] for c in result.chunks]
    scores = [c["score"] for c in result.chunks]
    docs = [c["doc_id"] for c in result.chunks]
    assert ranks == list(range(1, len(result.chunks) + 1)), "rank 应从 1 连续递增"
    assert scores == sorted(scores, reverse=True), "分数应单调不增"
    assert len(docs) == len(set(docs)), "同一 doc 只应保留最高分 chunk（文档级去重）"
    assert all(c["source"] in {"vector", "triple", "both", "hop"} for c in result.chunks)
    assert len(result.chunks) <= 5


def test_top_k_parameter_respected(retriever):
    assert len(retriever.search("星辰科技", top_k=2).chunks) <= 2
    assert len(retriever.search("星辰科技", top_k=3).chunks) <= 3


# ---------------------------------------------------------------------------
# 3. 可复现性
# ---------------------------------------------------------------------------
def test_search_is_deterministic(retriever):
    a = retriever.search("星图平台的客户华西医院位于哪个城市？", top_k=5)
    b = retriever.search("星图平台的客户华西医院位于哪个城市？", top_k=5)
    assert [(c["chunk_id"], c["score"]) for c in a.chunks] == [(c["chunk_id"], c["score"]) for c in b.chunks]
    assert a.triples == b.triples


# ---------------------------------------------------------------------------
# 4. 链路诊断与子图
# ---------------------------------------------------------------------------
def test_diagnostic_fields_consistent(retriever):
    result = retriever.search("星辰科技的创始人是谁？", top_k=3)
    # _full_rank 是 PPR 前完整候选排序，Top-K 必须是它的前缀子集
    full_ids = [cid for cid, _ in result._full_rank]
    assert set(c["chunk_id"] for c in result.chunks) <= set(full_ids)
    assert result._full_rank == sorted(result._full_rank, key=lambda x: -x[1])
    # 每个召回 chunk 都能归因到 seed 来源
    assert all(cid in result._seed_chunks for cid in result._seed_chunks)
    # dual_seed 路径统计完整
    assert {"seed_chunks", "selected_entities", "hop_chunks", "ppr_nodes"} <= set(result._dual_stats)
    assert result._dual_stats["seed_chunks"] >= 1
    # 计时字段齐全（性能回归基线）
    assert {"embed", "recall", "locate", "hop", "ppr", "total"} <= set(result.timing)


def test_subgraph_structure(retriever):
    result = retriever.search("星辰科技的创始人是谁？", top_k=5)
    sub = result.subgraph
    assert sub["mode"] == "trace"
    nodes = {n["id"]: n for n in sub["nodes"]}
    chunk_nodes = [n for n in sub["nodes"] if n.get("type") == "chunk"]
    assert chunk_nodes, "子图应包含 Top-K chunk 节点"
    for n in chunk_nodes:
        assert n["doc_id"] and n["text"], "chunk 节点需带 doc_id/text（可视化与溯源依赖）"
    # 边不得悬空
    for e in sub["edges"]:
        assert e["source"] in nodes and e["target"] in nodes, f"悬空边: {e}"
        assert e["type"] in {"triple", "text_conclude", "chunk_hop"}


def test_graph_version_reported(retriever, pipeline):
    result = retriever.search("星辰科技", top_k=3)
    assert result.graph_version == pipeline.graph.version
