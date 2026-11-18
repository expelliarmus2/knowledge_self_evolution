"""多跳检索组件测试：**手工构造黄金图**，绕过 LLM 抽取噪声。

为什么需要这一层：mock 端到端（test_retrieval_quality.py / test_smoke.py）里
MockLLMClient 抽不出中文事实三元组，导致 CHUNK_HOP 扩散、三元组 n 跳扩展、
图证据回链这些**检索核心能力实际没有被验证**（hop_chunks=0）。

本文件直接 `add_chunk + add_triple` 构造一个小而确定的黄金图，
把"抽取"与"检索"解耦，逐项验证：
  1. Top-1 定位与三元组证据回链
  2. 三元组 n 跳扩展（种子实体 1 跳外的证据也能召回）
  3. CHUNK_HOP 扩散（共享实体在两个 chunk 间搭桥）
  4. 文档级去重（同 doc 多 chunk 只留最高分）
  5. 演进感知（图删边 → 向量库同步 → 检索剔除）

黄金图（3 篇文档 4 个 chunk）：

    doc_lin  c_lin : 星辰科技 -创始人-> 林远
    doc_plat c_plat: 星辰科技 -发布方-> 星图平台 -客户-> 华西医院
    doc_plat c_plat2: 星图平台 的检索能力（用于去重）
    doc_hosp c_hosp: 华西医院 -位于-> 成都
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from graph.knowledge_graph import KnowledgeGraph  # noqa: E402
from openie.embedder import MockEmbedder  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from utils.config import Config  # noqa: E402
from vectorstore.manager import VectorStoreManager  # noqa: E402

CHUNKS = {
    "c_lin": ("星辰科技的创始人是林远，林远毕业于清华大学计算机系。", "doc_lin"),
    "c_plat": ("星辰科技发布了星图平台，星图平台集成了检索能力，客户包括华西医院。", "doc_plat"),
    "c_plat2": ("星图平台集成了向量检索与图检索两种能力。", "doc_plat"),
    "c_hosp": ("华西医院位于成都，是一家大型综合医院。", "doc_hosp"),
    # 陪跑 chunk：与查询词无词面重叠，只锚定"星图平台"实体，
    # 向量路召不回、三元组定位路也进不来 → 只能靠 CHUNK_HOP 扩散发现
    "c_x1": ("星图平台的财务模块服务于银行客户，支持对账与清算。", "doc_x1"),
    "c_x2": ("星图平台的移动端版本提供消息推送与日程管理。", "doc_x2"),
    "c_x3": ("星图平台的数据报表功能在制造业中应用广泛。", "doc_x3"),
    "c_x4": ("星图平台支持多语言界面，覆盖东南亚市场。", "doc_x4"),
    "c_x5": ("星图平台的权限体系满足合规审计要求。", "doc_x5"),
}

TRIPLES = [
    ("星辰科技", "创始人", "林远", "c_lin"),
    ("星辰科技", "发布方", "星图平台", "c_plat"),
    ("星图平台", "客户", "华西医院", "c_plat"),
    ("华西医院", "位于", "成都", "c_hosp"),
]


def _build(tmp: Path, **retrieval_overrides):
    retrieval_cfg = {"vector_seed": 2}  # 收紧向量路，让陪跑 chunk 进不了向量种子
    retrieval_cfg.update(retrieval_overrides)
    cfg = Config.load(overrides={
        "paths": {"processed_dir": str(tmp / "processed"), "logs_dir": str(tmp / "logs")},
        "embedding": {"provider": "mock", "dim": 256, "cache_enable": False},
        "llm": {"provider": "mock"},
        "retrieval": retrieval_cfg,
    })
    graph = KnowledgeGraph()
    for order, (cid, (text, doc)) in enumerate(CHUNKS.items()):
        graph.add_chunk(cid, text=text, doc_id=doc, order=order)
    for head, rel, tail, cid in TRIPLES:
        graph.add_triple(head, rel, tail, chunk_id=cid)
    # 没有三元组锚定的 chunk 显式建 text_conclude 边（共享"星图平台"实体）
    for cid in ("c_plat2", "c_x1", "c_x2", "c_x3", "c_x4", "c_x5"):
        graph.link_entity_to_chunk("星图平台", cid)
    embedder = MockEmbedder(dim=256)
    vdb = VectorStoreManager(cfg, embedder, tmp / "processed")
    vdb.sync_from_graph(graph, incremental=False)
    retriever = Retriever(cfg, graph, vdb, embedder, llm=None, prompts=None)
    return graph, vdb, embedder, retriever


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("golden")
    return _build(tmp)


@pytest.fixture(scope="module")
def retriever(world):
    return world[3]


def _triple_set(result) -> set:
    return {(t["head"], t["relation"], t["tail"]) for t in result.triples}


# ---------------------------------------------------------------------------
# 1. Top-1 定位 + 三元组证据回链
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("question,expected_doc,expected_triple", [
    ("星辰科技的创始人是谁？", "doc_lin", ("星辰科技", "创始人", "林远")),
    ("华西医院位于哪个城市？", "doc_hosp", ("华西医院", "位于", "成都")),
])
def test_top1_and_triple_evidence(retriever, question, expected_doc, expected_triple):
    result = retriever.search(question, top_k=5)
    assert result.chunks[0]["doc_id"] == expected_doc
    assert expected_triple in _triple_set(result), f"应命中三元组 {expected_triple}"


# ---------------------------------------------------------------------------
# 2. 三元组 n 跳扩展：问题只提"星图平台的客户"，答案"成都"在 2 跳外
# ---------------------------------------------------------------------------
def test_multihop_triple_expansion(retriever):
    result = retriever.search("星图平台的客户位于哪里？", top_k=5)
    triples = _triple_set(result)
    # 种子：星图平台 → 客户 → 华西医院（1 跳）
    assert ("星图平台", "客户", "华西医院") in triples
    # 端到端：2 跳外的证据（华西医院 → 位于 → 成都）被捞回最终候选集
    assert ("华西医院", "位于", "成都") in triples, "2 跳外的证据应进入候选三元组"
    # 答案 chunk 也在全量候选里（实体定位兜底；top_k=9 = 全部 chunk）
    assert "doc_hosp" in {c["doc_id"] for c in retriever.search("星图平台的客户位于哪里？", top_k=9).chunks}


def test_expand_triples_bfs_and_decay(world):
    """_expand_triples 纯函数：BFS 跳数正确 + 分数逐跳衰减 + 去重。"""
    graph = world[0]
    from graph.knowledge_graph import KnowledgeGraph
    assert isinstance(graph, KnowledgeGraph)
    retriever = world[3]
    collected = retriever._expand_triples({"星图平台"}, hops=2, decay=0.5)
    keys = {(t["head"], t["relation"], t["tail"]) for t in collected}
    # 1 跳：星图平台 → 客户 → 华西医院；星辰科技 → 发布方 → 星图平台
    assert ("星图平台", "客户", "华西医院") in keys
    assert ("星辰科技", "发布方", "星图平台") in keys
    # 2 跳：华西医院 → 位于 → 成都
    assert ("华西医院", "位于", "成都") in keys
    # 逐跳衰减 + 不重复收集
    hop1 = [t for t in collected if t["expanded_hop"] == 1]
    hop2 = [t for t in collected if t["expanded_hop"] == 2]
    assert hop1 and hop2
    assert max(t["score"] for t in hop2) < min(t["score"] for t in hop1)
    seen = {(t["head"], t["relation"], t["tail"]) for t in collected}
    assert len(seen) == len(collected)


def test_expand_skips_nonexistent_seeds(world):
    retriever = world[3]
    assert retriever._expand_triples({"不存在的实体XYZ"}, hops=2, decay=0.5) == []


# ---------------------------------------------------------------------------
# 3. CHUNK_HOP 扩散：共享实体（星图平台）把种子 chunk 桥接到陪跑 chunk
#
# 为什么用独立世界：默认配置下实体定位路（_locate_chunks）会把"共享实体的所有
# chunk"都变成种子（locate_max_chunks=80），hop 没有增量目标；收紧定位上限后，
# 陪跑 chunk 只能经由 CHUNK_HOP 在线扩散（重要实体 → 共享 chunk）被发现。
# ---------------------------------------------------------------------------
def test_chunk_hop_diffusion(tmp_path):
    graph, vdb, embedder, retriever = _build(tmp_path, locate_max_chunks=1)
    result = retriever.search("星图平台有哪些检索能力？", top_k=5)
    assert result._dual_stats["hop_chunks"] >= 1, "dual_seed 应产生在线 CHUNK_HOP 边"
    assert any(h.get("hop", 0) >= 1 for h in result._hop_chunks.values())
    # 扩散到的 chunk 必须能归因到 via（共享实体所在的种子 chunk）
    for cid, info in result._hop_chunks.items():
        assert info.get("via"), f"hop chunk {cid} 缺少 via 归因"
    # 陪跑 chunk 只能经由 hop 扩散进入候选（向量路被 vector_seed=2 收紧、
    # 定位路被 locate_max_chunks=1 收紧，均不含 c_x*）
    hop_docs = set(result._hop_chunks)
    assert any(cid.startswith("c_x") for cid in hop_docs), f"扩散应发现陪跑 chunk: {hop_docs}"


# ---------------------------------------------------------------------------
# 4. 文档级去重
# ---------------------------------------------------------------------------
def test_document_level_dedup(retriever):
    result = retriever.search("星图平台", top_k=5)
    docs = [c["doc_id"] for c in result.chunks]
    assert len(docs) == len(set(docs)), f"doc_plat 有 2 个 chunk，只应保留 1 个: {docs}"


# ---------------------------------------------------------------------------
# 5. 演进感知：删边 → 向量库同步 → 检索剔除
# ---------------------------------------------------------------------------
def test_evolution_removal_is_visible(tmp_path):
    graph, vdb, embedder, retriever = _build(tmp_path)
    before = retriever.search("华西医院位于哪个城市？", top_k=5)
    assert ("华西医院", "位于", "成都") in _triple_set(before)

    assert graph.remove_triple("华西医院", "位于", "成都")
    vdb.sync_from_graph(graph, incremental=True)

    after = retriever.search("华西医院位于哪个城市？", top_k=5)
    assert ("华西医院", "位于", "成都") not in _triple_set(after), "已删除的三元组不应再被召回"
    assert graph.version > 0
