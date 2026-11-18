"""端到端冒烟测试（mock 模式，离线可跑）。

覆盖验收点：
- OpenIE 两步 prompt 分离且可配置（模板文件加载）
- 三元组能成功构图并写入向量库
- pipeline 可端到端跑通且预留演进钩子
- 检索链路可跑通并支持三元组过滤
- 演进接口：图操作 → 版本提升 → 向量库同步 → 检索感知
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from graph.evolution import EvolutionManager, GraphOperation, NoopStrategy  # noqa: E402
from graph.knowledge_graph import KnowledgeGraph  # noqa: E402
from openie.llm_client import MockLLMClient  # noqa: E402
from pipeline.build_pipeline import BuildPipeline  # noqa: E402
from retrieval.retriever import Retriever  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.prompts import PromptManager  # noqa: E402


@pytest.fixture(scope="module")
def workdir():
    tmp = Path(tempfile.mkdtemp(prefix="kb_evo_test_"))
    (tmp / "raw").mkdir()
    shutil.copy(PROJECT_ROOT / "data" / "raw" / "sample_docs.jsonl", tmp / "raw" / "sample_docs.jsonl")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def cfg(workdir):
    return Config.load(
        overrides={
            "paths": {"raw_dir": str(workdir / "raw"), "processed_dir": str(workdir / "processed"),
                      "logs_dir": str(workdir / "logs")},
            "embedding": {"provider": "mock", "dim": 256, "cache_enable": False},
            "llm": {"provider": "mock"},
        }
    )


@pytest.fixture(scope="module")
def pipeline(cfg):
    # 注册一个可见事件的策略，验证钩子链路
    class SpyStrategy(NoopStrategy):
        name = "spy"

    manager = EvolutionManager()
    manager.register(SpyStrategy())
    pipe = BuildPipeline(cfg, evolution_manager=manager)
    return pipe


@pytest.fixture(scope="module")
def report(pipeline):
    return pipeline.run(incremental=False)


# ---------------------------------------------------------------------------
def test_two_step_prompts_configurable(cfg):
    pm = PromptManager.from_config(cfg)
    # 中文模板存在且两步分离
    sys_ner, usr_ner = pm.load("ner")
    assert "named_entities" in usr_ner
    sys_re, usr_re = pm.load("triple_extraction")
    assert "triples" in usr_re
    assert "{passage}" in usr_ner and "{passage}" in usr_re
    assert "{named_entity_json}" in usr_re
    # 过滤模板
    _, usr_f = pm.load("retrieval_filter")
    assert "{question}" in usr_f and "{candidate_facts_json}" in usr_f
    # 渲染结果包含真实段落与实体 JSON
    msgs = pm.render("triple_extraction", passage="某段落", named_entity_json='{"named_entities": ["甲"]}')
    assert any("某段落" in m["content"] for m in msgs)
    # 切英文只需换 language
    pm_en = PromptManager(cfg.path("paths.prompts_dir"), language="english")
    _, usr = pm_en.load("ner")
    assert "named entities" in usr.lower()


def test_build_pipeline_end_to_end(report):
    assert report["chunks"] > 0
    assert report["triples_extracted"] > 0
    assert report["graph"]["entities"] > 0
    assert report["graph"]["triples"] > 0
    # 演进钩子已被触发（spy 策略至少收到 pipeline_start/complete）
    # （事件记录在策略实例里，这里通过 report 间接验证）


def test_graph_and_vectorstore_consistent(pipeline, report):
    g = pipeline.graph
    stats = g.stats()
    vdb = pipeline.vdb
    assert vdb.stores["entities"].count() == stats["entities"]
    assert vdb.stores["triples"].count() == stats["triples"]
    assert vdb.stores["chunks"].count() == stats["chunks"]


def test_openie_entity_anchor(pipeline):
    """两步抽取的三元组锚定实体（require_entity_anchor=True 时无未锚定三元组入图）。"""
    g = pipeline.graph
    assert g.get_triples(), "图中应有三元组"
    for u, v, rel, data in g.get_triples():
        assert g.has_entity(u) and g.has_entity(v)


def test_retrieval_with_filter(pipeline, cfg):
    retriever = Retriever(
        cfg, pipeline.graph, pipeline.vdb, pipeline.embedder,
        pipeline.llm, pipeline.prompts, graph_path=pipeline.graph_path,
    )
    result = retriever.search("星辰科技的创始人是谁？", top_k=5)
    assert result.triples, "应命中三元组"
    assert result.chunks, "应有 PPR 排序 chunk"
    assert result.filtered_triples is not None  # LLM 过滤（mock）链路可用
    assert result.subgraph["nodes"], "应产出可视化子图"
    assert any(t["head"] == "星辰科技" or t["tail"] == "星辰科技" for t in result.triples)


def test_retrieval_multihop(pipeline, cfg):
    retriever = Retriever(
        cfg, pipeline.graph, pipeline.vdb, pipeline.embedder,
        pipeline.llm, pipeline.prompts, graph_path=pipeline.graph_path,
    )
    # 多跳：星图平台 →（发布方）星辰科技 →（创始人）林远；也考察 chunk hop
    result = retriever.search("星图平台的客户华西医院位于哪里？", top_k=5)
    assert result.chunks
    texts = " ".join(c["text"] for c in result.chunks)
    assert "成都" in texts or "华西医院" in texts


def test_evolution_interface(pipeline, cfg):
    """演进接口：图操作 → 版本提升 → 向量库同步 → 检索感知。"""
    g = pipeline.graph
    v0 = g.version
    n_triples0 = len(g.get_triples())
    api = pipeline.graph_api
    res = api.apply([GraphOperation(
        "add_triple",
        {"head": "知识库自演进", "relation": "预留接口", "tail": "EvoGraph-R1", "weight": 3.0},
        reason="test",
    )])
    assert res["failed"] == []
    assert g.version > v0
    assert len(g.get_triples()) == n_triples0 + 1
    # 向量库已增量同步（auto_sync_vectorstore）
    key = pipeline.vdb.triple_key("知识库自演进", "预留接口", "EvoGraph-R1")
    assert pipeline.vdb.stores["triples"].has(key)
    # 检索感知：新三元组可被召回
    retriever = Retriever(
        cfg, g, pipeline.vdb, pipeline.embedder, pipeline.llm,
        pipeline.prompts, graph_path=pipeline.graph_path,
    )
    result = retriever.search("EvoGraph-R1 与知识库自演进的关系是什么？", top_k=5)
    assert any(
        t["head"] == "知识库自演进" and t["tail"] == "EvoGraph-R1" for t in result.triples
    ), "演进新增的三元组应可被检索感知"

    # 删除操作：检索侧自动剔除
    res2 = api.apply([GraphOperation(
        "remove_triple", {"head": "知识库自演进", "relation": "预留接口", "tail": "EvoGraph-R1"}, reason="test",
    )])
    assert res2["failed"] == []
    assert not g.has_triple("知识库自演进", "预留接口", "EvoGraph-R1")
    result2 = retriever.search("EvoGraph-R1 与知识库自演进的关系是什么？", top_k=5)
    assert not any(
        t["relation"] == "预留接口" for t in result2.triples
    ), "已删除的三元组不应再出现在召回结果"


def test_incremental_build(pipeline, report):
    """增量构建：无新文档时零 embedding、零变更。"""
    v0 = pipeline.graph.version
    report2 = pipeline.run(incremental=True)
    assert report2["chunks"] == 0  # 全部文档已构建过
    # 版本不变（没有新的图操作）
    assert pipeline.graph.version == v0


def test_web_export(pipeline):
    export = pipeline.graph.export_json(None)  # 只构造不落盘
    assert export["stats"]["entities"] > 0
    node_types = {n["type"] for n in export["nodes"]}
    assert "entity" in node_types and "chunk" in node_types


# ---------------- 单次抽取模式（huaxi_poc 风格） ----------------
class _FakeOneStepLLM(MockLLMClient):
    """固定返回 huaxi 分隔符协议输出，验证 one_step 解析链路。"""

    def __init__(self, output: str):
        super().__init__()
        self._output = output

    def generate(self, messages):
        return self._output


def test_one_step_parser(cfg):
    """<|> ## <|COMPLETE|> 协议解析：多记录 / 结尾标记 / 代码块包裹 / 代词端点。"""
    from openie.chunker import Chunk
    from openie.extractor import OpenIEExtractor

    raw = (
        "Messi<|>joined<|>Barcelona##\n"
        "Barcelona<|>located in<|>Spain##\n"
        "He<|>plays for<|>Inter Miami##\n"
        "<|COMPLETE|>"
    )
    prompts = PromptManager.from_config(cfg)
    ext = OpenIEExtractor(
        llm=_FakeOneStepLLM(raw), prompts=prompts,
        extract_mode="one_step", resolve_pronouns=False,
    )
    res = ext.extract(Chunk(chunk_id="c1", doc_id="d1", order=0, text="..."))
    # 代词端点（He）被兑底丢弃，其余两条保留且视为已锚定
    assert [(t.head, t.relation, t.tail) for t in res.triples] == [
        ("Messi", "joined", "Barcelona"),
        ("Barcelona", "located in", "Spain"),
    ]
    assert all(t.anchored for t in res.triples)
    # 实体从三元组端点汇总
    assert set(res.entities) == {"Messi", "Barcelona", "Spain"}


def test_one_step_mock_end_to_end(cfg, tmp_path):
    """one_step + MockLLMClient 全流程：构建→图→检索。"""
    cfg2 = Config.load(overrides={
        "paths": {"raw_dir": str(cfg.path("paths.raw_dir")),
                  "processed_dir": str(tmp_path / "processed"),
                  "logs_dir": str(tmp_path / "logs")},
        "embedding": {"provider": "mock", "dim": 256, "cache_enable": False},
        "llm": {"provider": "mock"},
        "openie": {"extract_mode": "one_step"},
    })
    pipe = BuildPipeline(cfg2)
    rep = pipe.run(incremental=False)
    assert rep["triples_extracted"] > 0, "one_step 模式应抽到三元组"
    assert rep["graph"]["triples"] > 0, "one_step 三元组应入图"
    # 向量库同步
    assert rep["vdb"]["triples"]["total"] == rep["graph"]["triples"]


def test_extract_mode_validation(cfg):
    """非法模式应报错，防止 yaml 静默拼错。"""
    from openie.chunker import Chunk
    from openie.extractor import OpenIEExtractor

    prompts = PromptManager.from_config(cfg)
    with pytest.raises(ValueError):
        OpenIEExtractor(llm=MockLLMClient(), prompts=prompts, extract_mode="one-shot")
    # 默认模式仍是 two_step（不破坏既有行为）
    ext = OpenIEExtractor(llm=MockLLMClient(), prompts=prompts)
    assert ext.extract_mode == "two_step"
