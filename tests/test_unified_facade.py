"""统一门面测试：注册表 / 路由 / 信封 / KB 适配器（全部 mock，离线）。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from cohort.engine import CohortEngine  # noqa: E402
from cohort.seed import seed_database  # noqa: E402
from unified.engine import EngineRegistry, KBEngineAdapter, UnifiedEngine  # noqa: E402
from unified.router import Router  # noqa: E402


class FakeKBResult:
    def __init__(self, chunks):
        self.chunks = chunks

    def to_dict(self):
        return {"chunks": self.chunks}


class FakeRetriever:
    """模拟现有 retrieval.Retriever（只实现 search）。"""

    def __init__(self):
        self.calls = []

    def search(self, query, top_k=None, hops=None, subgraph_mode=None):
        self.calls.append((query, top_k, hops, subgraph_mode))
        return FakeKBResult([{"rank": 1, "chunk_id": "c1", "score": 0.9, "source": "vector", "hop": 0}])


class FakeCohortEngine(CohortEngine):
    name = "cohort"


@pytest.fixture()
def registry(tmp_path):
    db = tmp_path / "demo.db"
    seed_database(db, count=6)
    reg = EngineRegistry()
    reg.register(FakeCohortEngine(db))
    reg.register(KBEngineAdapter(FakeRetriever(), name="kb"))
    return reg


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
def test_registry_register_get(registry):
    assert set(registry.names()) == {"kb", "cohort"}
    assert registry.get("cohort").describe()["index_types"] == ["sqlite"]
    with pytest.raises(KeyError):
        registry.get("nope")


def test_registry_requires_name():
    reg = EngineRegistry()
    with pytest.raises(ValueError, match="name"):
        reg.register(object())


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
def test_explicit_query_type(registry):
    router = Router(registry)
    assert router.route("随便什么", query_type="cohort") == "cohort"
    with pytest.raises(KeyError):
        router.route("x", query_type="nope")


def test_keyword_routes_to_cohort(registry):
    router = Router(registry)
    assert router.route("纳入年龄 18-75 的肝细胞癌患者入组") == "cohort"
    assert router.route("受试者用药排除标准") == "cohort"


def test_default_routes_to_kb(registry):
    router = Router(registry)
    assert router.route("星辰科技的创始人是谁？") == "kb"


def test_dict_query_routes_to_cohort(registry):
    router = Router(registry)
    assert router.route({"include": [{"type": "age", "min": 18}]}) == "cohort"


def test_router_default_engine_fallback(registry):
    kb_only = EngineRegistry()
    kb_only.register(KBEngineAdapter(FakeRetriever(), name="kb"))
    router = Router(kb_only, default_engine="kb")
    assert router.route("入组年龄筛选") == "kb"  # cohort 未注册 → 回退 kb


# ---------------------------------------------------------------------------
# 统一入口 + 信封
# ---------------------------------------------------------------------------
def test_unified_search_cohort(registry):
    u = UnifiedEngine(registry)
    result = u.search({"include": [{"type": "age", "min": 18, "max": 75}],
                       "exclude": [], "reference_date": "2026-11-18"})
    assert result.engine == "cohort"
    assert result.payload["summary"]["total"] == 6
    assert "latency_ms" in result.meta
    assert result.meta["engine"]["index_types"] == ["sqlite"]


def test_unified_search_kb_adapter(registry):
    u = UnifiedEngine(registry)
    result = u.search("星辰科技")
    assert result.engine == "kb"
    assert result.payload["chunks"][0]["chunk_id"] == "c1"


def test_unified_result_to_dict(registry):
    u = UnifiedEngine(registry)
    result = u.search("星辰科技")
    d = result.to_dict()
    assert d["engine"] == "kb"
    assert isinstance(d["query"], str)
    assert d["payload"]["chunks"][0]["chunk_id"] == "c1"


def test_kb_adapter_passes_params(registry):
    retriever = registry.get("kb").retriever
    registry.get("kb").search("q", top_k=7, hops=2)
    assert retriever.calls[-1] == ("q", 7, 2, None)
