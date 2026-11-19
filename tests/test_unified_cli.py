"""统一 CLI / 路由按输入形态分流：str→RAG，QueryIR→科研入组。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cohort.query_ir import QueryIR  # noqa: E402


@pytest.fixture(scope="module")
def demo_db(tmp_path_factory):
    from cohort.seed import seed_database

    db = tmp_path_factory.mktemp("kse") / "demo.db"
    seed_database(db, count=10)
    return db


@pytest.fixture(scope="module")
def cohort_engine(demo_db):
    from cohort.engine import CohortEngine

    return CohortEngine(demo_db, seed_if_missing=False)


@pytest.fixture(scope="module")
def registry(cohort_engine):
    from unified.engine import EngineRegistry

    r = EngineRegistry()
    r.register(cohort_engine)
    return r


# ---------------------------------------------------------------------------
# 路由：QueryIR 实例 → cohort
# ---------------------------------------------------------------------------
def test_queryir_instance_routes_to_cohort(registry):
    from unified.router import Router

    ir = QueryIR(include=[{"type": "age", "min": 18, "max": 75}])
    assert Router(registry).route(ir) == "cohort"


def test_dict_ir_still_routes_to_cohort(registry):
    from unified.router import Router

    assert Router(registry).route({"include": [{"type": "age", "min": 18}]}) == "cohort"


def test_str_routes_to_default_kb(registry):
    from unified.engine import EngineRegistry
    from unified.router import Router

    # 只有 cohort 注册时，自然语言回退到 cohort（无 kb）；有 kb 时走 kb
    class FakeKB:
        name = "kb"

        def search(self, query, **params):
            return {"engine": "kb"}

        def describe(self):
            return {"name": "kb"}

    r = EngineRegistry()
    r.register(cohort_engine)
    r.register(FakeKB())
    assert Router(r).route("星辰科技的创始人是谁？") == "kb"


def test_queryir_instance_search_via_unified(registry):
    from unified.engine import UnifiedEngine

    ir = QueryIR(include=[{"type": "age", "min": 18, "max": 75}])
    result = UnifiedEngine(registry).search(ir)
    assert result.engine == "cohort"
    assert result.payload["summary"]["total"] == 10


# ---------------------------------------------------------------------------
# QueryIR.from_dict / json 往返
# ---------------------------------------------------------------------------
def test_ir_from_json_dict():
    ir = QueryIR.from_dict(json.loads(
        '{"include": [{"type": "age", "min": 18, "max": 75}],'
        ' "exclude": [{"type": "medication", "drug": "索拉非尼"}]}'))
    assert len(ir.include) == 1 and ir.include[0].type == "age"
    assert len(ir.exclude) == 1 and ir.exclude[0].get("drug") == "索拉非尼"


# ---------------------------------------------------------------------------
# CLI 冒烟（端到端 subprocess）
# ---------------------------------------------------------------------------
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("KB_PATHS__PROCESSED_DIR", None)
    return subprocess.run(
        [sys.executable, "-m", "unified.cli", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=120, env=env)


def test_cli_accepts_search_subcommand(demo_db):
    """兼容 pipeline.run search 的写法：unified.cli search --query ... 与不带前缀等价。"""
    ir = json.dumps({"include": [{"type": "age", "min": 18, "max": 75}]})
    with_prefix = _run_cli("search", "--ir", ir, "--db", str(demo_db), "--json")
    without = _run_cli("--ir", ir, "--db", str(demo_db), "--json")
    assert with_prefix.returncode == 0, with_prefix.stderr[-800:]
    assert json.loads(with_prefix.stdout)["engine"] == "cohort"
    assert json.loads(with_prefix.stdout)["payload"] == json.loads(without.stdout)["payload"]


def test_cli_bad_subcommand_rejected():
    proc = _run_cli("ask", "--query", "x")
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr


def test_cli_query_natural_language_needs_kb(demo_db):
    """自然语言 = RAG；KB 引擎缺失时明确报错提示，而不是静默跑 cohort。"""
    proc = _run_cli("--query", "星辰科技的创始人是谁？", "--mock",
                    "--processed", "/nonexistent_kb_dir")
    assert proc.returncode != 0
    assert "KB 引擎不可用" in proc.stderr
    assert "--ir" in proc.stderr  # 指向正确的入组用法


def test_cli_query_with_real_demo_kb():
    """有 KB 索引时 --query 自然语言 → engine=kb（RAG）。"""
    demo_kb = ROOT / "data" / "processed_demo_cn"
    if not (demo_kb / "graph.pkl").exists():
        pytest.skip("本地无 data/processed_demo_cn 演示库")
    proc = _run_cli("--query", "星辰科技的创始人是谁？", "--mock",
                    "--processed", str(demo_kb))
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "engine=kb" in proc.stdout
    assert "doc_star_001" in proc.stdout


def test_cli_ir_json_routes_to_cohort(demo_db):
    ir = json.dumps({
        "include": [{"type": "age", "min": 18, "max": 75},
                    {"type": "diagnosis", "disease_std": "肝细胞癌"}],
        "exclude": [{"type": "medication", "drug": "索拉非尼"}],
    }, ensure_ascii=False)
    proc = _run_cli("--ir", ir, "--db", str(demo_db))
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "engine=cohort" in proc.stdout
    assert "汇总" in proc.stdout
    assert "编译 SQL" in proc.stdout


def test_cli_ir_json_output(demo_db):
    ir = json.dumps({"include": [{"type": "age", "min": 18, "max": 75}]})
    proc = _run_cli("--ir", ir, "--db", str(demo_db), "--json")
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout)
    assert out["engine"] == "cohort"
    assert out["payload"]["summary"]["total"] == 10
    assert "compiled_sql" in out["payload"]


def test_cli_ir_from_file(tmp_path, demo_db):
    f = tmp_path / "ir.json"
    f.write_text(json.dumps({"include": [{"type": "age", "min": 18, "max": 75}]}),
                 encoding="utf-8")
    proc = _run_cli("--ir", f"@{f}", "--db", str(demo_db), "--json")
    assert proc.returncode == 0, proc.stderr[-800:]
    assert json.loads(proc.stdout)["engine"] == "cohort"


def test_cli_query_accepts_ir_json(demo_db):
    """--query 也接受 IR JSON（统一形态分流）：dict IR → cohort。"""
    ir = json.dumps({"include": [{"type": "age", "min": 18, "max": 75}]})
    proc = _run_cli("--query", ir, "--db", str(demo_db), "--json")
    assert proc.returncode == 0, proc.stderr[-800:]
    out = json.loads(proc.stdout)
    assert out["engine"] == "cohort"
    assert out["payload"]["summary"]["total"] == 10


def test_cli_query_natural_language_not_parsed_as_ir():
    """普通自然语言（非 JSON）不应被误判为入组 IR。"""
    from unified.cli import maybe_ir_from_query

    assert maybe_ir_from_query("星辰科技的创始人是谁？") is None
    assert maybe_ir_from_query("入组年龄 18-75 的患者") is None
    assert maybe_ir_from_query('{"include": [{"type": "age", "min": 18}]}') is not None


def test_cli_requires_query_or_ir():
    proc = _run_cli("--db", "/tmp/x.db")
    assert proc.returncode != 0
    assert "必须提供" in proc.stderr


def test_cli_bad_ir_json():
    proc = _run_cli("--ir", "{not json", "--db", "/tmp/x.db")
    assert proc.returncode != 0
