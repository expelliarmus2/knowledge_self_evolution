"""科研入组 demo 引擎测试（SQLite，全部离线）。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from cohort.engine import CohortEngine  # noqa: E402
from cohort.query_ir import QueryIR  # noqa: E402
from cohort.seed import DEMO_IR, seed_database  # noqa: E402

REF = "2026-11-18"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    db = tmp_path_factory.mktemp("cohort") / "demo.db"
    seed_database(db, count=30)
    return CohortEngine(db)


def _by_id(result, patient_id: int) -> dict:
    return next(d for d in result.decisions if d["patient_id"] == patient_id)


def _eligibility_map(result) -> dict:
    return {d["patient_id"]: d["eligibility"] for d in result.decisions}


# ---------------------------------------------------------------------------
# 原型覆盖（相对 DEMO_IR：18-75 岁、肝细胞癌确诊 2020-2025、AFP>400、排除索拉非尼）
# ---------------------------------------------------------------------------
def test_archetype_pass(engine):
    d = _by_id(engine.search(DEMO_IR), 1)
    assert d["name"] == "P001-合格"
    assert d["eligibility"] == "PASS"
    assert d["missing"] == []


def test_archetype_age_fail(engine):
    d = _by_id(engine.search(DEMO_IR), 2)
    assert d["eligibility"] == "FAIL"
    assert d["conditions"][0] == {"group": "include", "index": 0, "type": "age", "status": "FAIL"}


def test_archetype_diagnosis_fail(engine):
    d = _by_id(engine.search(DEMO_IR), 3)
    assert d["eligibility"] == "FAIL"
    assert d["conditions"][1]["status"] == "FAIL"  # diagnosis 条件


def test_archetype_exclude_veto(engine):
    d = _by_id(engine.search(DEMO_IR), 4)
    assert d["eligibility"] == "FAIL"  # exclude 触发 → 否决
    assert d["conditions"][3]["group"] == "exclude"
    assert d["conditions"][3]["status"] == "TRIGGERED"


def test_archetype_unknown_missing_lab(engine):
    d = _by_id(engine.search(DEMO_IR), 5)
    assert d["eligibility"] == "UNKNOWN"
    assert "include[2].lab" in d["missing"]


def test_archetype_unknown_missing_diagnosis_time(engine):
    d = _by_id(engine.search(DEMO_IR), 6)
    assert d["eligibility"] == "UNKNOWN"
    assert "include[1].diagnosis" in d["missing"]


def test_summary_shape(engine):
    s = engine.search(DEMO_IR).summary
    assert s["total"] == 30
    assert s["PASS"] >= 1 and s["FAIL"] >= 1 and s["UNKNOWN"] >= 1


# ---------------------------------------------------------------------------
# 编译语义
# ---------------------------------------------------------------------------
def test_age_boundary(engine):
    # min=56 → P001(56.x) PASS；min=57 → P001 FAIL
    ir = QueryIR.from_dict({"include": [{"type": "age", "min": 57}], "reference_date": REF})
    assert _by_id(engine.search(ir), 1)["eligibility"] == "FAIL"


def test_include_intersection(engine):
    # 只有年龄条件：所有 1960-1999 填充患者 + 原型 → 多数 PASS；加 AFP 条件后收窄
    ir_age = QueryIR.from_dict({"include": [{"type": "age", "min": 18, "max": 75}], "reference_date": REF})
    ir_full = QueryIR.from_dict(DEMO_IR)
    n_age = sum(1 for d in engine.search(ir_age).decisions if d["eligibility"] == "PASS")
    n_full = sum(1 for d in engine.search(ir_full).decisions if d["eligibility"] == "PASS")
    assert n_age > n_full


def test_no_exclude_no_unknown(engine):
    # 去掉 exclude 后，P004（已用索拉非尼）不再被否决，但其余条件全中 → PASS
    ir = dict(DEMO_IR)
    ir["exclude"] = []
    d = _by_id(engine.search(ir), 4)
    assert d["eligibility"] == "PASS"


def test_exclude_unknown(engine):
    # 用药状态为空的患者（填充 i%7==0 且 i%? 无 null status；构造显式用例）
    ir = QueryIR.from_dict({
        "include": [{"type": "age", "min": 18, "max": 75}],
        "exclude": [{"type": "medication", "drug": "索拉非尼"}],
        "reference_date": REF,
    })
    d = _by_id(engine.search(ir), 1)  # P001 无用药记录 → NOT_TRIGGERED → PASS
    assert d["eligibility"] == "PASS"
    assert d["conditions"][1]["status"] == "NOT_TRIGGERED"


def test_generated_sql_executes(engine):
    result = engine.search(DEMO_IR)
    assert "EXISTS" in result.sql
    assert "in0_pass" in result.sql and "in2_unknown" in result.sql
    assert result.params["ref"] == REF
    # SQL 已被 engine 成功执行（decisions 非空即证明可执行）
    assert len(result.decisions) == 30


def test_ir_validation():
    with pytest.raises(ValueError, match="include"):
        QueryIR.from_dict({"include": []})
    with pytest.raises(ValueError, match="op"):
        QueryIR.from_dict({"include": [{"type": "lab", "item": "AFP", "op": "~", "value": 1}]})
    with pytest.raises(ValueError, match="disease_std"):
        QueryIR.from_dict({"include": [{"type": "diagnosis"}]})
    with pytest.raises(ValueError, match="类型"):
        QueryIR.from_dict({"include": [{"type": "unknown_cond"}]})


def test_engine_requires_structured_ir(engine):
    with pytest.raises(ValueError, match="IR"):
        engine.search("自然语言入组条件……")
