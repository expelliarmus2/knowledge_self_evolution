# 科研入组筛查 demo 引擎（SQLite）

面向"结构化患者事实 + 确定性条件判定"的**基础 demo**：把自然语言入组条件写成结构化
IR，编译成 SQL，对每个患者返回 `PASS / FAIL / UNKNOWN` 三级判定。

> 物理索引 = SQLite 关系库（5 表）。年龄/数值/时间属性筛选在关系模型下更直接，
> 这是与"知识库多跳问答（图 + 向量）"并行的第二个索引，二者通过 `unified/` 统一门面共用入口。

## 运行

```bash
# 1. 生成 demo 库（合成 30 名"已抽取好"的结构化患者）
python -m cohort.seed --count 30

# 2. 统一门面 demo（自动建库 + 同一入口跑两条查询）
python -m unified.demo --json
python -m unified.demo                # 人读输出
```

## 表结构（cohort/schema.sql）

| 表 | 关键字段 | 说明 |
|---|---|---|
| `patients` | id, name, birth_date, sex | 患者主表（年龄由 birth_date 计算） |
| `diagnoses` | patient_id, disease, disease_std, status, diagnosed_at | 诊断；`diagnosed_at` 可空 → UNKNOWN |
| `labs` | patient_id, item, value, unit, sampled_at | 检验；`value` 可空 → UNKNOWN |
| `medications` | patient_id, drug, status, started_at, ended_at | 用药；`status` 可空 → UNKNOWN |
| `events` | patient_id, event_type, event_name, event_at | 事件；`event_at` 可空 → UNKNOWN |

种子数据为**合成**（`cohort/seed.py`），直接生成结构化行，模拟"LLM 粗抽取 + 细筛"后的产物，
覆盖：合格 / 年龄不符 / 诊断不符 / 排除用药 / 缺检查值（UNKNOWN）/ 缺诊断时间（UNKNOWN）。

## 入组条件 IR

```json
{
  "reference_date": "2026-11-18",
  "include": [
    {"type": "age",       "min": 18, "max": 75},
    {"type": "diagnosis", "disease_std": "肝细胞癌", "status": "确诊",
     "window_start": "2020-01-01", "window_end": "2025-12-31"},
    {"type": "lab",       "item": "AFP", "op": ">", "value": 400},
    {"type": "event",     "event_type": "surgery", "event_name": "肝切除",
     "window_start": "2024-01-01", "window_end": "2024-12-31"}
  ],
  "exclude": [
    {"type": "medication", "drug": "索拉非尼", "status": "已用"}
  ]
}
```

条件类型：`age` / `diagnosis` / `lab` / `medication` / `event`。
`include` 之间为 AND 求交；`exclude` 为否决（触发即 FAIL）。

## 判定语义（PASS / FAIL / UNKNOWN）

| 条件状态 | 含义 |
|---|---|
| `PASS` | 数据满足条件 |
| `FAIL` | 数据明确不满足 |
| `UNKNOWN` | 关键字段缺失，**不自动纳入**（缺 lab.value / 缺诊断或事件时间 / 用药状态无法确认） |

判定优先级：exclude 触发（否决）> include FAIL > include UNKNOWN > exclude UNKNOWN。

## 模块

| 文件 | 职责 |
|---|---|
| `schema.sql` | 五表 DDL |
| `db.py` | SQLite 连接 / 建表 / 种子载入 |
| `seed.py` | 合成结构化患者数据 + `DEMO_IR` + CLI |
| `query_ir.py` | IR 数据结构与校验 |
| `compiler.py` | IR → SQL 编译 + 逐患者决策 |
| `engine.py` | `CohortEngine`（统一门面 BaseEngine 实现） |

## 测试

```bash
python -m pytest tests/test_cohort_engine.py -q     # 13 用例：原型覆盖 + 编译语义 + IR 校验
```
