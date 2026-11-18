# 设计稿：统一索引与检索（门面统一）+ 科研入组 SQLite Demo

- 日期：2026-11-18
- 分支：`feature/unified-index-retrieval`
- 状态：已审批（审批补充：交付文档须对两个问题做描述，并附示例、图结构或数据存储内容/结构展示）

## 1. 目标与边界

在不合并两套底层索引的前提下，为 `knowledge_self_evolution` 提供**统一查询入口（门面）**，覆盖两个场景：

| 场景 | 引擎 | 物理索引 | 查询形态 |
|---|---|---|---|
| 场景 1：知识库多跳问答 | 现有 `retrieval.Retriever`（ChunkHop 双路+扩散+PPR+LLM 过滤） | NetworkX 图 + FAISS 三库（entity/triple/chunk） | 自然语言 → 证据子图 + 排序 chunk |
| 场景 2：科研入组筛查 | 新增 `cohort.CohortEngine`（**demo 级**） | SQLite 关系库（5 表） | 入组条件 IR → 每患者 PASS/FAIL/UNKNOWN |

**边界（明确不做）**：
- 不做物理索引合并（不把患者图塞进 NetworkX，也不把 chunk 图搬进 SQLite）。
- 科研入组只保证基本 demo：不实现完整 CQL、完整 Allen 时间代数、LLM 粗抽取/细筛两遍链路。
- 种子数据为**已抽取好的结构化行**（模拟 LLM 抽取产物），不跑 openie。

## 2. 总体架构

```
统一查询入口  UnifiedEngine.search(query, query_type=None, **params)
  → Router（query_type 显式优先；否则关键词规则）
  → EngineRegistry
      ├─ KBEngineAdapter（包装现有 retrieval.Retriever） → KBResult
      └─ CohortEngine（SQLite + IR→SQL 编译）           → CohortResult
  → UnifiedResult{query, engine, payload, meta}
```

统一的是：入口协议、结果信封、注册表、路由、config、CLI。
物理索引保持双轨（KB=图+FAISS；cohort=SQLite）。

## 3. 新增模块

```
unified/
  engine.py     # BaseEngine 协议 + EngineRegistry + UnifiedEngine
  router.py     # 规则路由：含 入组/筛查/符合/年龄/患者/用药 → cohort，否则 KB
  result.py     # UnifiedResult / KBResult / CohortResult 信封
  demo.py       # 同一条查询入口分别跑 KB 与 cohort 各一例
cohort/
  schema.sql    # patients / diagnoses / labs / medications / events 五表
  db.py         # SQLite 打开、建表、载种子
  seed.py       # 合成"已抽取好"的结构化患者数据（30 例）
  query_ir.py   # 入组条件 IR（include/exclude；age/diagnosis/lab/medication/event）
  compiler.py   # IR → SQL：include AND 求交、exclude NOT EXISTS 否决、缺字段→UNKNOWN
  engine.py     # CohortEngine.search() → decisions + 命中明细 + 生成 SQL
  README.md     # 表结构 / IR 格式 / demo 命令
config/default.yaml   # 新增 [unified, cohort] 段（sqlite 路径等）
tests/test_cohort_engine.py
tests/test_unified_facade.py
docs/scenarios.md                    # 文档 A：两个场景问题表述（含示例与结构展示）
docs/storage_format_scenarios.md     # 文档 B：场景 ↔ 存储格式选型（四层框架，含示例）
```

## 4. cohort demo 引擎设计

### 4.1 表结构（schema.sql）

```sql
CREATE TABLE patients(
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, birth_date TEXT NOT NULL, sex TEXT);

CREATE TABLE diagnoses(
  id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id),
  disease TEXT NOT NULL, disease_std TEXT NOT NULL,          -- 标准名（模拟已细筛）
  status TEXT NOT NULL,                                      -- 确诊/病理确诊/疑似/否定/待排
  diagnosed_at TEXT);                                        -- 可空 → UNKNOWN 来源

CREATE TABLE labs(
  id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id),
  item TEXT NOT NULL, value REAL, unit TEXT, sampled_at TEXT);  -- value 可空 → UNKNOWN 来源

CREATE TABLE medications(
  id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id),
  drug TEXT NOT NULL, status TEXT NOT NULL,                  -- 已用/计划/停用/否认
  started_at TEXT, ended_at TEXT);

CREATE TABLE events(
  id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id),
  event_type TEXT NOT NULL, event_name TEXT NOT NULL, event_at TEXT);
```

### 4.2 入组条件 IR（query_ir.py）

```json
{
  "include": [
    {"type": "age",        "min": 18, "max": 75},
    {"type": "diagnosis",  "disease_std": "肝细胞癌", "status": "确诊"},
    {"type": "lab",        "item": "AFP", "op": ">", "value": 400},
    {"type": "event",      "event_type": "surgery", "event_name": "肝切除",
     "window_start": "2024-01-01", "window_end": "2024-12-31"}
  ],
  "exclude": [
    {"type": "medication", "drug": "索拉非尼", "status": "已用"}
  ]
}
```

### 4.3 编译语义（compiler.py）

- **include 条件之间为 AND**（各自子查询求交；`EXISTS` 子查询）。
- **exclude 为否决**（`NOT EXISTS`）。
- **age**：`(julianday(:ref) - julianday(p.birth_date)) / 365.25` 与 `[min,max]` 比较。
- **lab**：`EXISTS(SELECT 1 FROM labs ... WHERE item=:i AND value :op :v)`；**无 value 行不满足**。
- **UNKNOWN 语义**：决策分三级
  - 显式满足/不满足 → `PASS` / `FAIL`；
  - 患者存在**关键字段缺失**（如缺 `diagnosed_at`、缺 lab `value`、用药状态无法确认）→ 该条件不得判定为满足，患者标记 `UNKNOWN`（**不自动纳入**），并在明细中列出缺失原因。
- 决策输出：每患者 `{patient_id, eligibility: PASS|FAIL|UNKNOWN, matched_conditions, missing_reasons, sql}`。

### 4.4 种子数据（seed.py）

- 30 名合成患者，直接生成结构化行（模拟已由 LLM 抽取的产物）。
- 覆盖路径：合格者（全条件命中）、年龄不符、诊断不符、exclude 用药者、缺 lab 值（→UNKNOWN）、缺诊断时间（→UNKNOWN）。

## 5. 统一门面设计

- `BaseEngine`：`search(query, **params) -> EngineResult`；`describe() -> {name, query_types, index_types}`。
- `EngineRegistry.register(engine)` / `get(name)`；KB 适配器包装现有 `Retriever`（签名 `search(query, top_k, hops, subgraph_mode) -> SearchResult`），payload 原样透传 chunks/triples/subgraph。
- `Router.route(query, query_type)`：`query_type` 显式优先；否则关键词命中（入组/筛查/符合/年龄/患者/用药/排除 → cohort；其余 → KB）。
- `UnifiedResult`：`{query, engine, payload, meta{latency_ms, engine_version, schema_version}}`。

## 6. CLI / 演示

```bash
python -m cohort.seed --count 30      # 生成 data/cohort/demo.db
python -m unified.demo                 # 同一入口：一条 KB 查询 + 一条入组查询
python -m pytest tests/ -q             # 原 9 用例 + 新增全部绿
```

## 7. 交付文档要求（含审批补充）

### 文档 A `docs/scenarios.md`：两个场景的问题表述

对两个场景分别描述：数据形态、查询形态、评价指标、难点，并**必须附**：
- 具体示例查询 + 输入数据片段 + 输出；
- 知识库场景：图结构展示（实体/chunk/三元组/CHUNK_HOP 边的 ASCII 子图示例）；
- 入组场景：关系表结构与示例行（含 PASS/FAIL/UNKNOWN 三例）。

### 文档 B `docs/storage_format_scenarios.md`：场景 ↔ 存储格式选型

以四层（查询语言 / 逻辑数据模型 / 物理执行引擎 / 物理存储结构）为骨架，**必须附**：
- 每层的概念示例（SQL 示例 vs Cypher 示例；行存/列存 DDL；邻接表/CSR 结构示例）；
- 关系 vs 图 分维度对比表；
- 场景→选型结论表（属性过滤/聚合/批量关联 → SQL；多跳/路径/邻居扩展 → 图引擎；混合 → 统一门面+双引擎，即本项目落地）；
- 跨层组合说明（SQL 连图、图引擎兼属性过滤、向量库为补充维度）。

## 8. 分支内提交顺序

1. 本设计稿 `docs/superpowers/specs/2026-11-18-unified-index-retrieval-design.md`
2. cohort 引擎 + unified 门面 + config + 测试
3. 文档 A、文档 B
4. 测试全绿后收尾

## 9. 测试验收

- `tests/test_cohort_engine.py`：年龄边界、include 求交、exclude 否决、UNKNOWN（缺值/缺时间）、生成 SQL 可执行。
- `tests/test_unified_facade.py`：注册表路由、显式 query_type、关键词路由、信封结构、KB 适配器（mock Retriever）。
- 回归：`tests/test_smoke.py` 原 9 用例仍绿。
