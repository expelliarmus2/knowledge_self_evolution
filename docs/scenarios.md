# 场景文档 A：两个检索场景的问题、图结构与存储结构

> 配套实现：`unified/`（统一门面）、`cohort/`（科研入组筛查 demo）、`retrieval/`（ChunkHop 多跳检索）。
> 本文回答两个问题：**两个场景各自要解决什么问题？各自用什么图/存储结构表达与支撑？**

---

## 0. 一句话概括

- **知识库多跳问答（ChunkHop）**：问题里藏着"关系链"，要沿着**图**把链走通，回答"谁/什么"。
- **科研入组筛查（cohort）**：问题是一堆**结构化条件**，要对**每个患者**做确定性判定（符合/不符合/数据不足），回答"哪些人"。

两个场景都叫"检索"，但**问题形态、数据结构、判定语义完全不同**——这正是"统一入口、双轨物理索引"设计的原因。

---

## 1. 场景一：知识库多跳问答（ChunkHop）

### 1.1 问题定义

输入是**自然语言问题**，答案需要**多步关联**才能得出。单文档内找不到答案，要跨越"实体 → 关系 → 实体"多次跳转：

> 例："**星辰科技的创始人是谁？**"
>
> 单步（hop0）："星辰科技" 这个实体
> 一阶（hop1）：`星辰科技 -(发布方)-> 星图平台`
> 二阶（hop2）：`星图平台 -(研发方)-> 星辰科技` 不行 → 换向
> `星辰科技 <-(创始人)- 林远`  → 答案：**林远**
>
> 跳转过程中每一步都有"候选实体 + 支撑 chunk"，跳错了（如把"星图平台"当作答案）就答错。

难点：问题短、实体名模糊（"星辰科技"可能指公司/产品）、关系方向不定、跨 hop 需要证据链回链。

### 1.2 数据形态：图结构（属性图）

该场景把知识组织成**有向属性图**（本仓库 `graph/knowledge_graph.py` 用 NetworkX DiGraph 实现）：

```text
                    ┌────────────────────────────┐
                    │  (chunk C3 "星图平台介绍")   │
                    └──────────────┬─────────────┘
                                   │ 提及
   ┌────────────┐   发布方    ┌─────▼─────┐
   │ 星辰科技   ├───────────►│  星图平台   │
   └──┬────────┘             └─────┬─────┘
      │                            │ 研发方
      │ 创始人                      │
      ▼                            ▼
   ┌────────────┐              ┌────────────┐
   │   林远      │              │  星辰智能   │
   └────────────┘              └────────────┘
```

图中的节点 = 实体（`星辰科技 / 星图平台 / 林远 / 星辰智能`），边 = 关系（`发布方 / 研发方 / 创始人`），
边上还带证据：`(head, tail, relation, chunk_id, hop)`——每一条边都能回链到支撑它的文本 chunk。

### 1.3 存储结构：三索引并行（`vectorstore/` + `graph/` + PPR 缓存）

| 索引 | 存储格式 | 用途 |
|---|---|---|
| 向量库（FAISS） | 三份浮点矩阵：entities / triples / chunks 的 embedding（HNSW/IVF 索引文件） | 语义召回：把问题 embedding 后按相似度找候选实体/chunk |
| 图（NetworkX） | `graph.pkl`（Python pickle 的邻接表内存结构） | 多跳扩展：从种子实体沿边扩展 1~2 跳，保留子图 |
| 倒排/PPR 缓存 | 每 seed 的 PageRank 分数表（字典→文件） | 全局排序：决定子图里先看谁 |

**ChunkHop 的物理执行流**（`retrieval/retriever.py`）：

```text
问题文本 ──embedding──► FAISS 召回 seed 实体/chunk（语义粗筛）
     └──► 图算子：沿边扩展 hop1/hop2，构建子图（结构扩展）
     └──► PPR 以 seed 为 personalized 起点给子图节点排序（全局打分）
     └──► 子图回链 chunk，按 (score, hop, rank) 输出
```

### 1.4 输入 / 输出示例（真实接口）

```python
result = retriever.search("星辰科技的创始人是谁？", top_k=5, hops=2)
result.triples    # 沿图找到的候选三元组
result.chunks     # 支撑 chunk，按 [score | source(ppr/vector/both) | hop] 排序
result.subgraph   # 可视化的 {nodes, edges} 子图
```

```jsonc
// result.subgraph（示意）
{
  "nodes": ["星辰科技", "星图平台", "林远"],
  "edges": [
    {"head": "星辰科技", "tail": "星图平台", "rel": "发布方", "chunk_id": "C3"},
    {"head": "星辰科技", "tail": "林远",     "rel": "创始人", "chunk_id": "C7"}
  ]
}
```

---

## 2. 场景二：科研入组筛查（cohort demo）

### 2.1 问题定义

输入是**结构化入组条件**（年龄段、疾病、检验阈值、时间窗、排除用药），
要对**每个患者**输出确定性判定：**符合（PASS）/ 不符合（FAIL）/ 数据不足（UNKNOWN）**。

> 例："招募 18–75 岁、2020–2025 年确诊肝细胞癌、AFP>400、做过肝切除手术的患者；
> 已用过索拉非尼的**排除**。"
>
> - 患者 A：56 岁、2023-03 确诊、AFP=1200、2024-05 肝切除、没用过索拉非尼 → **PASS**（入组）
> - 患者 B：86 岁，其余全符合 → **FAIL**（年龄超上限）
> - 患者 C：AFP 检验值缺失（电子病历该字段为空）→ **UNKNOWN**（不能因缺数据而错误纳入）

难点：**三值逻辑**（缺失 ≠ 不满足）、时间窗过滤（确诊时间必须在窗口内）、
排除标准是**一票否决**、每个条件必须可追溯到"哪条证据、缺了什么"。

### 2.2 数据形态：关系模型（5 张表，SQLite 单文件）

该场景把患者事实组织成**关系表**（`cohort/schema.sql`），而不是图——
因为判定是"对每行的属性筛选 + 表间 EXISTS 关联"，关系代数的集合语义刚好对应
"include 求交、exclude 求否"：

```text
patients(id, name, birth_date, sex)              ── 主表，年龄由 birth_date 派生
diagnoses(patient_id, disease, disease_std, status, diagnosed_at)
labs(patient_id, item, value, unit, sampled_at)  ── value 可空 → UNKNOWN
medications(patient_id, drug, status, started_at, ended_at)
events(patient_id, event_type, event_name, event_at)
```

物理存储 = **单个 SQLite 文件**（`data/cohort/demo.db`），内部是 B+Tree 行存。

### 2.3 示例行（真实种子数据，6 个原型患者）

**patients（节选）**

| id | name | birth_date | sex |
|---|---|---|---|
| 1 | P001-合格 | 1970-05-01 | M |
| 2 | P002-年龄不符 | 1940-01-01 | M |
| 5 | P005-缺检查值 | 1968-09-09 | F |
| 6 | P006-缺诊断时间 | 1972-11-11 | M |

**diagnoses / labs / medications（节选）**

| 表 | patient_id | 关键字段 | 含义 |
|---|---|---|---|
| diagnoses | 1 | 肝细胞癌 / 确诊 / 2023-03-01 | 时间在窗口内 → 证据满足 |
| diagnoses | 6 | 肝细胞癌 / 确诊 / **NULL** | 缺诊断时间 → include[1] UNKNOWN |
| labs | 1 | AFP / **1200.0** / 2023-03-01 | 阈值满足 |
| labs | 5 | AFP / **NULL** | 缺检验值 → include[2] UNKNOWN |
| medications | 4 | 索拉非尼 / 已用 | exclude 触发 → 一票否决 |

### 2.4 判定流程：IR → SQL → 逐患者决策

入组条件写成结构化 IR（`cohort/seed.py::DEMO_IR`，JSON）：

```json
{
  "reference_date": "2026-11-18",
  "include": [
    {"type": "age", "min": 18, "max": 75},
    {"type": "diagnosis", "disease_std": "肝细胞癌", "status": "确诊",
     "window_start": "2020-01-01", "window_end": "2025-12-31"},
    {"type": "lab", "item": "AFP", "op": ">", "value": 400},
    {"type": "event", "event_type": "surgery", "event_name": "肝切除",
     "window_start": "2024-01-01", "window_end": "2024-12-31"}
  ],
  "exclude": [{"type": "medication", "drug": "索拉非尼", "status": "已用"}]
}
```

`cohort/compiler.py` 把它编译成**一条 SELECT**（每条件产出布尔列，`cohort/engine.py` 执行）：

```sql
SELECT p.id, p.name,
  (julianday(:ref) - julianday(p.birth_date))/365.25 AS age,
  CASE WHEN EXISTS (SELECT 1 FROM diagnoses d
        WHERE d.patient_id = p.id AND d.disease_std=:in1_disease
          AND d.status=:in1_status
          AND d.diagnosed_at BETWEEN :in1_ws AND :in1_we) THEN 1 ELSE 0 END AS in1_pass,
  CASE WHEN EXISTS (SELECT 1 FROM diagnoses d
        WHERE d.patient_id = p.id AND d.disease_std=:in1_disease
          AND d.diagnosed_at IS NULL) THEN 1 ELSE 0 END AS in1_unknown,
  ...  -- labs/events/medications 同理，每条件一对 (pass, unknown)
FROM patients p;
```

判定优先级：**exclude 触发 → FAIL ＞ include 不满足 → FAIL ＞ include 缺数据 → UNKNOWN ＞ 其余 PASS**。

### 2.5 输入 / 输出示例（真实 demo 输出）

```text
python -m unified.demo
=== 统一门面 demo（已注册引擎: ['cohort', 'kb']）===
[1] 科研入组筛查（engine=cohort, IR 自动路由）
    汇总: {'total': 30, 'PASS': 6, 'FAIL': 20, 'UNKNOWN': 4}
      P001-合格      → PASS     缺失: 无
      P002-年龄不符   → FAIL     缺失: 无
      P004-排除用药   → FAIL     缺失: 无
      P005-缺检查值   → UNKNOWN  缺失: ['include[2].lab']
      P006-缺诊断时间 → UNKNOWN  缺失: ['include[1].diagnosis']
```

每条判定都携带**证据**：`conditions: [{group, index, type, status}]` + `missing: ["include[2].lab"]`，
可直接回溯到是哪张表哪个字段缺了数据。

---

## 3. 统一门面：为什么要"统一入口、双轨索引"

两个场景的**物理索引不该合并**：

| 维度 | 知识库多跳问答 | 科研入组筛查 |
|---|---|---|
| 问题形态 | 自然语言，答案在关系链末端 | 结构化条件，答案在满足集合 |
| 数据组织 | 属性图（实体-关系-证据） | 关系表（患者-诊断-检验-用药） |
| 关键算子 | 向量相似度 + 图多跳扩展 + PPR | 属性过滤 + EXISTS 关联 + 三值判定 |
| 缺失数据 | 模糊（证据不足 → 低分） | 关键（缺字段 → UNKNOWN，不得纳入） |
| 规模特征 | 知识广度大、链深小（≤2~3 跳） | 患者多、每患者行少、判定要可审计 |

因此 `unified/` 只做**入口层统一**（同一 `search()`、同一结果信封、注册表 + 路由），
物理层保持双轨：

```text
                    ┌──────────────────────────────┐
                    │  UnifiedEngine.search(q, ...) │  ← 唯一入口
                    └──────────────┬───────────────┘
                                   │ Router（query_type 显式 / 关键词 / 输入形态）
              ┌────────────────────┴───────────────────┐
              ▼                                         ▼
   ┌──────────────────────┐                 ┌──────────────────────┐
   │ KBEngineAdapter       │                 │ CohortEngine         │
   │ 包装 retrieval.Retriever│                │ SQLite 编译执行       │
   ├──────────────────────┤                 ├──────────────────────┤
   │ NetworkX 图 + FAISS 三库│                │ SQLite 5 表（B+Tree）  │
   │ + PPR 缓存            │                 │ IR→SQL→逐患者判定      │
   └──────────────────────┘                 └──────────────────────┘
            知识库多跳问答                              科研入组筛查
```

```python
# 同一入口，两条查询（unified/demo.py）
result1 = unified.search(DEMO_IR)                    # dict IR → 路由 cohort
result2 = unified.search("星辰科技的创始人是谁？", top_k=5)  # 自然语言 → 路由 kb

# 统一检索 CLI（unified/cli.py）：按输入形态自动分流
#   python -m unified.cli --query "星辰科技的创始人是谁？" --mock --processed data/processed_demo_cn
#   python -m unified.cli --ir '{"include": [...]}' --db data/cohort/demo.db
# 路由规则：QueryIR 实例 / 入组 dict → cohort；str 自然语言 → kb（RAG）
```

## 4. 结论

- 场景一的问题是**沿关系链找答案**：用**图**存关系、**向量**存语义、PPR 排全局序；
- 场景二的问题是**对每个患者做条件判定**：用**关系表**存事实、**SQL 集合语义**表达 include/exclude、三值逻辑处理缺失；
- 两者共用**统一查询入口**与**结果信封**，但**物理存储各留各的**——门面统一，索引不合并。

## 5. 评测闭环（怎么证明检索“做对了”)

两个场景各自配一条**离线可重复的评测线**（`eval/run_eval.py`，gold 题集见 `eval/gold/`）：

| 场景 | 评测命令 | 指标 | mock 下的基线（当前） |
|---|---|---|---|
| 知识库多跳问答 | `python -m eval.run_eval retrieval --corpus main` | Recall@k / MRR / 按 hop 分组 / 延迟 | main：Recall@1=0.80，MRR=0.875 |
| 多跳合成语料 | `... --corpus multihop` | 同上（hop>=2 子集单独统计） | multihop：hop2 Recall@1=0.667 |
| 科研入组筛查 | `python -m eval.run_eval cohort` | 三值混淆 / accuracy / macro-F1 / UNKNOWN 漏判率 | 30/30：accuracy=1.0，漏判率=0.0 |

要点：

1. **mock 测不出多跳**——离线端到端里 LLM 抽取是噪声，`hop_chunks` 恒为 0；多跳能力由 `tests/test_retrieval_golden_graph.py` 的**黄金图**（手工 `add_triple` 绕过抽取）单独验证，`--corpus multihop` 则负责给指标线喂跨文档两跳样本。
2. **同一 gold 两种跑法**：`--mode mock` 是确定性回归基线（CI 用），`--mode real` 连真实 embedding/LLM 服务看抽取质量对多跳的增益（对比 `multihop` 的 hop2 Recall@1 从 0.667 能否提升）。
3. **筛查指标里最该盯 `unknown_miss_rate`**：gold=UNKNOWN（数据缺失）却给出 PASS/FAIL，等于把缺失当证据——比普通误判更危险；当前 SQLite 引擎为 0.0。
4. **真 PPR 依赖 scipy**：缺失时 dual_seed 自动退化为 personalization 排序（检索指标不变，但延迟从 ~7ms 升到 ~170ms 的真 PPR 才是线上主路径）；`requirements.txt` 已收录。

## 6. 延迟构成（为什么第一次检索慢）

`pipeline.run search` / `unified.cli` 单进程首次检索 ~0.9s，其余 ~8–20ms。用 cProfile 定位：

| 阶段 | 首次耗时 | 稳态耗时 | 说明 |
|---|---|---|---|
| `scipy.sparse` 首次 import | **~0.78s** | 0 | networkx `pagerank` 走 `_pagerank_scipy` → `to_scipy_sparse_array` 触发 |
| 其余（embed/recall/locate/hop/ppr） | ~0.1s | 8–20ms | 检索本身 |

结论：**慢的是冷启动依赖加载，不是检索算法**。CLI 每次都是新进程 → 每次吃这 0.8s；
`serve` 常驻或脚本内复用引擎时，第二次起 total ≈ 8ms（实测 run2=20ms、run3=8ms）。

若真在意单次 CLI 延迟，可在 `config` 里设 `retrieval.chunkhop_mode: legacy` + `ppr_mode: personal`
（跳过 scipy 的 pagerank 路径，延迟回落到 ~7ms），代价是失去真 PPR 的全局排序。

## 7. 统一入口的输入形态约定

`unified.cli --query` / `UnifiedEngine.search(query)`：

| 输入 | 路由 | 引擎 |
|---|---|---|
| 自然语言字符串 | `kb` | ChunkHop 多跳检索（RAG） |
| `QueryIR` 实例 / 含 `include` 的 IR dict / IR JSON 字符串 | `cohort` | SQLite 编译执行（科研入组） |
| `--ir '<json>'` / `--ir @file.json` | `cohort`（显式） | 同上 |
| `--type kb|cohort` | 显式覆盖 | — |
