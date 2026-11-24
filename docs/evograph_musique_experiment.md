# EvoGraph 实验：自进化图谱对 MuSiQue 多跳检索精度的影响

分支：`feature/evograph-musique`

## 1. 实验问题

> 让 LLM 评价检索轨迹里的三元组、把"有用"的三元组回写图谱（自进化），
> 能否提升后续问题的多跳检索精度？与"不回写"的基线相比提升多少？

## 2. 数据与切分

数据：`data/test/musique_dev.json` — MuSiQue dev，**1000 题**，每题 20 个段落
（`is_supporting` 布尔标注 2~4 个支持段落，`question_decomposition` 给出 2/3/4-hop 子问题链）。

| 切分 | 题数 | 用途 |
|---|---|---|
| evolve set（前 500，原序） | 500 | 检索 + LLM 轨迹评价 + 三元组回写（**训练侧**） |
| eval set（后 500，原序） | 500 | 仅检索，度量精度（**评测侧**） |

可选 `--shuffle --seed` 打乱后再切，用于稳健性检查；默认原序，保证可复现。

**语料**：两批题目的 20000 个段落按 `title + paragraph_text` 去重后建库
（去重后约数千段），每段落 = 1 个 chunk；维护 `(qid, paragraph_idx) → chunk_id`
映射，用于判断检索命中的 chunk 是否属于该题的支持段落。

## 3. 两个实验臂

| 臂 | 图谱 | 流程 |
|---|---|---|
| **A. baseline（无自进化）** | 构建后的原始图谱，保持不变 | eval set 500 题直接检索 |
| **B. evolved（自进化）** | 原始图谱 + evolve set 回写的三元组 | ① evolve set 500 题检索 → LLM 评价 → 回写；② eval set 500 题检索 |

两臂使用**同一份原始图谱的副本**、**同一 eval set**、**同一检索配置**，
唯一差异是图谱是否经过自进化。图谱规模变化（三元组数/实体数/version）一并记录。

## 4. 指标

| 指标 | 定义 |
|---|---|
| `Recall@k` (k=1,5,10) | top-k 检索 chunk 覆盖该题支持段落的比例（按题平均） |
| `MRR@10` | 第一个命中支持段落的排名的倒数 |
| `hit@k` | 该题至少命中 1 个支持段落的比例 |
| `graph_delta` | 进化前后三元组数 / 实体数 / version 变化 |
| `evolve_stats` | 评价了多少轨迹、判定有用多少三元组、实际写入多少、去重丢弃多少 |
| `split_overlap` | evolve/eval 两批题共享实体的比例（泄漏风险指示） |

答案级指标（EM/F1）需要 LLM 读检索结果生成答案，real 模式下可选开启
（`--answer-metric`），mock 模式下不计算。

## 5. 自进化实现

对 evolve set 的每题：

1. 正常检索（`top_k` chunk + 命中三元组 + 子图）；
2. 把检索轨迹交给 LLM 评价，输入包含：原问题、`question_decomposition`（若可用）、
   召回 chunk 摘要、候选三元组列表（`head | relation | tail`，带来源 chunk）；
3. LLM 输出 JSON：`{"useful": [{"head":..., "relation":..., "tail":...}], "reason": "..."}`；
4. 过滤规则（防污染）：
   - 只接受**候选集合内**的三元组（不允许 LLM 凭空造新三元组）；
   - 每题最多回写 `max_write_per_q` 条（默认 8）；
   - 已存在同三元组则只累加权重，不新增边；
   - 回写边带 `meta: {evolved: true, source_qid: ..., confidence: ...}`，
     便于审计与"降权/回滚"实验；
5. 记录每条回写，写入 `experiments/evograph/artifacts/evolution_log.jsonl`。

## 6. 评价：这个方案合理吗？检索需要返回三元组吗？

### 6.1 检索必须返回三元组（含 provenance）

**是，必须。** 理由三条：

1. **没有三元组就没有"进化对象"**——回写的单位是三元组，检索若不返回
   `{head, relation, tail}` 及其来源 chunk，LLM 评价无从下手，回写也无 provenance；
2. **可审计、可回滚**——带 `source_chunk` / `retrieval_path`（哪一跳进来）才能
   定位污染来源、做降权或撤销；
3. **当前实现已具备基础**：`SearchResult.triples`（`head/relation/tail/score`）+
   `SearchResult.chunks`（含 `chunk_id`、`source`）已能支撑；建议补强到
   每条三元组内嵌 `chunk_ids` 与 `hop` 来源（`triples.jsonl` 里已有 `chunk_ids`，
   检索侧透传即可）。

### 6.2 方案合理，但有三个硬伤需要正视

| 风险 | 说明 | 缓解（本实验已做/建议） |
|---|---|---|
| **信用分配（credit assignment）** | 多跳题里"某条三元组是否有用"依赖上下文组合，LLM 在无答案参照时判断主观、易漂移 | 本实验提供子问题分解作参考；建议 real 模式用答案 EM/F1 做弱监督对齐（有 support 标注时用标注做 ground truth 校准） |
| **自我强化偏差** | 错误三元组回写后更容易被后续检索召回，误差累积 | 只允许写候选内三元组、每题限额、`evolved` 标记 + 可降权；报告中单独统计 `evolve` 写入量与 `split_overlap` |
| **泄漏/重合** | MuSiQue 题意高度共享实体（同一 wikipage 反复出现），eval set 的收益可能来自"同实体标注"而非泛化 | 报告 `split_overlap`；建议追加 **strict split**（按实体隔离）作为补充实验 |

### 6.3 更严谨的补充臂（建议后续加）

- **臂 C（shuffle 对照）**：把 evolve/eval 两批题对调重跑，若提升只在"先跑的那批"
  上出现，说明是顺序/记忆效应而非进化收益；
- **臂 D（写入但禁用）**：三元组写回图谱但检索时屏蔽 `evolved` 边，
  用来分离"图谱结构变化"与"检索缓存效应"；
- **臂 E（随机写入对照）**：随机选同样数量的候选三元组回写（不经 LLM 评价），
  若与 B 提升相当，说明收益来自"多写边"而非"LLM 评价质量"。

> 结论倾向：方案方向合理（用检索轨迹做在线知识强化），但**必须**有三元组 provenance、
> 写入限额、`evolved` 标记、以及与随机写入/shuffle 对照，否则很容易把
> "多连边带来的召回膨胀"误读为"自进化提升精度"。

## 7. 运行

```bash
# 冒烟（20 题 / 20 题，mock 全链路，验证流程）
./.venv/bin/python -m experiments.evograph.run_experiment --limit 20 --mode mock

# 全量（500/500，mock 基线）
./.venv/bin/python -m experiments.evograph.run_experiment --mode mock

# 全量（真实 embedding/LLM 服务）
./.venv/bin/python -m experiments.evograph.run_experiment --mode real
```

产物：`experiments/evograph/artifacts/`（gitignored）
- `split.json`：切分与映射
- `evolution_log.jsonl`：每条回写三元组（含理由）
- `result_baseline.json` / `result_evolved.json` / `comparison.json`
