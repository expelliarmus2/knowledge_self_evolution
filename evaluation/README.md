# 精度评测（recall@K）

对构建好的知识图谱跑测试集 recall@K 评测。

## 快速开始

```bash
# 项目根目录执行
python -m evaluation.eval_recall --processed data/processed_musique \
    --test data/test/musique_dev.json --top-ks 5,10,15,20,30 --workers 8
```

常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--processed` | `paths.processed_dir` | 知识库产物目录（graph.pkl + *.faiss） |
| `--test` | `data/test/musique_dev.json` | 测试问题文件 |
| `--top-ks` | `5,10,15,20,30` | 评测的 K 列表 |
| `--workers` | 8 | 并行检索线程数 |
| `--limit` | 全部 | 只评测前 N 条（快速验证用） |
| `--llm-filter` | 关 | 启用 LLM 三元组相关性过滤（会额外调 LLM，慢） |
| `--out` | `evaluation/results` | 结果输出目录 |

## 指标定义

- **doc_recall@K**：金标证据文档（`paragraphs[].title`）在 Top-K 检索 chunk 中的命中比例，
  按问题平均（每条问题 = 命中金标数 / 金标总数）。
- **full_hit@K**：问答级「全部金标文档都出现在 Top-K」的比例。
- **answer_hit@K**：金标 answer（含 aliases）字符串出现在 Top-K chunk 文本中的比例（词面匹配）。
- **coverage**：金标文档在图内是否有 chunk 节点 —— 抽取失败会压低 coverage，从而压低 recall 上限。

## 结果文件（`evaluation/results/<实验名>/recall_<时间戳>.*`）

- `recall_*.json`   完整报表（覆盖度 + overall + 金标全覆盖子集 + 失败列表）
- 早期（2026-11-10/11）扁平时间戳结果已归档至 `evaluation/results/recall_legacy/`
- `recall_*.csv`    表格（Excel 可直接打开）
- `recall_*.detail.jsonl`  逐条明细：gold / 检索 Top-K doc_id / 各 K 命中（错误分析用）

报表中 `*` 列为**金标全覆盖子集**指标（只统计金标文档全部在图内的题目），
用于排除抽取缺失干扰、单独评估检索器本身。

## 已知约束（2026-11-10 全量构建）

- 全量 11661 篇中约 **4918 个 chunk 抽取失败**（dashscope 内容审核 400）→
  ≈28% 语料文档在图内无 chunk，全量 recall 有天然上限；
- 测试集 1000 条问题的金标文档若在图外，该条无法被召回（报表中 coverage 会体现）。

## one_step 重建对比（2026-11-10，deepseek-v4.1-flash-1017 + enable_thinking=false）

| 维度 | two_step (qwen3.8-max) | one_step (deepseek flash) |
|---|---|---|
| 构建时长 | 3h02m (10957s) | 41min (2447s) |
| chunk 入图 | 12344/17262 (失败 4918) | **17262/17262 (0 失败)** |
| 图内文档 | 7092 | **9843** |
| 实体污染率 | 16.0% | **9.8%** |
| 同(头,尾)多关系组 | 5625 | **3902** |
| 金标全覆盖 | 633/1000 | **1000/1000** |
| doc_recall @5/@10/@30 | 0.150/0.202/0.305 | **0.167/0.221/0.335** |
| answer_hit @5/@10/@30 | 0.416/0.508/0.627 | **0.436/0.547/0.674** |
| 单条查询 | 1.4s | **0.95s** |

- 结果文件：`evaluation/results/recall_legacy/recall_20261110_172135.{json,csv,detail.jsonl}`（one_step）；对照 `recall_20261110_152013.*`（two_step）
- 产物：`data/processed_musique_onestep/`（graph.pkl / entities / triples / 三套 vdb，与 two_step 图并存）
- 说明：flash 默认 thinking 模式会把 4096 max_tokens 烧在推理上且输出为空（23.2s/chunk）；已通过 `llm.enable_thinking: false`（LLMClient extra_body 透传）关闭，降至 1.9s/chunk。

## 金标定义修正（2026-11-10，重要）

**此前所有 doc_recall 数字都是错的**：测试集每条含 ~20 段（2-4 篇 supporting + 16-18 篇干扰段落），
旧脚本把 20 段全当金标 → 分母膨胀 5-10 倍。标准 MuSiQue 指标只用 `is_supporting` 段落（2-4 篇，去重）。

修正后（supporting-only，`--supporting-only`）：

| 指标 | two_step 图 | one_step 图 | HippoRAG 论文 (MuSiQue, ColBERTv2) |
|---|---|---|---|
| doc_recall@2 | 0.362 | **0.429** | 0.409 |
| doc_recall@5 | 0.476 | **0.565** | 0.519 |
| doc_recall@10 | 0.536 | **0.637** | — |
| doc_recall@30 | 0.619 | **0.734** | — |
| full_hit@30 | 0.338 | **0.458** | — |

**失败归因（one_step 图，1000 题、2601 篇 supporting 金标）：**
- 60.6% 进 Top-30；10.9% 进候选池但排 30 名外（排序集中度问题）
- 28.5% 从未进候选池，其中 **95% 图上 2 跳内可达**（被 skip_hop=auto 关闭 hop 扩散 + locate/宽度截断挡掉），仅 **5% 是真正抽取缺口**
- 结论：三元组抽取不是 recall 瓶颈；瓶颈 = ① 评测金标定义（已修）② 候选池截断策略 ③ Top-30 排序集中度

结果文件：`evaluation/results/recall_legacy/recall_20261110_174907.*`（one_step）、`recall_20261110_175011.*`（two_step）；
旧 all-20-gold 结果 `recall_20261110_152013/172135.*` 已废弃。

## 与 HippoRAG2 论文的差距分析（2026-11-10 晚）

论文：*From RAG to Memory*（arXiv 2502.14802）Table 2/3，**MuSiQue passage recall@5 = 74.7**
（Table 3 消融 `w/o Filter` = 73.0）。我们修正金标后 doc_recall@5 = **0.565**，差距 18 分。

**为追差距做的 6 组配置对照（全量 1000 题，one_step 图）：**

| 配置 | doc@5 | doc@10 | doc@30 | full@30 |
|---|---|---|---|---|
| **baseline**（personal，hop 不参与排序，seeds 10/10/20） | **0.565** | **0.637** | **0.734** | **0.458** |
| hop 开 + boost 0.05 + width 50 | 0.544 | 0.626 | 0.713 | 0.424 |
| hop 开 + boost 0.05 + width 50 + min_hop_intersection 2 | 0.552 | 0.628 | 0.730 | 0.451 |
| seeds 扩容 20/20/50 + locate 150 | 0.551 | 0.637 | 0.737 | 0.456 |
| **ppr_mode: pagerank + hop 开** | **0.145** | 0.224 | 0.433 | 0.236 |
| hop 开 + boost 0.02 + width 300 | 0.565 | 0.637 | 0.723 | 0.441 |

**结论：配置级旋钮已穷尽**，baseline 是局部最优。关键证据：
- pagerank 图扩散崩到 0.145@5 —— 年份/国家类枢纽实体把 PPR 质量抽干（这也解释了为何当初选 personal）
- hop 扩散能让候选池变大，但 hop chunk 权重由「共享实体数」决定 → 偏好枢纽 → 挤压好种子
- 种子扩容无效 —— 多跳金标（如 `Mercia`）与问题词面无重叠，embedding 召不回

**HippoRAG2 达到 74.7 依赖的三件事（消融证实）：**
1. 强实体链接 query→node（`w/o Query to node` 直接掉到 44.9）
2. **干净的图** + passage 节点参与 PPR（我们的图仍有 9.8% 污染实体/5541 条硬污染，制造枢纽边）
3. LLM filter 重排（贡献 ~1.7 分）

**下一步（按性价比）：**
1. 构图期清洗污染实体（5541 条硬污染）→ 重建 → 重试 hop/PPR（解锁扩散的前提）
2. 改进实体链接（两段式实体召回 / entity_seed 与 vdb_entities 精度）
3. 加 chunk 级 LLM 重排（注意：当前 `--llm-filter` 只过滤三元组、不参与 chunk 排序，对 recall 无影响）

**已修复的隐藏 bug**：`skip_hop` 配置用 `str(cfg.get(...)).lower()` 解析，
YAML `false` → 字符串 `"false"`（真值！）→ `if skip_hop:` 恒真，hop 扩散从未真正跑过。
已改为布尔感知解析（`retrieval/retriever.py`）。

---

## 二、ChunkHop dual_seed 模式（对齐 Green-ChunkHop main 分支）

参考实现定位：原始仓库 `/data/yf/KG_Bench/Green-ChunkHop`（已不在），本地保留重构版
`/mnt/e/yf/chunkhop`（git remote: `gitcode.com/lll5464/chunkhop`）。其 dual_seed 算法为：
**双路入图（实体 + chunk 同时作为 PPR 种子）→ 重要实体选取（`rank_entities`：
`0.5×三元组种子分 + 0.5×chunk PPR 传导分`，top-30）→ 在线建 CHUNK_HOP 边 → 真 PPR（alpha=0.6）**。

### 新实现（`retrieval/chunkhop_dual.py`）

`DualSeedChunkHop.run(seed_chunks, entity_hits, triple_hits, query_entities)` 四步：

1. **实体种子分**：实体路相似度 + 三元组端点分（与定位路同口径）
2. **重要实体选取** `select_entities`：传导分 = Σ(种子chunk分/deg) / ln(1+度)
   —— hub 惩罚（England/artist 这类泛化枢纽即使高频出现也压下去）+ 查询实体 ×2.5 加成
3. **在线 CHUNK_HOP 边** `expand_online`：只经已选重要实体在 chunk 间建边，
   边权 = 共享重要实体数 × 查询交集加成 × 逐跳衰减；度 >600 的枢纽实体直接跳过
4. **真 PPR**：个人化质量只注入「已选实体 + 有边种子 chunk」（孤立节点注入=teleport 份额霸榜的坑），
   alpha=0.6、text_conclude 只挂已选实体边

### 检索器接入（`retrieval/retriever.py`）

- `retrieval.chunkhop_mode: dual_seed | legacy`（默认 dual_seed）
- dual 时跳过旧 hop_expander，`dual.run()` 产出 ppr_scores；`ppr_mode` 不再生效
- **`dual_score_blend: 0.3`**：最终分 = blend×PPR分 + (1-blend)×种子分 ——
  纯 PPR 把质量摊到 hop chunk 上 Top-5 被稀释（v0 实测 @5 掉 7pt），混合后保精度又保覆盖
- `_recall` 三路召回接上 `min_similarity` 过滤（FAISS cosine，IP 内积）
- 顺带修复 `_expand_triples` 的 `set(self.graph.entities())` 全图扫描（每次查询 79k 节点迭代）

### 评测结果（全量 1000 题，supporting-only，one_step 图）

| 配置 | doc@5 | doc@10 | doc@30 | full@10 | full@30 | ans@5 |
|---|---|---|---|---|---|---|
| legacy（personal 种子直排，165725） | 0.578 | 0.647 | 0.746 | 0.329 | 0.476 | 0.442 |
| **dual_seed + blend 0.3**（173203） | 0.566 | **0.649** | 0.746 | **0.332** | 0.466 | 0.431 |

结论：**dual_seed 与 legacy 基本打平**（@10 +0.2pt，@5 -1.2pt，@30 持平）。要点：
- 100 题子集上纯 PPR（blend=1）@30=0.746 vs legacy 0.716 —— **在线 hop 图候选池确实更丰富**，
  问题在 Top-5 排序（PPR 质量被摊开）
- 打平是图污染的天花板：重要实体选取依赖实体质量，而图里仍有 9.8% 污染实体
  （`Cyber Rights` 案例：Godwin 定律的 Godwin 被错误链接进 Godiva 问题候选）
- blend 扫描：0.3≈legacy@5 且 @10 最好；0.5 @5=0.532；0.7 @5=0.520（100 题子集）
- 耗时 0.58s/题（legacy 0.9s/题），无 LLM 依赖

**与旧 hop_expander 的本质区别**：旧方案离线预计算全局 chunk 邻接 + BFS 剪枝，
跳数权重看「共享实体数」→ 偏好枢纽；dual_seed 每查询只经 Top-K 重要实体在线建边，
从机制上避免枢纽噪声，且实体选取受查询题眼加成。

## 三、Agentic Search（IRCoT / Planner）

### 架构（新增 `agentic/` 包，不动现有检索链路）

```
agentic/
  tools.py    # Tools：search()（Retriever 封装）/ neighbor_chunks / entity_triples
  memory.py   # EvidencePool：doc 级去重证据池 + 三元组池 + 已见实体
  ircot.py    # IRCoT：LLM 生成推理句 → 用推理句检索 → 证据追加 → 循环；FINAL ANSWER 或证据饱和早停
  planner.py  # 分解子问题 → 并行检索 → 合并证据 → 作答
prompts/ircot_step_en.txt / agentic_plan_en.txt / agentic_answer_en.txt
evaluation/eval_agentic.py  # EM / F1 / evidence_recall / 步数 / LLM 调用数
pipeline/run.py ask         # 单题 CLI：python -m pipeline.run ask --processed ... --question "..."
config agentic:             # mode / max_steps / top_k_per_step / evidence_max_docs / include_triples ...
```

IRCoT 核心循环：每轮 `prompts.render("ircot_step", question, evidence, cot)` → LLM 输出
`FINAL ANSWER: <ans>` 或 `NEXT: <推理句>` → 用推理句调 `Tools.search()`（内部已是 dual_seed 检索）
→ 证据池按 doc_id 去重追加。连续两轮无新增 → 证据饱和早停；超步数用
`agentic_answer` 模板生成最终答案。证据渲染同时带命中三元组（图证据，区别于纯段落检索）。

### 评测（MuSiQue dev 前 200 题，flash-1017，8 并发）

| mode | EM | F1 | evidence_recall | avg_docs | LLM calls | 耗时 |
|---|---|---|---|---|---|---|
| **ircot** | **0.335** | **0.455** | **0.818** | 10.5 | 5.0 | 5.7s/题（146s 全量） |
| planner | 0.240 | 0.348 | 0.736 | 23.6 | 2.0 | 2.5s/题（64s 全量） |

结论：
- **IRCoT 明显优于一次性分解式 planner**（F1 +10.7pt，证据召回 +8pt）—— 逐步推理句检索
  能动态纠偏，planner 的静态子问题分解会漏跳
- evidence_recall 0.818 是当前检索器候选池的天花板（≈单跳 @30 覆盖），
  18% 金标进不了池 = 图/实体质量问题，不是 agent 逻辑问题
- 成本：5 次 LLM + 5 次检索 ≈ 5.7s/题（flash，思考关闭）；全量 1000 题 ≈ 12min/8 并发
- 对照：MuSiQue 论文里 GPT-3.5 系 IRCoT/类似多跳方法的 QA F1 大致落在 0.3~0.55 区间，
  本结果在合理区间；瓶颈同样是检索上游（先做构图期清洗 → 证据召回上限才能抬升）

**注**：evidence_recall 的 gold 定义与 eval_recall 一致（is_supporting 段落文档级去重）。
