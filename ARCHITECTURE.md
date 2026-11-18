# 架构文档（ARCHITECTURE）

> 各部分如何做、如何扩展；自进化（EvoGraph-R1 / RSI）如何接入；
> 可视化构建与检索的操作步骤；数据位置如何配置。

---

## 1. 总体架构

```
┌─────────────────────────────── pipeline/（编排层）───────────────────────────────┐
│  BuildPipeline.run()                                                              │
│   文档 → chunk → [OpenIE 两步抽取] →(流式逐批)→ [构图] →(全图)→ [向量入库] → 产物  │
│     │              openie/              graph/            vectorstore/            │
│     └── EvolutionManager.emit(event)  ← 12 个事件注入点（演进策略在此介入）─────┐ │
└────────────────────────────────────────────────────────────────────────────────┼─┘
                                                                                   │
┌──────────────────────────── retrieval/（检索层）────────────────────────────────▼─┐
│  Retriever.search(query)                                                          │
│   向量召回(三元组/实体/chunk 三路并行) → 实体定位chunk → CHUNK_HOP扩散 → PPR 排序   │
│   → LLM 三元组过滤 → 子图 JSON（供可视化）      ▲ 检索感知：graph.version / mtime   │
└──────────────────────────────────────────────────┼────────────────────────────────┘
                                                    │
┌──────────────────── graph/evolution.py（演进层 ★）─┴─────────────┐
│  GraphEvolutionAPI.apply(ops) → apply_operation ×N → version++  │
│      → changelog 追加 → VectorStoreManager 增量同步（只嵌新增）    │
└──────────────────────────────────────────────────────────────────┘
                                                    │
┌──────────────────── web/（可视化层）───────────────▼─────────────┐
│  Flask: /api/search /api/evolve /api/graph  +  vis-network 渲染   │
└──────────────────────────────────────────────────────────────────┘
```

核心设计原则：**每层只依赖下一层的稳定接口**。EvoGraph-R1 接入时只与
`graph/evolution.py` 的 API 交互，不触碰 openie / vectorstore / retrieval 内部。

---

## 2. 各部分应该如何做

### 2.1 配置层（`config/default.yaml` + `utils/config.py`）

**做什么**：全部参数集中一处，代码零硬编码。

**怎么做**：
- `Config.load(path=..., overrides=...)` → 点路径访问 `cfg.get("llm.model")`；
- 相对路径自动按项目根解析：`cfg.path("paths.raw_dir")`；
- 优先级：**CLI 参数 > 环境变量（`KB_` 前缀双下划线分层）> `--config` 指定 yaml > default.yaml > 代码默认**。

**换领域/换模型**：改 yaml（模型、地址、维度）；换领域只改 `prompts/` 模板。

### 2.2 OpenIE 层（`openie/`）——两种可切换抽取模式

**做什么**：把 chunk 文本变成带锚定的三元组。模式由 `openie.extract_mode` 配置，
可按「性能 vs 精度」权衡切换，无需改代码：

**模式一 two_step（默认，HippoRAG2 两步式，精度优先）**：

| 步骤 | 位置 | 输入 → 输出 | 关键机制 |
|---|---|---|---|
| 1. 切块 | `chunker.py` | 文档 → `[Chunk]` | `chunking.chunk_size/overlap` 可配 |
| 2. NER | `extractor.extract_entities` | `{passage}` → `["实体", ...]` | 模板 `prompts/ner_{lang}.txt` |
| 3. RE | `extractor.extract_triples` | `{passage}+{named_entity_json}` → `[[s,p,o],...]` | 模板 `prompts/triple_extraction_{lang}.txt`，端点强制锚定实体列表 |
| 4. 校验 | `extractor._anchor_ok / _resolve` | 原始三元组 → 合法三元组 | 锚定校验 + 代词解析 |

**模式二 one_step（huaxi_poc 风格单次抽取，性能优先）**：

| 步骤 | 位置 | 输入 → 输出 | 关键机制 |
|---|---|---|---|
| 1. 单次抽取 | `extractor.extract_triples_one_step` | `{passage}` → `头<|>关系<|>尾##...` | 模板 `prompts/openie_one_step_{lang}.txt`，分隔符协议 `<|>` `##` `<|COMPLETE|>` |
| 2. 解析+兑底 | `extractor._parse_one_step_output` | 原始输出 → `[[s,p,o],...]` | 代码块包裹容错、代词端点丢弃；实体从端点汇总（全部视为已锚定） |

LLM 调用次数减半，适合大规模语料。真实 API 实测约快 3.8 倍（44s/篇 → 11.7s/篇），
代价是没有独立实体列表锚定，实体变体依赖 prompt 约束（模板内置"不用代词/类别名/时间"规则）。

**扩展点**：
- 换抽取模式 → `openie.extract_mode: two_step | one_step`（一杆子开关）；
- 换抽取模型 → 只改 `llm:` 配置段；
- 离线/测试 → `provider: mock`（确定性规则实现，两种模式均支持）；
- 换语言 → `project.language: english`，模板走 `_en` 后缀（回退无后缀）；
- 新增抽取策略（如多轮抽取、置信度过滤）→ 在 `extract()` 内加分支，其余层无感知。

**并行方式**（吸收 huaxi_poc）：`extract_batched_stream()` 用
`ThreadPoolExecutor(llm.max_workers)` 提交全部 chunk，`as_completed` 按完成顺序流式产出，
每凑满 `openie.batch_size` 吐一批 → 抽取与构图**重叠执行**。

### 2.3 构图层（`graph/`）——图是唯一事实源

**做什么**：三元组建图；图上的版本号是全系统演进的时钟。

**图结构**（`knowledge_graph.py`，NetworkX MultiDiGraph）：

| 元素 | 节点/边 | 属性 |
|---|---|---|
| 实体 | node `type=entity` | `display / entity_type / chunk_ids` |
| chunk | node `type=chunk` | `doc_id / order / text` |
| 三元组 | edge key=关系名 | `weight / count / chunk_ids / meta` |
| text_conclude | 实体→chunk | `count`（实体出现在哪些 chunk，检索定位用） |
| synonym | 实体↔实体 | `score`（可选，向量近邻合并） |

**怎么做**：
- `GraphBuilder.build(batch_stream, chunk_texts, on_batch_done)`：逐批消费流式抽取结果，
  `add_triple` 自动累计权重、`link_entity_to_chunk` 维护定位边；
- 实体名归一（`normalize_name`）+ 规范名映射（`_name2canonical`），大小写策略可配；
- **每次写操作 `version += 1` 且写入 `changelog`**——这是演进审计/回放/策略学习的数据底座。

### 2.4 向量库层（`vectorstore/`）——三库分离 + 增量同步

**做什么**：实体、三元组、chunk 三类对象分别入 FAISS（IndexIDMap2 + IndexFlatIP，key 级增删改查）。

**怎么做**：
- `FaissStore`：key→id 映射存 `.meta.json`；`add / remove / upsert / search / save`；
- `VectorStoreManager.sync_from_graph(graph, incremental=True)`：对比图与库，
  **只嵌入新增项**（Embedding 调度：`embedding.batch_size` 批量 + `max_workers` 有界并发）；
- 语义：删除图元素后 sync 会移除对应向量（演进一致性）。

### 2.5 检索层（`retrieval/`）——Green-ChunkHop 式多跳

**做什么**：查询 → 相关 chunk 排序 + 可用三元组 + 可视化子图。

**检索流程**（`retriever.py: search()`，9 步计时输出在 `timing`）：

```
 1. query embedding
 2. 三路并行种子召回（线程）：三元组库(triple_seed) / 实体库(entity_seed) / chunk库(vector_seed)
 3. 演进校验：召回的三元组/实体与当前图核对，已删除的自动失效   ★ 检索感知演进
 4. 实体→chunk 定位：text_conclude 边；chunk 分 = Σ(实体分/度)
 5. CHUNK_HOP 扩散：共享实体 chunk 邻接 BFS（weight = shared_count × (1+查询实体交集)）
    （hop_expander 邻接缓存绑定 graph.version，图演进后自动重建）      ★
 6. 三元组 n 跳扩展（可选）：从命中实体沿三元组边 BFS，分数逐跳衰减
 7. PPR：实体+chunk 混合子图上 networkx.pagerank（personalized=种子分数，alpha=ppr_alpha）
 8. LLM 三元组过滤：prompts/retrieval_filter_{lang}.txt（失败自动回退词面重叠）
 9. 组装 SearchResult：filtered_triples / ranked chunks / subgraph(nodes,edges JSON)
```

**可调参数**（`retrieval:` 段）：seed 数量、hops、max_hop_width、ppr_alpha、
final_top_k、llm_filter_enable、use_triple_expand 等。

### 2.6 可视化层（`web/`）

见 §4 操作步骤。

### 2.7 流水线层（`pipeline/`）

**做什么**：把 2.2–2.5 串成端到端，并在固定安全点广播事件。

`BuildPipeline.run()` 时序（★=演进注入点，见 §3）：

```
pipeline_start ★
  before_chunking ★ → 切块 → after_chunking ★
  before_extract ★
    循环批次: [并行抽取一批] → [立即构图该批] → on_extract_batch ★（策略可返回图操作）
  after_extract ★
  before_graph_build ★ → 导出产物 → after_graph_build ★
  before_vectorize ★ → 增量向量同步 → after_vectorize ★
  持久化（graph.pkl / *.faiss / graph_export.json / build_report.json）
pipeline_complete ★          （on_error ★ 任意阶段异常时触发）
```

---

## 3. 自进化如何加（EvoGraph-R1 / RSI 对接）

系统在**三个层面**预留了接入点，按需选择；接入代码**不需要修改任何核心模块**。

### 3.1 第一层：图操作 API（演进执行面）

任何演进系统最终产出「对图的增删改」，直接走统一入口：

```python
from graph.evolution import GraphOperation, GraphEvolutionAPI
from graph.knowledge_graph import KnowledgeGraph
from openie.embedder import build_embedder
from vectorstore.manager import VectorStoreManager

graph = KnowledgeGraph.load("data/processed/graph.pkl")
vdb = VectorStoreManager(cfg, embedder, "data/processed")
api = GraphEvolutionAPI(graph, vdb, auto_sync_vectorstore=True)   # ← 向量库自动增量同步

result = api.apply([
    GraphOperation("add_triple",    {"head": "A", "relation": "新关系", "tail": "B", "weight": 2.0}, reason="策略决策"),
    GraphOperation("update_triple",  {"head": "A", "relation": "旧关系", "tail": "B", "weight": 5.0}, reason="置信度上调"),
    GraphOperation("remove_entity", {"name": "过时实体"}, reason="淘汰"),
    GraphOperation("add_synonym_edge", {"a": "AI", "b": "人工智能", "score": 0.9}),
])
print(result)   # {"applied": [...], "failed": [...], "version": 217, "stats": {...}}
graph.save("data/processed/graph.pkl")
```

**8 种操作**：`add_triple / remove_triple / update_triple / add_entity /
remove_entity / update_entity / add_synonym_edge / batch`。
每次 apply 后 `version` 单调递增 + changelog 追加 + 向量库增量同步 →
**检索侧自动感知**（HopExpander 缓存按 version 失效、召回结果与图核对、
Retriever 按文件 mtime 热重载）。

> 运行期通道：`POST /api/evolve {"ops": [...]}`（Web）或
> `python -m pipeline.run evolve --ops '[...]'`（CLI）——同一 API。

### 3.2 第二层：流水线策略注入（构建过程介入面）

**场景**：EvoGraph-R1 想在构建过程中实时干预（例如根据当前批次抽取质量调整图结构）。

实现一个策略类，在感兴趣的事件点返回图操作即可：

```python
# my_evolution.py（放项目内或加入 PYTHONPATH）
from graph.evolution import EvolutionStrategy, GraphOperation

class R1FeedbackStrategy(EvolutionStrategy):
    name = "evograph_r1_feedback"

    def on_event(self, event: str, ctx) -> list:
        # ctx: EvolutionContext(config, graph, vector_manager, embedder, llm)
        #      ctx.event_data ← 本次事件负载（emit 时注入）
        if event == "on_extract_batch":
            # 例：R1 策略审阅本批三元组 → 决定图演进
            batch = ctx.event_data.get("batch", [])
            ops = []
            for r in batch:
                for t in r["triples"]:
                    if 判定需要修正(t):          # ← R1 的策略逻辑（RL/规则/LLM 均可）
                        ops.append(GraphOperation(
                            "update_triple",
                            {"head": t["head"], "relation": t["relation"],
                             "tail": t["tail"], "weight": 修正后权重},
                            reason="R1 feedback",
                        ))
            return ops                            # 流水线在安全点统一应用
        if event == "pipeline_complete":
            # 例：构建结束后触发一轮全局图优化（可用 ctx.llm / ctx.embedder）
            return self.全局优化(ctx.graph)
        return []
```

yaml 启用（无需改代码）：

```yaml
evolution:
  enabled: true
  strategies: ["my_evolution:R1FeedbackStrategy"]   # "模块:类名"
  auto_sync_vectorstore: true
```

事件负载说明见 `pipeline/hooks.py`（12 个事件 + 完整注释示例）。

### 3.3 第三层：检索反馈闭环（RSI 数据面）

**场景**：把检索/使用结果喂回演进策略（自改进回路）。

- **输入信号**：`SearchResult`（`filtered_triples` 命中与丢弃、`chunks` 排序、
  `timing`、`subgraph`）+ Web `/api/search` 的 JSON——直接可作为 R1 的 reward/状态特征；
- **执行回路**：策略分析信号 → 产出 `GraphOperation` 列表 →
  `GraphEvolutionAPI.apply()`（或 `/api/evolve`）→ version++ → 检索感知，闭环完成；
- **历史数据**：`graph.changelog`（op + payload + reason + version）可直接导出做
  策略训练/回放：`graph.changelog`（列表，最新 max_changelog 条）。

### 3.4 检索如何"感知"演进（已实现，无需接入方处理）

| 机制 | 位置 | 行为 |
|---|---|---|
| version 绑定缓存 | `hop_expander.py` | `graph.version` 变化 → 邻接缓存自动失效重建 |
| 召回校验 | `retriever._validate_triples/_entities` | 向量库中已被删除的元素自动剔除 |
| 文件热重载 | `retriever._ensure_fresh()` | 外部进程演进图后，检索自动重载新图 |

---

## 4. 可视化：构建与检索操作步骤

### 步骤 ① 构建知识库

```bash
cd knowledge_self_evolution
python -m pipeline.run build --mock          # 离线演示（无需模型服务）
# 真实模式：先在 config/default.yaml 配好 embedding/llm，然后去掉 --mock
python -m pipeline.run build                  # 增量（新文档才处理）
python -m pipeline.run build --full          # 全量重建
```

构建产物落在 `paths.processed_dir`（默认 `data/processed/`）：
`graph.pkl / graph_export.json / *.faiss / openie_results.jsonl / build_report.json`。

### 步骤 ② 启动可视化服务

```bash
python -m pipeline.run serve --mock                      # http://localhost:5570
python -m pipeline.run serve --mock --port 5571 --processed /path/to/kb
```

### 步骤 ③ 浏览器操作（http://localhost:5570）

1. **全图浏览**：页面自动加载 `graph_export.json`（或点「全图浏览」）——
   蓝点=实体、紫方=chunk、蓝实线=三元组边；
2. **检索子图**：顶部输入查询（如 `星辰科技的创始人是谁？`）→ 回车或点「检索子图」：
   - 图区：命中实体高亮放大；seed chunk 按来源着色（绿=向量路、橙=三元组路、
     蓝=双向）；hop chunk 橙色方点；**金色加粗三元组边 = LLM 过滤保留**；
     橙虚线 = CHUNK_HOP 边（悬停显示共享实体）；灰虚线 = text_conclude 定位边；
   - 右侧「检索结果」页：LLM 过滤后三元组 / 全部命中三元组（含扩展跳数与分数）/
     PPR 排序 chunk 文本（`[来源|hop数] score`）；
   - 点击任意节点/边 → 「详情」页显示完整 JSON；
3. **演进演示**：右侧「演进操作」页输入操作 JSON →「应用演进操作」：

   ```json
   [{"op": "add_triple", "payload": {"head": "星图平台", "relation": "演进关联", "tail": "知识库自演进"}, "reason": "demo"}]
   ```

   应用后：图版本提升 → 向量库自动增量同步 → 全图刷新；再检索同一主题，
   新增的边已出现在命中子图中（闭环验证）。

### 步骤 ④ API 直调（脚本/接入用）

```bash
# 检索（返回完整子图 JSON）
curl -X POST localhost:5570/api/search -H "Content-Type: application/json" \
     -d '{"query": "星辰科技的创始人是谁？", "top_k": 10, "hops": 2}'
# 演进
curl -X POST localhost:5570/api/evolve -H "Content-Type: application/json" \
     -d '{"ops": [{"op": "add_triple", "payload": {"head": "A", "relation": "r", "tail": "B"}}]}'
# 全图 / 统计
curl localhost:5570/api/graph ; curl localhost:5570/api/stats
```

CLI 检索：`python -m pipeline.run search --query "..." [--json] [--processed DIR]`

---

## 5. 数据位置配置（三级覆盖）

**优先级**：CLI 参数 > 环境变量 > `--config` yaml > `config/default.yaml`

| 位置 | 配置方式 | 示例 |
|---|---|---|
| yaml | `paths.raw_dir` / `paths.processed_dir` / `paths.prompts_dir` / `paths.logs_dir` | `raw_dir: /data/my_docs` |
| 环境变量 | `KB_PATHS__RAW_DIR` / `KB_PATHS__PROCESSED_DIR` / `KB_PATHS__PROMPTS_DIR` / `KB_PATHS__LOGS_DIR` | `export KB_PATHS__RAW_DIR=/data/docs` |
| CLI（build） | `--input`（别名 `--raw`）/ `--output`（别名 `--processed`） | `build --input /d/in --output /d/kb` |
| CLI（search/evolve/serve） | `--processed` | `search --query "..." --processed /d/kb` |

约定：
- `raw_dir`：输入文档。`.jsonl`（每行 `{"doc_id": ..., "text": ...}`）、
  `.txt`/`.md`（每文件一篇，文件名即 doc_id）；目录或单文件均可；
- `processed_dir`：知识库产物目录（图 + 向量库 + 导出）。**换目录 = 换知识库**——
  可以同时维护多套知识库，用 `--processed` 切换检索/服务目标；
- 路径支持相对（按项目根解析）与绝对路径；
- 单独 yaml 配置文件：`python -m pipeline.run build --config config/my_domain.yaml`
  （在 default 基础上深合并覆盖）。

示例——为一套新领域数据建独立知识库并可视化：

```bash
python -m pipeline.run build --input /data/medical_docs --output /data/medical_kb   # 不加 --mock 为真实模式
python -m pipeline.run serve --processed /data/medical_kb
```
