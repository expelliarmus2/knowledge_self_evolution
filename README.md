# 知识库自演进（knowledge_self_evolution）

基于 **HippoRAG2** 的知识库构建、检索与可视化系统，并为 **EvoGraph-R1 自进化能力**（或 RSI 部分）对接预留接口。

> **架构详解、各模块如何做、自进化接入、可视化操作步骤、数据路径配置 → 见 [ARCHITECTURE.md](ARCHITECTURE.md)**
> 本 README 为快速上手与验收对照。

```
文档 → [两步式 OpenIE] → 三元组 → [构图] → 知识图谱 + 向量库
                                        ↓
              检索：双路召回 → CHUNK_HOP 扩散 → PPR → LLM 三元组过滤 → 子图可视化
                                        ↓
              演进钩子：图操作注入（增删改）→ 增量更新 → 检索感知
```

## 目录结构

```
knowledge_self_evolution/
├── config/default.yaml      # 全局配置（Embedding/LLM/路径/切块/检索/演进 全部 yaml 驱动）
├── prompts/                 # prompt 模板（换领域只换模板，不改代码）
│   ├── ner_zh.txt / ner_en.txt                    # 第一步：NER
│   ├── triple_extraction_zh.txt / _en.txt          # 第二步：实体条件化三元组抽取
│   ├── retrieval_filter_zh.txt / _en.txt           # 检索相关性过滤
│   └── README.md                                  # 模板协议说明
├── openie/                  # 两步式抽取
│   ├── llm_client.py        #   OpenAI 兼容客户端（重试退避）+ MockLLMClient（离线规则实现）
│   ├── embedder.py          #   向量嵌入（批量+有界并发+磁盘缓存）+ MockEmbedder
│   ├── chunker.py           #   文档切块（大小/重叠可配）
│   └── extractor.py         #   两步抽取编排：NER→RE、锚定校验、代词解析、并行/流式
├── graph/                   # 构图与演进
│   ├── knowledge_graph.py   #   MultiDiGraph：实体/chunk 节点，triple/text_conclude/synonym 边
│   │                        #   version 单调递增 + changelog；apply_operation 统一入口
│   ├── graph_builder.py     #   流式构图（逐批消费抽取结果）
│   └── evolution.py         #   ★ 演进核心：GraphOperation(8种) / EvolutionStrategy /
│                            #     EvolutionManager(事件广播) / GraphEvolutionAPI(门面+向量库自动同步)
├── vectorstore/
│   ├── faiss_store.py       #   FAISS IndexIDMap2 封装（key 级增删改查、.meta.json 持久化）
│   └── manager.py           #   实体/三元组/chunk 三库管理 + sync_from_graph 增量同步
├── retrieval/
│   ├── retriever.py         #   主检索器（双路召回+hop扩散+PPR+LLM过滤+子图组装）
│   ├── hop_expander.py      #   CHUNK_HOP 多跳扩散（共享实体邻接，缓存绑定 graph.version）
│   └── filterer.py          #   LLM 三元组相关性过滤（带词面重叠回退）
├── pipeline/
│   ├── build_pipeline.py    #   端到端流水线：抽取→流式构图→向量入库（事件注入点齐全）
│   ├── hooks.py             #   ★ 演进钩子说明与接入示例（EvoGraph-R1 注入点一览）
│   └── run.py               #   CLI：build / search / evolve / serve
├── web/                     # HTML 可视化（Flask + vis-network）
│   ├── app.py               #   /api/graph /api/stats /api/search /api/evolve
│   ├── index.html, static/  #   检索子图渲染、chunk 排序、演进操作面板
│   └── lib/                 #   vis-network.min.js（本地）
├── cohort/                  # 科研入组筛查 demo（SQLite，统一门面第二个引擎）
│   ├── schema.sql           #   5 表：patients/diagnoses/labs/medications/events
│   ├── query_ir.py          #   入组条件 IR（include/exclude + 时间窗）
│   ├── compiler.py          #   IR → SQL（每条件一对 pass/unknown 布尔列）
│   ├── engine.py            #   CohortEngine（BaseEngine 实现）
│   └── seed.py              #   合成 30 例结构化患者 + DEMO_IR
├── unified/                 # 统一检索门面：BaseEngine/注册表/路由/结果信封
│   ├── engine.py            #   EngineRegistry / KBEngineAdapter / UnifiedEngine
│   ├── router.py            #   query_type 显式优先，否则关键词/输入形态
│   └── demo.py              #   同一入口跑 KB 问答 + 入组筛查
├── docs/                    # 场景文档
│   ├── scenarios.md         #   场景 A：两场景问题 + 图/存储结构展示
│   └── storage_format_scenarios.md  # 场景 B：四层次框架下的存储选型
├── eval/                    # 评测：gold 题集 + 指标（Recall@k/MRR/三值混淆）
│   ├── metrics.py           #   纯函数指标（无外部依赖）
│   ├── run_eval.py          #   CLI：retrieval / cohort，mock 或 real 模式
│   └── gold/                #   retrieval_gold / multihop_docs / cohort_gold
├── data/raw/                # 输入文档（.jsonl：{doc_id, text}；或 .txt/.md 每文件一篇）
├── data/processed/          # 产物：graph.pkl / graph_export.json / *.faiss / build_report.json
├── data/cohort/demo.db      # 科研入组 demo SQLite（seed 自动生成）
├── tests/                   # test_smoke / test_cohort_engine / test_unified_facade
│                           #   + test_retrieval_quality(质量) + test_retrieval_golden_graph(黄金图)
│                           #   + test_eval_metrics(评测指标)
└── utils/                   # config(yaml+环境变量覆盖) / prompts / runtime(并行池) / text(解析清洗)
```

## 快速开始

环境要求：Python 3.10+，`pip install -r requirements.txt  -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com --proxy ""`

```bash
# 1. 离线演示（mock LLM + mock embedding，无需任何服务）
#    注意：--mock 是 256 维向量；若 data/raw 含大语料（如 corpus_musique.json）
#    或 processed_dir 已是 1024 维真实库，必须用独立 --output 目录，否则报
#    AssertionError: d == self.d（旧索引维度不匹配）。
mkdir -p /tmp/kb_raw && cp data/raw/sample_docs.jsonl /tmp/kb_raw/   # 只放中文示例
python -m pipeline.run build --mock --input /tmp/kb_raw --output /tmp/kb_demo
python -m pipeline.run search --query "星辰科技的创始人是谁？" --mock --processed /tmp/kb_demo
python -m pipeline.run serve  --mock --processed /tmp/kb_demo          # Web http://localhost:5570

# 1b. 也可以直接用仓库现成的中文演示库（无需 build）
python -m pipeline.run search --query "星辰科技的创始人是谁？" --mock --processed data/processed_demo_cn

# 2. 真实模式：config/default.yaml 配置模型后（embedding 服务 + LLM 服务）
python -m pipeline.run build                  # 增量构建（已构建文档自动跳过）
python -m pipeline.run build --full           # 全量重建
python -m pipeline.run search --query "..." --json

# 3. 自定义数据位置（优先级：CLI > 环境变量 KB_PATHS__* > --config yaml > default.yaml）
python -m pipeline.run build --input /data/my_docs --output /data/my_kb
python -m pipeline.run serve --processed /data/my_kb            # 服务指向该知识库
KB_PATHS__PROCESSED_DIR=/data/my_kb python -m pipeline.run search --query "..." --mock

# 4. 演进接口演示（EvoGraph-R1 将通过同一 API 注入）
python -m pipeline.run evolve --ops '[
  {"op": "add_triple",   "payload": {"head": "A", "relation": "r", "tail": "B"}},
  {"op": "remove_entity","payload": {"name": "废弃实体"}},
  {"op": "update_triple","payload": {"head": "A", "relation": "r", "tail": "B", "weight": 5.0}}
]'
# 应用后：图版本提升 → 向量库增量同步 → 下一次检索立即感知新结构

# 5. 测试
python -m pytest tests/ -q

# 6. 统一门面 demo（科研入组筛查 + 知识库问答同一入口）
python -m cohort.seed --count 30   # 生成合成结构化患者 demo 库（data/cohort/demo.db）
# KB 侧默认连真实库（1024 维）；离线下用 --mock + 覆盖到 256 维演示库：
KB_PATHS__PROCESSED_DIR=data/processed_demo_cn python -m unified.demo --mock --json
# 只跑入组筛查（KB 不可用自动降级跳过）：
python -m unified.demo --mock

# 6b. 科研入组筛查单独跑（Python 交互）
python -c "
from cohort.engine import CohortEngine
from cohort.seed import DEMO_IR
r = CohortEngine('data/cohort/demo.db').search('科研入组筛查', DEMO_IR, reference_date='2026-11-18')
print(r.summary)   # {'total': 30, 'PASS': 6, 'FAIL': 20, 'UNKNOWN': 4}
print(r.sql)       # IR 编译出的 SQL
"

# 7. 评测（离线 mock 确定性；real 模式需真实 embedding/LLM 服务）
python -m eval.run_eval retrieval --corpus main --mode mock
python -m eval.run_eval retrieval --corpus multihop --mode mock   # 多跳合成语料
python -m eval.run_eval cohort                                   # 30 例三值混淆（accuracy/F1/UNKNOWN 漏判率）

> **场景文档**：两个检索场景的问题、图/存储结构展示 → [docs/scenarios.md](docs/scenarios.md)；
> 用“查询语言/逻辑模型/执行引擎/物理存储”四层次解释存储选型 → [docs/storage_format_scenarios.md](docs/storage_format_scenarios.md)。
> **检索测试策略**：四层（纯函数 → 黄金图组件 → mock 端到端质量 → 指标评测），
> 详见 `tests/test_retrieval_quality.py` 与 `tests/test_retrieval_golden_graph.py` 的模块 docstring。
```

## 配置（config/default.yaml）

所有参数集中 yaml；可用环境变量 `KB_<SECTION>__<KEY>` 覆盖（双下划线分层），如
`KB_LLM__PROVIDER=mock`、`KB_EMBEDDING__DIM=1024`。

| 段 | 关键项 | 说明 |
|---|---|---|
| `paths` | raw_dir / processed_dir / prompts_dir / logs_dir | ★ 数据位置（CLI/环境变量可覆盖，见 ARCHITECTURE §5） |
| `embedding` | provider: openai\|mock、model、base_url、api_key、dim、batch_size、max_workers | 向量服务（默认 bge-m3 @ localhost:8091，dim 1024） |
| `llm` | provider: openai\|mock、model、base_url、api_key、temperature、max_workers | 抽取/过滤 LLM（默认 qwen3-32b @ localhost:8008） |
| `chunking` | chunk_size 512 / overlap 64 | 文档切块策略 |
| `openie` | require_entity_anchor、resolve_pronouns、batch_size | 两步抽取校验（锚定/代词解析） |
| `retrieval` | triple_seed/entity_seed/vector_seed、hops、max_hop_width、ppr_alpha、final_top_k、llm_filter_enable、use_triple_expand | 检索参数（Green-ChunkHop 式双路+扩散+PPR） |
| `evolution` | enabled、strategies、auto_sync_vectorstore | ★ 演进策略注入（EvoGraph-R1 对接点） |
| `web` | host、port 5570、graph_node_limit | 可视化服务 |

**Embedding / LLM 均经 yaml 配置生效**：换模型只改 `model/base_url/api_key`；
完全离线时 provider 改 `mock`（确定性规则实现，保证 CI 可跑）。

## 两步式 OpenIE（HippoRAG2 协议）

1. **NER**（`prompts/ner_{lang}.txt`）：`{passage}` → `{"named_entities": [...]}`
2. **RE**（`prompts/triple_extraction_{lang}.txt`）：`{passage}` + `{named_entity_json}` →
   `{"triples": [["主语","谓语","宾语"], ...]}`

抽取器强制：**每个三元组至少锚定实体列表中一个实体**（`require_entity_anchor`），
**代词解析为具体名称**（`resolve_pronouns`）。模板整体可替换——换领域只换 `prompts/`，代码不动。
语言由 `project.language` 决定（`chinese`/`english` 对应 `_zh`/`_en` 后缀，回退无后缀模板）。

## 检索链路（Green-ChunkHop 式）

```
query → embedding
  ├─ 并行三路种子召回：三元组库 / 实体库 / chunk 向量库（线程并行）
  ├─ 实体→chunk 定位（text_conclude 边，分数=Σ实体分/度）
  ├─ CHUNK_HOP 扩散：共享实体 chunk 邻接 BFS（weight = shared_count × (1+查询实体交集)）
  ├─ 三元组 n 跳扩展（可选，分数逐跳衰减）
  ├─ PPR：实体+chunk 混合子图上 networkx.pagerank（personalized）
  ├─ LLM 过滤候选三元组（prompts/retrieval_filter_*.txt；失败回退词面重叠）
  └─ 输出：filtered_triples + ranked chunks + 可视化子图（nodes/edges JSON）
```

## EvoGraph-R1 对接预留（★ 核心设计）

后续接入自进化能力**无需重构任何核心模块**，三个层面均已预留：

1. **图结构层**（`graph/knowledge_graph.py` + `graph/evolution.py`）
   - `apply_operation(GraphOperation)` 统一入口，8 种操作：
     `add_triple / remove_triple / update_triple / add_entity / remove_entity /
      update_entity / add_synonym_edge / batch`
   - 每次操作 `version` 单调递增并写入 changangelog（审计/回放/策略学习的数据基础）
   - `GraphEvolutionAPI.apply(ops)`：应用操作列表 → **自动增量同步向量库**（只嵌入新增项）
     → 返回 `{applied, failed, version, stats}`

2. **流水线层**（`pipeline/hooks.py` 事件一览）
   - `EvolutionManager` 按序广播 `PIPELINE_EVENTS`：
     `pipeline_start / before_chunking / after_chunking / before_extract / on_extract_batch /
      after_extract / before_graph_build / after_graph_build / before_vectorize /
      after_vectorize / pipeline_complete / on_error`
   - 策略在 `on_extract_batch` 等注入点可**即时返回图操作**，流水线在安全点统一应用
   - 接入方式：实现 `EvolutionStrategy.on_event(event, ctx) -> List[GraphOperation]`，
     yaml 配置 `evolution.strategies: ["my_pkg.mod:MyStrategy"]` 即生效（见 pipeline/hooks.py 示例）

3. **检索感知层**
   - `HopExpander` 邻接缓存绑定 `graph.version`，图演进后自动失效重建
   - `Retriever._ensure_fresh()`：图文件更新（外部进程演进）自动重载
   - 召回结果与当前图做存在性校验：被演进删除的三元组/实体自动失效
   - Web `POST /api/evolve` 已演示完整闭环：应用操作 → 版本提升 → 向量同步 → 下次检索感知

## Web 可视化

`python -m pipeline.run serve [--mock]` → http://localhost:5570

- **检索子图**：实体节点（命中高亮）、chunk 节点（seed 来源着色：绿=向量路、橙=三元组路、
  紫=未命中）、三元组边（LLM 过滤保留的金色加粗）、CHUNK_HOP 边（橙虚线，标注共享实体）、
  text_conclude 定位边（灰虚线）
- **结果面板**：LLM 过滤后三元组 / 命中三元组（含扩展跳数）/ PPR 排序 chunk
- **演进面板**：直接输入图操作 JSON 调用 `/api/evolve`，全图即时刷新
- **全图浏览**：读取 `data/processed/graph_export.json`（节点上限 `web.graph_node_limit`）

## 数据流与产物

```
data/raw/*.jsonl ──build──▶ data/processed/
                             ├── graph.pkl            # NetworkX 图（增量构建时加载）
                             ├── graph_export.json    # web 全图数据
                             ├── entities.faiss + .meta.json
                             ├── triples.faiss  + .meta.json
                             ├── chunks.faiss   + .meta.json
                             ├── openie_results.jsonl # 抽取明细
                             └── build_report.json    # 构建报告
```

## 已验证（mock 离线端到端）

- [x] 两步 prompt 分离且可配置（中/英模板渲染、占位符替换、热替换）
- [x] 三元组建图 + 实体/三元组/chunk 三库入库，图与库计数一致
- [x] pipeline 端到端跑通（流式并行抽取 + 逐批构图 + 增量向量化），演进钩子全事件链路可用
- [x] 检索链路跑通：双路召回 → hop 扩散 → PPR → LLM 过滤（Top1 chunk 命中、多跳查询可达）
- [x] 演进闭环：add/remove 三元组 → 版本提升 → 向量库增量同步 → 检索感知（新增可召回、删除自动失效）
- [x] HTML 子图可视化 + 演进操作面板（/api/search、/api/evolve、/api/graph 实测通过）
- [x] Embedding/LLM 均经 yaml 配置生效（openai 真实模式 + mock 离线模式切换）
- [x] 增量构建：无新文档时零 embedding、图版本不变

测试：`python -m pytest tests/ -q`（9 个用例，覆盖以上全部验收点）
