# Prompt 模板说明

本目录存放所有可替换的 prompt 模板。**换领域时只换模板文件，不改代码。**

## 命名与加载规则

- 模板名在 `config/default.yaml` 的 `prompts:` 段配置（`ner` / `triple_extraction` / `retrieval_filter`）。
- 加载时按语言自动选择：`{name}_{lang}.txt`（lang 取 `project.language`，即 `zh` / `en`），
  若不存在则回退到 `{name}.txt`。
- 模板为纯文本，支持两种段落标签：
  - `<system>...</system>`：system 消息内容（可省略）
  - `<user>...</user>`：user 消息内容，含占位符
  - 若无标签，整个文件视为 user 内容。

## 占位符

| 模板 | 占位符 |
|---|---|
| `ner` | `{passage}` — 待抽取实体的大文本 |
| `triple_extraction` | `{passage}` — 原文；`{named_entity_json}` — 第一步 NER 的实体列表 JSON |
| `openie_one_step` | `{passage}` — 原文 |
| `retrieval_filter` | `{question}` — 查询；`{candidate_facts_json}` — 候选三元组 JSON；`{max_facts}` — 最多保留条数 |

## 输出协议（代码侧解析依赖，改模板时必须保持）

| 模板 | 输出 |
|---|---|
| `ner` | `{"named_entities": [...]}` |
| `triple_extraction` | `{"triples": [[s, p, o], ...]}` |
| `openie_one_step` | `头<|>关系<|>尾##头<|>关系<|>尾##...`，末尾 `<|COMPLETE|>`（huaxi 分隔符协议） |
| `retrieval_filter` | `{"fact": [[s, p, o], ...]}` |

输出允许被 ```json 代码块包裹，代码会自动清洗。

## 抽取模式（config: openie.extract_mode）

- `two_step`（默认）：先 `ner` 后 `triple_extraction`，两次 LLM 调用，三元组端点锚定实体列表，精度优先。
- `one_step`：单次调用 `openie_one_step` 直接产出三元组，LLM 次数减半，性能优先
  （huaxi_poc 风格；实体从三元组端点汇总，无独立锚定，代词端点兜底丢弃）。

## Agentic Search 提示词（新增）

| 模板 | 占位符 | 输出协议 | 用途 |
|---|---|---|---|
| `ircot_step_{lang}.txt` | `{question}` `{evidence}` `{cot}` | `FINAL ANSWER: <答案>` 或 `NEXT: <一句推理>` | IRCoT 每轮决策 |
| `agentic_plan_{lang}.txt` | `{question}` | JSON 数组 `["子问题1", ...]` | Planner 问题分解 |
| `agentic_answer_{lang}.txt` | `{question}` `{evidence}` `{triples}` | 仅答案本身 | 证据→短答案 |

配套代码：`agentic/ircot.py`、`agentic/planner.py`；配置段 `agentic:`；
评测：`python -m evaluation.eval_agentic --processed <dir> --mode ircot --limit 200`。
