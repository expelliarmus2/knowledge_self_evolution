# HippoRAG2 OpenIE Prompt 落地记录

目标：当前项目（knowledge_self_evolution）的三元组抽取 prompt 参考 HippoRAG2 官方
OpenIE prompt（优化版：NER-conditioned RE + one-shot 多轮示例）。

## 改动

1. `utils/prompts.py`（备份 `utils/prompts.py.bak`）
   - 新增 `load_messages()`：支持 `<system>/<user>/<assistant>` 多轮模板解析
   - `render()` 改为按消息块顺序输出（原仅 system+user 两轮）
   - `load()` 保持旧签名兼容

2. `prompts/ner_en.txt`（备份 `.bak`）＝ HippoRAG2 `templates/ner.py`
   - system：从段落抽取命名实体，输出 `{"named_entities": [...]}`
   - one-shot：Radio City 段落 → 实体列表

3. `prompts/triple_extraction_en.txt`（备份 `.bak`）＝ HippoRAG2 `templates/triple_extraction.py`
   - system：从段落+实体列表构建 RDF 图，输出 `{"triples": [["s","p","o"],...]}`
   - 要求：每条三元组至少含一个（最好两个）给定实体；代词解析；简洁自然短语谓语；
     只陈述文中明确事实
   - one-shot：Radio City 段落 + 实体列表 → 12 条三元组

## 全量重建（deepseek-v4.1-flash-1017，two_step，整文档 passage）

产物：`data/processed_musique_h2prompt/`（build_report.json：11656 docs / 11656 chunks /
135713 triples extracted / 101431 entities / 图内 132165 triple 边 / 3234s）

| 指标 | one_step(doclevel) | h2prompt | 作者图 |
|---|---|---|---|
| 抽取三元组 | 100555(99247) | **135713(132165)** | 137100 |
| 实体 | 81636 | 101431 | 88470 |
| 唯一 relation | 22027 | 33585 | - |
| shared>=5 文档对候选 | 172079（碎片） | **2243** | 1398 |

结论：HippoRAG2 prompt 把三元组数量拉回作者水平（差 3.6%，one_step 差 27%）且**实体
碎片化大幅缓解**（候选池 2243 ≈ 作者 1398，one_step 因碎片实体共享≥5 遍地达 172079）。
副作用：chunk_hop 预建边几乎无处可加（Coverage10K 仅覆盖 13% chunk，对 recall 无增益）。

## 评测

### 线 A：Green-ChunkHop 作者口径（fuzzy 0.8 内容匹配，hpopt full + idf_triple overlay，1000 题）

| 图 | @5 | @10 | @15 | @20 | @30 |
|---|---|---|---|---|---|
| **h2prompt (HippoRAG2 prompt)** | **0.5972** | 0.6852 | 0.7436 | 0.7759 | 0.8173 |
| doclevel one_step + full overlay | 0.5721 | 0.6820 | 0.7347 | 0.7675 | 0.8123 |
| doclevel + Coverage10K | 0.5716 | 0.6819 | 0.7339 | 0.7675 | 0.8132 |
| 作者图 + hpopt full | 0.6171 | 0.7287 | 0.7821 | 0.8191 | 0.8538 |
| deepseek 切分 + hpopt 迁移 | 0.5697 | 0.6577 | 0.6960 | 0.7187 | 0.7442 |

→ 同一 deepseek 抽取管线（整文档）下，换 HippoRAG2 prompt 使 @5 **+2.5pt**（0.5721→0.5972），
全 K 反超 one_step；与作者图差距从 4.5pt 缩到 2.0pt@5（@10+ 仍差 4.4~5.9pt，根源仍在
三元组/实体数据层与图结构）。

### 线 B：项目 dual_seed 口径（title 级 doc_recall，supporting-only，1000 题）

| 图 | @5 | @10 | @15 | @20 | @30 |
|---|---|---|---|---|---|
| h2prompt（无预建边） | 0.5745 | 0.6631 | 0.6865 | 0.7023 | 0.7544 |
| h2prompt + Coverage10K(ms5) | 0.5680 | 0.6600 | 0.6855 | 0.7023 | 0.7545 |
| h2prompt + Coverage10K(ms3) | 0.5673 | 0.6562 | 0.6844 | 0.7023 | 0.7552 |
| doclevel one_step 纯在线 | 0.5978 | 0.6627 | 0.7083 | 0.7237 | 0.7237 |
| doclevel one_step + Coverage10K | 0.6182 | 0.6822 | 0.7158 | 0.7542 | 0.8221 |

→ 项目口径下 h2prompt @5 略低（-2.3pt）但 **@30 高 +3.1pt**（0.7544 vs 0.7237）。
原因：项目 dual_seed 的在线 hop 依赖共享实体（碎片实体到处命中→doclevel 在线 hop 强），
h2prompt 实体规范后在线 hop 弱、长尾反而更广。

## 两口径结论相反的原因（与 doclevel 一致的模式）

- Green 口径（fuzzy 内容匹配）：惩罚整文档 passage 的 CLS 区分度，奖励干净三元组/实体
  → HippoRAG2 prompt 赢。
- 项目口径（title 级）：对整文档友好，奖励稠密 hop 边 → 碎片实体+预建边（doclevel）赢。

## 复现命令

```bash
# 模板已在仓库内；全量重建（32 并发 two_step，整文档）：
KB_OPENIE__EXTRACT_MODE=two_step KB_CHUNKING__CHUNK_SIZE=1000000 \
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python \
  -m pipeline.run build --input data/raw/corpus_musique.json \
  --output data/processed_musique_h2prompt --full

# Green 口径（作者评测器）：
OUT=data/processed_musique_h2green \
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/build_doclevel_graph.py
OUT=data/processed_musique_h2green \
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/reorder_igraph.py
OUT=data/processed_musique_h2green \
  /mnt/e/yf/models/.venv-bge/bin/python tools/embed_doclevel.py
OUT=data/processed_musique_h2green \
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/build_faiss_doclevel.py
DOC=data/processed_musique_h2green DATASET=musique_h2green \
  /home/expelliarmus/miniconda3/envs/chunkhop/bin/python tools/run_hpopt_doclevel.py

# 项目口径：
/home/expelliarmus/miniconda3/envs/chunkhop/bin/python -m evaluation.eval_recall \
  --processed data/processed_musique_h2prompt --test data/test/musique_dev.json \
  --supporting-only --top-ks 5,10,15,20,30 --limit 1000 \
  --out evaluation/results/h2prompt_dualseed
```

## 结果文件

- Green：`worktrees/ablation_main/Green-ChunkHop/exp_weight_ablation/results/musique_h2green_chunk_hop_eval_results.json`
- 项目：`evaluation/results/h2prompt_dualseed/recall_20261114_005822.json`、
  `h2prompt_dualseed_b10k/recall_20261114_010807.json`、`h2prompt_ms3_b10k/recall_20261114_011808.json`
- 图：`data/processed_musique_h2prompt{,_b10k,_b10k_ms3}/`、`data/processed_musique_h2green/`
- 日志：`logs_build_h2prompt.log`、`logs_embed_h2green.log`、`logs_green_h2green.log`、
  `logs_overlay_h2green.log`、`logs_eval_h2prompt*.log`
