# h2official 官方链复现笔记（Green-ChunkHop ablation/main 官方代码）

本文件记录：把我方抽取数据（h2prompt，deepseek + HippoRAG2 官方 prompt）灌进
**Green-ChunkHop `ablation/main` 官方构图/检索链**的全过程，含踩坑、修复、
最终数字与可复现命令。

## 1. 最终结果（Green 口径，fuzzy 0.8，1000 题全量）

| 配置 | R@5 | R@10 | R@15 | R@20 | R@30 |
|---|---|---|---|---|---|
| 作者原图（musique_llama，官方 hpopt 配置） | 0.6328 | 0.7332 | 0.7862 | 0.8186 | 0.8481 |
| 作者原图（Green 口径同链复跑） | 0.6171 | 0.7287 | 0.7821 | 0.8191 | 0.8538 |
| **h2official 官方链（本轮最佳）** | **0.5962** | **0.6960** | **0.7421** | **0.7727** | **0.8177** |
| h2green（自研转换链，同数据） | 0.5972 | 0.6852 | 0.7436 | 0.7759 | 0.8173 |
| h2official 修复前（stale mapping） | 0.0471 | — | — | — | — |

- 与作者图差距：**2.1pt@5 / 3.3pt@10 / 4.0pt@15 / 4.6pt@20 / 3.6pt@30**。
- 结论：官方链已把 h2official 数据拉到**与自研转换链（h2green）持平并小幅超越 @10/@30**；
  剩余差距归因**数据层**（作者用 llama 抽取，137100 三元组 / 88470 实体；
  我方 deepseek + HippoRAG2 prompt，132165 三元组 / 102748 实体，实体更碎、量少 3.7%）。
- 复现稳定性：最佳配置两次全量跑 `0.5970` / `0.5962`（±0.0008）。

最佳配置参数：
```
--hs 对应用 env: TSC=5 VSC=200 GRAPH_SUFFIX=_t NO_OVERLAY 关掉默认 overlay,
改为 DENSE_HOP_ADJ=_overlay/idf_triple_adjacency_dict.pkl DENSE_HOP_K=20 DENSE_HOP_SIM_MIN=0.0
其余仍为 hpopt full: top_k 50, ppr_alpha 0.8, dense_final_weight 0.03,
bridge 32/50/0.65, projected_hop_strength 1.0, hops 3, fuzzy 0.8
```

## 2. 官方链完整流程（命令）

工作目录：
- 官方代码：`/mnt/e/yf/worktrees/kg_ablation_main/Green-ChunkHop`（HEAD 58a8d82）
- 数据/产物：`/mnt/e/yf/knowledge_self_evolution/data/h2official`
- Green 口径 runner：`/mnt/e/yf/knowledge_self_evolution/tools/run_hpopt_doclevel.py`

```bash
P=/home/expelliarmus/miniconda3/envs/chunkhop/bin/python     # faiss + igraph
OUT=/mnt/e/yf/knowledge_self_evolution/data/h2official
REPO=/mnt/e/yf/worktrees/kg_ablation_main/Green-ChunkHop

# 1) h2prompt 产物 -> networkx（本篇修复后的版本：n-gram text_conclude + title 实体 + MultiDiGraph）
$P tools/h2_to_nx.py \
   --triples data/processed_musique_h2prompt/triples.jsonl \
   --entities data/processed_musique_h2prompt/entities.json \
   --chunks data/processed_musique_h2prompt/chunks.jsonl \
   --out $OUT/musique_h2_graph_t.pkl

# 2) 官方链：nx -> jsonl -> igraph -> CHUNK_HOP 边
cd $REPO
$P src/build_graph/convert_nx_to_jsonl.py --input-pkl $OUT/musique_h2_graph_t.pkl --output-jsonl $OUT/graph_export_t.jsonl
$P src/build_graph/build_graph.py --json-filepath $OUT/graph_export_t.jsonl --save-path $OUT/graph_export_t
$P src/build_graph/build_passage_hop.py --graph-path $OUT/graph_export_t --min-shared-entities 5 --save-suffix _with_chunk_hop_t
cp $OUT/graph_export_t_with_chunk_hop_t.pkl $OUT/graph_export_with_chunk_hop_t.pkl

# 3) triple_mapping 必须与当前图同源重建（顶点 id 随图变化）
cd /mnt/e/yf/knowledge_self_evolution
OUT=$OUT GRAPH=$OUT/graph_export_with_chunk_hop_t.pkl $P tools/gen_triple_mapping_official.py

# 4) 向量（进程内 BGE，等价官方 vector_builder；官方脚本受并发/HTTP 限制）
#    passages 可复用 doclevel 向量（#\d+ 归一化命中 11647/11647）；triples 用文本级缓存复用
OUT=$OUT /mnt/e/yf/models/.venv-bge/bin/python tools/embed_official_vectors.py   # 首次全量
OUT=$OUT OLD_MAP=$OUT/triple_mapping_m1.json NEW_MAP=$OUT/triple_mapping.json \
  OLD_VEC=$OUT/triple_vecs_m1.npy NEW_VEC=$OUT/triple_vecs.npy \
  /mnt/e/yf/models/.venv-bge/bin/python tools/reuse_triple_vecs.py               # 增量复用

# 5) FAISS（FlatIP + HNSW M32/efC200/efS256）
OUT=$OUT $P tools/build_faiss_official.py

# 6) dense overlay（idf_triple 邻接，作者 0.6171 配置的一部分）
$P tools/build_overlay_h2.py            # 产物 _overlay_t_runtime/idf_triple.adjacency.pkl
#    转 dict 格式 {i:[(j,sim),...]}（retriever 要 dict，不是 list records）
$P - <<'EOF'
import pickle; from collections import defaultdict
recs = pickle.load(open('data/h2official/_overlay_t_runtime/idf_triple.adjacency.pkl','rb'))
adj = defaultdict(list)
for a,b,w,eff,names in recs:
    adj[a].append((b,w)); adj[b].append((a,w))
for i in adj: adj[i].sort(key=lambda x:-x[1])
pickle.dump(dict(adj), open('data/h2official/_overlay/idf_triple_adjacency_dict.pkl','wb'))
EOF

# 7) Green 口径全量评测
cd $REPO/exp_weight_ablation
env REPO=$PWD DOC=$OUT DATASET=musique_h2official GRAPH_SUFFIX=_t \
    DENSE_HOP_ADJ=$OUT/_overlay/idf_triple_adjacency_dict.pkl \
    DENSE_HOP_K=20 DENSE_HOP_SIM_MIN=0.0 TSC=5 VSC=200 \
    $P /mnt/e/yf/knowledge_self_evolution/tools/run_hpopt_doclevel.py
# 结果：results/musique_h2official_chunk_hop_eval_results.json
```

前置：`exp_weight_ablation/dataset/musique_h2official` 软链到 `data/h2official`
（runner 从 `dataset/<DATASET>/` 读 graph/faiss/mapping）。

## 3. 关键修复（按出现顺序，全是硬坑）

1. **`original_text` 节点名必须是 `"title\n全文"`**
   官方链的 fuzzy 内容匹配靠这个；只放 title → R@5 0.3739。
2. **图重建后 triple_mapping / triple 向量必须同源重建**
   顶点 id 每次 build_graph 都变（本次 `Erik Hort` 旧 mapping head_id=3299，新图=13644），
   stale mapping 让 `locate_chunks_from_entities` 全部错位 → R@5 = 0.0471。
   修法：`tools/gen_triple_mapping_official.py`（语义对齐官方 `vector_builder.py` L128，
   head/tail 文本取顶点 `text`，fallback `id`）。
3. **`run_hpopt_doclevel.py` 默认自动追加 overlay 参数**
   默认 `--dense-hop-adj DOC/_overlay/idf_triple.adjacency.pkl --dense-hop-k 20`，
   不设 `NO_OVERLAY=1` 且 overlay 是 list 格式 → `AttributeError: 'list' object has no attribute 'items'`。
4. **dense overlay 双要件**：`--dense-hop-k > 0`（默认 0 = 不加载）+ 邻接文件必须是
   **dict** `{i: [(j, sim), ...]}`（list records 是 `build_rare_triple_edges.py` 的原始格式）。
   weight 尺度 = `0.01 × rarity score`（中位 0.018），所以 `sim_min` 用 **0.0**，
   用默认 0.3 会把 99% 边滤掉。
5. **官方 `vector_builder.py` 的 `_embed_texts` 硬编码 32 并发** → 必须限流；
   本机 8800 嵌入服务在 GTX 1070 Ti 上反复崩（`ThreadingHTTPServer` 默认 queue=5 也会拒连），
   最终改**进程内编码**（`tools/embed_official_vectors.py` + `tools/build_faiss_official.py`），
   产物与官方等价。
6. **大预算 CHUNK_HOP 不是瓶颈**：`--min-shared-entities 1`（100000 条 / 50000 对）
   与 `--min-shared-entities 5`（2804 条 / 1402 对）在修复 mapping 后**几乎同分**
   （0.5751 vs 0.5729@5）。作者 0.63 系的高分来源不是 CHUNK_HOP 密度。
7. **n-gram text_conclude 不是瓶颈**：加 n-gram 匹配（394213 边，对齐 h2green 的 386372）
   只 +0.1pt（0.5751 → 0.5762）。
8. **title 实体不是瓶颈**：+5541 title 实体只 +0.1pt（0.582）。
9. **MultiDiGraph 保留同 (head,tail) 多关系**：triple 边 127098 → 132165（+5067），
   +0.6pt（0.5762 → 0.582）。用文本级缓存复用只需重编码 5067 条（21s）。
10. **真正有效的是检索参数**：
    - `triple_seed_count` 10 → **5**：@5 +0.5pt（种子更纯）
    - `vector_seed_count` 50 → **200**：@5/@10 再 +0.4pt
    - 组合后 0.5877 → **0.5962@5**；过度（TSC=20）会崩到 0.5619。

## 4. 消融小结（同一图 + 同链，n=1000）

| 变更 | R@5 | R@30 |
|---|---|---|
| 修复 stale mapping（mse5 图） | 0.5729 | 0.8099 |
| + CHUNK_HOP 大预算 100000 | 0.5751 | 0.8083 |
| + n-gram text_conclude | 0.5762 | 0.8106 |
| + MultiDiGraph 全关系 132165 | 0.5820 | 0.8121 |
| + title 实体 5541 | 0.5830 | 0.8117 |
| + dense overlay（dict, K20, sim 0） | 0.5877 | 0.8171 |
| + TSC=5 / VSC=200（最佳） | **0.5962** | **0.8177** |

## 5. 产物清单（`data/h2official`）

| 文件 | 说明 |
|---|---|
| `musique_h2_graph_t.pkl` | nx 图（MultiDiGraph，114395 节点 / 532474 边） |
| `graph_export_t.pkl` | 官方 igraph 图（108854→114395 顶点 / 632474~526378 边，视 suffix） |
| `graph_export_with_chunk_hop_t.pkl` | +100000 CHUNK_HOP 边（GRAPH_SUFFIX=_t 用这份） |
| `triple_mapping.json` | 132165 条，head_id/tail_id 与当前图同源 |
| `triple_vecs.npy` | 132165×1024，zero=0 |
| `vdb_triples.faiss` / `vdb_triples_hnsw.faiss` | 132165 向量 |
| `vdb_passages*.faiss` | 11647 向量（doclevel 复用） |
| `_overlay_t_runtime/idf_triple.adjacency.pkl` | 171977 对 / 343954 有向（list records） |
| `_overlay/idf_triple_adjacency_dict.pkl` | 转 dict，retriever 可直接消费 |
| `triple_mapping_m1.json` + `triple_vecs_m1.npy` | 127098 条旧口径备份（增量复用用） |

## 6. 未做/可继续

- **实体规范化合并**（原文 97207 → 规范 95560，仅 1.7%，预期 <0.5pt，成本 35min 重嵌入）。
- **实体碎片化治理**（我方 102748 实体 vs 作者 88470）：这是剩余 2pt 的主嫌，但属数据层。
- **改用作者 llama 数据**才能验证"链已追平、差距纯数据"的最终结论。
