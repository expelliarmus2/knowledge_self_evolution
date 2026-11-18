#!/usr/bin/env python
"""重新构图：deepseek one_step 抽取三元组 + 整文档 passage（不切分语料）。

对齐作者图（graph_export_with_chunk_hop.pkl）处理：
  - passage = 每篇文档整段（"title\\ntext"），11656 篇（与作者图一致，非 17262 切分块）
  - 实体 = 三元组 head/tail + passage titles（去 GENERIC 单字）
  - text_conclude 边：实体短语 n-gram(1..5) 精确匹配 passage 文本（entity -> passage）
  - 三元组边：head -> tail（relation = 关系名）
  - CHUNK_HOP 边：共享实体 >= min_shared_entities(5，对齐作者图) 的 passage 对（双向，
    weight = shared_count/100）
  - 桥边/projected 边（对齐 rare_triple 检索语义，构图期预计算）：
      bridge_entities.json：degree ∈ [2,50] 且支撑 >=2 passage 的实体表
      项目格式图中预建 bridge_entity_to_chunk / bridge_chunk_to_entity /
      entity_projected_chunk 边（全图版，weight 对齐 rare_triple）

产物（data/processed_musique_doclevel/）：
  1. graph_export.json        （作者 JSONL 格式，节点/关系）
  2. graph_export_with_chunk_hop.pkl （作者 igraph 格式，可直接供 Green-ChunkHop 评测）
  3. graph.pkl                （项目 KnowledgeGraph 兼容格式：整文档 chunk + 全边类型）
  4. bridge_entities.json     （构图期预计算桥实体表）
  5. passage_texts.txt / triple_texts.txt + triple_mapping_raw.json （供嵌入阶段）
  6. build_report.json

用法（chunkhop env，含 igraph/networkx/faiss）：
  python tools/build_doclevel_graph.py
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJ = Path("/mnt/e/yf/knowledge_self_evolution")
WORKTREE = Path("/mnt/e/yf/worktrees/ablation_main/Green-ChunkHop")
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(WORKTREE / "src" / "build_graph"))

from build_graph import build_graph_from_file, load_graph, save_graph  # noqa: E402
from build_passage_hop import ChunkHopBuilder  # noqa: E402

import igraph as ig  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(os.environ.get("OUT", str(PROJ / "data" / "processed_musique_doclevel")))

CORPUS = PROJ / "data" / "raw" / "corpus_musique.json"
TRIPLES = Path(os.environ.get("TRIPLES", str(PROJ / "data" / "processed_musique_onestep" / "triples.jsonl")))
CHUNKS = Path(os.environ.get("CHUNKS", str(PROJ / "data" / "processed_musique_onestep" / "chunks.jsonl")))

MIN_SHARED = 5  # 对齐作者 musique 图 stats.json: min_shared_entities=5
BRIDGE_MAX_DEGREE = 50
BRIDGE_MIN_DEGREE = 2
NGRAM_MAX = 5

GENERIC = {
    "film", "movie", "war", "album", "song", "band", "novel", "city",
    "united", "states", "american", "british", "french", "german",
}


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def norm_phrase(text: str) -> tuple[str, ...]:
    ws = [norm_word(w) for w in text.split()]
    return tuple(w for w in ws if w)


def main():
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- 1. 数据读取
    corpus = json.load(open(CORPUS, encoding="utf-8"))
    print(f"corpus: {len(corpus)} 篇")
    by_title: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(corpus):
        by_title[c["title"]].append(i)

    def map_doc(doc_id: str):
        m = re.match(r"^(.*?)(?:#(\d+))?$", doc_id)
        title, n = m.group(1), int(m.group(2) or 1)
        lst = by_title.get(title, [])
        return lst[n - 1] if n <= len(lst) else None

    chunk2doc = {}
    for l in open(CHUNKS, encoding="utf-8"):
        c = json.loads(l)
        chunk2doc[c["chunk_id"]] = c["doc_id"]

    # 三元组聚合到文档级
    doc_triples: dict[int, set] = defaultdict(set)
    n_skipped = 0
    for l in open(TRIPLES, encoding="utf-8"):
        t = json.loads(l)
        heads, tails = set(), set()
        for cid in t.get("chunk_ids", []):
            doc = map_doc(chunk2doc.get(cid, ""))
            if doc is not None:
                heads.add(doc)
                tails.add(doc)
        if not heads:
            n_skipped += 1
            continue
        h, r, tail = t["head"], t["relation"], t["tail"]
        for doc in heads | tails:
            doc_triples[doc].add((h, r, tail))
    print(f"三元组文档级聚合完成：{sum(len(v) for v in doc_triples.values())} 条（跳过 {n_skipped} 条无文档归属）")

    # ------------------------------------------- 2. passages / triples 列表
    passage_titles: list[str] = []
    passages: dict[int, str] = {}  # pid -> "title\ntext"
    for i, c in enumerate(corpus):
        passage_titles.append(c["title"])
        passages[i] = c["title"] + "\n" + c["text"]

    triples: list[tuple[str, str, str]] = []
    for doc, ts in sorted(doc_triples.items()):
        triples.extend(ts)
    # 文档级去重（保留顺序）
    seen_t: set[tuple] = set()
    uniq = []
    for h, r, t in triples:
        k = (norm_phrase(h), r, norm_phrase(t))
        if k not in seen_t:
            seen_t.add(k)
            uniq.append((h, r, t))
    triples = uniq
    print(f"文档级三元组（去重后）: {len(triples)}")

    # ------------------------------------------- 3. 实体集
    ent_display: dict[tuple[str, ...], str] = {}
    for h, _r, t in triples:
        for name in (h, t):
            ph = norm_phrase(name)
            if ph and ph not in ent_display:
                ent_display[ph] = name
    for title in passage_titles:
        ph = norm_phrase(title)
        if ph and ph not in ent_display:
            ent_display[ph] = title
    ent_display = {
        k: v for k, v in ent_display.items()
        if not (len(k) == 1 and k[0] in GENERIC)
    }
    print(f"实体集: {len(ent_display)}（三元组 head/tail + titles）")

    # ------------------------------------------- 4. text_conclude n-gram 匹配
    print("实体<->passage n-gram 匹配 ...")
    ent_by_first: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for ph in ent_display:
        ent_by_first[ph[0]].add(ph)
    ent_to_passages: dict[tuple[str, ...], set[int]] = defaultdict(set)
    for pid, text in passages.items():
        ws = [w for w in (norm_word(w) for w in text.split()) if w]
        seen_ng: set[tuple[str, ...]] = set()
        for n in range(1, NGRAM_MAX + 1):
            for i in range(len(ws) - n + 1):
                ng = tuple(ws[i:i + n])
                if ng in seen_ng:
                    continue
                seen_ng.add(ng)
                for ent in ent_by_first.get(ng[0], ()):
                    if ent == ng:
                        ent_to_passages[ent].add(pid)
    n_tc = sum(len(v) for v in ent_to_passages.values())
    print(f"text_conclude 边: {n_tc}（实体数 {len(ent_to_passages)}）")

    # ------------------------------------------- 5. 写 graph_export.json
    print("写 graph_export.json ...")
    node_id_of_ent: dict[tuple[str, ...], str] = {}
    n_passages = len(passage_titles)
    lines = []
    for pid, title in enumerate(passage_titles):
        lines.append(json.dumps({
            "type": "node", "id": str(pid), "labels": ["Node", "OriginalText"],
            "properties": {
                "node_type": "original_text", "text": passages[pid],
                "doc_id": title, "chunk_id": "", "type": "original_text",
                "file_path": title,
            },
        }, ensure_ascii=False))
    for ei, (ph, name) in enumerate(sorted(ent_display.items(), key=lambda x: x[1])):
        eid = str(n_passages + ei)
        node_id_of_ent[ph] = eid
        lines.append(json.dumps({
            "type": "node", "id": eid, "labels": ["Node", "Entity"],
            "properties": {"node_type": "entity", "text": name, "doc_id": "",
                           "chunk_id": "", "type": "entity"},
        }, ensure_ascii=False))
    # text_conclude
    for ph, pids in ent_to_passages.items():
        eid = node_id_of_ent[ph]
        name = ent_display[ph]
        for pid in sorted(pids):
            lines.append(json.dumps({
                "type": "relationship", "id": hashlib.md5(
                    f"tc:{eid}:{pid}".encode()).hexdigest()[:10],
                "label": "RELATED_TO",
                "properties": {"relation": "text_conclude", "weight": 1.0},
                "start": {"id": eid, "labels": ["Node", "Entity"],
                          "properties": {"node_type": "entity", "text": name}},
                "end": {"id": str(pid), "labels": ["Node", "OriginalText"],
                        "properties": {"node_type": "original_text",
                                       "text": passages[pid][:80]}},
            }, ensure_ascii=False))
    # triple 边
    ent_id_by_name = {ent_display[ph]: eid for ph, eid in node_id_of_ent.items()}
    n_tr = 0
    for h, r, t in triples:
        hp, tp = norm_phrase(h), norm_phrase(t)
        if hp not in node_id_of_ent or tp not in node_id_of_ent:
            continue
        lines.append(json.dumps({
            "type": "relationship", "id": hashlib.md5(
                f"tr:{node_id_of_ent[hp]}:{r}:{node_id_of_ent[tp]}".encode()).hexdigest()[:10],
            "label": "RELATED_TO",
            "properties": {"relation": r, "weight": 1.0},
            "start": {"id": node_id_of_ent[hp], "labels": ["Node", "Entity"],
                      "properties": {"node_type": "entity", "text": h}},
            "end": {"id": node_id_of_ent[tp], "labels": ["Node", "Entity"],
                    "properties": {"node_type": "entity", "text": t}},
        }, ensure_ascii=False))
        n_tr += 1
    jsonl_path = OUT / "graph_export.json"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"graph_export.json: {n_passages + len(node_id_of_ent)} 节点 + {n_tc + n_tr} 关系"
          f"（text_conclude={n_tc}, triple={n_tr}）")

    # ------------------------------------------- 6. igraph + CHUNK_HOP
    print("构建 igraph（作者 build_graph 管线）...")
    g = build_graph_from_file(str(jsonl_path))
    print(f"igraph: {g.vcount()} 节点, {g.ecount()} 边")
    builder = ChunkHopBuilder(
        g, min_shared_entities=MIN_SHARED, top_k_chunks=50000,
        enable_1hop_edge=False, top_k_connections=50,
    )
    pairs = builder.find_shared_entity_pairs_fast()
    print(f"CHUNK_HOP qualifying pairs: {len(pairs)}")
    builder.add_chunk_hop_edges(pairs, include_1hop=False)
    save_graph(g, str(OUT / "graph_export_with_chunk_hop"))
    print(f"graph_export_with_chunk_hop.pkl: {g.vcount()} 节点, {g.ecount()} 边")

    # ------------------------------------------- 7. 项目格式 graph.pkl（全边类型）
    print("构建项目格式 graph.pkl（含 bridge/projected 边）...")
    nxg = _build_project_graph(passage_titles, passages, ent_display, ent_to_passages,
                               triples, node_id_of_ent)
    with open(OUT / "graph.pkl", "wb") as f:
        pickle.dump({
            "g": nxg,
            "version": 1,
            "changelog": [],
            "case_insensitive_entities": False,
            "max_changelog": 1000,
        }, f)
    print(f"graph.pkl: {nxg.number_of_nodes()} 节点, {nxg.number_of_edges()} 边")

    # ------------------------------------------- 8. bridge 实体表
    bridge_entities = {
        "min_degree": BRIDGE_MIN_DEGREE, "max_degree": BRIDGE_MAX_DEGREE,
        "entities": [],
    }
    for ph, pids in ent_to_passages.items():
        deg = len(pids)
        if BRIDGE_MIN_DEGREE <= deg <= BRIDGE_MAX_DEGREE and len(pids) >= 2:
            bridge_entities["entities"].append({
                "name": ent_display[ph], "degree": deg,
                "passages": sorted(pids),
            })
    bridge_entities["entities"].sort(key=lambda e: -e["degree"])
    with open(OUT / "bridge_entities.json", "w", encoding="utf-8") as f:
        json.dump(bridge_entities, f, ensure_ascii=False)
    print(f"bridge 实体表: {len(bridge_entities['entities'])} 个")

    # ------------------------------------------- 9. 嵌入输入
    with open(OUT / "passage_texts.txt", "w", encoding="utf-8") as f:
        for pid in range(n_passages):
            f.write(passages[pid].replace("\n", " ") + "\n")
    triple_texts = []
    triple_mapping_raw = []
    for h, r, t in triples:
        hp, tp = norm_phrase(h), norm_phrase(t)
        if hp not in node_id_of_ent or tp not in node_id_of_ent:
            continue
        triple_texts.append(f"{h} {r} {t}")
        triple_mapping_raw.append({
            "head": h, "head_id": node_id_of_ent[hp], "head_type": "Entity",
            "relation": r,
            "tail": t, "tail_id": node_id_of_ent[tp], "tail_type": "Entity",
        })
    with open(OUT / "triple_texts.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(triple_texts) + "\n")
    with open(OUT / "triple_mapping_raw.json", "w", encoding="utf-8") as f:
        json.dump(triple_mapping_raw, f, ensure_ascii=False)
    print(f"嵌入输入: passage {n_passages} 条 / triple {len(triple_texts)} 条")

    # ------------------------------------------- 10. meta
    meta = {
        "passages": n_passages,
        "entities": len(node_id_of_ent),
        "text_conclude_edges": n_tc,
        "triple_edges": n_tr,
        "chunk_hop_pairs": len(pairs),
        "chunk_hop_edges": len(pairs) * 2,
        "bridge_entities": len(bridge_entities["entities"]),
        "min_shared_entities": MIN_SHARED,
        "triples_doc_level_dedup": len(triples),
        "triples_raw": sum(len(v) for v in doc_triples.values()),
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(OUT / "build_report.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("DONE", json.dumps(meta))


def _build_project_graph(passage_titles, passages, ent_display, ent_to_passages,
                         triples, node_id_of_ent):
    """项目 KnowledgeGraph schema 的 networkx MultiDiGraph（含全部边类型）。"""
    import networkx as nx
    nxg = nx.MultiDiGraph()
    # chunk 节点：整文档（pid -> "doc#pid"? 用 "title" 作为 chunk_id 会出现重复 title 冲突，
    # 故 chunk_id 用 "doc:{pid}"，doc_id 保留 title）
    for pid, title in enumerate(passage_titles):
        nxg.add_node(f"doc:{pid}",
                     type="chunk", text=passages[pid], doc_id=title, order=0, meta={})
    # entity 节点 + text_conclude 边
    for ph, name in ent_display.items():
        nxg.add_node(name, type="entity", display=name, entity_type="unknown",
                     freq=len(ent_to_passages.get(ph, ())), chunk_ids=[], meta={})
    ent_to_pid: dict[str, set[int]] = {}
    for ph, pids in ent_to_passages.items():
        name = ent_display[ph]
        ent_to_pid[name] = set(pids)
        for pid in pids:
            nxg.add_edge(name, f"doc:{pid}", key="text_conclude",
                         relation="text_conclude", count=1)
    # triple 边
    for h, r, t in triples:
        hp, tp = norm_phrase(h), norm_phrase(t)
        if hp not in node_id_of_ent or tp not in node_id_of_ent:
            continue
        nxg.add_edge(h, t, key=r, relation=r, weight=1.0, count=1, chunk_ids=[])
        # triple 关联的 passage 链接（h/t 的 text_conclude 已由 n-gram 覆盖，这里补齐
        # 若 n-gram 未命中但三元组确实在该 passage 抽取时 —— 本项目原逻辑 add_triple 会
        # link_entity_to_chunk，故此处直接跳过（text_conclude 由 n-gram 统一负责））
    # CHUNK_HOP 边：共享实体 >= MIN_SHARED（由 igraph 的 pairs 结果重算）
    # 直接复用 igraph 阶段的 pairs：ent_to_passages 交集 >= MIN_SHARED 的 passage 对
    pid_sets: dict[int, set[str]] = defaultdict(set)
    for ph, pids in ent_to_passages.items():
        name = ent_display[ph]
        for pid in pids:
            pid_sets[pid].add(name)
    pair_cnt = 0
    sorted_pids = sorted(pid_sets.keys())
    for ai in range(len(sorted_pids)):
        a = sorted_pids[ai]
        sa = pid_sets[a]
        for b in sorted_pids[ai + 1:]:
            shared = sa & pid_sets[b]
            if len(shared) >= MIN_SHARED:
                w = len(shared) / 100.0
                nxg.add_edge(f"doc:{a}", f"doc:{b}", key="chunk_hop",
                             relation="chunk_hop", weight=w,
                             shared_entity_count=len(shared),
                             shared_entities=json.dumps(sorted(shared)[:100]))
                nxg.add_edge(f"doc:{b}", f"doc:{a}", key="chunk_hop",
                             relation="chunk_hop", weight=w,
                             shared_entity_count=len(shared),
                             shared_entities=json.dumps(sorted(shared)[:100]))
                pair_cnt += 1
    print(f"  项目图 CHUNK_HOP pairs: {pair_cnt}")
    # bridge / projected 边（对齐 rare_triple 语义，构图期预建）
    n_bridge_e2c = n_bridge_c2e = n_proj = 0
    for ph, pids in ent_to_passages.items():
        deg = len(pids)
        if not (BRIDGE_MIN_DEGREE <= deg <= BRIDGE_MAX_DEGREE and len(pids) >= 2):
            continue
        name = ent_display[ph]
        for pid in sorted(pids):
            cid = f"doc:{pid}"
            if not nxg.has_edge(name, cid, "bridge_entity_to_chunk"):
                nxg.add_edge(name, cid, key="bridge_entity_to_chunk",
                             relation="bridge_entity_to_chunk", weight=1.0)
                n_bridge_e2c += 1
            nxg.add_edge(cid, name, key="bridge_chunk_to_entity",
                         relation="bridge_chunk_to_entity",
                         weight=1.0 / np.sqrt(deg))
            n_bridge_c2e += 1
        # entity_projected_chunk：支撑 passage 两两
        spids = sorted(pids)
        proj_w = 1.0 / deg
        for ai in range(len(spids)):
            for bj in range(ai + 1, len(spids)):
                nxg.add_edge(f"doc:{spids[ai]}", f"doc:{spids[bj]}",
                             key="entity_projected_chunk",
                             relation="entity_projected_chunk", weight=proj_w)
                nxg.add_edge(f"doc:{spids[bj]}", f"doc:{spids[ai]}",
                             key="entity_projected_chunk",
                             relation="entity_projected_chunk", weight=proj_w)
                n_proj += 2
    print(f"  项目图 bridge 边: e2c={n_bridge_e2c} c2e={n_bridge_c2e} projected={n_proj}")
    return nxg


if __name__ == "__main__":
    main()
