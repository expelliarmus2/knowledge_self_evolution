"""h2prompt (HippoRAG2 prompt 抽取) 三元组 → 官方 ablation/main networkx 图格式。

输出与 data/musique_llama/musique_llama_graph.pkl 同构：
- node: name=文本, attrs {type: 'original_text'|'entity', file_id, chunk_id}
- edge: text_conclude(实体↔文档) + relation 三元组边(head→tail, weight)

用法:
  python tools/h2_to_nx.py \
    --triples data/processed_musique_h2prompt/triples.jsonl \
    --entities data/processed_musique_h2prompt/entities.json \
    --chunks data/processed_musique_h2prompt/chunks.jsonl \
    --out data/h2official/musique_h2_graph.pkl
"""
import argparse, json, os, pickle, re
from collections import defaultdict
import networkx as nx

NGRAM_MAX = 5
GENERIC = {
    "film", "movie", "war", "album", "song", "band", "novel", "city",
    "united", "states", "american", "british", "french", "german",
}


def norm_word(w):
    return re.sub(r"[^a-z0-9]", "", str(w).lower())


def norm_phrase(text):
    ws = [norm_word(w) for w in str(text).split()]
    return tuple(w for w in ws if w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # chunk_id -> (doc_id, text)
    chunk2doc = {}
    doc_texts = {}
    with open(args.chunks, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunk2doc[c["chunk_id"]] = c["doc_id"]
            doc_texts[c["doc_id"]] = doc_texts.get(c["doc_id"], "") + (" " if doc_texts.get(c["doc_id"]) else "") + c["text"]
    print(f"chunks: {len(chunk2doc)}, docs with text: {len(doc_texts)}")

    # triples
    triples = []
    with open(args.triples, encoding="utf-8") as f:
        for line in f:
            triples.append(json.loads(line))
    print(f"triples: {len(triples)}")

    ents = json.load(open(args.entities, encoding="utf-8"))

    g = nx.MultiDiGraph()

    # 文档节点（original_text）：name = "title\n全文"（与作者 musique_llama_graph 同构）
    node_of_doc = {}
    for d in sorted(doc_texts):
        name = f"{d}\n{doc_texts[d]}"
        node_of_doc[d] = name
        g.add_node(name, type="original_text", file_id=d, chunk_id="")
    print(f"docs: {len(node_of_doc)}")

    # 实体节点：三元组 head/tail 并集（对齐作者图实体集合语义）
    entity_docs = defaultdict(set)   # entity -> {doc}
    for t in triples:
        for side in ("head", "tail"):
            e = t[side]
            if not g.has_node(e):
                g.add_node(e, type="entity", file_id="", chunk_id="")
            for cid in t.get("chunk_ids", []):
                d = chunk2doc.get(cid)
                if d:
                    entity_docs[e].add(d)
            # 兜底：entities.json 里的 chunk_ids
            meta = ents.get(e)
            if meta and not entity_docs[e]:
                for cid in meta.get("chunk_ids", []):
                    d = chunk2doc.get(cid)
                    if d:
                        entity_docs[e].add(d)

    # passage 标题作为实体（对齐 h2green/作者图：标题实体常为查询种子）
    n_title_ent = 0
    for d in sorted(doc_texts):
        if not g.has_node(d):
            phr = norm_phrase(d)
            if not phr or (len(phr) == 1 and phr[0] in GENERIC):
                continue
            g.add_node(d, type="entity", file_id="", chunk_id="")
            entity_docs[d].add(d)
            n_title_ent += 1
    print(f"title 实体新增: {n_title_ent}")

    # text_conclude 边（实体 ↔ 所属文档）
    # 1) 三元组/实体元数据里引用的 chunk 所属文档
    # 2) n-gram 精确匹配：实体短语出现在文档全文中（对齐作者图/h2green 的语义）
    if not os.environ.get("NO_NGRAM"):
        phr_of_ent = {}
        ent_of_phr = {}
        ent_by_first = defaultdict(set)
        for e in entity_docs:
            phr = norm_phrase(e)
            if not phr:
                continue
            if len(phr) == 1 and phr[0] in GENERIC:
                continue
            phr_of_ent[e] = phr
            ent_of_phr[phr] = e
            ent_by_first[phr[0]].add(phr)
        print(f"n-gram 匹配: 实体 {len(phr_of_ent)}，文档 {len(doc_texts)}")
        for d, text in doc_texts.items():
            ws = [w for w in (norm_word(w) for w in text.split()) if w]
            seen = set()
            for n in range(1, NGRAM_MAX + 1):
                for i in range(len(ws) - n + 1):
                    ng = tuple(ws[i:i + n])
                    if ng in seen:
                        continue
                    seen.add(ng)
                    for phr in ent_by_first.get(ng[0], ()):
                        if phr == ng:
                            entity_docs[ent_of_phr[phr]].add(d)
    
    n_ec = 0
    for e, ds in entity_docs.items():
        for d in ds:
            g.add_edge(e, node_of_doc[d], relation="text_conclude", weight=1.0)
            n_ec += 1
    print(f"text_conclude edges: {n_ec}")

    # 三元组边
    for t in triples:
        g.add_edge(t["head"], t["tail"],
                   relation=t["relation"], weight=float(t.get("weight", 1.0)))

    print(f"nx graph: {g.number_of_nodes()} nodes / {g.number_of_edges()} edges")
    with open(args.out, "wb") as f:
        pickle.dump(g, f)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
