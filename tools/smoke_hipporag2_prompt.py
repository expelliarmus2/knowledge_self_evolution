#!/usr/bin/env python
"""冒烟对比：HippoRAG2 官方 OpenIE 模板（two_step, one-shot NER-conditioned RE）vs 旧 one_step。

用法（chunkhop env，需网络访问 dashscope）:
  python tools/smoke_hipporag2_prompt.py [N]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openie.chunker import Chunk  # noqa: E402
from openie.extractor import OpenIEExtractor  # noqa: E402
from openie.llm_client import build_llm  # noqa: E402
from utils.config import Config  # noqa: E402
from utils.prompts import PromptManager  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CORPUS = Path("data/raw/corpus_musique.json")


def main():
    cfg = Config.load()
    llm = build_llm(cfg)
    prompts = PromptManager.from_config(cfg)
    common = dict(llm=llm, prompts=prompts, language="english",
                  require_entity_anchor=False, resolve_pronouns=True, max_entities=64)
    new = OpenIEExtractor(**common, extract_mode="two_step",
                           ner_template="ner", re_template="triple_extraction")
    old = OpenIEExtractor(**common, extract_mode="one_step",
                          one_step_template="openie_one_step")

    corpus = json.load(open(CORPUS, encoding="utf-8"))[:N]
    for i, doc in enumerate(corpus):
        text = f"{doc['title']}\n{doc['text']}"
        ch = Chunk(chunk_id=f"doc:{i}", doc_id=doc["title"], text=text, order=0)
        r_new = new.extract(ch)
        r_old = old.extract(ch)
        ents = r_new.entities
        short = [e for e in ents if len(e.split()) == 1 and len(e) <= 4]
        print(f"\n=== [{i}] {doc['title']} | {len(text.split())} words ===")
        print(f"HippoRAG2: entities={len(ents)} (短碎片 {len(short)}: {short[:8]}) "
              f"triples={len(r_new.triples)}")
        print(f"  sample: {[(t.head, t.relation, t.tail) for t in r_new.triples[:4]]}")
        print(f"one_step : triples={len(r_old.triples)}")
        print(f"  sample: {[(t.head, t.relation, t.tail) for t in r_old.triples[:4]]}")
        if r_new.error or r_old.error:
            print(f"  ERROR new={r_new.error} old={r_old.error}")


if __name__ == "__main__":
    main()
