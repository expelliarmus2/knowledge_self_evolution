"""证据池：文档级去重 + 三元组池 + 已见实体，供 agent 渲染 LLM 上下文。"""
from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

_WS = re.compile(r"\s+")


@dataclass
class EvidencePool:
    max_docs: int = 32
    max_chars: int = 6000
    include_triples: bool = True
    max_triples: int = 12
    docs: "OrderedDict[str, dict]" = field(default_factory=OrderedDict)
    triples: "OrderedDict[tuple, float]" = field(default_factory=OrderedDict)
    entities: Set[str] = field(default_factory=set)

    # ------------------------------------------------------------------
    def add_chunks(self, chunks: List[dict], step: int = 0, query: str = "") -> int:
        """doc 级去重追加，返回新增文档数（满额后不再追加）。"""
        n_new = 0
        for c in chunks or ():
            doc = str(c.get("doc_id") or "")
            if not doc or doc in self.docs:
                continue
            if len(self.docs) >= self.max_docs:
                break
            self.docs[doc] = {
                "text": _WS.sub(" ", str(c.get("text") or "")).strip(),
                "score": float(c.get("score") or 0.0),
                "step": step,
                "query": query,
                "chunk_id": c.get("chunk_id", ""),
            }
            n_new += 1
        return n_new

    def add_triples(self, triples: List[dict]) -> int:
        if not self.include_triples:
            return 0
        n_new = 0
        for t in triples or ():
            key = (str(t.get("head", "")), str(t.get("relation", "")), str(t.get("tail", "")))
            if not all(key) or key in self.triples:
                continue
            if len(self.triples) >= self.max_triples:
                break
            self.triples[key] = float(t.get("score") or 0.0)
            n_new += 1
        return n_new

    def add_entities(self, names) -> None:
        for n in names or ():
            if n:
                self.entities.add(str(n))

    # ------------------------------------------------------------------
    def render(self, max_chars: Optional[int] = None) -> str:
        limit = int(max_chars or self.max_chars)
        out: List[str] = []
        total = 0
        for doc, info in self.docs.items():
            line = f"[{doc}] {info['text']}"
            if out and total + len(line) > limit:
                break
            out.append(line)
            total += len(line)
        return "\n".join(out)

    def render_triples(self) -> str:
        rows = [f"{h} -{r}-> {t}" for (h, r, t) in list(self.triples)[: self.max_triples]]
        return "\n".join(rows) if rows else "(none)"

    def doc_ids(self) -> List[str]:
        return list(self.docs)

    def __len__(self) -> int:
        return len(self.docs)
