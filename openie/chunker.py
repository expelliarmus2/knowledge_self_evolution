"""文档切块：按分隔符优先、字符数兜底的滑动窗口切块。"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from utils.config import Config


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    order: int
    text: str
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "order": self.order,
            "text": self.text,
            "meta": self.meta,
        }


class Chunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, separators: Optional[List[str]] = None):
        self.chunk_size = max(64, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))
        self.separators = separators or ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";"]

    @classmethod
    def from_config(cls, cfg: Config) -> "Chunker":
        return cls(
            chunk_size=int(cfg.get("chunking.chunk_size", 512)),
            chunk_overlap=int(cfg.get("chunking.chunk_overlap", 64)),
            separators=cfg.get("chunking.separators") or None,
        )

    def _split_units(self, text: str) -> List[str]:
        """按分隔符层级递归切分为原子单元（句子/行），保持文本不丢失。"""
        units = [text]
        for sep in self.separators:
            nxt: List[str] = []
            for unit in units:
                parts = unit.split(sep)
                for i, p in enumerate(parts):
                    nxt.append(p)
                    if i < len(parts) - 1:
                        nxt.append(sep)
            # 合并"分隔符单元"与其前文，形成句子级单元
            merged: List[str] = []
            for u in nxt:
                if u in self.separators and merged:
                    merged[-1] = merged[-1] + u
                else:
                    merged.append(u)
            units = [u for u in merged if u]
        # 超长单元按字符硬切
        final: List[str] = []
        for u in units:
            while len(u) > self.chunk_size:
                final.append(u[: self.chunk_size])
                u = u[self.chunk_size :]
            if u:
                final.append(u)
        return final

    def create_chunks(self, text: str, doc_id: str) -> List[Chunk]:
        units = self._split_units(text.strip())
        chunks: List[Chunk] = []
        buf = ""
        order = 0

        def flush():
            nonlocal buf, order
            if buf.strip():
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}::chunk{order}",
                        doc_id=doc_id,
                        order=order,
                        text=buf.strip(),
                    )
                )
                order += 1
            buf = ""

        for unit in units:
            if len(buf) + len(unit) <= self.chunk_size:
                buf += unit
                continue
            flush()
            if len(unit) <= self.chunk_size:
                buf = unit
            else:  # 已硬切过，不会到这里
                buf = unit
        flush()

        # chunk_overlap：滑窗补充（对相邻块首部回补 overlap 字符）
        if self.chunk_overlap > 0 and len(chunks) >= 2:
            for i in range(1, len(chunks)):
                prev_tail = chunks[i - 1].text[-self.chunk_overlap :]
                if prev_tail and prev_tail not in chunks[i].text[: self.chunk_overlap * 2]:
                    chunks[i].text = prev_tail + chunks[i].text
        return chunks
