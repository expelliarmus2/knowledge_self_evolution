"""自进化核心：LLM 评价检索轨迹三元组 → 回写图谱。

- real 模式：LLM JSON 评价（只允许从候选三元组里选，不凭空造）；
- mock 模式：确定性启发式（head/tail 与问题 token 重叠）——只验证流程，不代表真实质量。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from graph.knowledge_graph import KnowledgeGraph
from openie.llm_client import LLMClient, MockLLMClient


@dataclass
class EvolveStats:
    evaluated: int = 0          # 评价了几条轨迹
    candidates: int = 0         # 候选三元组总数
    judged_useful: int = 0      # 判定有用
    written: int = 0            # 实际新写入（去重后）
    deduped: int = 0            # 已存在只累权
    rejected: int = 0           # 判定无用/过滤
    log: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d.pop("log", None)
        return d


_EVAL_PROMPT = """你是知识图谱自进化审核员。给定一个问题与检索轨迹，判断哪些候选三元组
对回答该问题"有用"（提供直接事实、或衔接多跳路径所需的桥接信息）。

只从候选三元组中挑选，禁止编造新三元组。输出 JSON：
{{"useful": [{{"head": "...", "relation": "...", "tail": "..."}}], "reason": "一句话说明"}}

问题: {question}
子问题链: {decomposition}
候选三元组:
{candidates}
JSON:"""

_EXTRACT_PROMPT = """从下列检索到的段落中抽取对回答问题有帮助的事实三元组（不超过 8 条）。
只包含段落中明确表述的事实，不要推测。输出 JSON：
{{"triples": [["head", "relation", "tail"], ...]}}

问题: {question}
段落:
{chunks}
JSON:"""


class Evolver:
    def __init__(self, graph: KnowledgeGraph, llm, *, mode: str = "real",
                 max_write_per_q: int = 8, extract_new: bool = True):
        self.graph = graph
        self.llm = llm
        self.mode = mode
        self.max_write_per_q = max_write_per_q
        self.extract_new = extract_new
        self.stats = EvolveStats()

    # ---------- 新知识抽取（真正的增量） ----------
    def extract(self, question: str, chunks: List[dict]) -> List[dict]:
        """从检索到的 chunk 文本抽取候选新三元组（与图谱已有边对比后只留新的）。"""
        texts = [c.get("text", "") for c in chunks[:6] if c.get("text")]
        if not texts:
            return []
        triples = self._extract_llm(question, texts) if self.mode != "mock" else self._extract_heuristic(texts)
        out, seen = [], set()
        for h, r, t in triples:
            h, r, t = str(h).strip(), str(r).strip(), str(t).strip()
            if not (h and r and t) or h == t or len(h) > 80 or len(t) > 80:
                continue
            key = (h, r, t)
            if key in seen:
                continue
            seen.add(key)
            # 只保留图谱尚未包含的边：同端点同 relation 已存在则走“加权”路径，不进候选
            if not self.graph.g.has_edge(h, t, key=r):
                out.append({"head": h, "relation": r, "tail": t,
                            "confidence": 0.5, "is_new": True})
        return out

    def _extract_llm(self, question: str, texts: List[str]) -> List[tuple]:
        chunk_block = "\n\n".join(f"[{i + 1}] {t[:800]}" for i, t in enumerate(texts))
        raw = self.llm.generate([{"role": "user",
                                  "content": _EXTRACT_PROMPT.format(question=question, chunks=chunk_block)}])
        try:
            payload = json.loads(_strip_json(raw))
            return [tuple(x) for x in payload.get("triples", []) if len(x) == 3]
        except Exception:
            return []

    def _extract_heuristic(self, texts: List[str]) -> List[tuple]:
        """mock：中文复用 _mock_re；英文用「大写专名 … 关系词 … 专名」弱规则。

        仅用于验证流程（mock embedding/LLM 无语义能力），不代表真实抽取质量。
        """
        from openie.llm_client import _mock_ner, _mock_re

        out: List[tuple] = []
        rel_words = {
            "born": "born_in", "located": "located_in", "directed": "directed_by",
            "founded": "founded_by", "released": "released_in", "starring": "starring",
            "won": "won", "is": "related_to", "was": "related_to",
        }
        for text in texts:
            for sent in re.split(r"(?<=[.!?])\s+", text):
                ents = _mock_ner(sent)
                out.extend(tuple(x) for x in _mock_re(sent, ents))
                if not ents:
                    continue
                # 英文专名：首字母大写的多词短语 / 年份 / 引号内容
                en_ents = list(dict.fromkeys(
                    re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z][a-z]{3,}|\d{4}|['\"]\w[^'\"]*['\"])\b", sent)
                ))
                if not en_ents:
                    continue
                lower = sent.lower()
                for w, rel in rel_words.items():
                    idx = lower.find(f" {w} ")
                    if idx == -1:
                        continue
                    left = [e for e in en_ents if sent.find(e) < idx]
                    right = [e for e in en_ents if sent.find(e) > idx]
                    if left and right:
                        out.append((left[-1], rel, right[0]))
                    break
        return out

    # ---------- 评价 ----------
    def judge(self, question: str, decomposition: str, triples: List[dict]) -> List[dict]:
        """返回判定有用的三元组子集（带 confidence）。"""
        self.stats.evaluated += 1
        self.stats.candidates += len(triples)
        if not triples:
            return []
        if self.mode == "mock":
            return self._judge_heuristic(question, triples)
        return self._judge_llm(question, decomposition, triples)

    def _judge_heuristic(self, question: str, triples: List[dict]) -> List[dict]:
        """确定性：候选三元组的 head/tail 与问题共享 ≥1 个词元即判有用。

        保留候选自带字段（如 is_new），便于 write_back 区分新增/加权。
        """
        q_tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_.'-]{1,}", question.lower()))
        q_tokens |= {t.lower() for t in re.findall(r"\b\d{4}\b|\b\w+\b", question.lower())}
        useful = []
        for t in triples:
            h, r, tail = str(t.get("head", "")), str(t.get("relation", "")), str(t.get("tail", ""))
            hit = any(tok in h.lower() or tok in tail.lower() for tok in q_tokens)
            if hit:
                item = {"head": h, "relation": r, "tail": tail, "confidence": 0.6}
                if t.get("is_new"):
                    item["is_new"] = True
                useful.append(item)
        return useful

    def _judge_llm(self, question: str, decomposition: str, triples: List[dict]) -> List[dict]:
        cand_lines = "\n".join(
            f"- {t.get('head')} | {t.get('relation')} | {t.get('tail')}" for t in triples
        )
        prompt = _EVAL_PROMPT.format(
            question=question, decomposition=decomposition or "(无)", candidates=cand_lines
        )
        raw = self.llm.generate([{"role": "user", "content": prompt}])
        try:
            payload = json.loads(_strip_json(raw))
            useful = payload.get("useful", [])
        except Exception:
            return []
        # 过滤：只接受候选集合内（允许小写/引号差异）
        cand_keys = {(str(t["head"]), str(t["relation"]), str(t["tail"])) for t in triples}
        out = []
        for u in useful:
            key = (str(u.get("head", "")), str(u.get("relation", "")), str(u.get("tail", "")))
            if key in cand_keys:
                out.append({"head": key[0], "relation": key[1], "tail": key[2], "confidence": 0.8})
        return out

    # ---------- 回写 ----------
    def write_back(self, question: str, qid: str, useful: List[dict]) -> None:
        if not useful:
            return
        # 新边优先（真正的知识增量）；剩余额度用于已有边加权（排序信号）
        ranked = sorted(useful, key=lambda x: not x.get("is_new"))
        for t in ranked[: self.max_write_per_q]:
            h, r, tail = t["head"], t["relation"], t["tail"]
            if not (h and r and tail):
                self.stats.rejected += 1
                continue
            # 判断是否新边：存在则累权
            is_new = not self.graph.g.has_edge(h, tail, key=r) and not self.graph.g.has_edge(tail, h, key=r)
            self.graph.add_triple(
                h, r, tail,
                weight=0.5,
                meta={"evolved": True, "source_qid": qid, "confidence": t.get("confidence", 0.5),
                      "is_new": bool(t.get("is_new", False)) or is_new},
            )
            if is_new:
                self.stats.written += 1
            else:
                self.stats.deduped += 1
            self.stats.judged_useful += 1
            self.stats.log.append({
                "qid": qid, "question": question, "head": h, "relation": r, "tail": tail,
                "is_new": is_new, "confidence": t.get("confidence", 0.5),
            })


def _strip_json(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    return raw[start:end + 1] if start != -1 and end != -1 else raw
