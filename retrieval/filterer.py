"""LLM 三元组相关性过滤（HippoRAG2 filter prompt 思路）。

候选三元组按向量相似度排序、截断后交 LLM 过滤；LLM 失败时回退到
词面重叠打分（保证检索链路可用性）。
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

from openie.llm_client import LLMClient
from utils.config import Config
from utils.prompts import PromptManager
from utils.text import as_triple_list, extract_json, strip_think

logger = logging.getLogger("kb")


class TripleFilterer:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptManager,
        template: str = "retrieval_filter",
        max_facts: int = 8,
        enabled: bool = True,
    ):
        self.llm = llm
        self.prompts = prompts
        self.template = template
        self.max_facts = max_facts
        self.enabled = enabled

    @classmethod
    def from_config(cls, cfg: Config, llm, prompts: PromptManager) -> "TripleFilterer":
        return cls(
            llm=llm,
            prompts=prompts,
            template=cfg.get("prompts.retrieval_filter", "retrieval_filter"),
            max_facts=int(cfg.get("retrieval.llm_filter_max_facts", 8)),
            enabled=bool(cfg.get("retrieval.llm_filter_enable", True)),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _lexical_overlap_score(query: str, fact: Tuple[str, str, str]) -> int:
        q_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", query.lower()))
        text = " ".join(fact).lower()
        f_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9]{2,}", text))
        # 词级重叠 + 实体包含加成
        score = len(q_tokens & f_tokens)
        for tok in q_tokens:
            if tok in text:
                score += 1
        return score

    def _fallback(self, query: str, facts: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        scored = sorted(
            ((self._lexical_overlap_score(query, f), f) for f in facts),
            key=lambda x: -x[0],
        )
        keep = [f for sc, f in scored if sc > 0][: self.max_facts]
        return keep

    # ------------------------------------------------------------------
    def filter(
        self,
        query: str,
        facts: List[Tuple[str, str, str]],
        scores: Optional[List[float]] = None,
    ) -> Tuple[List[Tuple[str, str, str]], Dict]:
        """返回 (过滤后三元组, 调试信息)。facts 为 (head, relation, tail) 列表。"""
        info: Dict = {"mode": "none", "n_candidates": len(facts)}
        if not facts:
            return [], info
        if not self.enabled:
            info["mode"] = "disabled"
            return facts[: self.max_facts], info
        # 按分数排序后截断，送入 LLM
        if scores is not None and len(scores) == len(facts):
            order = sorted(range(len(facts)), key=lambda i: -scores[i])
            facts_sorted = [facts[i] for i in order]
        else:
            facts_sorted = list(facts)
        capped = facts_sorted[: max(self.max_facts * 2, self.max_facts)]

        import json as _json
        try:
            messages = self.prompts.render(
                self.template,
                question=query,
                candidate_facts_json=_json.dumps(
                    {"fact": [list(f) for f in capped]}, ensure_ascii=False, indent=2
                ),
                max_facts=self.max_facts,
            )
            raw = self.llm.generate(messages)
            parsed = as_triple_list(extract_json(strip_think(raw)))
            if parsed is None:
                raise ValueError("LLM 输出无法解析为 JSON")
            # 只保留候选中存在的事实（禁止编造）
            cap_set = {tuple(f) for f in capped}
            keep = [tuple(f) for f in parsed if tuple(f) in cap_set]
            info["mode"] = "llm"
            info["n_kept"] = len(keep)
            return keep, info
        except Exception as err:  # noqa: BLE001
            logger.warning("LLM 过滤失败，回退词面重叠打分: %s", err)
            keep = self._fallback(query, capped)
            info["mode"] = "fallback"
            info["n_kept"] = len(keep)
            return keep, info
