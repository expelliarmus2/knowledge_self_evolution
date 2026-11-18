"""IRCoT：迭代式 CoT 检索。

每轮：LLM 生成"下一句推理"（或直接给出最终答案）→ 用该推理句检索 →
证据池按 doc 去重追加 → 下一轮。早停条件：模型给出 FINAL ANSWER 或
连续两轮证据无新增且已达到最少步数。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from agentic.memory import EvidencePool
from agentic.tools import Tools

logger = logging.getLogger("kb.agentic")

_FINAL_RE = re.compile(r"FINAL\s+ANSWER\s*:\s*(.+)", re.IGNORECASE)
_NEXT_RE = re.compile(r"NEXT\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class IrcotResult:
    answer: str
    evidence_docs: List[str] = field(default_factory=list)
    steps: List[dict] = field(default_factory=list)   # [{sentence, n_new, stop}]
    stop_reason: str = ""
    llm_calls: int = 0


def run_ircot(question: str, tools: Tools, llm, prompts, cfg) -> IrcotResult:
    max_steps = int(cfg.get("agentic.max_steps", 6))
    k_step = int(cfg.get("agentic.top_k_per_step", 5))
    min_steps = int(cfg.get("agentic.min_steps", 1))
    pool = EvidencePool(
        max_docs=int(cfg.get("agentic.evidence_max_docs", 32)),
        max_chars=int(cfg.get("agentic.evidence_max_chars", 6000)),
        include_triples=bool(cfg.get("agentic.include_triples", True)),
        max_triples=int(cfg.get("agentic.max_triples", 12)),
    )
    cot: List[str] = []
    llm_calls = 0
    steps: List[dict] = []
    consecutive_stale = 0

    def _seed_entities(text: str) -> None:
        """把问题/推理句中的候选实体名直接加入实体池（供后续 expand 工具）。"""
        # 简单启发：双引号内 + 大写连续词（后续可换 LLM NER）
        for m in re.finditer(r'"([^"]+)"', text):
            pool.add_entities([m.group(1)])
        for m in re.finditer(r"\b([A-Z][a-zA-Z0-9'\-&\. ]{2,})\b", text):
            pool.add_entities([m.group(1).strip()])

    for step in range(1, max_steps + 1):
        ev = pool.render()
        messages = prompts.render(
            "ircot_step",
            question=question,
            evidence=ev or "(none yet)",
            cot="\n".join(cot) or "(none yet)",
        )
        out = (llm.generate(messages) or "").strip()
        llm_calls += 1
        _seed_entities(out)

        m = _FINAL_RE.search(out)
        if m:
            ans = m.group(1).strip()
            steps.append({"sentence": "", "n_new": 0, "stop": "final"})
            return IrcotResult(ans, pool.doc_ids(), steps, "final_answer", llm_calls)

        m = _NEXT_RE.search(out)
        sentence = (m.group(1).strip() if m else out).strip()
        if not sentence:
            sentence = question  # 模型没按格式 → 退化为原问题检索
        cot.append(sentence)

        r = tools.search(sentence, k=k_step)
        n_new = pool.add_chunks(r["chunks"], step, sentence)
        n_tri = pool.add_triples(r["triples"])
        pool.add_entities(
            [e.get("name") for e in r["entities"]]
            + [x.get("entity") for x in r.get("selected_entities", [])]
        )
        steps.append({"sentence": sentence, "n_new": n_new, "triples": n_tri, "stop": ""})

        if n_new == 0 and n_tri == 0:
            consecutive_stale += 1
        else:
            consecutive_stale = 0
        if step >= min_steps and consecutive_stale >= 2:
            break  # 证据饱和：连续两轮无新增，不再检索

    # 达到步数上限或证据饱和 → 用最终证据作答
    ans = (llm.generate(prompts.render(
        "agentic_answer",
        question=question,
        evidence=pool.render() or "(none)",
        triples=pool.render_triples(),
    )) or "").strip()
    llm_calls += 1
    return IrcotResult(ans, pool.doc_ids(), steps, "evidence_answer", llm_calls)
