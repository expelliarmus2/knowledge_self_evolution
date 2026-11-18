"""Planner：LLM 分解子问题 → 并行检索 → 合并证据 → 答案。

与 IRCoT 的区别：子问题彼此独立（可并行检索），IRCoT 则是逐步迭代。
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentic.ircot import IrcotResult
from agentic.memory import EvidencePool
from agentic.tools import Tools

logger = logging.getLogger("kb.agentic")


def _parse_json_list(text: str) -> list:
    text = (text or "").strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(x).strip() for x in data if str(x).strip()]


def run_planner(question: str, tools: Tools, llm, prompts, cfg) -> IrcotResult:
    max_workers = int(cfg.get("agentic.planner_workers", 4))
    k_sub = int(cfg.get("agentic.planner_top_k", 10))
    pool = EvidencePool(
        max_docs=int(cfg.get("agentic.evidence_max_docs", 32)),
        max_chars=int(cfg.get("agentic.evidence_max_chars", 6000)),
        include_triples=bool(cfg.get("agentic.include_triples", True)),
        max_triples=int(cfg.get("agentic.max_triples", 12)),
    )
    llm_calls = 1
    subs = _parse_json_list(llm.generate(prompts.render("agentic_plan", question=question)))
    if not subs:
        subs = [question]

    steps: list = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(subs))) as ex:
        futs = {ex.submit(tools.search, s, k=k_sub): s for s in subs}
        for f in as_completed(futs):
            sub = futs[f]
            try:
                r = f.result()
            except Exception as err:  # noqa: BLE001
                logger.warning("子问题检索失败 %s: %s", sub, err)
                continue
            n_new = pool.add_chunks(r["chunks"], 1, sub)
            n_tri = pool.add_triples(r["triples"])
            pool.add_entities(
                [e.get("name") for e in r["entities"]]
                + [x.get("entity") for x in r.get("selected_entities", [])]
            )
            steps.append({"sub": sub, "n_new": n_new, "triples": n_tri})

    ans = (llm.generate(prompts.render(
        "agentic_answer",
        question=question,
        evidence=pool.render() or "(none)",
        triples=pool.render_triples(),
    )) or "").strip()
    llm_calls += 1
    return IrcotResult(ans, pool.doc_ids(), steps, "planner", llm_calls)
