"""轨迹级评价（方式A）：对 IRCoT 每轮（推理句 + 本轮证据）打分，
只从高价值轮次的证据 chunk 抽取知识回写图谱。

- real：LLM 逐轮评级 high/medium/low；
- mock：确定性回退（本轮有新增证据 → high，否则 low）；
- LLM 失败/漏评 → 回退 n_new 启发式，保证整轮实验不中断。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger("kb.evograph")

_TRAJ_PROMPT = """<user>
Question: {question}

Multi-round retrieval trajectory (each round: reasoning sentence, then evidence retrieved that round):
{trajectory}

Rate each round's VALUE for answering the question:
- high: this round's evidence directly supports answering (bridge fact / missing hop / key entity)
- medium: partially relevant background
- low: irrelevant, redundant or noise

Output STRICT JSON only, no markdown fence:
{{"rated_steps": [{{"step": 1, "value": "high", "reason": "short reason"}}]}}
</user>"""

_MAX_SEL_CHUNKS = 8          # 抽取来源 chunk 上限（控制 extract prompt 规模）
_TEXT_PREVIEW = 90


def _usable(steps: List[dict]) -> List[dict]:
    return [s for s in (steps or []) if s.get("sentence")]


def _render_steps(steps: List[dict]) -> str:
    lines = []
    for i, s in enumerate(steps, 1):
        docs = "; ".join(
            f"[{c.get('doc_id')}] {(c.get('text') or '')[:_TEXT_PREVIEW]}"
            for c in (s.get("step_chunks") or [])[:5]
        )
        lines.append(f"Step {i}: NEXT: {s['sentence']}\n  evidence: {docs or '(no new docs)'}")
    return "\n".join(lines)


def judge_trajectory(
    question: str,
    steps: List[dict],
    llm,
    mode: str = "real",
    max_chars: int = 3500,
) -> Tuple[Dict[int, str], List[dict]]:
    """轨迹级评价。

    返回 (每轮评级 {step: high|medium|low}, 选中 chunk 列表)。
    选中 chunk = high/medium 轮次的 step_chunks，按 (chunk_id, doc_id) 去重，
    截前 _MAX_SEL_CHUNKS 条（按轮次顺序，分数高者优先稳定排序）。
    """
    usable = _usable(steps)
    if not usable:
        return {}, []

    values: Dict[int, str] = {}
    if mode != "mock":
        traj = _render_steps(usable)[:max_chars]
        try:
            raw = (llm.generate([{
                "role": "user",
                "content": _TRAJ_PROMPT.format(question=question, trajectory=traj),
            }]) or "").strip()
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0)) if m else {}
            for item in data.get("rated_steps", []):
                try:
                    values[int(item.get("step", 0))] = str(item.get("value", "low")).lower()
                except (TypeError, ValueError):
                    continue
        except Exception as err:  # noqa: BLE001
            logger.warning("轨迹评价失败，回退 n_new 启发式: %s", err)
            values = {}
    # mock / 漏评回退：有新增证据 → high
    for i, s in enumerate(usable, 1):
        values.setdefault(i, "high" if s.get("n_new", 0) > 0 else "low")

    selected: List[dict] = []
    seen = set()
    for i, s in enumerate(usable, 1):
        if values.get(i, "low") not in ("high", "medium"):
            continue
        for c in s.get("step_chunks") or []:
            key = (c.get("chunk_id"), c.get("doc_id"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(dict(c))
    # 上限截断：分数高者优先（None 兜底 0），同分保持轮次序
    selected.sort(key=lambda c: -(float(c.get("score") or 0.0)))
    return values, selected[:_MAX_SEL_CHUNKS]


def selected_triples(steps: List[dict], values: Dict[int, str]) -> List[dict]:
    """高价值轮次命中的图内三元组（加权路径候选）。"""
    out: List[dict] = []
    for i, s in enumerate(_usable(steps), 1):
        if values.get(i, "low") in ("high", "medium"):
            out.extend(dict(t) for t in (s.get("step_triples") or []))
    return out


# ------------------------- 三元组删除（净化维度） -------------------------
_DELETE_PROMPT = """<user>
Question: {question}

Evidence (chunks retrieved during multi-round search):
{evidence}

Graph triples hit during retrieval (numbered):
{triples}

Mark triples that are clearly WRONG or NOISY for DELETION:
- contradicted by the evidence above, OR
- nonsensical / broken extraction (e.g. a date or place as an entity acting as a person), OR
- relation does not hold between head and tail at all.

Rules:
- Only delete when clearly wrong. When unsure, KEEP.
- Do NOT delete triples that are merely irrelevant to this question.

Output STRICT JSON only, no markdown fence:
{{"deletions": [{{"index": 1, "reason": "short reason"}}]}}
(index refers to the numbered triple above; empty list if nothing to delete)
</user>"""

_MAX_DEL_PER_Q = 4          # 每题删除上限（保守，防误删）


def judge_deletions(
    question: str,
    hit_triples: List[dict],
    evidence_chunks: List[dict],
    llm,
    mode: str = "real",
    max_chars: int = 3000,
) -> List[dict]:
    """轨迹评价的删除维度：图内命中三元组中明显错误/噪声的边。

    返回待删列表 [{"head","relation","tail","reason"}]；mock 模式无法判断，保守不删。
    """
    if not hit_triples or mode == "mock":
        return []
    # 编号去重（head/relation/tail 唯一）
    uniq, seen = [], set()
    for t in hit_triples:
        key = (str(t.get("head")), str(t.get("relation")), str(t.get("tail")))
        if key in seen or not all(key):
            continue
        seen.add(key)
        uniq.append(dict(t))

    evidence = " | ".join(
        f"[{c.get('doc_id')}] {(c.get('text') or '')[:110]}"
        for c in evidence_chunks[:6]
    )[:max_chars // 2]
    lines = "\n".join(
        f"{i}. {t['head']} -{t['relation']}-> {t['tail']}"
        for i, t in enumerate(uniq, 1)
    )
    try:
        raw = (llm.generate([{
            "role": "user",
            "content": _DELETE_PROMPT.format(
                question=question, evidence=evidence or "(none)", triples=lines),
        }]) or "").strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        out = []
        for item in data.get("deletions", []):
            try:
                idx = int(item.get("index", 0)) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(uniq):
                d = dict(uniq[idx])
                d["reason"] = str(item.get("reason", ""))[:200]
                if d not in out:
                    out.append(d)
            if len(out) >= _MAX_DEL_PER_Q:
                break
        return out
    except Exception as err:  # noqa: BLE001
        logger.warning("删除评价失败，保守跳过: %s", err)
        return []
