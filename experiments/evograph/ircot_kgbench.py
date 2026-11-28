"""KG_Bench 风格 IRCoT + ChunkHop 检索方案评测。

方案（对齐 /mnt/e/yf/KG_Bench ircot_codex_musique.jsonnet）：
  - 20-shot CoT prompt（gold_with_2_distractors_context_cot_qa_codex.txt 提取）
  - 逐句生成：LLM 续写一条句子（上下文=累积段落各截 350 词 + Q + CoT so far）
  - 每步检索：查询 = 最后一条非推理句（去 wh 词）或原问题；ChunkHop /search top-6
  - 证据累积：段落 id 去重，global_max=15
  - 退出：句子命中 ".* answer is:? (.*)" / 空生成 / 满 10 句
  - 流程顺序：先检索（问题作首查询）→ 生成 → 检索 → 生成 …

检索器 = ChunkHop（Green-ChunkHop musique_llama 图，服务 :5567），非 dual_seed。

指标（同 eval 500 题）：pool_recall@10 / pool_recall（≤15）/ answer EM/contains。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

ART = Path(__file__).resolve().parent / "artifacts"
PROMPT_FILE = Path(__file__).resolve().parent / "prompts" / "ircot_musique_codex_20shot.txt"
PASSAGE_MAP = Path("/mnt/e/yf/KG_Bench/Green-ChunkHop/dataset/musique_llama/passage_mapping.json")
CH_URL = "http://localhost:5567/search"

_MAX_SENT = 10          # max_num_sentences
_RETRIEVE_K = 6         # retrieval_count（每步）
_GLOBAL_MAX = 15        # global_max_num_paras
_MAX_PARA_WORDS = 350   # max_para_num_words
_WS = re.compile(r"\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
_ANSWER_RE = re.compile(r".* answer is:? (.*)\.?$", re.IGNORECASE)
_REASONING_STARTERS = ("thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence")
_ARITH_RE = re.compile(r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
_WH_WORDS = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s or "")).strip().lower()


def _first_sentence(text: str) -> str:
    parts = _SENT_SPLIT.split(text.strip())
    return parts[0].strip() if parts else ""


def _is_reasoning(sentence: str) -> bool:
    low = sentence.lower()
    if low.startswith(_REASONING_STARTERS) or _ARITH_RE.match(sentence):
        return True
    return False


def _remove_wh(text: str) -> str:
    return " ".join(w for w in text.split(" ") if w.strip().lower() not in _WH_WORDS).strip()


_PROTOCOL = (
    "You are continuing a chain-of-thought answer, ONE sentence at a time.\n"
    "Below are examples of the format (context passages, then Q, then A with step-by-step "
    "sentences ending in 'So the answer is: <answer>.').\n"
    "Now continue the CURRENT example: given the passages and the question, output ONLY the "
    "next single sentence of the reasoning chain (do not repeat previous sentences). "
    "When the reasoning is complete, output the final sentence in the exact form "
    "'So the answer is: <answer>.' and nothing else."
)


class ChunkHopClient:
    """ChunkHop 检索服务客户端（:5567）。"""

    def __init__(self, url: str = CH_URL, top_k: int = _RETRIEVE_K, timeout: int = 60):
        self.url, self.top_k, self.timeout = url, top_k, timeout
        self.calls = 0

    def search(self, query: str) -> list[dict]:
        if not query.strip():
            return []
        self.calls += 1
        body = json.dumps({"question": query, "final_top_k": self.top_k}).encode()
        for _ in range(3):
            try:
                req = urllib.request.Request(self.url, data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read()).get("chunks", [])
            except Exception:
                time.sleep(2)
        return []


def _para_to_text(p: dict) -> str:
    text = " ".join(p["text"].split(" ")[:_MAX_PARA_WORDS])
    title = p.get("title") or p["text"].split("\n", 1)[0]
    if text.startswith(title):
        return text.strip()
    return f"Wikipedia Title: {title}\n{text.strip()}"


def run_ircot_kgbench(question: str, llm, shots: str, ch: ChunkHopClient):
    """KG_Bench IRCoT 主循环。返回 (answer, sentences, paras 顺序列表, stop_reason)。"""
    paras: list[dict] = []          # 累积段落（发现顺序）
    seen: set[int] = set()
    sentences: list[str] = []
    stop = "max_sentences"

    def add_new(chunks: list[dict]) -> int:
        n = 0
        for c in chunks:
            gi = c.get("graph_index")
            if gi is None or gi in seen or len(paras) >= _GLOBAL_MAX:
                continue
            seen.add(gi)
            paras.append({"graph_index": gi, "text": c.get("text", ""),
                          "title": (c.get("text", "").split("\n", 1) or [""])[0],
                          "ppr_score": c.get("ppr_score", 0.0)})
            n += 1
        return n

    def gen_next() -> str:
        context = "\n\n".join(_para_to_text(p) for p in paras)
        cot = " ".join(sentences)
        parts = [x for x in (context, f"Q: {question}", f"A: {cot}") if x]
        prompt = "\n\n\n".join([_PROTOCOL, shots, "\n\n".join(parts).strip() + "\n"]).strip()
        raw = llm.generate([{"role": "user", "content": prompt}]) or ""
        return _first_sentence(raw)

    # 顺序与 commaqa 一致：先检索（问题作查询），再逐句生成
    add_new(ch.search(_remove_wh(question)))
    while len(sentences) < _MAX_SENT:
        sent = gen_next()
        if not sent or (_norm(sent) == _norm(sentences[-1]) if sentences else False):
            stop = "empty_generation" if not sent else "degenerate_loop"
            break
        sentences.append(sent)
        m = _ANSWER_RE.match(sent)
        if m:
            stop = "answer_regex"
            break
        # 查询 = 最后一条非推理句（无则问题）
        factual = [s for s in sentences if not _is_reasoning(s)]
        query = factual[-1].strip() if factual else question
        query = _remove_wh(query)
        add_new(ch.search(query))

    answer = "EMPTY"
    for s in reversed(sentences):
        m = _ANSWER_RE.match(s)
        if m:
            answer = m.group(1).strip()
            break
    if answer == "EMPTY":
        m2 = _ANSWER_RE.match(" ".join(sentences))
        answer = m2.group(1).strip() if m2 else " ".join(sentences)[-200:]
    return answer, sentences, paras, stop


def build_gold_lookup():
    """passage_index ↔ MuSiQue gold 段落（title+text 精确 + 词集 ≥0.8 模糊）。"""
    pm = json.loads(PASSAGE_MAP.read_text(encoding="utf-8"))
    exact: dict[str, int] = {}
    words_by_idx: dict[int, set] = {}
    for e in pm:
        text = e["text"]
        title = text.split("\n", 1)[0]
        body = text.split("\n", 1)[1] if "\n" in text else ""
        exact[_norm(f"{title}\n{body}")] = e["index"]
        words_by_idx[e["index"]] = set(_norm(f"{title} {body}").split())
    return exact, words_by_idx


def match_gold(title: str, text: str, exact: dict, words_by_idx: dict) -> int | None:
    key = _norm(f"{title}\n{text}")
    if key in exact:
        return exact[key]
    gw = set(_norm(f"{title} {text}").split())
    best, best_j = None, 0.0
    for idx, w in words_by_idx.items():
        j = len(gw & w) / max(1, len(gw | w))
        if j > best_j:
            best, best_j = idx, j
    return best if best_j >= 0.8 else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="KG_Bench IRCoT + ChunkHop 检索评测")
    ap.add_argument("--limit", type=int, default=0, help="冒烟：前 N 题（0=全量 500）")
    ap.add_argument("--tag", default="ircot_kgbench")
    args = ap.parse_args(argv)

    from openie.llm_client import build_llm
    from utils.config import Config

    shots = PROMPT_FILE.read_text(encoding="utf-8")
    exact, words_by_idx = build_gold_lookup()
    split = json.loads((ART / "split_real_n500_ircot.json").read_text())
    qmap = {q["id"]: q for q in json.loads(Path("data/test/musique_dev.json").read_text(encoding="utf-8"))}
    te_ids = split["eval_ids"][: args.limit] if args.limit else split["eval_ids"]

    llm = build_llm(Config.load())
    ch = ChunkHopClient()

    t0 = time.time()
    recs, no_cover = [], 0
    for i, qid in enumerate(te_ids, 1):
        q = qmap[qid]
        gold = set()
        for p in q.get("paragraphs", []):
            if not p.get("is_supporting"):
                continue
            idx = match_gold(p["title"], p["paragraph_text"], exact, words_by_idx)
            if idx is not None:
                gold.add(idx)
        if not gold:
            no_cover += 1
        try:
            answer, sentences, paras, stop = run_ircot_kgbench(q["question"], llm, shots, ch)
            got = [p["graph_index"] for p in paras]
            top10 = set(got[:10])
            rec = {
                "qid": qid, "steps": len(sentences), "llm_calls": len(sentences),
                "retrieval_calls": ch.calls, "stop_reason": stop, "pool_size": len(paras),
                "pool_recall@10": len(gold & top10) / len(gold) if gold else 0.0,
                "pool_recall": len(gold & set(got)) / len(gold) if gold else 0.0,
                "answer": str(answer)[:200], "gold_answer": q.get("answer", ""),
                "answer_em": int(_norm(answer) == _norm(q.get("answer", ""))),
                "answer_contains": int(bool(_norm(q.get("answer", "")))
                                        and _norm(q.get("answer", "")) in _norm(str(answer))),
                "gold_in_corpus": len(gold) > 0,
            }
        except Exception as err:  # noqa: BLE001
            rec = {"qid": qid, "error": f"{type(err).__name__}: {err}"}
        recs.append(rec)
        if i % 25 == 0 or i == len(te_ids):
            ok = [r for r in recs if "error" not in r]
            print(f"[{i}/{len(te_ids)}] pool_recall@10="
                  f"{sum(r['pool_recall@10'] for r in ok) / max(1, len(ok)):.4f} "
                  f"em={sum(r['answer_em'] for r in ok) / max(1, len(ok)):.4f} "
                  f"err={len(recs) - len(ok)}", flush=True)

    ok = [r for r in recs if "error" not in r]
    out = {
        "meta": {"n_eval": len(te_ids), "n_ok": len(ok), "no_gold_coverage": no_cover,
                 "scheme": "KG_Bench IRCoT (20-shot CoT, sent-at-a-time) + ChunkHop retriever",
                 "elapsed_s": round(time.time() - t0, 1)},
        "summary": {
            "pool_recall@10": round(sum(r["pool_recall@10"] for r in ok) / max(1, len(ok)), 4),
            "pool_recall": round(sum(r["pool_recall"] for r in ok) / max(1, len(ok)), 4),
            "answer_em": round(sum(r["answer_em"] for r in ok) / max(1, len(ok)), 4),
            "answer_contains": round(sum(r["answer_contains"] for r in ok) / max(1, len(ok)), 4),
            "avg_steps": round(sum(r["steps"] for r in ok) / max(1, len(ok)), 2),
            "gold_coverage": round(sum(1 for r in ok if r.get("gold_in_corpus")) / max(1, len(ok)), 4),
        },
        "records": recs,
    }
    path = ART / f"{args.tag}_n{len(te_ids)}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False))
    print(f"saved → {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
