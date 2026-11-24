"""数据准备：MuSiQue dev 1000 题 → 500(evolve) / 500(eval) 切分 + 段落语料建库。"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from utils.config import Config


@dataclass
class MuSiQueSplit:
    """1000 题切分 + 段落→doc_id 映射（段落按 text 去重）。"""

    questions: List[dict] = field(default_factory=list)
    evolve_ids: List[str] = field(default_factory=list)
    eval_ids: List[str] = field(default_factory=list)
    qid_doc_ids: Dict[str, List[str]] = field(default_factory=dict)  # 每题的 doc_ids（20 段）
    doc_meta: Dict[str, dict] = field(default_factory=dict)          # doc_id → {title, text}

    def supporting_doc_ids(self, q: dict) -> List[str]:
        """该题 support 段落对应的 doc_id 集合（过滤掉未映射到 KB 的段落）。"""
        out = []
        for p in q["paragraphs"]:
            if p.get("is_supporting"):
                did = self.qid_doc_ids[q["id"]][p["idx"]]
                if did:
                    out.append(did)
        return out


def _para_doc_id(text: str) -> str:
    return f"p{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"


def load_split(dev_path: Path | str, *, shuffle: bool = False, seed: int = 42,
               max_questions: Optional[int] = None) -> MuSiQueSplit:
    dev = json.load(open(dev_path, encoding="utf-8"))
    if max_questions:
        dev = dev[:max_questions]
    questions = list(dev)
    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(questions)
    n = len(questions)
    half = n // 2
    split = MuSiQueSplit(questions=questions)
    seen: Dict[str, str] = {}  # text → doc_id（跨题去重）
    for q in questions:
        qid = q["id"]
        split.qid_doc_ids[qid] = []
        for p in q["paragraphs"]:
            text = p["paragraph_text"]
            did = seen.get(text)
            if did is None:
                did = _para_doc_id(text)
                seen[text] = did
                split.doc_meta[did] = {"title": p.get("title", ""), "text": text}
            split.qid_doc_ids[qid].append(did)
    split.evolve_ids = [q["id"] for q in questions[:half]]
    split.eval_ids = [q["id"] for q in questions[half:]]
    return split


def load_split_processed(dev_path: Path | str, corpus_path: Path | str, chunks_path: Path | str,
                         *, shuffle: bool = False, seed: int = 42,
                         max_questions: Optional[int] = None) -> MuSiQueSplit:
    """real 模式：段落 doc_id 映射到已有真实 KB（data/processed_musique）的 doc_id。

    映射链：dev 段落 (title, text) → corpus_musique.json 同 title 组内的文本序号 k
          → doc_id = title（k=0）或 title#(k+1)；再校验 doc_id 在 KB chunks 中存在。
    未覆盖的段落 doc_id = None（支持段落计入 coverage 统计、检索判定时跳过）。
    """
    import collections

    dev = json.load(open(dev_path, encoding="utf-8"))
    if max_questions:
        dev = dev[:max_questions]
    questions = list(dev)
    if shuffle:
        import random

        rng = random.Random(seed)
        rng.shuffle(questions)
    n = len(questions)
    half = n // 2
    split = MuSiQueSplit(questions=questions)

    corpus = json.load(open(corpus_path, encoding="utf-8"))
    title_groups: Dict[str, List[str]] = collections.defaultdict(list)
    for d in corpus:
        title_groups[d["title"]].append(d["text"])
    proc_docs: set = set()
    for line in open(chunks_path, encoding="utf-8"):
        proc_docs.add(json.loads(line)["doc_id"])

    covered_support = covered_all = 0
    total_support = total_all = 0
    for q in questions:
        qid = q["id"]
        split.qid_doc_ids[qid] = []
        for p in q["paragraphs"]:
            text = p["paragraph_text"]
            did: Optional[str] = None
            seq = title_groups.get(p.get("title", ""))
            if seq is not None:
                for k, t in enumerate(seq):
                    if t == text:
                        did = p["title"] if k == 0 else f"{p['title']}#{k + 1}"
                        break
            if did and did in proc_docs:
                split.qid_doc_ids[qid].append(did)
                covered_all += 1
            else:
                split.qid_doc_ids[qid].append(None)
            total_all += 1
            if p.get("is_supporting"):
                total_support += 1
                if did and did in proc_docs:
                    covered_support += 1
    split.evolve_ids = [q["id"] for q in questions[:half]]
    split.eval_ids = [q["id"] for q in questions[half:]]
    split.doc_coverage = {  # type: ignore[attr-defined]
        "support": round(covered_support / max(1, total_support), 4),
        "all": round(covered_all / max(1, total_all), 4),
        "support_total": total_support, "support_covered": covered_support,
    }
    return split


def write_corpus(split: MuSiQueSplit, out_dir: Path) -> Path:
    """把去重段落写成 docs.jsonl（BuildPipeline 输入格式）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = [{"doc_id": did, "text": meta["text"], "meta": {"title": meta["title"]}}
            for did, meta in sorted(split.doc_meta.items())]
    path = out_dir / "docs.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


def build_mock_kb(split: MuSiQueSplit, workspace: Path | str, cfg: Config) -> Path:
    """mock 模式构建 KB（embedding mock 256 + llm mock）。返回 processed_dir。

    幂等基于语料内容 hash（docs 数 + 内容 md5），而非仅看目录是否存在——
    否则小语料冒烟建的库会被全量实验误复用。
    """
    import hashlib as _h

    from pipeline.build_pipeline import BuildPipeline

    workspace = Path(workspace)
    raw_dir = workspace / "raw"
    docs_path = write_corpus(split, raw_dir)
    processed = workspace / "processed_mock"

    corpus_hash = _h.md5(docs_path.read_bytes()).hexdigest()[:16]
    manifest = processed / "corpus_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("hash") == corpus_hash:
        return processed

    shutil.rmtree(processed, ignore_errors=True)
    overrides = {
        "embedding": {"provider": "mock", "dim": 256},
        "llm": {"provider": "mock"},
        "paths": {"raw_dir": str(raw_dir), "processed_dir": str(processed)},
    }
    cfg2 = Config.load(overrides=overrides)
    pipeline = BuildPipeline(cfg2)
    pipeline.run(input_path=docs_path, incremental=False)
    manifest.write_text(json.dumps({"hash": corpus_hash, "docs": len(split.doc_meta)}), encoding="utf-8")
    return processed
