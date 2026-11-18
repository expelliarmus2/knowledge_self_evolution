"""OpenIE 抽取器（两种可切换模式，yaml 配置 openie.extract_mode）：

模式一 two_step（默认，HippoRAG2 两步式，精度优先）：
  Step 1（NER）：从 chunk 抽取实体列表；
  Step 2（RE）：基于原文 + 实体列表抽取 [主语, 谓语, 宾语] 三元组，
               每个三元组至少锚定实体列表中的一个实体，代词解析为具体名称。

模式二 one_step（huaxi_poc 风格单次抽取，性能优先）：
  单次 LLM 调用直接产出全部三元组，分隔符协议 <|> ## <|COMPLETE|>；
  实体列表从三元组端点汇总（无独立 NER，故全部视为已锚定）；
  LLM 调用次数减半，适合大规模语料 / 低延迟场景。

两种模式的 prompt 均来自 prompts/ 模板，可整体替换（换领域只换模板）。
抽取以 chunk 为单位，可并行（吸收 huaxi_poc 的 ThreadPoolExecutor 并行方式），
并支持流式产出（as_completed 顺序），供流水线边抽取边构图。
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional

from openie.chunker import Chunk
from openie.llm_client import LLMClient, MockLLMClient
from utils.config import Config
from utils.prompts import PromptManager
from utils.text import (
    as_entity_list,
    as_triple_list,
    extract_json,
    is_pronoun,
    normalize_name,
    strip_think,
)

logger = logging.getLogger("kb")


@dataclass
class Triple:
    head: str
    relation: str
    tail: str
    chunk_id: Optional[str] = None
    doc_id: Optional[str] = None
    anchored: bool = True   # 是否锚定了 NER 实体
    resolved_pronoun: Optional[str] = None  # 代词解析记录: 原词 -> 实体

    def to_list(self) -> List[str]:
        return [self.head, self.relation, self.tail]

    def to_dict(self) -> dict:
        return {
            "head": self.head,
            "relation": self.relation,
            "tail": self.tail,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "anchored": self.anchored,
            "resolved_pronoun": self.resolved_pronoun,
        }


@dataclass
class OpenIEResult:
    chunk_id: str
    doc_id: str
    entities: List[str] = field(default_factory=list)
    triples: List[Triple] = field(default_factory=list)
    dropped: List[Triple] = field(default_factory=list)  # 未锚定实体被丢弃的三元组
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "entities": self.entities,
            "triples": [t.to_dict() for t in self.triples],
            "dropped": [t.to_dict() for t in self.dropped],
            "error": self.error,
        }


class OpenIEExtractor:
    def __init__(
        self,
        llm: LLMClient,
        prompts: PromptManager,
        language: str = "chinese",
        require_entity_anchor: bool = True,
        resolve_pronouns: bool = True,
        max_entities: int = 64,
        ner_template: str = "ner",
        re_template: str = "triple_extraction",
        extract_mode: str = "two_step",
        one_step_template: str = "openie_one_step",
    ):
        self.llm = llm
        self.prompts = prompts
        self.require_entity_anchor = require_entity_anchor
        self.resolve_pronouns = resolve_pronouns
        self.max_entities = max_entities
        self.ner_template = ner_template
        self.re_template = re_template
        if extract_mode not in ("two_step", "one_step"):
            raise ValueError(f"openie.extract_mode 非法: {extract_mode}（two_step | one_step）")
        self.extract_mode = extract_mode
        self.one_step_template = one_step_template
        _ = language  # 语言由 PromptManager 模板后缀控制

    @classmethod
    def from_config(cls, cfg: Config, llm: LLMClient, prompts: PromptManager) -> "OpenIEExtractor":
        return cls(
            llm=llm,
            prompts=prompts,
            language=str(cfg.get("project.language", "chinese")),
            require_entity_anchor=bool(cfg.get("openie.require_entity_anchor", True)),
            resolve_pronouns=bool(cfg.get("openie.resolve_pronouns", True)),
            max_entities=int(cfg.get("openie.max_entities", 64)),
            ner_template=cfg.get("prompts.ner", "ner"),
            re_template=cfg.get("prompts.triple_extraction", "triple_extraction"),
            extract_mode=str(cfg.get("openie.extract_mode", "two_step")).lower(),
            one_step_template=cfg.get("prompts.openie_one_step", "openie_one_step"),
        )

    # ---------------- Step 1: NER ----------------
    def extract_entities(self, passage: str) -> List[str]:
        messages = self.prompts.render(self.ner_template, passage=passage)
        raw = self.llm.generate(messages)
        entities = as_entity_list(extract_json(strip_think(raw)))
        # 清洗：过短、代词剔除；去重保序
        seen, out = set(), []
        for e in entities:
            e = normalize_name(e)
            if not e or len(e) < 1 or is_pronoun(e):
                continue
            if e in seen:
                continue
            seen.add(e)
            out.append(e)
        return out[: self.max_entities]

    # ---------------- Step 2: 实体条件化三元组抽取 ----------------
    def extract_triples(self, passage: str, entities: List[str]) -> List[List[str]]:
        import json as _json
        named_entity_json = _json.dumps({"named_entities": entities}, ensure_ascii=False, indent=2)
        messages = self.prompts.render(
            self.re_template, passage=passage, named_entity_json=named_entity_json
        )
        raw = self.llm.generate(messages)
        return as_triple_list(extract_json(strip_think(raw)))

    # ---------------- 两步组合 + 校验 ----------------
    def _anchor_ok(self, value: str, entity_set: set) -> bool:
        """三元组端点锚定校验：精确 / 包含（大小写不敏感）。"""
        v = normalize_name(value)
        if v in entity_set:
            return True
        vl = v.lower()
        for e in entity_set:
            el = e.lower()
            if vl == el or el in vl or vl in el:
                return True
        return False

    def _resolve(self, value: str, entities: List[str]) -> tuple:
        """代词解析：返回 (最终值, 解析记录)。"""
        if not self.resolve_pronouns or not is_pronoun(value):
            return value, None
        # 启发式：解析为实体列表第一个非代词实体（列表保序，首位通常是主语/主题实体）
        for e in entities:
            if not is_pronoun(e):
                return e, f"{value}->{e}"
        return value, None

    def extract(self, chunk: Chunk) -> OpenIEResult:
        """按配置模式分发：two_step（默认）| one_step（huaxi 单次抽取）。"""
        if self.extract_mode == "one_step":
            return self._extract_one_step(chunk)
        return self._extract_two_step(chunk)

    # ---------------- 模式一：两步抽取 ----------------
    def _extract_two_step(self, chunk: Chunk) -> OpenIEResult:
        result = OpenIEResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
        try:
            entities = self.extract_entities(chunk.text)
            result.entities = entities
            entity_set = set(entities)
            raw_triples = self.extract_triples(chunk.text, entities)
            for s, p, o in raw_triples:
                if not s or not p or not o:
                    continue
                s2, r1 = self._resolve(s, entities)
                o2, r2 = self._resolve(o, entities)
                anchored = self._anchor_ok(s2, entity_set) or self._anchor_ok(o2, entity_set)
                tri = Triple(
                    head=s2, relation=p, tail=o2,
                    chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                    anchored=anchored,
                    resolved_pronoun=";".join(x for x in (r1, r2) if x) or None,
                )
                if self.require_entity_anchor and not anchored:
                    result.dropped.append(tri)
                else:
                    result.triples.append(tri)
        except Exception as err:  # noqa: BLE001
            result.error = f"{type(err).__name__}: {err}"
            logger.error("OpenIE 抽取失败 chunk=%s: %s", chunk.chunk_id, err)
        return result

    # ---------------- 模式二：单次抽取（huaxi_poc 风格） ----------------
    _TUPLE_SEP = "<|>"
    _RECORD_SEP = "##"
    _COMPLETE_MARK = "<|COMPLETE|>"

    def _parse_one_step_output(self, raw: str) -> List[List[str]]:
        """解析 huaxi 分隔符协议：`头<|>关系<|>尾##头<|>关系<|>尾##...<|COMPLETE|>`。"""
        text = strip_think(raw).strip()
        text = text.replace(self._COMPLETE_MARK, "")
        # 兼容 LLM 可能加的代码块包裹
        if text.startswith("```"):
            text = text.strip("`").lstrip("json").strip()
        triples: List[List[str]] = []
        for rec in text.split(self._RECORD_SEP):
            parts = [p.strip().strip("`").strip() for p in rec.split(self._TUPLE_SEP)]
            if len(parts) < 3 or not parts[0] or not parts[1] or not parts[2]:
                continue
            # 尾实体若含多余分隔片段则拼回（容错）
            tail = "<|>".join(parts[2:]).strip()
            triples.append([parts[0], parts[1], tail])
        return triples

    def extract_triples_one_step(self, passage: str) -> List[List[str]]:
        """单次 LLM 调用直接产出三元组（无独立 NER，LLM 调用次数减半）。"""
        messages = self.prompts.render(self.one_step_template, passage=passage)
        raw = self.llm.generate(messages)
        return self._parse_one_step_output(raw)

    def _extract_one_step(self, chunk: Chunk) -> OpenIEResult:
        result = OpenIEResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)
        try:
            raw_triples = self.extract_triples_one_step(chunk.text)
            entities: List[str] = []
            seen: set = set()
            for s, p, o in raw_triples:
                if is_pronoun(s) or is_pronoun(o):
                    # prompt 已约束不用代词，此处兑底丢弃违反约束的端点
                    logger.debug("one_step 丢弃代词端点三元组: %s", (s, p, o))
                    continue
                s2, o2 = normalize_name(s), normalize_name(o)
                if not s2 or not o2:
                    continue
                for e in (s2, o2):
                    if e not in seen:
                        seen.add(e)
                        entities.append(e)
                result.triples.append(Triple(
                    head=s2, relation=p.strip(), tail=o2,
                    chunk_id=chunk.chunk_id, doc_id=chunk.doc_id,
                    anchored=True,   # 无独立实体列表，端点即实体，全部视为已锚定
                ))
            result.entities = entities[: self.max_entities]
        except Exception as err:  # noqa: BLE001
            result.error = f"{type(err).__name__}: {err}"
            logger.error("OpenIE one_step 抽取失败 chunk=%s: %s", chunk.chunk_id, err)
        return result

    # ---------------- 并行 / 流式 ----------------
    def extract_parallel(self, chunks: List[Chunk], max_workers: int = 8, progress=None) -> List[OpenIEResult]:
        """并行抽取，按输入顺序返回（huaxi_poc 式线程池并行）。"""
        results: Dict[str, OpenIEResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = {pool.submit(self.extract, c): c for c in chunks}
            it = as_completed(futures)
            if progress is not None:
                it = progress(it, total=len(futures), desc="OpenIE 抽取")
            for fut in it:
                res = fut.result()
                results[res.chunk_id] = res
        return [results[c.chunk_id] for c in chunks if c.chunk_id in results]

    def extract_stream(self, chunks: List[Chunk], max_workers: int = 8) -> Iterator[OpenIEResult]:
        """流式抽取：按完成顺序逐个产出，供流水线边抽取边构图（流式并行优化）。"""
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            futures = [pool.submit(self.extract, c) for c in chunks]
            for fut in as_completed(futures):
                yield fut.result()

    def extract_batched_stream(
        self, chunks: List[Chunk], max_workers: int = 8, batch_size: int = 16
    ) -> Iterator[List[OpenIEResult]]:
        """流式批产出：每凑满 batch_size（或抽取结束）产出一批，用于增量构图。"""
        batch: List[OpenIEResult] = []
        for res in self.extract_stream(chunks, max_workers=max_workers):
            batch.append(res)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
