"""构建流水线：OpenIE 抽取 → 流式构图 → 向量入库，端到端串联。

组织方式吸收 huaxi_poc 的并行与调度：
- LLM 抽取：ThreadPoolExecutor(llm.max_workers) 并行、按完成顺序流式产出；
- 构图：每完成一批（openie.batch_size）立即增量写图（抽取与构图重叠执行）；
- 向量化：Embedder 批量 + 有界并发，且只嵌入新增项（增量模式）。

演进钩子（EvoGraph-R1 对接预留）：
- 全流程事件注入点见 graph.evolution.PIPELINE_EVENTS；
- 每批构图完成后发射 on_extract_batch，策略可返回图操作即时应用；
- BuildPipeline.graph_api 暴露 GraphEvolutionAPI 供外部演进系统调用。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from graph.evolution import (
    EvolutionContext,
    EvolutionManager,
    GraphEvolutionAPI,
    GraphOperation,
)
from graph.graph_builder import GraphBuilder
from graph.knowledge_graph import KnowledgeGraph
from openie.chunker import Chunk, Chunker
from openie.embedder import build_embedder
from openie.extractor import OpenIEExtractor
from openie.llm_client import build_llm
from utils.config import Config
from utils.prompts import PromptManager
from vectorstore.manager import VectorStoreManager

logger = logging.getLogger("kb")


def load_documents(path: Path) -> List[dict]:
    """读取文档：.jsonl / .json（顶层数组，MuSiQue corpus 格式）/ .txt / .md（每文件一篇）。

    json/jsonl 元素兼容字段：
      - {"doc_id"/"id"/"title", "text"/"content"/"paragraph_text", ...}
    """
    path = Path(path)
    docs: List[dict] = []
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix in (".jsonl", ".json", ".txt", ".md"))
        for f in files:
            if f.suffix in (".jsonl", ".json"):
                docs.extend(_load_json_any(f))
            else:
                docs.append({"doc_id": f.stem, "text": f.read_text(encoding="utf-8")})
    elif path.suffix in (".jsonl", ".json"):
        docs = _load_json_any(path)
    else:
        docs = [{"doc_id": path.stem, "text": path.read_text(encoding="utf-8")}]
    return [d for d in docs if d.get("text", "").strip()]


def _doc_from_record(rec: dict, fallback_id: str) -> dict:
    """从一条 JSON 记录提取 doc_id / text（兼容 MuSiQue corpus 等多种字段命名）。"""
    text = rec.get("text") or rec.get("content") or rec.get("paragraph_text") or ""
    doc_id = (
        rec.get("doc_id")
        or rec.get("id")
        or (rec.get("title") if rec.get("title") and str(rec.get("title")).strip() else None)
        or fallback_id
    )
    return {"doc_id": str(doc_id), "text": str(text), "meta": {k: v for k, v in rec.items() if k not in ("text", "content")}}


def _load_json_any(f: Path) -> List[dict]:
    """兼容 .jsonl（每行一条）与 .json（顶层数组 / {data: [...]}）；doc_id 自动去重。"""
    docs: List[dict] = []
    if f.suffix == ".jsonl":
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    docs.append(_doc_from_record(rec, f"{f.stem}::{len(docs)}"))
        _dedup_doc_ids(docs)
        return docs
    with open(f, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("data") or data.get("docs") or data.get("documents") or []
    if not isinstance(data, list):
        return []
    for i, rec in enumerate(data):
        if isinstance(rec, dict):
            docs.append(_doc_from_record(rec, f"{f.stem}::{i}"))
        elif isinstance(rec, str) and rec.strip():
            docs.append({"doc_id": f"{f.stem}::{i}", "text": rec, "meta": {}})
    _dedup_doc_ids(docs)
    return docs


def _dedup_doc_ids(docs: List[dict]) -> None:
    """同 doc_id 的文档自动加序号后缀（MuSiQue corpus 存在重复 title）。"""
    seen: dict = {}
    for d in docs:
        base = d["doc_id"]
        n = seen.get(base, 0)
        seen[base] = n + 1
        if n > 0:
            d["doc_id"] = f"{base}#{n + 1}"


class BuildPipeline:
    def __init__(self, cfg: Config, evolution_manager: Optional[EvolutionManager] = None):
        self.cfg = cfg
        self.processed_dir = cfg.path("paths.processed_dir", "data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.graph_path = self.processed_dir / "graph.pkl"

        # ---- 组件装配（全部由 yaml 配置驱动）----
        self.prompts = PromptManager.from_config(cfg)
        self.llm = build_llm(cfg)
        self.embedder = build_embedder(cfg)
        self.extractor = OpenIEExtractor.from_config(cfg, self.llm, self.prompts)
        self.chunker = Chunker.from_config(cfg)

        # ---- 增量构建：已有图则加载 ----
        if self.graph_path.exists():
            self.graph = KnowledgeGraph.load(self.graph_path)
            logger.info("加载已有图: %s", self.graph.stats())
        else:
            self.graph = KnowledgeGraph(
                case_insensitive_entities=bool(cfg.get("graph.case_insensitive_entities", False)),
                max_changelog=int(cfg.get("graph.max_changelog", 1000)),
            )
        self.builder = GraphBuilder(cfg=cfg.as_dict() if hasattr(cfg, "as_dict") else cfg, graph=self.graph)
        self.vdb = VectorStoreManager(cfg, self.embedder, self.processed_dir)

        # ---- 演进（EvoGraph-R1 注入点）----
        self.evolution = evolution_manager or EvolutionManager()
        self.evolution.ctx = EvolutionContext(
            config=cfg, graph=self.graph, vector_manager=self.vdb,
            embedder=self.embedder, llm=self.llm,
        )
        if bool(cfg.get("evolution.enabled", True)):
            self.evolution.load_from_config(cfg)
            if not self.evolution.strategies:
                self.evolution.register(_default_strategy(cfg))
        # 图演进门面 API（外部演进系统 / web /api/evolve 使用）
        self.graph_api = GraphEvolutionAPI(
            self.graph, self.vdb,
            auto_sync_vectorstore=bool(cfg.get("evolution.auto_sync_vectorstore", True)),
        )

    # ------------------------------------------------------------------
    def _apply_ops(self, ops: List[GraphOperation]) -> None:
        """在安全点应用演进策略返回的图操作。"""
        for op in ops:
            try:
                self.graph.apply_operation(op)
            except Exception as err:  # noqa: BLE001
                logger.error("演进图操作失败 %s: %s", op.op, err)

    def _chunk_docs(self, docs: List[dict], incremental: bool) -> List[Chunk]:
        known_docs = {self.graph.g.nodes[c].get("doc_id") for c in self.graph.chunks()}
        all_chunks: List[Chunk] = []
        for doc in docs:
            if incremental and doc["doc_id"] in known_docs:
                logger.info("增量模式跳过已构建文档: %s", doc["doc_id"])
                continue
            all_chunks.extend(self.chunker.create_chunks(doc["text"], doc["doc_id"]))
        return all_chunks

    # ------------------------------------------------------------------
    def run(
        self,
        input_path: Optional[Path] = None,
        incremental: bool = True,
        limit: Optional[int] = None,
    ) -> dict:
        t_start = time.time()
        self.evolution.emit("pipeline_start", graph=self.graph.stats())

        # 1. 文档与切块
        raw_path = Path(input_path) if input_path else self.cfg.path("paths.raw_dir", "data/raw")
        docs = load_documents(raw_path)
        if limit:
            docs = docs[: int(limit)]
        self.evolution.emit("before_chunking", docs=[d["doc_id"] for d in docs])
        chunks = self._chunk_docs(docs, incremental)
        chunk_texts = {c.chunk_id: c.text for c in chunks}
        self.evolution.emit("after_chunking", n_chunks=len(chunks))
        logger.info("文档 %d 篇 → chunk %d 个", len(docs), len(chunks))
        if not chunks:
            logger.warning("没有需要构建的新 chunk（全部已增量构建？）")

        # 2. OpenIE 流式抽取 + 增量构图（抽取与构图重叠：流式并行优化）
        self.evolution.emit("before_extract", n_chunks=len(chunks))
        self.evolution.emit("before_graph_build")

        batch_size = int(self.cfg.get("openie.batch_size", 16))
        llm_workers = int(self.cfg.get("llm.max_workers", 16))
        n_results = 0
        n_triples = 0

        def on_batch_done(idx: int, batch, graph: KnowledgeGraph) -> None:
            ops = self.evolution.emit("on_extract_batch", batch_idx=idx, batch=[r.to_dict() for r in batch])
            self._apply_ops(ops)

        def batch_stream() -> Iterator[List]:
            nonlocal n_results, n_triples
            for batch in self.extractor.extract_batched_stream(chunks, max_workers=llm_workers, batch_size=batch_size):
                n_results += len(batch)
                n_triples += sum(len(r.triples) for r in batch)
                yield batch

        self.builder.build(batch_stream(), chunk_texts, on_batch_done=on_batch_done)
        self.evolution.emit("after_extract", n_results=n_results, n_triples=n_triples)
        self.evolution.emit("after_graph_build", graph=self.graph.stats())

        # 3. 向量入库（增量：只嵌入新增项）
        self.evolution.emit("before_vectorize", graph=self.graph.stats())
        vdb_report = self.vdb.sync_from_graph(self.graph, incremental=incremental, progress=True)
        self.evolution.emit("after_vectorize", vdb_report=vdb_report)

        # 4. 产物导出
        artifacts = self.builder.export_artifacts(self.processed_dir)
        self.graph.save(self.graph_path)
        node_limit = self.cfg.get("web.graph_node_limit", 500)
        export = self.graph.export_json(self.processed_dir / "graph_export.json", node_limit=int(node_limit))
        self.vdb.save()

        report = {
            "docs": len(docs),
            "chunks": len(chunks),
            "openie_results": n_results,
            "triples_extracted": n_triples,
            "artifacts": artifacts,
            "vdb": vdb_report,
            "graph": self.graph.stats(),
            "graph_export": {"nodes": len(export["nodes"]), "edges": len(export["edges"])},
            "elapsed_sec": round(time.time() - t_start, 2),
            "incremental": incremental,
        }
        with open(self.processed_dir / "build_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        self.evolution.emit("pipeline_complete", report=report)
        logger.info("构建完成: %s", json.dumps(report, ensure_ascii=False))
        return report


def _default_strategy(cfg: Config):
    """默认空策略（保证钩子链路可用；EvoGraph-R1 接入时替换为真实策略）。"""
    from graph.evolution import NoopStrategy

    return NoopStrategy()
