"""统一门面 demo：同一查询入口分别跑 KB 检索与科研入组筛查。

用法::

    python -m unified.demo                 # 自动生成 demo.db + 走统一入口（KB 索引存在则一并演示）
    python -m unified.demo --db /tmp/x.db  # 指定入组库
    python -m unified.demo --json          # 输出 JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.config import Config


def build_kb_engine(cfg: Config):
    """按 pipeline.run.cmd_search 同款方式构建 KB 引擎；

    图不存在或依赖服务（embedding/LLM）不可用时返回 None，由调用方降级。
    """
    processed_dir = cfg.path("paths.processed_dir", "data/processed")
    graph_path = processed_dir / "graph.pkl"
    if not graph_path.exists():
        return None

    try:
        from graph.knowledge_graph import KnowledgeGraph
        from openie.embedder import build_embedder
        from openie.llm_client import build_llm
        from retrieval.retriever import Retriever
        from unified.engine import KBEngineAdapter
        from utils.prompts import PromptManager
        from vectorstore.manager import VectorStoreManager

        graph = KnowledgeGraph.load(graph_path)
        embedder = build_embedder(cfg)
        llm = build_llm(cfg)
        prompts = PromptManager.from_config(cfg)
        vdb = VectorStoreManager(cfg, embedder, processed_dir)
        retriever = Retriever(cfg, graph, vdb, embedder, llm, prompts, graph_path=graph_path)
        return KBEngineAdapter(retriever)
    except Exception as exc:  # 服务未启动 / 库损坏等 → 降级
        print(f"[kb] KB 引擎构建失败，跳过: {type(exc).__name__}: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统一门面 demo")
    parser.add_argument("--db", default=None, help="入组 demo SQLite 路径")
    parser.add_argument("--count", type=int, default=None, help="种子患者数")
    parser.add_argument("--mock", action="store_true", help="KB 侧使用 mock embedding/LLM")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    overrides = {"embedding": {"provider": "mock", "dim": 256}, "llm": {"provider": "mock"}} if args.mock else {}
    cfg = Config.load(overrides=overrides)

    # ---- 1. 注册 cohort 引擎（缺库自动生成）----
    db = Path(args.db) if args.db else cfg.path("cohort.sqlite_path", "data/cohort/demo.db")
    from cohort.engine import CohortEngine
    from cohort.seed import DEMO_IR

    cohort_engine = CohortEngine(db, seed_count=args.count or cfg.get("cohort.seed_count", 30))

    # ---- 2. 注册 KB 引擎（索引存在时）----
    kb_engine = build_kb_engine(cfg)

    from unified.engine import EngineRegistry, UnifiedEngine

    registry = EngineRegistry()
    registry.register(cohort_engine)
    if kb_engine is not None:
        registry.register(kb_engine)
    unified = UnifiedEngine(registry)

    out: dict = {"registry": registry.describe()}

    # ---- 3. 同一入口：科研入组筛查（结构化 IR，dict 自动路由到 cohort）----
    result = unified.search(DEMO_IR)
    out["cohort"] = result.to_dict()
    cohort_summary = result.payload["summary"]

    # ---- 4. 同一入口：知识库问答（自然语言，路由到 kb）----
    if kb_engine is not None:
        try:
            kb_result = unified.search("星辰科技的创始人是谁？", top_k=5)
            payload = kb_result.payload
            out["kb"] = {
                "engine": kb_result.engine,
                "graph_version": payload.get("graph_version"),
                "top_chunks": payload.get("chunks", [])[:5],
                "triples": payload.get("triples", [])[:10],
                "subgraph": payload.get("subgraph", {}),
            }
        except Exception as exc:
            print(f"[kb] KB 查询失败，降级跳过: {type(exc).__name__}: {exc}")
            out["kb"] = None
    else:
        out["kb"] = None

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    # ---- 5. 人读输出 ----
    print(f"\n=== 统一门面 demo（已注册引擎: {registry.names()}）===")
    print(f"\n[1] 科研入组筛查（engine=cohort, IR 自动路由）")
    print(f"    汇总: {cohort_summary}")
    print(f"    引用日期: {result.payload['reference_date']}")
    for d in result.payload["decisions"]:
        if d["patient_id"] <= 6:  # 只展示 6 个原型患者
            print(f"      {d['name']:<18} → {d['eligibility']:<7} 缺失: {d['missing'] or '无'}")
    print(f"    编译 SQL 片段: …{result.payload['compiled_sql'][:120]}…")

    if out["kb"] is not None:
        print(f"\n[2] 知识库问答（engine=kb, 自然语言自动路由）")
        print(f"    graph v{out['kb']['graph_version']}，Top5 chunk:")
        for c in out["kb"]["top_chunks"]:
            print(f"      #{c['rank']} [{c['source']}|hop{c['hop']}] {c['chunk_id']} score={c['score']}")
    else:
        print("\n[2] 知识库问答: 跳过（KB 服务未启动或索引未构建；先启动 embedding/LLM 服务并运行 python -m pipeline.run build）")
    print("\n提示: python -m unified.demo --json 可查看完整 JSON。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
