"""CLI 入口（项目根目录运行）：

    python -m pipeline.run build  --mock                       # 离线演示：构建知识库
    python -m pipeline.run build  --input data/raw --limit 3   # 真实模式（yaml 配置模型）
    python -m pipeline.run build  --full                       # 全量重建（默认增量）
    python -m pipeline.run build  --input /path/to/docs --output /path/to/kb   # 自定义数据位置
    python -m pipeline.run search --query "..." --mock --json  # 检索
    python -m pipeline.run search --query "..." --processed /path/to/kb       # 指定知识库产物目录
    python -m pipeline.run evolve --ops '[{"op":"add_triple","payload":{...}}]'  # 演进接口演示
    python -m pipeline.run serve  --mock                       # Web 可视化服务
    python -m pipeline.run serve  --processed /path/to/kb --port 5571

数据位置配置优先级：CLI 参数 > 环境变量（KB_PATHS__RAW_DIR 等）> --config yaml > config/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import Config  # noqa: E402


def _load_cfg(args, path_overrides: dict = None) -> Config:
    """组装配置：mock 开关 + 数据路径覆盖（CLI 最高优先）。"""
    overrides: dict = {}
    if getattr(args, "mock", False):
        overrides["embedding"] = {"provider": "mock", "dim": 256}
        overrides["llm"] = {"provider": "mock"}
    if path_overrides:
        overrides["paths"] = path_overrides
    return Config.load(path=args.config, overrides=overrides)


def _path_overrides(args, keys=("raw", "processed", "output")) -> dict:
    """从 CLI 参数提取 paths 覆盖（raw→paths.raw_dir, processed/output→paths.processed_dir）。"""
    out: dict = {}
    raw = getattr(args, "raw", None)
    processed = getattr(args, "processed", None) or getattr(args, "output", None)
    if raw:
        out["raw_dir"] = str(Path(raw).resolve())
    if processed:
        out["processed_dir"] = str(Path(processed).resolve())
    _ = keys
    return out


def cmd_build(args) -> None:
    from pipeline.build_pipeline import BuildPipeline
    from utils.runtime import setup_logging

    cfg = _load_cfg(args, _path_overrides(args))
    setup_logging(cfg.get("logging.level", "INFO"), cfg.path("paths.logs_dir", "logs"))
    pipeline = BuildPipeline(cfg)
    report = pipeline.run(
        input_path=Path(args.input) if args.input else None,
        incremental=not args.full,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_search(args) -> None:
    from openie.embedder import build_embedder
    from openie.llm_client import build_llm
    from retrieval.retriever import Retriever
    from utils.prompts import PromptManager
    from utils.runtime import setup_logging
    from vectorstore.manager import VectorStoreManager

    cfg = _load_cfg(args, _path_overrides(args))
    setup_logging(cfg.get("logging.level", "INFO"), cfg.path("paths.logs_dir", "logs"))
    processed_dir = cfg.path("paths.processed_dir", "data/processed")
    graph_path = processed_dir / "graph.pkl"
    if not graph_path.exists():
        sys.exit("图不存在，请先运行 build")

    from graph.knowledge_graph import KnowledgeGraph

    graph = KnowledgeGraph.load(graph_path)
    embedder = build_embedder(cfg)
    llm = build_llm(cfg)
    prompts = PromptManager.from_config(cfg)
    vdb = VectorStoreManager(cfg, embedder, processed_dir)
    retriever = Retriever(cfg, graph, vdb, embedder, llm, prompts, graph_path=graph_path)
    result = retriever.search(args.query, top_k=args.top_k, hops=args.hops,
                              subgraph_mode=args.subgraph_mode)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"\n[graph v{result.graph_version}] 查询: {result.query}")
        print(f"\n— LLM 过滤后三元组（{result.filter_info.get('mode')}）:")
        for t in result.filtered_triples:
            print(f"   {t[0]} ──{t[1]}──> {t[2]}")
        print(f"\n— PPR 排序 Top{len(result.chunks)} chunk:")
        for c in result.chunks:
            print(f"   #{c['rank']} [{c['source']}|hop{c['hop']}] {c['chunk_id']} (score={c['score']})")
            print(f"      {c['text'][:80]}...")
        print(f"\n— 子图: {len(result.subgraph.get('nodes', []))} nodes / {len(result.subgraph.get('edges', []))} edges"
              f" (mode={result.subgraph.get('mode', 'trace')})")
        print(f"— 耗时: {json.dumps(result.timing, ensure_ascii=False)}")


def cmd_ask(args) -> None:
    """Agentic Search：IRCoT / Planner 多步检索 + 答案生成。"""
    from agentic import run_ircot, run_planner, Tools
    from openie.embedder import build_embedder
    from openie.llm_client import build_llm
    from retrieval.retriever import Retriever
    from utils.prompts import PromptManager
    from utils.runtime import setup_logging
    from vectorstore.manager import VectorStoreManager
    from graph.knowledge_graph import KnowledgeGraph

    cfg = _load_cfg(args, _path_overrides(args))
    setup_logging(cfg.get("logging.level", "INFO"), cfg.path("paths.logs_dir", "logs"))
    processed_dir = cfg.path("paths.processed_dir", "data/processed")
    graph_path = processed_dir / "graph.pkl"
    if not graph_path.exists():
        sys.exit("图不存在，请先运行 build")

    graph = KnowledgeGraph.load(graph_path)
    embedder = build_embedder(cfg)
    llm = build_llm(cfg)
    prompts = PromptManager.from_config(cfg)
    vdb = VectorStoreManager(cfg, embedder, processed_dir)
    retriever = Retriever(cfg, graph, vdb, embedder, None, prompts, graph_path=graph_path)
    tools = Tools(retriever)

    mode = args.mode or str(cfg.get("agentic.mode", "ircot")).lower()
    runner = run_ircot if mode == "ircot" else run_planner
    res = runner(args.question, tools, llm, prompts, cfg)

    if args.json:
        print(json.dumps({
            "question": args.question,
            "mode": mode,
            "answer": res.answer,
            "stop_reason": res.stop_reason,
            "llm_calls": res.llm_calls,
            "evidence_docs": res.evidence_docs,
            "steps": res.steps,
        }, ensure_ascii=False, indent=2))
        return
    print(f"\nQuery: {args.question}")
    print(f"Mode : {mode}  stop={res.stop_reason}  llm_calls={res.llm_calls}")
    print(f"\nAnswer: {res.answer}")
    print(f"\n证据文档（{len(res.evidence_docs)}）:")
    for i, d in enumerate(res.evidence_docs, 1):
        print(f"  {i}. {d}")
    print("\n推理/检索步:")
    for i, s in enumerate(res.steps, 1):
        if s.get("stop") == "final":
            print(f"  {i}. [FINAL]")
        elif "sub" in s:
            print(f"  {i}. {s['sub']}  +{s.get('n_new', 0)} docs")
        else:
            print(f"  {i}. {s.get('sentence', '')}  +{s.get('n_new', 0)} docs")


def cmd_evolve(args) -> None:
    """演进接口演示：CLI 直接应用图操作（EvoGraph-R1 将通过同一 API 注入）。"""
    from graph.evolution import GraphOperation
    from graph.knowledge_graph import KnowledgeGraph
    from openie.embedder import build_embedder
    from utils.runtime import setup_logging
    from vectorstore.manager import VectorStoreManager

    cfg = _load_cfg(args, _path_overrides(args))
    setup_logging(cfg.get("logging.level", "INFO"), cfg.path("paths.logs_dir", "logs"))
    processed_dir = cfg.path("paths.processed_dir", "data/processed")
    graph_path = processed_dir / "graph.pkl"
    graph = KnowledgeGraph.load(graph_path)
    embedder = build_embedder(cfg)
    vdb = VectorStoreManager(cfg, embedder, processed_dir)

    from graph.evolution import GraphEvolutionAPI

    api = GraphEvolutionAPI(graph, vdb, auto_sync_vectorstore=bool(cfg.get("evolution.auto_sync_vectorstore", True)))
    ops = [GraphOperation(o.get("op", ""), o.get("payload", {}), o.get("reason", "")) for o in json.loads(args.ops)]
    result = api.apply(ops)
    graph.save(graph_path)
    graph.export_json(processed_dir / "graph_export.json", node_limit=int(cfg.get("web.graph_node_limit", 500)))
    vdb.save()
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_serve(args) -> None:
    from utils.runtime import setup_logging
    from web.app import create_app

    cfg = _load_cfg(args, _path_overrides(args))
    setup_logging(cfg.get("logging.level", "INFO"), cfg.path("paths.logs_dir", "logs"))
    app = create_app(cfg)
    app.run(
        host=cfg.get("web.host", "0.0.0.0"),
        port=int(args.port or cfg.get("web.port", 5570)),
        debug=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="知识库自演进 · 构建与检索 CLI")
    parser.add_argument("--config", default=None, help="yaml 配置路径（默认 config/default.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="端到端构建流水线")
    p_build.add_argument("--input", "--raw", dest="input", default=None, help="输入文档路径（默认 paths.raw_dir）")
    p_build.add_argument("--output", "--processed", dest="output", default=None, help="产物目录（默认 paths.processed_dir）")
    p_build.add_argument("--full", action="store_true", help="全量重建（默认增量）")
    p_build.add_argument("--limit", type=int, default=None, help="只处理前 N 篇文档")
    p_build.add_argument("--mock", action="store_true", help="离线 mock 模式")
    p_build.set_defaults(func=cmd_build)

    p_search = sub.add_parser("search", help="检索")
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=None)
    p_search.add_argument("--hops", type=int, default=None)
    p_search.add_argument("--json", action="store_true", help="输出完整 JSON")
    p_search.add_argument("--subgraph-mode", choices=["trace", "ppr"], default=None,
                          help="子图模式：trace=溯源子图（默认）| ppr=PPR 展开全子图")
    p_search.add_argument("--processed", default=None, help="知识库产物目录（默认 paths.processed_dir）")
    p_search.add_argument("--mock", action="store_true", help="离线 mock 模式")
    p_search.set_defaults(func=cmd_search)

    p_ask = sub.add_parser("ask", help="Agentic Search（IRCoT/Planner 多步检索+作答）")
    p_ask.add_argument("--question", required=True)
    p_ask.add_argument("--mode", choices=["ircot", "planner"], default=None)
    p_ask.add_argument("--json", action="store_true", help="输出完整 JSON")
    p_ask.add_argument("--processed", default=None, help="知识库产物目录（默认 paths.processed_dir）")
    p_ask.add_argument("--mock", action="store_true", help="离线 mock 模式")
    p_ask.set_defaults(func=cmd_ask)

    p_evolve = sub.add_parser("evolve", help="应用图演进操作（演示 EvoGraph-R1 接口）")
    p_evolve.add_argument("--ops", required=True, help='JSON 数组 [{"op":"add_triple","payload":{...}}]')
    p_evolve.add_argument("--processed", default=None, help="知识库产物目录（默认 paths.processed_dir）")
    p_evolve.add_argument("--mock", action="store_true", help="离线 mock 模式")
    p_evolve.set_defaults(func=cmd_evolve)

    p_serve = sub.add_parser("serve", help="Web 可视化服务")
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--processed", default=None, help="知识库产物目录（默认 paths.processed_dir）")
    p_serve.add_argument("--mock", action="store_true", help="离线 mock 模式")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
