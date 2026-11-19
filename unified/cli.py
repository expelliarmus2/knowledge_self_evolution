"""统一检索 CLI：一个入口，按输入形态自动分流。

- ``--query "自然语言问题"``    → 知识库多跳问答（RAG / ChunkHop）
- ``--ir '{"include": [...]}'`` → 科研入组筛查（QueryIR → SQLite 编译执行）
- ``--ir @file.json``           → 从文件读取 QueryIR
- ``--type kb|cohort``          → 显式指定引擎（跳过自动路由）

用法::

    python -m unified.cli --query "星辰科技的创始人是谁？" --mock --processed data/processed_demo_cn
    python -m unified.cli --ir '{"include": [{"type": "age", "min": 18, "max": 75}]}' --db data/cohort/demo.db
    python -m unified.cli --ir @/tmp/ir.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.config import Config


def build_kb_engine(cfg: Config):
    """按 pipeline.run.cmd_search 同款方式构建 KB 引擎（服务/索引不可用时降级 None）。"""
    from unified.demo import build_kb_engine as _demo_build

    return _demo_build(cfg)


def build_cohort_engine(cfg: Config, db: str | None, count: int | None):
    from cohort.engine import CohortEngine

    path = Path(db) if db else cfg.path("cohort.sqlite_path", "data/cohort/demo.db")
    return CohortEngine(path, seed_count=count or cfg.get("cohort.seed_count", 30))


def load_ir(spec: str):
    """--ir / --query 的 JSON 参数：JSON 字符串 或 @文件路径 → QueryIR。"""
    from cohort.query_ir import QueryIR

    raw = spec
    if spec.startswith("@"):
        raw = Path(spec[1:]).read_text(encoding="utf-8")
    return QueryIR.from_dict(json.loads(raw))


def maybe_ir_from_query(text: str) -> object | None:
    """--query 统一入口：字符串是可解析的入组 IR（含 include 的 JSON dict / @文件）时返回 QueryIR。

    这样 ``--query`` 一个参数即可覆盖两种形态：

    - ``--query '{"include": [...]}'`` → 科研入组（QueryIR）
    - ``--query '自然语言问题'``        → RAG（ChunkHop）
    """
    try:
        return load_ir(text) if text.strip().startswith(("{", "@")) else None
    except (ValueError, json.JSONDecodeError, OSError):
        return None

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="统一检索 CLI（str→RAG，QueryIR→科研入组）")
    parser.add_argument("--query", default=None, help="自然语言问题 → RAG")
    parser.add_argument("--ir", default=None, help="入组 QueryIR：JSON 字符串或 @文件 → 科研入组")
    parser.add_argument("--type", choices=["kb", "cohort"], default=None, help="显式引擎（跳过路由）")
    parser.add_argument("--mock", action="store_true", help="KB 侧 mock embedding/LLM（256 维）")
    parser.add_argument("--processed", default=None, help="KB 产物目录（默认 paths.processed_dir）")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--hops", type=int, default=None)
    parser.add_argument("--subgraph-mode", choices=["trace", "ppr"], default=None)
    parser.add_argument("--db", default=None, help="cohort SQLite 路径（缺库自动生成）")
    parser.add_argument("--count", type=int, default=None, help="种子患者数")
    parser.add_argument("--reference-date", default=None, help="cohort 引用日期（默认取 IR/seed）")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if not args.query and not args.ir:
        parser.error("必须提供 --query（自然语言→RAG）或 --ir（QueryIR→科研入组）之一")

    overrides: dict = {}
    if args.mock:
        overrides = {"embedding": {"provider": "mock", "dim": 256}, "llm": {"provider": "mock"}}
    if args.processed:
        # 与 KB_PATHS__* 约定一致：在 Config.load 前注入环境变量覆盖
        import os

        os.environ["KB_PATHS__PROCESSED_DIR"] = args.processed
    cfg = Config.load(overrides=overrides)

    # 输入形态决定引擎：QueryIR / IR JSON → cohort；纯自然语言 → kb（RAG）
    # --type 仅用于显式覆盖
    query_type = args.type
    if args.ir is not None:
        query: object = load_ir(args.ir)
    else:
        query = maybe_ir_from_query(args.query) or args.query

    from unified.engine import EngineRegistry, UnifiedEngine

    # 按需构建引擎：KB 引擎的构建代价很高（networkx/openai/faiss import、
    # graph.pkl 反序列化、FAISS 索引加载），入组查询不需要它。
    need_kb = query_type == "kb" or (isinstance(query, str) and query_type != "cohort")
    registry = EngineRegistry()
    kb_engine = None
    if need_kb:
        kb_engine = build_kb_engine(cfg)
        if kb_engine is None:
            # 自然语言 = RAG 语义：KB 不可用时前置给出明确提示
            print(
                "[unified] KB 引擎不可用（索引未构建或 embedding/LLM 服务未启动）；"
                "自然语言 --query 走 RAG 需要 KB 引擎。\n"
                "  请先构建：python -m pipeline.run build --mock --input <仅含示例文档的目录> --output <新目录>\n"
                "  或直接指向现成演示库：--processed data/processed_demo_cn\n"
                "  科研入组请改用：--ir '{\"include\": [...]}'（QueryIR 形态，不依赖 KB）",
                file=sys.stderr,
            )
            return 1
        registry.register(kb_engine)
    else:
        # cohort-only：不 import networkx/openai/faiss，不加载图与向量库
        registry.register(build_cohort_engine(cfg, args.db, args.count))

    # KB 侧参数透传（cohort 引擎会忽略未识别的参数）；reference_date 仅 cohort 使用
    params: dict = {k: v for k, v in {
        "top_k": args.top_k, "hops": args.hops, "subgraph_mode": args.subgraph_mode,
        "reference_date": args.reference_date,
    }.items() if v is not None}

    unified = UnifiedEngine(registry)
    try:
        result = unified.search(query, query_type=query_type, **params)
    except Exception as exc:
        print(f"[unified] 查询失败: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    # ---- 人读输出 ----
    from cohort.query_ir import QueryIR as _QueryIR

    print(f"=== 统一检索（engine={result.engine}，输入形态={'QueryIR' if isinstance(query, _QueryIR) else 'str'}）===")
    p = result.payload
    if result.engine == "cohort":
        print(f"  汇总: {p['summary']}   引用日期: {p['reference_date']}")
        for d in p["decisions"]:
            print(f"    {d['name']:<20} → {d['eligibility']:<7} 缺失: {d['missing'] or '无'}")
        print(f"  编译 SQL 片段: …{p['compiled_sql'][:140]}…")
    else:
        print(f"  graph v{p.get('graph_version')}，Top{len(p.get('chunks', []))} chunk:")
        for c in p.get("chunks", [])[:5]:
            print(f"    #{c['rank']} [{c['source']}|hop{c['hop']}] {c['chunk_id']} score={c['score']}")
        sg = p.get("subgraph", {}) or {}
        print(f"  子图: {len(sg.get('nodes', []))} nodes / {len(sg.get('edges', []))} edges"
              f" (mode={sg.get('mode')}, v{sg.get('version')})")
    print(f"  耗时: {result.meta['latency_ms']}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
