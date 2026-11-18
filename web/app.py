"""Web 可视化服务：检索子图渲染 + 图演进接口演示。

端点：
- GET  /            → 可视化页面（web/index.html）
- GET  /api/graph   → 构图产物（graph_export.json）全图数据
- GET  /api/stats   → 图 / 向量库统计
- POST /api/search  → 检索：双路召回 + hop 扩散 + PPR + LLM 过滤，返回子图
- POST /api/evolve  → 演进接口（EvoGraph-R1 注入点演示）：
                      应用图操作（增删改节点与边），自动同步向量库并提升版本号，
                      下次检索立即感知新图结构。
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from graph.evolution import GraphEvolutionAPI, GraphOperation
from graph.knowledge_graph import KnowledgeGraph
from openie.embedder import build_embedder
from openie.llm_client import build_llm
from retrieval.retriever import Retriever
from utils.config import Config
from utils.prompts import PromptManager
from vectorstore.manager import VectorStoreManager

logger = logging.getLogger("kb")

WEB_DIR = Path(__file__).resolve().parent


def create_app(cfg: Config) -> Flask:
    app = Flask(__name__, static_folder=str(WEB_DIR / "static"), static_url_path="/static")

    # 静态资源禁用浏览器缓存（开发迭代阶段避免旧 JS/CSS 残留）
    @app.after_request
    def _no_static_cache(resp):
        if resp.status_code == 200 and any(
            resp.mimetype in ("text/css", "application/javascript", "text/html")
            or resp.content_type and "javascript" in resp.content_type
            for _ in [0]
        ):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return resp

    processed_dir = cfg.path("paths.processed_dir", "data/processed")
    graph_path = processed_dir / "graph.pkl"
    export_path = processed_dir / "graph_export.json"

    state = {"retriever": None, "api": None}
    lock = threading.RLock()

    # ---- 懒初始化（首次请求时装配检索链路）----
    def _init():
        if state["retriever"] is not None:
            return
        if not graph_path.exists():
            raise FileNotFoundError("图不存在，请先运行 build")
        graph = KnowledgeGraph.load(graph_path)
        embedder = build_embedder(cfg)
        llm = build_llm(cfg)
        prompts = PromptManager.from_config(cfg)
        vdb = VectorStoreManager(cfg, embedder, processed_dir)
        retriever = Retriever(cfg, graph, vdb, embedder, llm, prompts, graph_path=graph_path)
        state["retriever"] = retriever
        state["api"] = GraphEvolutionAPI(
            graph, vdb, auto_sync_vectorstore=bool(cfg.get("evolution.auto_sync_vectorstore", True))
        )

    # ---- 页面 ----
    @app.route("/")
    def index():
        return send_from_directory(str(WEB_DIR), "index.html")

    @app.route("/lib/<path:filename>")
    def lib(filename):
        return send_from_directory(str(WEB_DIR / "lib"), filename)

    # ---- 图数据 ----
    @app.route("/api/graph")
    def api_graph():
        if not export_path.exists():
            return jsonify({"error": "graph_export.json 不存在，请先运行 build"}), 404
        with open(export_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))

    @app.route("/api/stats")
    def api_stats():
        with lock:
            _init()
            return jsonify({
                "graph": state["retriever"].graph.stats(),
                "vdb": state["retriever"].vdb.stats(),
            })

    # ---- 检索 ----
    @app.route("/api/search", methods=["POST"])
    def api_search():
        data = request.get_json(force=True, silent=True) or {}
        query = str(data.get("query", "")).strip()
        if not query:
            return jsonify({"error": "query 不能为空"}), 400
        overrides = {}
        if data.get("top_k"):
            overrides["final_top_k"] = int(data["top_k"])
        if data.get("hops") is not None:
            overrides["hops"] = int(data["hops"])
        subgraph_mode = data.get("subgraph_mode")
        if data.get("trace_max_entities"):
            overrides["trace_max_entities"] = max(5, int(data["trace_max_entities"]))
        with lock:
            try:
                _init()
                eff_cfg = cfg.with_overrides({"retrieval": overrides}) if overrides else cfg
                retriever = state["retriever"]
                retriever.cfg = eff_cfg
                result = retriever.search(query, top_k=data.get("top_k"), hops=data.get("hops"),
                                          subgraph_mode=subgraph_mode)
                return jsonify(result.to_dict())
            except FileNotFoundError as err:
                return jsonify({"error": str(err)}), 404
            except Exception as err:  # noqa: BLE001
                logger.exception("检索失败")
                return jsonify({"error": f"{type(err).__name__}: {err}"}), 500

    # ---- 演进接口（EvoGraph-R1 注入点）----
    @app.route("/api/evolve", methods=["POST"])
    def api_evolve():
        data = request.get_json(force=True, silent=True) or {}
        raw_ops = data.get("ops") or []
        if not raw_ops:
            return jsonify({"error": "ops 不能为空"}), 400
        with lock:
            try:
                _init()
                api: GraphEvolutionAPI = state["api"]
                ops = [GraphOperation(o.get("op", ""), o.get("payload", {}), o.get("reason", "")) for o in raw_ops]
                result = api.apply(ops)
                # 演进后持久化 + 刷新导出（检索侧经 _ensure_fresh 感知）
                api.graph.save(graph_path)
                api.graph.export_json(export_path, node_limit=int(cfg.get("web.graph_node_limit", 500)))
                vdb = state["retriever"].vdb
                vdb.save()
                state["retriever"]._graph_mtime = None  # 强制下次检索重载
                return jsonify(result)
            except Exception as err:  # noqa: BLE001
                logger.exception("演进操作失败")
                return jsonify({"error": f"{type(err).__name__}: {err}"}), 500

    @app.route("/api/health")
    def health():
        return jsonify({"ok": True, "graph_ready": graph_path.exists()})

    return app


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="知识库自演进 · Web 可视化")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--processed", default=None, help="知识库产物目录（默认 paths.processed_dir）")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    overrides: dict = {}
    if args.mock:
        overrides = {"embedding": {"provider": "mock", "dim": 256}, "llm": {"provider": "mock"}}
    if args.processed:
        overrides = {"paths": {"processed_dir": str(Path(args.processed).resolve())}}
    cfg = Config.load(path=args.config, overrides=overrides)
    app = create_app(cfg)
    app.run(
        host=args.host or cfg.get("web.host", "0.0.0.0"),
        port=args.port or int(cfg.get("web.port", 5570)),
        debug=False,
    )


if __name__ == "__main__":
    main()
