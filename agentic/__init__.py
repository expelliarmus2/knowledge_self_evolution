"""Agentic Search：迭代 CoT 检索（IRCoT）与分解式规划检索（Planner）。

与单跳检索的关系：agentic 层把 `Retriever.search()` 当作工具，不修改检索内部逻辑。
证据池按 doc_id 去重（复用检索器的文档级去重语义），每轮用上一句推理句检索 ——
这是 IRCoT 的核心：检索信号从"原始问题"变为"逐步逼近的推理句"。
"""
from .ircot import IrcotResult, run_ircot  # noqa: F401
from .planner import run_planner  # noqa: F401
from .memory import EvidencePool  # noqa: F401
from .tools import Tools  # noqa: F401

__all__ = ["IrcotResult", "run_ircot", "run_planner", "EvidencePool", "Tools"]
