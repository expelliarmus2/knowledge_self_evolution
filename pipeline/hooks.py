"""流水线演进钩子定义（EvoGraph-R1 注入点一览）。

事件时序（BuildPipeline.run 内按序发射）：

    pipeline_start
      ├─ before_chunking ── after_chunking
      ├─ before_extract
      │    └─ on_extract_batch × N   # 每批抽取完成（流式），策略可返回图操作即时应用
      ├─ after_extract
      ├─ before_graph_build ── after_graph_build
      ├─ before_vectorize ── after_vectorize
      └─ pipeline_complete
    on_error（任意阶段异常）

策略返回值：List[GraphOperation]，在安全点由流水线统一应用到图
（应用后 graph.version 递增，检索侧自动感知）。

自定义策略示例（EvoGraph-R1 接入方式）：

    from graph.evolution import EvolutionStrategy, GraphOperation

    class MyEvolutionStrategy(EvolutionStrategy):
        name = "evograph_r1_demo"

        def on_event(self, event, ctx):
            if event == "on_extract_batch":
                # 例：基于抽取批次决策图演进（增删改节点与边）
                return [GraphOperation(
                    "add_triple",
                    {"head": "A", "relation": "演进补充关系", "tail": "B", "weight": 2.0},
                    reason="EvoGraph-R1 决策",
                )]
            return []

然后在 config 中启用：
    evolution:
      enabled: true
      strategies: ["my_pkg.my_module:MyEvolutionStrategy"]
"""
from graph.evolution import (
    PIPELINE_EVENTS,
    EvolutionContext,
    EvolutionManager,
    EvolutionStrategy,
    GraphOperation,
)

__all__ = [
    "PIPELINE_EVENTS",
    "EvolutionContext",
    "EvolutionManager",
    "EvolutionStrategy",
    "GraphOperation",
]
