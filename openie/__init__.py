"""OpenIE 模块：HippoRAG2 两步式三元组抽取（NER → 实体条件化 RE）。

组件：
- llm_client : OpenAI 兼容 LLM 客户端（+ Mock 离线实现）
- embedder   : OpenAI 兼容 Embedding 客户端，批量+有界并发+重试（+ Mock）
- chunker    : 文档切块
- extractor  : 两步式抽取器（实体锚定校验、代词解析）
"""
from openie.llm_client import LLMClient, MockLLMClient, build_llm
from openie.embedder import Embedder, MockEmbedder, build_embedder
from openie.extractor import OpenIEExtractor, OpenIEResult, Triple

__all__ = [
    "LLMClient", "MockLLMClient", "build_llm",
    "Embedder", "MockEmbedder", "build_embedder",
    "OpenIEExtractor", "OpenIEResult", "Triple",
]
