"""检索模块（Green-ChunkHop 思路）：

- hop_expander : CHUNK_HOP 多跳扩散（共享实体的 chunk 邻接，带剪枝与查询感知权重）
- filterer     : LLM 对候选三元组做相关性过滤（HippoRAG2 filter）
- retriever    : 双路召回（三元组库 + 实体库 + chunk 向量库并行）→ hop 扩散 → PPR 排序
                 → LLM 过滤；演进感知（graph.version 变化自动重建缓存）
"""
from retrieval.retriever import Retriever, SearchResult
from retrieval.hop_expander import HopExpander
from retrieval.filterer import TripleFilterer

__all__ = ["Retriever", "SearchResult", "HopExpander", "TripleFilterer"]
