"""向量库模块：实体 / 三元组 / chunk 三个 FAISS 库的统一管理。

- faiss_store : IndexIDMap2(IndexFlatIP) 封装，支持稳定 key 的增删改查
- manager     : 与知识图谱联动的增量同步（只嵌入新增/变更项）
"""
from vectorstore.faiss_store import FaissStore
from vectorstore.manager import VectorStoreManager

__all__ = ["FaissStore", "VectorStoreManager"]
