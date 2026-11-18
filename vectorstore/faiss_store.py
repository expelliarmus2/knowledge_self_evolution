"""FAISS 向量库封装：IndexIDMap2(IndexFlatIP)，key 级增删改查。

- 向量 L2 归一化后使用内积检索 = cosine 相似度；
- 每个 item 有稳定业务 key（实体名 / 三元组串 / chunk_id），ID 映射持久化；
- remove/update 支持增量演进（EvoGraph-R1 图操作后的向量库同步依赖于此）。
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("kb")


class FaissStore:
    def __init__(self, name: str, dim: int, path: Optional[Path] = None):
        import faiss

        self.name = name
        self.dim = int(dim)
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self.key2id: Dict[str, int] = {}
        self.id2key: Dict[int, str] = {}
        self.metas: Dict[str, dict] = {}
        self._next_id = 0

        if self.path is not None and self.path.exists():
            self.index = faiss.read_index(str(self.path))
            meta_path = self._meta_path()
            if meta_path.exists():
                state = json.loads(meta_path.read_text(encoding="utf-8"))
                self.key2id = {k: int(v) for k, v in state["key2id"].items()}
                self.id2key = {int(k): v for k, v in state["id2key"].items()}
                self.metas = state.get("metas", {})
                self._next_id = int(state.get("next_id", len(self.key2id)))
            else:
                # 索引存在但元数据丢失：按位置重建映射
                for i in range(self.index.ntotal):
                    self.id2key[i] = f"__pos{i}__"
                self.key2id = {v: k for k, v in self.id2key.items()}
                self._next_id = self.index.ntotal
        else:
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    # ------------------------------------------------------------------
    def _meta_path(self) -> Path:
        return self.path.with_suffix(".meta.json")

    def _validate(self, vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != self.dim:
            raise ValueError(f"[{self.name}] 向量维度不匹配: {vecs.shape[1]} != {self.dim}")
        return vecs

    # ------------------------------------------------------------------
    def add(self, items: List[dict], replace: bool = True) -> int:
        """items: [{"key": str, "vector": list|ndarray, "meta": dict}]。

        - key 已存在：replace=True 时更新（删除重加），否则跳过；
        返回实际新增/更新的数量。
        """
        if not items:
            return 0
        import faiss

        with self._lock:
            new_items: List[dict] = []
            for item in items:
                key = str(item["key"])
                if key in self.key2id:
                    if replace:
                        self._remove_keys_locked([key])
                        new_items.append(item)
                    continue
                else:
                    new_items.append(item)
            if not new_items:
                return 0
            vecs = self._validate([it["vector"] for it in new_items])
            ids = np.array([self._next_id + i for i in range(len(new_items))], dtype="int64")
            self.index.add_with_ids(vecs, ids)
            for offset, item in enumerate(new_items):
                key = str(item["key"])
                idx = int(ids[offset])
                self.key2id[key] = idx
                self.id2key[idx] = key
                self.metas[key] = item.get("meta") or {}
            self._next_id += len(new_items)
            return len(new_items)

    def _remove_keys_locked(self, keys: Iterable[str]) -> int:
        import faiss

        ids = np.array([self.key2id[k] for k in keys if k in self.key2id], dtype="int64")
        if len(ids) == 0:
            return 0
        removed = self.index.remove_ids(faiss.IDSelectorBatch(ids))
        for idx in ids:
            key = self.id2key.pop(int(idx), None)
            if key is not None:
                self.key2id.pop(key, None)
                self.metas.pop(key, None)
        return int(getattr(removed, "n", len(ids)) or len(ids))

    def remove(self, keys: Iterable[str]) -> int:
        keys = [str(k) for k in keys]
        with self._lock:
            return self._remove_keys_locked(keys)

    def update(self, items: List[dict]) -> int:
        """删除旧 key 后重加（向量的更新 = 删 + 增）。"""
        keys = [str(it["key"]) for it in items]
        with self._lock:
            self._remove_keys_locked(keys)
            return self.add(items, replace=False)

    # ------------------------------------------------------------------
    def search(self, query_vec, top_k: int) -> List[Tuple[str, float, dict]]:
        """检索：返回 [(key, score, meta)]，按相似度降序。"""
        with self._lock:
            if self.index.ntotal == 0:
                return []
            vecs = self._validate(query_vec)
            k = min(int(top_k), self.index.ntotal)
            dists, idxs = self.index.search(vecs, k)
        out: List[Tuple[str, float, dict]] = []
        for d, i in zip(dists[0], idxs[0]):
            if i < 0:
                continue
            key = self.id2key.get(int(i))
            if key is None:
                continue
            out.append((key, float(d), dict(self.metas.get(key, {}))))
        return out

    def has(self, key: str) -> bool:
        return str(key) in self.key2id

    def get_meta(self, key: str) -> dict:
        return dict(self.metas.get(str(key), {}))

    def count(self) -> int:
        return len(self.key2id)

    def keys(self) -> List[str]:
        return list(self.key2id.keys())

    # ------------------------------------------------------------------
    def save(self) -> None:
        import faiss

        if self.path is None:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.path))
            state = {
                "key2id": self.key2id,
                "id2key": {str(k): v for k, v in self.id2key.items()},
                "metas": self.metas,
                "next_id": self._next_id,
                "dim": self.dim,
            }
            self._meta_path().write_text(
                json.dumps(state, ensure_ascii=False), encoding="utf-8"
            )

    def stats(self) -> dict:
        return {"name": self.name, "dim": self.dim, "count": self.count()}
