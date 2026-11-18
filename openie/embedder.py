"""Embedding 客户端：批量 + 有界并发 + 重试退避 + 可选磁盘缓存。

调度优化吸收自 huaxi_poc（thread_map 并行）与 chunkhop（embedding_cache）：
1. 请求批量化（batch_size），减少网络往返；
2. 有界并发（max_workers），打满 embedding 服务而不过载；
3. 指数退避重试；
4. 增量构建时只对新增/变更项计算（由 VectorStoreManager 保证）；
5. 磁盘向量缓存可选（cache_enable），跨构建复用。
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from utils.config import Config
from utils.runtime import run_parallel

logger = logging.getLogger("kb")


class Embedder:
    """OpenAI 兼容 embeddings 客户端。"""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "dummy",
        dim: int = 1024,
        normalize: bool = True,
        batch_size: int = 32,
        max_workers: int = 8,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
        timeout: int = 60,
        cache_file: Optional[Path] = None,
    ):
        from openai import OpenAI
        self.model = model
        self.dim = dim
        self.normalize = normalize
        self.batch_size = max(1, batch_size)
        self.max_workers = max(1, max_workers)
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self._cache: Dict[str, list] = {}
        self._cache_file = cache_file
        self._cache_lock = threading.Lock()
        if cache_file is not None and Path(cache_file).exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    for line in f:
                        rec = json.loads(line)
                        self._cache[rec["key"]] = rec["vec"]
                logger.info("加载 embedding 缓存: %d 条", len(self._cache))
            except Exception as err:  # noqa: BLE001
                logger.warning("embedding 缓存加载失败（忽略）: %s", err)

    # ---- 缓存 ----
    def _cache_key(self, text: str) -> str:
        return hashlib.sha1(f"{self.model}::{text}".encode("utf-8")).hexdigest()

    def _cache_get(self, text: str) -> Optional[list]:
        if self._cache_file is None:
            return None
        with self._cache_lock:
            return self._cache.get(self._cache_key(text))

    def _cache_put(self, text: str, vec: list) -> None:
        if self._cache_file is None:
            return
        with self._cache_lock:
            self._cache[self._cache_key(text)] = vec
            try:
                with open(self._cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": self._cache_key(text), "vec": vec}, ensure_ascii=False) + "\n")
            except Exception as err:  # noqa: BLE001
                logger.warning("embedding 缓存写入失败: %s", err)

    # ---- 核心 ----
    def _embed_batch_remote(self, texts: List[str]) -> List[list]:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.client.embeddings.create(input=texts, model=self.model)
                vecs = [d.embedding for d in resp.data]
                if len(vecs) != len(texts):
                    raise ValueError(f"embedding 返回数量不匹配: {len(vecs)} != {len(texts)}")
                return vecs
            except Exception as err:  # noqa: BLE001
                last_err = err
                if attempt < self.max_retries:
                    sleep = self.retry_backoff * (2 ** attempt) + random.random() * 0.5
                    logger.warning("embedding 调用失败(第%d次): %s，%.1fs 后重试", attempt + 1, err, sleep)
                    time.sleep(sleep)
        raise RuntimeError(f"embedding 调用最终失败: {last_err}") from last_err

    def embed(self, text: str) -> List[float]:
        text = str(text).replace("\n", " ").strip()
        if not text:
            text = " "
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vec = self._embed_batch_remote([text])[0]
        self._cache_put(text, vec)
        return vec

    def embed_batch(self, texts: List[str], progress: bool = False) -> np.ndarray:
        """批量并行 embedding，返回 (n, dim) 的矩阵（已按 normalize 配置归一化）。"""
        texts = [str(t).replace("\n", " ").strip() or " " for t in texts]
        # 切批：先查缓存，未命中项分批
        batches: List[List[str]] = []
        results: Dict[int, list] = {}
        pending: List[tuple] = []  # (orig_idx, text)
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                results[i] = cached
            else:
                pending.append((i, t))
        for s in range(0, len(pending), self.batch_size):
            batches.append(pending[s : s + self.batch_size])

        def _do(batch: List[tuple]) -> None:
            idxs = [i for i, _ in batch]
            vecs = self._embed_batch_remote([t for _, t in batch])
            for t, v in zip(batch, vecs):
                self._cache_put(t[1], v)
            for i, v in zip(idxs, vecs):
                results[i] = v

        if batches:
            run_parallel(_do, batches, max_workers=self.max_workers, progress=progress)

        mat = np.array([results[i] for i in range(len(texts))], dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            mat = mat / norms
        return mat


class MockEmbedder:
    """离线 mock：字符 n-gram 哈希袋 + L2 归一化，确定性、共享词即可相似。"""

    def __init__(self, dim: int = 256, normalize: bool = True, **kwargs):
        self.dim = dim
        self.normalize = normalize

    def _vec(self, text: str) -> list:
        text = str(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = []
        # 中文字符 bigram
        for m in __import__("re").finditer(r"[\u4e00-\u9fff]{2,}", text):
            s = m.group()
            tokens += [s[i : i + 2] for i in range(len(s) - 1)]
        # 英文/数字词
        for m in __import__("re").finditer(r"[A-Za-z0-9]+", text):
            tokens.append(m.group().lower())
        for tok in set(tokens):
            h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest()[:8], 16)
            vec[h % self.dim] += 1.0
        if self.normalize:
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec /= n
        return vec.tolist()

    def embed(self, text: str) -> List[float]:
        return self._vec(text)

    def embed_batch(self, texts: List[str], progress: bool = False) -> np.ndarray:
        return np.array([self._vec(t) for t in texts], dtype=np.float32)


def build_embedder(cfg: Config) -> Union[Embedder, MockEmbedder]:
    """根据配置构造 embedding 客户端（provider: openai | mock）。"""
    provider = str(cfg.get("embedding.provider", "openai")).lower()
    if provider == "mock":
        return MockEmbedder(dim=int(cfg.get("embedding.dim", 256)) or 256)
    cache_file = None
    if cfg.get("embedding.cache_enable", False):
        cache_file = Path(cfg.get("embedding.cache_file") or "data/processed/embedding_cache.jsonl")
        if not cache_file.is_absolute():
            cache_file = cfg.root / cache_file
    return Embedder(
        model=cfg.require("embedding.model"),
        base_url=cfg.require("embedding.base_url"),
        api_key=cfg.get("embedding.api_key", "dummy"),
        dim=int(cfg.require("embedding.dim")),
        normalize=bool(cfg.get("embedding.normalize", True)),
        batch_size=int(cfg.get("embedding.batch_size", 32)),
        max_workers=int(cfg.get("embedding.max_workers", 8)),
        max_retries=int(cfg.get("embedding.max_retries", 3)),
        retry_backoff=float(cfg.get("embedding.retry_backoff", 2.0)),
        timeout=int(cfg.get("embedding.timeout", 60)),
        cache_file=cache_file,
    )
