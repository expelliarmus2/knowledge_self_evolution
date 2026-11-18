"""日志与并行调度工具。"""
from __future__ import annotations

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, List, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # 无 tqdm 时优雅降级
    _tqdm = None


def setup_logging(level: str = "INFO", log_dir: Optional[Path] = None, name: str = "kb") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:  # 幂等
        return logger
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}_{int(time.time())}.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def run_parallel(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int = 8,
    desc: Optional[str] = None,
    progress: bool = True,
) -> List[R]:
    """有界线程池并行执行（吸收 huaxi_poc 的 thread_map 调度方式）。

    - 保持输入顺序返回
    - 单项失败立即抛出首个异常（保证流水线可感知错误）
    - tqdm 可选
    """
    items = list(items)
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        results = []
        it = items
        if _tqdm is not None and progress:
            it = _tqdm(items, desc=desc)
        for item in it:
            results.append(fn(item))
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        results: List[R] = []
        iterator = futures
        if _tqdm is not None and progress:
            iterator = _tqdm(futures, total=len(futures), desc=desc)
        for fut in iterator:
            results.append(fut.result())
    return results
