"""全局配置加载：yaml + 环境变量覆盖（KB_ 前缀，双下划线分层）。

优先级：环境变量 > 用户配置文件 > 内置 default.yaml > 代码默认值
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
ENV_PREFIX = "KB_"


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 优先。"""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_env_value(raw: str) -> Any:
    """尝试将环境变量字符串解析为 bool/int/float/JSON，失败则保留字符串。"""
    low = raw.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def load_env_overrides(prefix: str = ENV_PREFIX) -> dict:
    """KB_EMBEDDING__MODEL=x → {'embedding': {'model': 'x'}}"""
    overrides: dict = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        node = overrides
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _parse_env_value(value)
    return overrides


class Config:
    """点路径访问的配置对象：cfg.get('embedding.model')。"""

    def __init__(self, data: dict, root: Path = PROJECT_ROOT):
        self._data = data
        self.root = root

    # ---- 访问 ----
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"缺少配置项: {dotted}")
        return value

    def section(self, dotted: str) -> dict:
        value = self.get(dotted, {})
        return dict(value) if isinstance(value, dict) else {}

    def as_dict(self) -> dict:
        return copy.deepcopy(self._data)

    # ---- 路径解析（相对路径基于项目根目录）----
    def path(self, dotted: str, default: Optional[str] = None) -> Path:
        raw = self.get(dotted, default)
        if raw is None:
            raise KeyError(f"缺少路径配置: {dotted}")
        p = Path(str(raw))
        return p if p.is_absolute() else (self.root / p)

    # ---- 构造 ----
    @classmethod
    def load(
        cls,
        path: Optional[str] = None,
        overrides: Optional[dict] = None,
        env: bool = True,
    ) -> "Config":
        data: dict = {}
        if Path(DEFAULT_CONFIG_PATH).exists():
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        if path:
            with open(path, "r", encoding="utf-8") as f:
                data = deep_merge(data, yaml.safe_load(f) or {})
        if env:
            data = deep_merge(data, load_env_overrides())
        if overrides:
            data = deep_merge(data, overrides)
        return cls(data)

    def with_overrides(self, overrides: dict) -> "Config":
        return Config(deep_merge(self._data, overrides), root=self.root)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"
