"""统一门面：引擎注册表、路由、统一查询入口。"""

from .engine import BaseEngine, EngineRegistry, KBEngineAdapter, UnifiedEngine
from .result import UnifiedResult
from .router import Router

__all__ = ["BaseEngine", "EngineRegistry", "KBEngineAdapter", "Router", "UnifiedEngine", "UnifiedResult"]
