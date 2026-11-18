"""Prompt 模板管理器：从 prompts/ 目录加载可替换模板，换领域只换模板不改代码。"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import List, Optional, Dict

from utils.config import Config

# 占位符形式：{variable}，仅替换已知变量；模板中的 JSON 示例花括号不受影响
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# 多轮标签：<system>/<user>/<assistant>（支持 HippoRAG2 式 one-shot 示例）
_TAG_RE = re.compile(r"<(system|user|assistant)>(.*?)</\1>", re.S)


class PromptManager:
    """模板加载规则：{name}_{lang}.txt → 回退 {name}.txt。

    模板文件支持 <system>...</system> 与 <user>...</user> 标签；
    无标签时整个文件视为 user 内容。
    """

    def __init__(self, prompts_dir: Path, language: str = "zh"):
        self.prompts_dir = Path(prompts_dir)
        self.language = language
        self._cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, cfg: Config) -> "PromptManager":
        return cls(
            prompts_dir=cfg.path("paths.prompts_dir", "prompts"),
            language=str(cfg.get("project.language", "chinese")).lower(),
        )

    def _lang_suffix(self) -> str:
        lang = self.language
        if lang in ("zh", "chinese", "cn"):
            return "zh"
        if lang in ("en", "english"):
            return "en"
        return lang

    def _resolve_file(self, name: str) -> Path:
        candidates = [
            self.prompts_dir / f"{name}_{self._lang_suffix()}.txt",
            self.prompts_dir / f"{name}.txt",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"未找到 prompt 模板 {name}（查找过 {[str(c) for c in candidates]}）"
        )

    def load(self, name: str) -> tuple:
        """返回 (system, user_template)。兼容旧调用；多轮模板由 load_messages 处理。"""
        blocks = self.load_messages(name)
        system = next((c for r, c in blocks if r == "system"), "")
        user = next((c for r, c in blocks if r == "user"), "")
        return system, user

    def load_messages(self, name: str) -> tuple:
        """返回 ((role, content), ...)；支持 <system>/<user>/<assistant> 多轮。

        无标签时整文件视为 user。带缓存（支持运行期热替换：clear_cache）。
        """
        with self._lock:
            if name in self._cache:
                return self._cache[name]
            path = self._resolve_file(name)
            raw = path.read_text(encoding="utf-8")
            blocks = [(m.group(1), m.group(2).strip()) for m in _TAG_RE.finditer(raw)]
            if not blocks:
                blocks = [("user", raw.strip())]
            self._cache[name] = tuple(blocks)
            return self._cache[name]

    def render(self, name: str, **kwargs) -> List[Dict[str, str]]:
        """渲染为 OpenAI messages 格式。

        自定义占位符替换（非 str.format）：仅替换 {var} 形式的已知变量，
        模板中的 JSON 示例（如 {"named_entities": [...]}）保持原样。
        """
        blocks = self.load_messages(name)

        def _sub(match: "re.Match") -> str:
            key = match.group(1)
            if key in kwargs:
                return str(kwargs[key])
            return match.group(0)  # 未知变量保留原样

        return [{"role": role, "content": _PLACEHOLDER_RE.sub(_sub, content)}
                for role, content in blocks]

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()
