"""文本 / JSON 清洗工具：LLM 输出解析、实体名归一化、代词处理。"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional

# 思考块（deepseek/qwen 等模型的<think>输出）
_THINK_RE = re.compile(r"<[!\[]?think[>\]]?[\s\S]*?<[/!]think[>\]]?", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """去掉 <think>...</think> 块。"""
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> Optional[Any]:
    """从 LLM 输出中稳健解析 JSON。

    依次尝试：直接解析 → 剥离 markdown 代码块 → 首尾大括号/方括号截取。
    """
    if not text:
        return None
    text = strip_think(text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except (json.JSONDecodeError, ValueError):
            pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def as_triple_list(obj: Any) -> List[List[str]]:
    """从任意 JSON 结构中提取 [[s,p,o],...]（兼容 {"triples":...}/{"fact":...}/裸列表）。"""
    if obj is None:
        return []
    if isinstance(obj, dict):
        for key in ("triples", "fact", "facts", "data", "relations"):
            if key in obj:
                return as_triple_list(obj[key])
        # 兼容 {"head":..,"relation":..,"tail":..} 单条结构
        if all(k in obj for k in ("head", "relation", "tail")):
            return [[str(obj["head"]), str(obj["relation"]), str(obj["tail"])]]
        return []
    if isinstance(obj, list):
        out: List[List[str]] = []
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                out.append([str(x).strip() for x in item[:3]])
            elif isinstance(item, dict) and all(k in item for k in ("head", "relation", "tail")):
                out.append([str(item["head"]), str(item["relation"]), str(item["tail"])])
        return out
    return []


def as_entity_list(obj: Any) -> List[str]:
    """从 NER JSON 输出中提取实体列表（兼容 {"named_entities":[...]}/裸列表）。"""
    if obj is None:
        return []
    if isinstance(obj, dict):
        for key in ("named_entities", "entities", "entity", "names"):
            if key in obj and isinstance(obj[key], list):
                return [str(x).strip() for x in obj[key] if str(x).strip()]
        return []
    if isinstance(obj, list):
        return [str(x).strip() for x in obj if str(x).strip()]
    return []


_WS_RE = re.compile(r"\s+")


def normalize_name(name: str, case_insensitive: bool = False) -> str:
    """实体名归一化：去首尾空白、折叠连续空白；可选英文小写。"""
    name = _WS_RE.sub(" ", str(name)).strip()
    return name.lower() if case_insensitive else name


# 常见代词（zh/en），用于三元组的代词解析
PRONOUNS_ZH = {"他", "她", "它", "他们", "她们", "它们", "该", "其", "此", "这", "那", "该公司", "该组织", "该系统"}
PRONOUNS_EN = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their", "this", "that", "the company"}
PRONOUNS_ALL = PRONOUNS_ZH | PRONOUNS_EN


def is_pronoun(name: str) -> bool:
    token = str(name).strip().lower()
    return token in PRONOUNS_ALL
