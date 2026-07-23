# -*- coding: utf-8 -*-
"""
GB/T 审查技能（Skill / Function Calling）

为 LLM 提供可调用的本地工具函数，执行确定性的规则检查，
让模型专注于理解和判断，把精确的规则比对交给代码。

用法：
    from core.skills import SKILL_SCHEMAS, execute_skill

    tools = SKILL_SCHEMAS["C2.1"]           # 获取维度对应的工具 schema
    result = execute_skill(name, args, ctx)  # 执行工具并返回 JSON 字符串
"""

from __future__ import annotations

import json
import re
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════════
# GB/T 1.1 规则知识库（复用自 counter_example.py 中的规则）
# ═══════════════════════════════════════════════════════════════════════════════

# 必备章节关键词（按类别分组）
_MANDATORY_KEYWORDS: dict[str, list[str]] = {
    "前言":         ["前言", "前  言"],
    "范围":         ["范围"],
    "规范性引用文件": ["规范性引用文件"],
    "术语和定义":    ["术语和定义", "术语"],
}

# GB/T 1.1 标准章节名称 → 常见非标准写法
_WRONG_TITLES: dict[str, list[str]] = {
    "前言":       ["序言", "说明", "编制说明", "前  言", "标准前言"],
    "引言":       ["概述", "背景", "介绍", "简介"],
    "范围":       ["适用范围", "标准范围", "应用范围", "范围与目的", "总则", "目的和范围"],
    "规范性引用文件": ["引用文件", "参考文件", "规范性文件", "引用标准", "参考标准",
                  "规范引用文献", "标准引用"],
    "术语和定义":  ["定义与术语", "定义", "术语", "术语定义", "名词解释",
                  "缩略语和术语", "定义与缩略语"],
    "参考文献":    ["参考资料", "文献参考", "引用文献", "参考标准列表"],
}

# GB/T 1.1 规定的标准章节顺序（顶级章节）
_STANDARD_ORDER = ["范围", "规范性引用文件", "术语和定义"]

# 层级深度上限
_MAX_DEPTH = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Skill 执行函数
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_sections(source_text: str) -> str:
    """从 source_text 中提取所有章节标题，返回 JSON 数组。

    匹配格式：
      [编号] 标题     —— 有编号章节（如 [5.1] 术语和定义）
      [附录A] 标题     —— 附录
      前言 / 引言 等   —— 无编号章节（按已知关键词匹配）
    """
    sections = []

    # 匹配 [编号] 标题
    for m in re.finditer(r"^\[([^\]]+)\]\s+(.+)$", source_text, re.MULTILINE):
        sections.append({"section_number": m.group(1), "title": m.group(2).strip()})

    # 匹配无编号的已知章节（独占一行且不以 [ 开头）
    known = ["前言", "前  言", "引言", "引  言", "目次", "目  次", "参考文献", "参 考 文 献"]
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped in known:
            sections.append({"section_number": None, "title": stripped.replace("  ", "")})

    return json.dumps(sections, ensure_ascii=False)


def _check_mandatory_sections(sections_json: str) -> str:
    """检查必备章节是否齐全。

    Args:
        sections_json: extract_sections 返回的 JSON 字符串
    Returns:
        JSON 格式的检查结果
    """
    sections = json.loads(sections_json)
    titles = [s["title"] for s in sections]

    missing = []
    found = []
    for category, keywords in _MANDATORY_KEYWORDS.items():
        hit = any(any(kw in t for kw in keywords) for t in titles)
        if hit:
            found.append(category)
        else:
            missing.append(category)

    return json.dumps({
        "found": found,
        "missing": missing,
        "has_issue": len(missing) > 0,
        "detail": f"缺少必备章节: {', '.join(missing)}" if missing else "所有必备章节均存在",
    }, ensure_ascii=False)


def _check_section_order(sections_json: str) -> str:
    """验证有编号的顶级章节是否按数字递增排列。

    Args:
        sections_json: extract_sections 返回的 JSON 字符串
    Returns:
        JSON 格式的检查结果
    """
    sections = json.loads(sections_json)

    # 只取有数字编号的顶级章节（编号不含"."）
    top_level = []
    for s in sections:
        num = s.get("section_number")
        if num and re.fullmatch(r"\d+", str(num)):
            top_level.append({"number": int(num), "title": s["title"]})

    issues = []
    for i in range(1, len(top_level)):
        prev, curr = top_level[i - 1], top_level[i]
        if curr["number"] <= prev["number"]:
            issues.append({
                "prev": f"{prev['number']} {prev['title']}",
                "curr": f"{curr['number']} {curr['title']}",
                "problem": f"章节 {curr['number']} 出现在 {prev['number']} 之后，顺序不正确",
            })

    # 额外检查前三章标准顺序（范围→规范性引用文件→术语和定义）
    order_map = {}
    for s in top_level:
        for i, std_name in enumerate(_STANDARD_ORDER):
            if std_name in s["title"]:
                order_map[std_name] = s["number"]

    for i in range(len(_STANDARD_ORDER) - 1):
        a, b = _STANDARD_ORDER[i], _STANDARD_ORDER[i + 1]
        if a in order_map and b in order_map and order_map[a] > order_map[b]:
            issues.append({
                "prev": f"{order_map[a]} {a}",
                "curr": f"{order_map[b]} {b}",
                "problem": f"「{a}」（第{order_map[a]}章）应排在「{b}」（第{order_map[b]}章）之前",
            })

    return json.dumps({
        "top_level_sections": top_level,
        "issues": issues,
        "has_issue": len(issues) > 0,
    }, ensure_ascii=False)


def _check_section_names(sections_json: str) -> str:
    """检查章节标题是否使用了非标准名称。

    Args:
        sections_json: extract_sections 返回的 JSON 字符串
    Returns:
        JSON 格式的检查结果
    """
    sections = json.loads(sections_json)
    issues = []

    for s in sections:
        title = s["title"]
        for std_name, wrong_list in _WRONG_TITLES.items():
            if title in wrong_list or title.strip() in wrong_list:
                issues.append({
                    "section_number": s["section_number"],
                    "current_title": title,
                    "standard_title": std_name,
                    "problem": f"章节标题「{title}」应为标准名称「{std_name}」",
                })

    return json.dumps({
        "issues": issues,
        "has_issue": len(issues) > 0,
    }, ensure_ascii=False)


def _check_appendix_types(sections_json: str) -> str:
    """核查附录的"规范性/资料性"属性标注是否合理。

    检查逻辑：
      - 附录标题中应含"规范性附录"或"资料性附录"属性标注
      - 如有标注，报告各附录的属性
      - 如缺少属性标注，标记为异常
    """
    sections = json.loads(sections_json)
    appendices = []

    for s in sections:
        title = s.get("title", "")
        sec_num = s.get("section_number", "")

        # 识别附录：编号含"附录"前缀，或标题含"附录"
        is_appendix = False
        if sec_num and ("附录" in str(sec_num)):
            is_appendix = True
        if "附录" in title and ("规范性" in title or "资料性" in title):
            is_appendix = True

        if not is_appendix:
            continue

        attr = None
        if "规范性" in title:
            attr = "规范性"
        elif "资料性" in title:
            attr = "资料性"

        appendices.append({
            "section_number": sec_num,
            "title": title,
            "attribute": attr,
            "missing_attribute": attr is None,
        })

    return json.dumps({
        "appendices": appendices,
        "total": len(appendices),
        "detail": "请结合文档正文判断各附录的属性标注是否正确（规范性附录包含要求性条款，资料性附录仅提供补充信息）",
    }, ensure_ascii=False)


def _check_section_depth(sections_json: str) -> str:
    """检查章节编号层级深度，找出超过 5 级的。

    层级计算：section_number 中的"."数 + 1
    例如 "5.1.2.3.4.5" → 6 级，超过上限。
    """
    sections = json.loads(sections_json)
    exceeded = []

    for s in sections:
        num = s.get("section_number")
        if not num or not re.match(r"[\d.]+", str(num)):
            continue
        depth = str(num).count(".") + 1
        if depth > _MAX_DEPTH:
            exceeded.append({
                "section_number": num,
                "title": s.get("title", ""),
                "depth": depth,
                "problem": f"编号「{num}」层级为 {depth} 级，超过 GB/T 1.1 规定的 {_MAX_DEPTH} 级上限",
            })

    return json.dumps({
        "exceeded": exceeded,
        "has_issue": len(exceeded) > 0,
        "max_depth_allowed": _MAX_DEPTH,
    }, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAI Tools Schema 定义
# ═══════════════════════════════════════════════════════════════════════════════

_EXTRACT_SECTIONS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "extract_sections",
        "description": "从 GB/T 标准文档正文中提取所有章节的编号和标题，返回结构化列表。建议首先调用此工具获取文档结构。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

_CHECK_MANDATORY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_mandatory_sections",
        "description": "检查文档是否包含 GB/T 1.1 规定的所有必备章节（前言、范围、规范性引用文件、术语和定义）。需要先调用 extract_sections 获取章节列表。",
        "parameters": {
            "type": "object",
            "properties": {
                "sections_json": {
                    "type": "string",
                    "description": "extract_sections 返回的 JSON 字符串",
                },
            },
            "required": ["sections_json"],
        },
    },
}

_CHECK_ORDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_section_order",
        "description": "验证文档顶级章节编号是否按递增顺序排列，以及前三章（范围→规范性引用文件→术语和定义）是否符合 GB/T 1.1 规定顺序。",
        "parameters": {
            "type": "object",
            "properties": {
                "sections_json": {
                    "type": "string",
                    "description": "extract_sections 返回的 JSON 字符串",
                },
            },
            "required": ["sections_json"],
        },
    },
}

_CHECK_NAMES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_section_names",
        "description": "检查章节标题是否使用了非标准名称（如"适用范围"应为"范围"，"定义"应为"术语和定义"等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "sections_json": {
                    "type": "string",
                    "description": "extract_sections 返回的 JSON 字符串",
                },
            },
            "required": ["sections_json"],
        },
    },
}

_CHECK_APPENDIX_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_appendix_types",
        "description": "核查文档中各附录的属性标注（规范性/资料性），检查是否存在归类错误。",
        "parameters": {
            "type": "object",
            "properties": {
                "sections_json": {
                    "type": "string",
                    "description": "extract_sections 返回的 JSON 字符串",
                },
            },
            "required": ["sections_json"],
        },
    },
}

_CHECK_DEPTH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_section_depth",
        "description": "检查章节编号的层级深度，找出超过 GB/T 1.1 规定的 5 级上限的章节。",
        "parameters": {
            "type": "object",
            "properties": {
                "sections_json": {
                    "type": "string",
                    "description": "extract_sections 返回的 JSON 字符串",
                },
            },
            "required": ["sections_json"],
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 按维度组织 Schema + 执行分发
# ═══════════════════════════════════════════════════════════════════════════════

# 维度 → 工具 Schema 列表
SKILL_SCHEMAS: dict[str, list[dict]] = {
    "C2.1": [
        _EXTRACT_SECTIONS_SCHEMA,
        _CHECK_MANDATORY_SCHEMA,
        _CHECK_ORDER_SCHEMA,
        _CHECK_NAMES_SCHEMA,
        _CHECK_APPENDIX_SCHEMA,
        _CHECK_DEPTH_SCHEMA,
    ],
    # 后续维度在此扩展：
    # "C2.2": [...],
    # "C3.1": [...],
}

# 工具名 → 执行函数映射
_SKILL_DISPATCH: dict[str, Any] = {
    "extract_sections":        lambda args, ctx: _extract_sections(ctx["source_text"]),
    "check_mandatory_sections": lambda args, ctx: _check_mandatory_sections(args["sections_json"]),
    "check_section_order":     lambda args, ctx: _check_section_order(args["sections_json"]),
    "check_section_names":     lambda args, ctx: _check_section_names(args["sections_json"]),
    "check_appendix_types":    lambda args, ctx: _check_appendix_types(args["sections_json"]),
    "check_section_depth":     lambda args, ctx: _check_section_depth(args["sections_json"]),
}


def execute_skill(name: str, arguments: dict, context: dict) -> str:
    """执行指定 skill 并返回结果字符串。

    Args:
        name: 工具名称
        arguments: 模型传入的参数（已从 JSON 解析为 dict）
        context: 运行时上下文，至少包含 {"source_text": "..."}

    Returns:
        工具执行结果的 JSON 字符串
    """
    fn = _SKILL_DISPATCH.get(name)
    if fn is None:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        return fn(arguments, context)
    except Exception as e:
        return json.dumps({"error": f"工具 {name} 执行失败: {str(e)}"}, ensure_ascii=False)
