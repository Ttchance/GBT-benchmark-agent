# -*- coding: utf-8 -*-
"""
反例构造框架

针对 GB/T 标准文档的不同审查维度，构造对应的错误反例文档。

支持维度：
  C2.1  L2 文档级  结构审查  —— 章节结构不符合 GB/T 1.1
  C2.2  L2 文档级  范围审查  —— 正文与 scope 不一致
  C3.1  L3 条款级  语气审查  —— 应/宜/可 误用
  C3.2  L3 条款级  术语审查  —— 术语与定义不一致
  C3.3  L3 条款级  引用审查  —— 规范性引用清单不完整或冗余

用法::

    from core.llm_client import OpenAILLMClient
    from core.parse_pdf import Section
    from core.counter_example import CounterExamplePipeline

    llm = OpenAILLMClient(config)
    pipeline = CounterExamplePipeline(llm)
    result = pipeline.run(all_sections, dimensions=["C2.1", "C3.1"])
    print(result.to_text())
"""

from __future__ import annotations

import copy
import json
import logging
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from core.parse_pdf import Section

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CounterExample:
    """单条反例记录"""
    dimension: str                   # "C2.1" / "C3.1" 等
    error_type: str                  # 具体错误类型描述
    original: Section                # 原始正确章节
    corrupted: Optional[Section]     # 注入错误后的章节；None 表示该章节被整体删除
    error_description: str           # 说明此处植入了什么错误，供标注使用


@dataclass
class CounterExampleDocument:
    """完整反例文档，包含所有维度的注入结果"""
    source_sections: list[Section]
    examples: list[CounterExample] = field(default_factory=list)

    def to_corrupted_sections(self) -> list[Section]:
        """将注入的反例合并回 sections 列表，其余章节保持原样。
        corrupted 为 None 的条目表示该章节被整体删除，不出现在输出中。
        """
        corrupted_map = {id(ex.original): ex.corrupted for ex in self.examples}
        result = []
        for s in self.source_sections:
            replacement = corrupted_map.get(id(s), s)
            if replacement is None:   # None → 删除该章节
                continue
            result.append(replacement)
        return result

    def to_text(self) -> str:
        """输出反例文档全文（已注入错误）。"""
        lines = []
        for s in self.to_corrupted_sections():
            heading = f"[{s.section_number}] {s.title}" if s.section_number else s.title
            lines.append(heading)
            if s.content:
                lines.append(s.content)
            lines.append("")
        return "\n".join(lines)

    def to_annotation(self) -> str:
        """输出错误标注清单，供评估使用。"""
        lines = ["# 反例标注清单\n"]
        for i, ex in enumerate(self.examples, 1):
            lines.append(f"## [{i}] 维度: {ex.dimension}  错误类型: {ex.error_type}")
            lines.append(f"位置: [{ex.original.section_number}] {ex.original.title}")
            lines.append(f"说明: {ex.error_description}")
            lines.append("")
        return "\n".join(lines)

    def to_json(self) -> str:
        """将反例结果序列化为 JSON 字符串。"""
        data = {
            "total_errors": len(self.examples),
            "corrupted_document": [
                {
                    "section_number": s.section_number,
                    "title": s.title,
                    "content": s.content,
                }
                for s in self.to_corrupted_sections()
            ],
            "error_annotations": [
                {
                    "index": i,
                    "dimension": ex.dimension,
                    "error_type": ex.error_type,
                    "location": {
                        "section_number": ex.original.section_number,
                        "title": ex.original.title,
                    },
                    "original_content": ex.original.content,
                    "corrupted_content": ex.corrupted.content if ex.corrupted is not None else None,
                    "error_description": ex.error_description,
                }
                for i, ex in enumerate(self.examples, 1)
            ],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------

class BaseCounterExampleGenerator(ABC):
    """反例生成器抽象基类"""

    dimension: str = ""      # 如 "C2.1"
    error_types: list[str] = []  # 该维度支持的错误类型列表

    def __init__(self, llm_client=None):
        self._llm = llm_client

    def _resolve_error_type(self, error_type: Optional[str]) -> Optional[str]:
        """Return a valid error type, or None when an explicit type is unsupported."""
        if error_type is None:
            return random.choice(self.error_types)
        if error_type not in self.error_types:
            logger.warning("[%s] unsupported error_type %s, skip", self.dimension, error_type)
            return None
        return error_type

    @abstractmethod
    def generate(
        self,
        sections: list[Section],
        error_type: Optional[str] = None,
    ) -> list[CounterExample]:
        """
        对输入的 sections 生成若干条反例。

        Args:
            sections: 原始正确章节列表

        Returns:
            CounterExample 列表
        """

    def _call_llm(self, prompt: str) -> str:
        """统一 LLM 调用入口，无 LLM 时返回空字符串。"""
        if self._llm is None:
            logger.warning("[%s] 未配置 LLM，跳过生成", self.dimension)
            return ""
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._llm.chat(messages, temperature=0.9, max_tokens=2048)
        except Exception as exc:
            logger.error("[%s] LLM 调用失败: %s", self.dimension, exc)
            return ""

    @staticmethod
    def _make_corrupted(original: Section, new_content: str) -> Section:
        """保留原编号/标题，仅替换 content。"""
        return Section(
            section_number=original.section_number,
            title=original.title,
            content=new_content,
        )

    @staticmethod
    def _brief_content(content: Optional[str], limit: int = 120) -> str:
        """压缩正文摘要，避免 error_description 塞入大段原文。"""
        if content is None:
            return "（空）"
        text = re.sub(r"\s+", " ", str(content)).strip()
        if not text or text.lower() == "none":
            return "（空）"
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    @staticmethod
    def _section_label(section_number: Optional[str], title: str) -> str:
        """统一章节定位展示。"""
        return f"{section_number} {title}" if section_number else title

    @staticmethod
    def _format_description(error: str, fix: str, basis: str) -> str:
        """统一反例说明：先指出错误，再给修改建议，最后给依据。"""
        return f"错误：{error}；修改建议：{fix}；依据：{basis}"


# ---------------------------------------------------------------------------
# C2.1  结构审查反例（完成）
# ---------------------------------------------------------------------------

class StructureErrorGenerator(BaseCounterExampleGenerator):
    """
    C2.1 文档级结构审查反例。

    错误类型：
      E-S-01  missing_mandatory_section : 缺少"范围"、"规范性引用文件"等必备章节
      E-S-02  wrong_section_order       : 章节顺序不符合 GB/T 1.1 规定
      E-S-03  nonstandard_section_name  : 必备章节名称不规范（如"定义"代替"术语和定义"）
      E-S-04  foreword_introduction_mix : 前言/引言内容混淆
      E-S-05  appendix_type_error       : 附录规范性/资料性归类错误
      E-S-06  section_depth_exceeded    : 章节编号层级超过 5 级
    """

    dimension = "C2.1"
    error_types = [
        "E-S-01", "E-S-02", "E-S-03",
        "E-S-04", "E-S-05", "E-S-06",
    ]

    # GB/T 1.1 规定的必备章节及其标准名称
    _MANDATORY_NAMES = {
        "范围":        "范围",
        "规范性引用文件": "规范性引用文件",
        "术语和定义":   "术语和定义",
    }
    # 各必备章节对应的非标准替代名称（用于 E-S-03）
    _NONSTANDARD_NAMES = {
        "范围":        "适用范围",
        "规范性引用文件": "参考文件",
        "术语和定义":   "定义",
    }
    # 多余章节注入内容（用于 E-S-04）
    _REDUNDANT_SECTION = Section(
        section_number=None,
        title="背景介绍",
        content="本章为额外补充的背景说明，不属于 GB/T 1.1 规定的必备要素。",
    )

    def generate(
        self,
        sections: list[Section],
        error_type: Optional[str] = None,
    ) -> list[CounterExample]:
        error_type = self._resolve_error_type(error_type)
        if error_type is None:
            return []
        # error_type = 'E-S-02'
        dispatch = {
            "E-S-01": self._missing_mandatory,
            "E-S-02": self._wrong_order,
            "E-S-03": self._nonstandard_name,
            "E-S-04": self._foreword_intro_mix,
            "E-S-05": self._appendix_type_error,
            "E-S-06": self._depth_exceeded,
        }
        return dispatch[error_type](sections)

    @staticmethod
    def _brief_content(content: Optional[str], limit: int = 120) -> str:
        """压缩正文摘要，避免 error_description 塞入大段原文。"""
        if content is None:
            return "（空）"
        text = re.sub(r"\s+", " ", str(content)).strip()
        if not text or text.lower() == "none":
            return "（空）"
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."

    @staticmethod
    def _section_label(section_number: Optional[str], title: str) -> str:
        """统一章节定位展示。"""
        return f"{section_number} {title}" if section_number else title

    @staticmethod
    def _format_description(error: str, fix: str, basis: str) -> str:
        """统一结构审查反例说明：先指出错误，再给修改建议，最后给依据。"""
        return f"错误：{error}；修改建议：{fix}；依据：{basis}"

    # ── E-S-01：缺少必备章节 ──────────────────────────────────────

    # 必备章节关键词（按类别分组）
    _MANDATORY_KEYWORDS: dict[str, list[str]] = {
        "前言":        ["前言"],
        "规范性引用文件": ["规范性引用文件"],
        "术语和定义":   ["术语和定义", "术语"],
        "核心技术要素":  ["要求", "性能", "规范", "指标", "试验", "检验", "检测"],
    }

    def _missing_mandatory(self, sections: list[Section]) -> list[CounterExample]:
        """随机删除 1~2 个必备章节及其所有子节，模拟必备要素缺失。

        删除操作直接作用于传入的 sections 列表（原地修改）；
        title 相同视为同一章节。
        """
        print("[错误注入] E-S-01: 必备要素缺失")
        # 1. 找出文档中实际存在的各类必备章节
        found: dict[str, Section] = {}
        for category, keywords in self._MANDATORY_KEYWORDS.items():
            for s in sections:
                if any(kw in s.title for kw in keywords):
                    found[category] = s
                    break   # 每类只取第一个匹配

        if not found:
            return []

        # 2. 随机决定删除 1 或 2 个
        n = random.randint(1, min(2, len(found)))
        chosen_categories = random.sample(list(found.keys()), n)

        # 3. 收集要删除的节（父节 + 子节）并构造反例记录
        examples: list[CounterExample] = []
        to_remove: list[Section] = []      # 待从 sections 移除的对象

        for category in chosen_categories:
            parent = found[category]

            # 父节反例
            examples.append(CounterExample(
                dimension=self.dimension,
                error_type="E-S-01",
                original=parent,
                corrupted=None,
                error_description=self._format_description(
                    error=(
                        f"必备章节「{self._section_label(parent.section_number, parent.title)}」"
                        f"被整体删除，标题「{parent.title}」和正文内容均从文档中缺失；"
                        f"原正文摘要为「{self._brief_content(parent.content)}」"
                    ),
                    fix=(
                        f"恢复章节「{self._section_label(parent.section_number, parent.title)}」"
                        f"及其原有正文，并保留该章节下属子节"
                    ),
                    basis=(
                        f"GB/T 1.1 要求标准文件按规定设置并保留必要结构要素，"
                        f"缺失「{category}」会导致文档结构不完整"
                    ),
                ),
            ))
            to_remove.append(parent)

            # 子节：编号前缀匹配（section_number 以"父编号."开头）
            if parent.section_number:
                prefix = parent.section_number + "."
                for s in sections:
                    if s.section_number and s.section_number.startswith(prefix):
                        examples.append(CounterExample(
                            dimension=self.dimension,
                            error_type="E-S-01",
                            original=s,
                            corrupted=None,
                            error_description=self._format_description(
                                error=(
                                    f"子节「{self._section_label(s.section_number, s.title)}」"
                                    f"随父章节「{self._section_label(parent.section_number, parent.title)}」"
                                    f"一同被删除，标题「{s.title}」和正文内容缺失；"
                                    f"原正文摘要为「{self._brief_content(s.content)}」"
                                ),
                                fix=(
                                    f"在父章节「{self._section_label(parent.section_number, parent.title)}」"
                                    f"下恢复子节「{self._section_label(s.section_number, s.title)}」及其原有正文"
                                ),
                                basis=(
                                    "GB/T 1.1 要求章节结构保持完整，删除必备章节时连带删除其子节"
                                    "会造成该结构要素内容缺失"
                                ),
                            ),
                        ))
                        to_remove.append(s)

        # 4. 原地删除：title 相同即视为对应节
        remove_titles = {s.title for s in to_remove}
        sections[:] = [s for s in sections if s.title not in remove_titles]

        return examples

    # ── E-S-02：章节顺序错误 ──────────────────────────────────────
    def _wrong_order(self, sections: list[Section]) -> list[CounterExample]:
        """随机交换两个顶级章节（含其所有子节）的位置，并同步更新 section_number。
        直接原地修改传入的 sections 列表。
        """
        print("[错误注入] E-S-02: 章节顺序错误")
        def _has_real_body_content(section: Section) -> bool:
            """判断 section.content 是否为正文而非目录占位。"""
            if section.content is None:
                return False

            text = str(section.content).strip()
            if not text or text.lower() == "none":
                return False

            if re.fullmatch(r"[.…·•\-—\s]+", text):
                return False
            if re.fullmatch(r"[0-9IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+", text):
                return False

            cleaned = text.replace("[续]", "").strip().strip("…．。 \t\n")
            return bool(cleaned)

        def _is_top_level(section: Section) -> bool:
            return bool(section.section_number) and "." not in section.section_number

        def _collect_top_level_groups() -> list[dict]:
            """按顶级章节在 sections 中出现的位置切分连续块，避免目录/正文混组。"""
            groups = []
            i = 0
            while i < len(sections):
                current = sections[i]
                if not _is_top_level(current):
                    i += 1
                    continue

                start = i
                i += 1
                while i < len(sections) and not _is_top_level(sections[i]):
                    i += 1

                block = sections[start:i]
                groups.append({
                    "start": start,
                    "end": i,
                    "number": current.section_number,
                    "section": current,
                    "is_body": any(_has_real_body_content(s) for s in block),
                })
            return groups

        def _find_toc_group(groups: list[dict], number: str, body_start: int) -> Optional[dict]:
            """找到某个正文顶级章节在目录中的对应块。"""
            matches = [
                g for g in groups
                if g["number"] == number and not g["is_body"] and g["start"] < body_start
            ]
            return matches[-1] if matches else None

        def _renumber_range(start: int, end: int, old: str, new: str) -> None:
            for s in sections[start:end]:
                if s.section_number == old:
                    s.section_number = new
                elif s.section_number and s.section_number.startswith(old + "."):
                    s.section_number = new + s.section_number[len(old):]

        def _rebuild_with_replacements(replacements: dict[int, tuple[int, list[Section]]]) -> list[Section]:
            new_list: list[Section] = []
            i = 0
            while i < len(sections):
                if i in replacements:
                    end, block = replacements[i]
                    new_list.extend(block)
                    i = end
                else:
                    new_list.append(sections[i])
                    i += 1
            return new_list

        # 1. 先按出现位置切分顶级章节块，只从正文块中选交换对象
        groups = _collect_top_level_groups()
        body_groups = [g for g in groups if g["is_body"]]
        if len(body_groups) < 2:
            return []

        # 2. 随机挑两个不同的正文顶级章节
        a, b = random.sample(body_groups, 2)
        if a["start"] > b["start"]:
            a, b = b, a

        num_a, num_b = a["number"], b["number"]

        # 保存正文顶级章节快照，供 CounterExample 使用
        snap_a = Section(section_number=num_a, title=a["section"].title, content=a["section"].content)
        snap_b = Section(section_number=num_b, title=b["section"].title, content=b["section"].content)

        # 3. 准备正文块和对应目录块
        body_block_a = list(sections[a["start"]:a["end"]])
        body_block_b = list(sections[b["start"]:b["end"]])

        toc_a = _find_toc_group(groups, num_a, a["start"])
        toc_b = _find_toc_group(groups, num_b, b["start"])
        toc_block_a = list(sections[toc_a["start"]:toc_a["end"]]) if toc_a else []
        toc_block_b = list(sections[toc_b["start"]:toc_b["end"]]) if toc_b else []

        # 4. 同步更新目录块与正文块中的编号
        _renumber_range(a["start"], a["end"], num_a, num_b)
        _renumber_range(b["start"], b["end"], num_b, num_a)
        if toc_a and toc_b:
            _renumber_range(toc_a["start"], toc_a["end"], num_a, num_b)
            _renumber_range(toc_b["start"], toc_b["end"], num_b, num_a)

        # 5. 重建 sections：目录块只和目录块交换，正文块只和正文块交换
        replacements = {
            a["start"]: (a["end"], body_block_b),
            b["start"]: (b["end"], body_block_a),
        }
        if toc_a and toc_b:
            replacements[toc_a["start"]] = (toc_a["end"], toc_block_b)
            replacements[toc_b["start"]] = (toc_b["end"], toc_block_a)

        sections[:] = _rebuild_with_replacements(replacements)

        # 6. 构造反例记录（记录原始 → 错误状态）
        return [
            CounterExample(
                dimension=self.dimension,
                error_type="E-S-02",
                original=snap_a,
                corrupted=Section(section_number=num_b, title=snap_a.title, content=snap_a.content),
                error_description=self._format_description(
                    error=(
                        f"章节「{snap_a.title}」的标题和正文被放置到错误编号「{num_b}」"
                        f"及错误顺序位置，并与章节「{snap_b.title}」互换；"
                        f"错误位置下的正文摘要为「{self._brief_content(snap_a.content)}」，"
                        f"该章节块共 {len(body_block_a)} 节"
                    ),
                    fix=(
                        f"将章节「{snap_a.title}」恢复为编号「{num_a}」并放回原始顺序位置，"
                        f"同时恢复其下属子节的原编号前缀"
                    ),
                    basis=(
                        "GB/T 1.1 要求标准正文的章、条结构按逻辑顺序组织，"
                        "顶级章节及其子节不应随意互换位置或编号"
                    ),
                ),
            ),
            CounterExample(
                dimension=self.dimension,
                error_type="E-S-02",
                original=snap_b,
                corrupted=Section(section_number=num_a, title=snap_b.title, content=snap_b.content),
                error_description=self._format_description(
                    error=(
                        f"章节「{snap_b.title}」的标题和正文被放置到错误编号「{num_a}」"
                        f"及错误顺序位置，并与章节「{snap_a.title}」互换；"
                        f"错误位置下的正文摘要为「{self._brief_content(snap_b.content)}」，"
                        f"该章节块共 {len(body_block_b)} 节"
                    ),
                    fix=(
                        f"将章节「{snap_b.title}」恢复为编号「{num_b}」并放回原始顺序位置，"
                        f"同时恢复其下属子节的原编号前缀"
                    ),
                    basis=(
                        "GB/T 1.1 要求标准正文的章、条结构按逻辑顺序组织，"
                        "顶级章节及其子节不应随意互换位置或编号"
                    ),
                ),
            ),
        ]

    # ── E-S-03：必备章节名称不规范 ───────────────────────────────

    # 标准名称 → 错误写法列表
    _WRONG_TITLES: dict[str, list[str]] = {
        "前言":       ["序言", "说明", "编制说明", "引言", "前  言", "标准前言"],
        "引言":       ["前言", "概述", "背景", "介绍", "简介"],
        "范围":       ["适用范围", "标准范围", "应用范围", "范围与目的", "总则", "目的和范围"],
        "规范性引用文件": ["引用文件", "参考文件", "规范性文件", "引用标准", "参考标准",
                      "规范引用文献", "标准引用"],
        "术语和定义":  ["定义与术语", "定义", "术语", "术语定义", "名词解释",
                      "缩略语和术语", "定义与缩略语"],
        "附录": {      # 附录单独处理，见下方逻辑
            "规范性": ["附录X（参考性）", "附录X（强制性）", "规范性附录X", "附录X", "附件X（规范性）"],
            "资料性": ["附录X（参考性）", "附录X（信息性）", "资料性附录X", "附录X（说明性）", "附件X（资料性）"],
        },
        "参考文献":    ["参考资料", "文献参考", "引用文献", "参考标准列表", "Bibliography"],
    }

    def _nonstandard_name(self, sections: list[Section]) -> list[CounterExample]:
        """随机选一个必备章节，将其标题替换为一种非标准错误写法，并原地修改 sections。"""
        print("[错误注入] E-S-03: 非标准章节名称")
        # 1. 找出所有命中必备关键词的 section（附录单独识别）
        candidates: list[tuple[Section, str]] = []   # (section, 标准名称)

        for s in sections:
            t = s.title
            # 普通必备章节（非附录）
            for std_name in self._WRONG_TITLES:
                if std_name == "附录":
                    continue
                if std_name in t:
                    candidates.append((s, std_name))
                    break
            # 附录：标题含"附录"且含"规范性"或"资料性"
            else:
                if "附录" in t:
                    if "规范性" in t:
                        candidates.append((s, "附录__规范性"))
                    elif "资料性" in t:
                        candidates.append((s, "附录__资料性"))

        if not candidates:
            return []

        # 2. 随机选一个候选章节
        target, std_name = random.choice(candidates)

        # 3. 确定错误写法列表
        if std_name == "附录__规范性":
            wrong_list = self._WRONG_TITLES["附录"]["规范性"]
            # 将占位符 X 替换为附录实际字母
            letter = re.search(r'附录\s*([A-Z])', target.title)
            ltr = letter.group(1) if letter else "A"
            wrong_list = [w.replace("X", ltr) for w in wrong_list]
        elif std_name == "附录__资料性":
            wrong_list = self._WRONG_TITLES["附录"]["资料性"]
            letter = re.search(r'附录\s*([A-Z])', target.title)
            ltr = letter.group(1) if letter else "A"
            wrong_list = [w.replace("X", ltr) for w in wrong_list]
        else:
            wrong_list = self._WRONG_TITLES[std_name]

        wrong_title = random.choice(wrong_list)

        # 4. 原地修改 sections 中对应 section 的 title（title 相同视为同一节）
        snap_title = target.title   # 保留原始标题快照
        for s in sections:
            if s.title == target.title:
                s.title = wrong_title

        # 5. 构造反例记录
        corrupted = Section(
            section_number=target.section_number,
            title=wrong_title,
            content=target.content,
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-S-03",
            original=Section(
                section_number=target.section_number,
                title=snap_title,
                content=target.content,
            ),
            corrupted=corrupted,
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, snap_title)}」"
                    f"的标题被错误改为非标准写法「{wrong_title}」；"
                    f"错误标题为「{wrong_title}」，正文摘要为「{self._brief_content(target.content)}」"
                ),
                fix=(
                    f"将标题「{wrong_title}」恢复为标准标题「{snap_title}」，"
                    f"章节正文内容保持不变"
                ),
                basis=(
                    "GB/T 1.1 对标准文件结构要素名称有规范表述，"
                    "范围、规范性引用文件、术语和定义、附录、参考文献等标题不应使用非标准替代名称"
                ),
            ),
        )]

    # ── E-S-0000：多余章节 ─────────────────────────────────────────
    def _redundant_section(self, sections: list[Section]) -> list[CounterExample]:
        """在规范性正文区插入一个 GB/T 1.1 未规定的多余章节。"""
        # 找第一个有编号的顶级章节作为插入位置的参照
        targets = [s for s in sections if s.section_number and "." not in s.section_number]
        if not targets:
            return []
        ref = targets[0]
        redundant = Section(
            section_number="0",
            title=self._REDUNDANT_SECTION.title,
            content=self._REDUNDANT_SECTION.content,
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-S-04",
            original=ref,
            corrupted=redundant,
            error_description=self._format_description(
                error=(
                    f"在章节「{self._section_label(ref.section_number, ref.title)}」之前"
                    f"插入了多余章节「{self._section_label(redundant.section_number, redundant.title)}」；"
                    f"多余章节正文摘要为「{self._brief_content(redundant.content)}」"
                ),
                fix=(
                    f"删除多余章节「{self._section_label(redundant.section_number, redundant.title)}」，"
                    f"保持原章节「{self._section_label(ref.section_number, ref.title)}」及后续结构顺序"
                ),
                basis="GB/T 1.1 规定标准文件结构要素应按需要设置，不应在正文结构中插入无依据的额外章节",
            ),
        )]

    # ── E-S-04：前言/引言混淆 ────────────────────────────────────
    def _foreword_intro_mix(self, sections: list[Section]) -> list[CounterExample]:
        """若前言与引言均存在，交换两者的 content，并原地修改 sections。"""
        print("[错误注入] E-S-04: 前言/引言混淆")

        def _has_real_body_content(section: Optional[Section]) -> bool:
            """仅将具有实质正文的节视为可交换对象，跳过目录项或占位内容。"""
            if section is None or section.content is None:
                return False
            text = str(section.content).strip()
            if not text or text.lower() == "none":
                return False

            cleaned = text.replace("[续]", "").strip().strip("…．。 \t\n")
            return bool(cleaned)

        foreword = next(
            (
                s for s in sections
                if ("前言" in s.title or "前  言" in s.title)
                and _has_real_body_content(s)
            ),
            None,
        )
        intro = next(
            (
                s for s in sections
                if ("引言" in s.title or "引  言" in s.title)
                and _has_real_body_content(s)
            ),
            None,
        )
        if not foreword or not intro:
            return []

        # 保存原始 content 快照
        fw_content_orig = foreword.content
        intro_content_orig = intro.content

        # 原地交换 content
        foreword.content = intro_content_orig
        intro.content    = fw_content_orig

        return [
            CounterExample(
                dimension=self.dimension,
                error_type="E-S-04",
                original=Section(section_number=foreword.section_number,
                                 title=foreword.title, content=fw_content_orig),
                corrupted=Section(section_number=foreword.section_number,
                                  title=foreword.title, content=intro_content_orig),
                error_description=self._format_description(
                    error=(
                        f"章节标题为「{foreword.title}」，但正文被错误替换为「{intro.title}」的内容；"
                        f"错误正文摘要为「{self._brief_content(intro_content_orig)}」"
                    ),
                    fix=(
                        f"将章节「{foreword.title}」的正文恢复为原前言内容"
                        f"「{self._brief_content(fw_content_orig)}」，并将引言内容放回「{intro.title}」章节"
                    ),
                    basis=(
                        "GB/T 1.1 中前言和引言属于不同结构要素，前言通常说明起草依据、提出归口、"
                        "起草单位等信息，引言通常说明文件背景、目的和系列文件关系，二者内容不应混用"
                    ),
                ),
            ),
            CounterExample(
                dimension=self.dimension,
                error_type="E-S-04",
                original=Section(section_number=intro.section_number,
                                 title=intro.title, content=intro_content_orig),
                corrupted=Section(section_number=intro.section_number,
                                  title=intro.title, content=fw_content_orig),
                error_description=self._format_description(
                    error=(
                        f"章节标题为「{intro.title}」，但正文被错误替换为「{foreword.title}」的内容；"
                        f"错误正文摘要为「{self._brief_content(fw_content_orig)}」"
                    ),
                    fix=(
                        f"将章节「{intro.title}」的正文恢复为原引言内容"
                        f"「{self._brief_content(intro_content_orig)}」，并将前言内容放回「{foreword.title}」章节"
                    ),
                    basis=(
                        "GB/T 1.1 中前言和引言属于不同结构要素，前言通常说明起草依据、提出归口、"
                        "起草单位等信息，引言通常说明文件背景、目的和系列文件关系，二者内容不应混用"
                    ),
                ),
            ),
        ]

    # ── E-S-05：附录归类错误 ─────────────────────────────────────
    def _appendix_type_error(self, sections: list[Section]) -> list[CounterExample]:
        """找 title 中含"附录"且明确标注"规范性"或"资料性"的 section，
        将其属性互换（规范性 ↔ 资料性），并原地修改 sections。
        """
        print("[错误注入] E-S-05: 附录归类错误")
        # 1. 收集所有含属性标注的附录节
        candidates: list[tuple[Section, str, str]] = []   # (section, old_attr, new_attr)
        for s in sections:
            if "附录" not in s.title:
                continue
            if "规范性" in s.title and "资料性" not in s.title:
                candidates.append((s, "规范性", "资料性"))
            elif "资料性" in s.title and "规范性" not in s.title:
                candidates.append((s, "资料性", "规范性"))

        if not candidates:
            return []

        # 2. 随机选一个
        target, old_attr, new_attr = random.choice(candidates)
        snap_title = target.title

        # 3. 原地修改：替换 title 中的属性标注（title 相同视为同一节）
        wrong_title = snap_title.replace(old_attr, new_attr, 1)
        for s in sections:
            if s.title == snap_title:
                s.title = wrong_title

        # 4. 构造反例记录
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-S-05",
            original=Section(
                section_number=target.section_number,
                title=snap_title,
                content=target.content,
            ),
            corrupted=Section(
                section_number=target.section_number,
                title=wrong_title,
                content=target.content,
            ),
            error_description=self._format_description(
                error=(
                    f"附录「{target.section_number or snap_title}」的标题属性被错误改为「{new_attr}」，"
                    f"错误标题为「{wrong_title}」；正文摘要为「{self._brief_content(target.content)}」"
                ),
                fix=(
                    f"将附录标题从「{wrong_title}」恢复为「{snap_title}」，"
                    f"即把附录属性由「{new_attr}」改回「{old_attr}」"
                ),
                basis=(
                    "GB/T 1.1 要求附录明确区分规范性附录和资料性附录，"
                    "二者效力不同，不能随意互换属性"
                ),
            ),
        )]

    # ── E-S-06：层级深度超限 ─────────────────────────────────────
    def _depth_exceeded(self, sections: list[Section]) -> list[CounterExample]:
        """将某个 3/4 级编号（如 5.1.2）改为 6 级（如 5.1.2.1.1.1），超过 5 级限制。
        直接在 sections 中原地修改 section_number。
        无深层章节时，取任意有编号的 section 强制补齐至 6 级。
        """
        print("[错误注入] E-S-06: 层级深度超限")
        TARGET_DOTS = 5   # 目标层级深度（5 个"."= 6 级）

        # 1. 优先找 3 级或 4 级编号（count(".") in {2, 3}），不够则取任意有编号的节
        candidates = [
            s for s in sections
            if s.section_number and 2 <= s.section_number.count(".") <= 3
        ]
        if not candidates:
            # 降级：取任意含编号的节（包括顶级如 "5"、二级如 "5.1"）
            candidates = [s for s in sections if s.section_number]
        if not candidates:
            return []

        target = random.choice(candidates)
        orig_num = target.section_number   # 保存快照

        # 2. 计算需要追加几个 ".1" 使总深度达到 TARGET_DOTS
        current_dots = orig_num.count(".")
        extra = TARGET_DOTS - current_dots            # 需要追加的层数
        if extra <= 0:
            extra = 2   # 已经很深时也至少再追加 2 级，确保超限
        over_num = orig_num + ".1" * extra

        # 3. 原地修改 sections 中对应节的 section_number
        for s in sections:
            if s.section_number == orig_num:
                s.section_number = over_num

        # 4. 构造反例记录（original 用快照，corrupted 反映修改后状态）
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-S-06",
            original=Section(
                section_number=orig_num,
                title=target.title,
                content=target.content,
            ),
            corrupted=Section(
                section_number=over_num,
                title=target.title,
                content=target.content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{target.title}」的编号由「{orig_num}」错误改为「{over_num}」，"
                    f"层级由 {orig_num.count('.') + 1} 级变为 {over_num.count('.') + 1} 级；"
                    f"错误标题为「{target.title}」，正文摘要为「{self._brief_content(target.content)}」"
                ),
                fix=(
                    f"将章节「{target.title}」的编号恢复为「{orig_num}」，"
                    "并保持章节编号层级不超过 5 级"
                ),
                basis="GB/T 1.1 对章条编号层级有上限要求，层级过深会破坏标准结构的清晰性和可读性",
            ),
        )]


# ---------------------------------------------------------------------------
# C2.2  范围审查反例（未完成）
# ---------------------------------------------------------------------------

#TODO:完成了第一、二、五功能
class ScopeErrorGenerator(BaseCounterExampleGenerator):
    """
    C2.2 文档级范围审查反例。

    错误类型：
      E-SC-01  out_of_scope_content  : 正文超出范围界定——正文规定了范围章节未提及的内容
      E-SC-02  scope_missing_content : 正文未覆盖范围承诺——范围声明的对象在正文中无对应要求
    #   E-SC-03  scope_too_vague       : 范围表述过于宽泛——范围删去关键限定语，边界模糊
    #   E-SC-04  scope_title_mismatch  : 范围与标题不一致——在范围中植入与文件标题矛盾的描述
      E-SC-03  scope_has_requirement : 范围包含要求性条款——在范围中植入"应/宜"等规范性语气
    """

    dimension = "C2.2"
    error_types = [
        "E-SC-01", "E-SC-02", "E-SC-03",
    ]

    def generate(
        self,
        sections: list[Section],
        error_type: Optional[str] = None,
    ) -> list[CounterExample]:
        scope = self._find_scope(sections)
        if scope is None:
            logger.warning("[C2.2] 未找到「范围」章节，跳过")
            return []

        error_type = self._resolve_error_type(error_type)
        if error_type is None:
            return []
        # error_type = 'E-SC-02'
        dispatch = {
            "E-SC-01": self._out_of_scope_content,                       # (sections, scope)
            "E-SC-02": lambda sec, _: self._scope_missing_content(sec),  # 在 sections 中定位正文范围节
            # "E-SC-03": self._scope_too_vague,#未实现                           # (sections, scope)
            # "E-SC-04": self._scope_title_mismatch,#未实现                      # (sections, scope)
            "E-SC-03": lambda sec, _: self._scope_has_requirement(sec), # 在 sections 中定位正文范围节
        }
        return dispatch[error_type](sections, scope)

    # ── 辅助 ──────────────────────────────────────────────────────
    @staticmethod
    def _find_scope(sections: list[Section]) -> Optional[Section]:
        return next((s for s in sections if "范围" in s.title), None)

    @staticmethod
    def _find_title_section(sections: list[Section]) -> Optional[Section]:
        """尝试从 section_number 为 None 或 '0' 的节中找到文件标题节。"""
        for s in sections:
            if not s.section_number and ("标准" in s.title or "规范" in s.title or "规程" in s.title):
                return s
        return None

    @staticmethod
    def _normalized_title(title: Optional[str]) -> str:
        return re.sub(r"\s+", "", str(title or ""))

    @classmethod
    def _is_scope_title(cls, section: Section) -> bool:
        return cls._normalized_title(section.title) == "范围"

    @staticmethod
    def _has_real_body_content(section: Section) -> bool:
        """过滤目录项、页码和空占位内容，只保留真实正文。"""
        if section.content is None:
            return False

        text = str(section.content).strip()
        if not text or text.lower() == "none":
            return False
        if re.fullmatch(r"[.…·•\-—\s]+", text):
            return False
        if re.fullmatch(r"[.…·•\-—\s]*[0-9IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+", text):
            return False
        if re.fullmatch(r"[0-9IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+", text):
            return False

        cleaned = text.replace("[续]", "").strip().strip("…．。 \t\n")
        return bool(cleaned)

    @classmethod
    def _find_body_scope(cls, sections: list[Section]) -> Optional[Section]:
        """找到正文中的「范围」章节，跳过目录中的同名条目。"""
        return next(
            (
                section for section in sections
                if cls._is_scope_title(section) and cls._has_real_body_content(section)
            ),
            None,
        )

    # ── E-SC-01：正文超出范围界定（LLM）─────────────────────────
    def _out_of_scope_content(
        self, sections: list[Section], scope: Section
    ) -> list[CounterExample]:
        """在某正文章节末尾追加超出范围声明的内容。"""
        candidates = [
            s for s in sections
            if s.content and "范围" not in s.title and len(s.content) > 30
        ]
        if not candidates:
            return []
        target = random.choice(candidates)
        snap_content = target.content

        prompt = f"""以下是 GB/T 标准「范围」章节的声明：
{scope.content}

以下是正文中某章节的原始内容：
【{target.section_number} {target.title}】
{target.content}

请在上述正文内容的末尾追加 1~2 句话，使其包含明显超出「范围」声明限定边界的内容（如涉及范围未提及的产品类型、使用场景或技术参数）。
只输出追加后的完整正文，不要解释。"""

        new_content = self._call_llm(prompt)
        if not new_content:
            return []

        target.content = new_content   # 原地修改

        # 打印追加内容（new_content 比 snap_content 多出的部分）
        appended = new_content[len(snap_content):].strip()
        print(
            f"\n[E-SC-01] 章节「{target.title}」注入内容：\n"
            f"  原始末尾：...{snap_content[-60:].strip()}\n"
            f"  追加内容：{appended}\n"
        )

        return [CounterExample(
            dimension=self.dimension,
            error_type="E-SC-01",
            original=Section(section_number=target.section_number,
                             title=target.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=target.section_number,
                        title=target.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, target.title)}」末尾"
                    f"被注入了超出「范围」声明限定边界的内容；注入内容为「{self._brief_content(appended)}」，"
                    f"错误正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"删除该超出范围的追加内容，或先在「范围」章节中明确扩展适用边界，"
                    f"并保证正文内容与范围声明一致"
                ),
                basis="GB/T 1.1 要求正文技术内容应与「范围」规定的适用对象和边界相对应，不应出现范围外要求",
            ),
        )]

    # ── E-SC-02：正文未覆盖范围承诺（LLM）──────────────────────
    def _scope_missing_content(self, sections: list[Section]) -> list[CounterExample]:
        """在正文「范围」章节追加一个正文未覆盖的适用对象声明。"""
        scope = self._find_body_scope(sections)
        if scope is None:
            logger.warning("[C2.2] 未找到 title 为「范围」且 content 为正文的章节，跳过 E-SC-02")
            return []

        snap_content = scope.content

        excluded_titles = {
            "范围",
            "前言",
            "引言",
            "目录",
            "目次",
            "规范性引用文件",
            "术语和定义",
            "参考文献",
            "索引",
        }

        def _is_technical_body_section(section: Section) -> bool:
            if section is scope:
                return False
            if not section.section_number or not re.match(r"\d", str(section.section_number)):
                return False
            if not self._has_real_body_content(section):
                return False

            title = self._normalized_title(section.title)
            return title not in excluded_titles

        technical_sections = [s for s in sections if _is_technical_body_section(s)]
        if not technical_sections:
            logger.warning("[C2.2] 未提取到正文技术内容摘要，跳过 E-SC-02")
            return []

        summary_parts: list[str] = []
        summary_chars = 0
        for section in technical_sections:
            item = (
                f"【{self._section_label(section.section_number, section.title)}】"
                f"{self._brief_content(section.content, 220)}"
            )
            if summary_chars + len(item) > 6000:
                break
            summary_parts.append(item)
            summary_chars += len(item)

        body_summary = "\n".join(summary_parts)

        prompt = f"""以下是 GB/T 标准「范围」章节的原始内容：
{snap_content}

以下是正文技术内容摘要（已排除「范围」、目录、前言、引言等非技术正文）：
{body_summary}

请基于上述范围和正文摘要，生成一句新增的适用范围声明。
要求：
1. 新增声明应看起来与本标准领域相关；
2. 新增声明应声明本文件适用于某类产品、对象或使用场景；
3. 该产品、对象或使用场景不能在正文摘要中已有对应技术要求、试验方法或判定规则；
4. 只输出这一句新增声明，不要输出原范围正文，不要解释；
5. 不要使用“应”“宜”“可”“必须”等要求性或建议性助动词，只作适用范围陈述。"""

        scope_sentence = self._call_llm(prompt)
        if not scope_sentence:
            return []

        scope_sentence = scope_sentence.strip().strip("`").strip()
        scope_sentence = re.sub(r"^```(?:json|text)?", "", scope_sentence, flags=re.IGNORECASE).strip()
        scope_sentence = re.sub(r"```$", "", scope_sentence).strip()
        scope_sentence = next((line.strip() for line in scope_sentence.splitlines() if line.strip()), "")
        scope_sentence = re.sub(r"^(新增适用对象|适用对象|新增范围声明|范围声明)[:：]\s*", "", scope_sentence).strip()
        scope_sentence = scope_sentence.strip("\"'“”")

        if not scope_sentence:
            return []
        if "适用" not in scope_sentence:
            scope_sentence = f"本文件也适用于{scope_sentence.rstrip('。；;')}。"
        elif not scope_sentence.endswith(("。", "；", ";")):
            scope_sentence = scope_sentence + "。"
        if scope_sentence in snap_content:
            logger.warning("[C2.2] LLM 生成的新增范围声明已存在于原范围中，跳过 E-SC-02")
            return []

        new_content = snap_content.rstrip() + "\n" + scope_sentence

        # 只修改「范围」章节，正文不补充对应技术要求。
        scope.content = new_content

        print(
            f"\n[E-SC-02] 「范围」章节追加了未覆盖的承诺内容：\n"
            f"  原始末尾：...{snap_content[-60:].strip()}\n"
            f"  追加内容：{scope_sentence}\n"
        )

        return [CounterExample(
            dimension=self.dimension,
            error_type="E-SC-02",
            original=Section(section_number=scope.section_number,
                             title=scope.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=scope.section_number,
                        title=scope.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(scope.section_number, scope.title)}」"
                    f"的标题为「{scope.title}」，但正文 content 中追加了正文未覆盖的适用对象或使用场景声明；"
                    f"错误内容为「{self._brief_content(scope_sentence)}」，"
                    f"正文技术内容中未设置该对象或场景对应的技术要求、试验方法或判定规则；"
                    f"错误范围正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"删除该无正文支撑的范围承诺，恢复原范围表述「{self._brief_content(snap_content)}」；"
                    f"若确需保留该适用对象或场景，应在正文中补充对应的技术要求、试验方法或判定规则"
                ),
                basis=(
                    "GB/T 1.1 要求「范围」章节给出的适用对象、界限和用途"
                    "应与正文实际规定内容相匹配，不能作出正文未覆盖的范围承诺"
                ),
            ),
        )]

    # ── E-SC-00000：范围表述过于宽泛（规则）──────────────────────
#     def _scope_too_vague(
#         self, sections: list[Section], scope: Section
#     ) -> list[CounterExample]:
#         """删除范围 content 中的关键限定短语，使边界模糊。"""
#         # 常见限定短语模式：数字范围、特定用途、特定产品类型等
#         _LIMIT_PATTERNS = [
#             re.compile(r'[，,]?适用于[^，。；\n]{2,20}'),
#             re.compile(r'[，,]?不适用于[^，。；\n]{2,20}'),
#             re.compile(r'\d+\s*[kKmMgG]?[VvAaWwHz]\b[^，。；\n]{0,10}'),
#             re.compile(r'额定[^，。；\n]{2,15}'),
#             re.compile(r'[（(][^（(）)\n]{2,20}[）)]'),   # 括号内限定
#         ]
#         snap_content = scope.content
#         new_content = scope.content
#         removed_any = False

#         for pat in _LIMIT_PATTERNS:
#             reduced = pat.sub("", new_content, count=1)
#             if reduced != new_content:
#                 new_content = reduced
#                 removed_any = True
#                 break   # 每次只删一处，保持可读性

#         if not removed_any or new_content.strip() == snap_content.strip():
#             # 降级：直接用 LLM 改写
#             prompt = f"""以下是 GB/T 标准「范围」章节的原始内容：
# {scope.content}

# 请改写该范围，删除或模糊其中的关键限定语（如具体产品类型、适用场合、参数范围等），使范围表述变得宽泛、边界不清晰。
# 只输出改写后的范围正文，不要解释。"""
#             new_content = self._call_llm(prompt)
#             if not new_content:
#                 return []

#         scope.content = new_content
#         if removed_any:
#             print(
#                 f"\n[E-SC-03] 「范围」章节删除了限定短语（规则）：\n"
#                 f"  修改前：{snap_content.strip()[:120]}\n"
#                 f"  修改后：{new_content.strip()[:120]}\n"
#             )
#         else:
#             print(
#                 f"\n[E-SC-03] 「范围」章节经 LLM 改写（表述宽泛化）：\n"
#                 f"  修改前：{snap_content.strip()[:120]}\n"
#                 f"  修改后：{new_content.strip()[:120]}\n"
#             )
#         return [CounterExample(
#             dimension=self.dimension,
#             error_type="E-SC-03",
#             original=Section(section_number=scope.section_number,
#                              title=scope.title, content=snap_content),
#             corrupted=self._make_corrupted(
#                 Section(section_number=scope.section_number,
#                         title=scope.title, content=snap_content),
#                 new_content,
#             ),
#             error_description=self._format_description(
#                 error=(
#                     f"「范围」章节的关键限定语被删除或模糊化，导致适用对象、参数范围或使用场景边界不清晰；"
#                     f"错误范围正文摘要为「{self._brief_content(new_content)}」"
#                 ),
#                 fix=(
#                     f"恢复原范围中的关键限定条件，使范围表述回到「{self._brief_content(snap_content)}」"
#                 ),
#                 basis="GB/T 1.1 要求「范围」应清楚界定文件的适用对象和边界，不能使用过宽泛或边界不明的表述",
#             ),
#         )]

    # ── E-SC-00000：范围与标题不一致（LLM）────────────────────────
#     def _scope_title_mismatch(
#         self, sections: list[Section], scope: Section
#     ) -> list[CounterExample]:
#         """在范围内容中植入与文件标题暗示的对象相矛盾的描述。"""
#         # 尝试从 ParsedDocument.title 推断，退而求其次用文件 section_number=None 的节
#         doc_title = next(
#             (s.content[:60] for s in sections if not s.section_number and s.content),
#             "（文件标题未知）",
#         )
#         snap_content = scope.content

#         prompt = f"""以下是 GB/T 标准的文件标题或封面摘要：
# {doc_title}

# 以下是该标准「范围」章节的原始内容：
# {scope.content}

# 请修改范围内容，使其中关于适用对象或用途的描述与上述文件标题产生明显偏差（例如：将适用对象替换为完全不同的产品领域，或声明不适用于文件标题所指向的对象）。
# 只输出修改后的范围正文，不要解释。"""

#         new_content = self._call_llm(prompt)
#         if not new_content:
#             return []

#         scope.content = new_content
#         print(
#             f"\n[E-SC-04] 「范围」章节与标题不一致（LLM 改写）：\n"
#             f"  文件标题：{doc_title.strip()[:80]}\n"
#             f"  修改前：  {snap_content.strip()[:120]}\n"
#             f"  修改后：  {new_content.strip()[:120]}\n"
#         )
#         return [CounterExample(
#             dimension=self.dimension,
#             error_type="E-SC-04",
#             original=Section(section_number=scope.section_number,
#                              title=scope.title, content=snap_content),
#             corrupted=self._make_corrupted(
#                 Section(section_number=scope.section_number,
#                         title=scope.title, content=snap_content),
#                 new_content,
#             ),
#             error_description=self._format_description(
#                 error=(
#                     f"「范围」章节的适用对象或用途被改写为与文件标题/封面摘要不一致；"
#                     f"文件标题或封面摘要为「{self._brief_content(doc_title)}」，"
#                     f"错误范围正文摘要为「{self._brief_content(new_content)}」"
#                 ),
#                 fix=(
#                     f"将「范围」内容恢复为与文件标题一致的适用对象和用途描述，"
#                     f"可恢复为原范围正文「{self._brief_content(snap_content)}」"
#                 ),
#                 basis="GB/T 1.1 要求文件名称、范围和正文技术内容保持一致，范围不应偏离标题所指向的标准化对象",
#             ),
#         )]

    # ── E-SC-03：范围包含要求性条款（LLM）────────────────────
    def _scope_has_requirement(
        self, sections: list[Section]
    ) -> list[CounterExample]:
        """用 LLM 在 title 为「范围」且 content 为正文的 section 中注入要求性表述。"""
        scope = self._find_body_scope(sections)
        if scope is None:
            logger.warning("[C2.2] 未找到 title 为「范围」且 content 为正文的章节，跳过 E-SC-05")
            return []

        snap_content = scope.content

        prompt = f"""以下是 GB/T 标准「范围」章节的原始正文：
{snap_content}

请在不改变原有范围基本含义的前提下，改写或追加一句话，使「范围」章节中混入明显的要求性或建议性表述。
要求：
1. 新增或改写的句子必须包含“应”或“宜”等要求性/建议性用语；
2. 该句应自然地出现在范围正文中，但其内容应属于具体要求、建议、试验规定或执行要求；
3. 保留原有范围正文的主要信息；
4. 只输出修改后的完整范围正文，不要解释。"""

        new_content = self._call_llm(prompt)
        if not new_content:
            return []

        new_content = new_content.strip()
        if new_content == snap_content.strip():
            logger.warning("[C2.2] LLM 未修改「范围」正文，跳过 E-SC-05")
            return []

        original_modal_count = len(re.findall(r"[应宜]", snap_content))
        new_modal_count = len(re.findall(r"[应宜]", new_content))
        if new_modal_count <= original_modal_count:
            logger.warning("[C2.2] LLM 未新增应/宜等要求性表述，跳过 E-SC-05")
            return []

        appended = new_content[len(snap_content):].strip() if new_content.startswith(snap_content) else ""
        injected_detail = appended or new_content

        # 原地修改 sections 中的范围正文节。
        scope.content = new_content
        print(
            f"\n[E-SC-03] 「范围」章节经 LLM 注入要求性表述：\n"
            f"  注入内容：{self._brief_content(injected_detail)}\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-SC-03",
            original=Section(section_number=scope.section_number,
                             title=scope.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=scope.section_number,
                        title=scope.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(scope.section_number, scope.title)}」"
                    f"的标题为「{scope.title}」，但正文 content 中混入了“应”“宜”等要求性或建议性表述；"
                    f"错误内容为「{self._brief_content(injected_detail)}」，"
                    f"错误范围正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"从「范围」章节删除该要求性或建议性表述，恢复为只陈述文件适用对象和边界的范围正文；"
                    f"如该要求确有必要，应移至正文相应技术要求、试验方法或执行要求章节"
                ),
                basis=(
                    "GB/T 1.1 要求「范围」章节用于说明文件的适用对象、界限和用途，"
                    "不应写入具体技术要求、建议性要求、试验规定或执行要求"
                ),
            ),
        )]


# ---------------------------------------------------------------------------
# C3.1  语气审查反例（完成）
# ---------------------------------------------------------------------------

class ModalErrorGenerator(BaseCounterExampleGenerator):
    """
    C3.1 条款级语气审查反例。

    错误类型：
      E-T-01  宜替换应（降强度）    : 强制要求误用推荐语气，降低条款约束力
      E-T-02  应替换宜（升强度）    : 推荐性建议误用强制语气，过度约束
      E-T-03  可替换应（降强度）    : 强制要求误用许可语气，条款失去约束力
      E-T-04  混用应与须           : 使用"须"代替"应"，GB/T 1.1 不采用"须"
      E-T-05  使用模糊语气词        : "尽量""最好""一般"等替代规定助动词
      E-T-06  不宜替换不应（降强度）: 禁止性条款误用不推荐语气
      E-T-07  助动词与条款内容矛盾  : 明显强制安全要求却使用"宜"或"可"（LLM）
    """

    dimension = "C3.1"
    error_types = [
        "E-T-01", "E-T-02", "E-T-03",
        "E-T-04", "E-T-05", "E-T-06", "E-T-07",
    ]

    def generate(
        self,
        sections: list[Section],
        error_type: Optional[str] = None,
    ) -> list[CounterExample]:
        #找候选 sections → 随机选一节 → 修改内容 → 原地写回 → 打印 → 返回 CounterExample
        error_type = self._resolve_error_type(error_type)
        if error_type is None:
            return []
        # error_type = 'E-T-02'
        dispatch = {
            "E-T-01": self._shall_to_should,
            "E-T-02": self._should_to_shall,
            "E-T-03": self._shall_to_may,
            "E-T-04": self._shall_to_must,
            "E-T-05": self._vague_modal,
            "E-T-06": self._not_shall_to_not_should,
            "E-T-07": self._modal_content_contradiction,
        }
        return dispatch[error_type](sections)

    # ── 内部辅助 ──────────────────────────────────────────────────

    @staticmethod
    def _replace_first(content: str, src: str, dst: str) -> tuple[str, int]:
        """将 content 中第一处 src 替换为 dst。
        返回 (新内容, 替换起始位置)；未找到时返回 (原内容, -1)。
        """
        idx = content.find(src)
        if idx == -1:
            return content, -1
        return content[:idx] + dst + content[idx + len(src):], idx

    def _replace_modal_llm(
        self, content: str, src: str, dst: str
    ) -> tuple[str, int]:
        """用 LLM 判断 content 中是否存在作为助动词的 src，若存在则替换第一处。
        返回 (新内容, 替换起始位置idx)；LLM 判定非助动词或调用失败时返回 (原内容, -1)。

        工作流：
          1. 截取前 600 字发给 LLM，要求判断并替换
          2. LLM 返回 NO_CHANGE → 认定无助动词用法，返回 (原内容, -1)
          3. LLM 返回改写后文字 → 与原文逐字比对找差异位置，合并回完整 content
          4. LLM 不可用 → 降级调用 _replace_first（规则兜底）
        """
        if src not in content:
            return content, -1

        # LLM 不可用时直接降级
        if self._llm is None:
            return self._replace_first(content, src, dst)

        excerpt = content[:600]
        modal_example = f"产品{src}符合…/各方{src}按照…"
        non_modal_example = f"对{src}/相{src}/适{src}"
        prompt = (
            f"以下是 GB/T 标准文档的部分正文：\n"
            f"---\n{excerpt}\n---\n\n"
            f"请判断其中「{src}」是否存在作为助动词的用法"
            f"（即表示强制、推荐或许可语气，例如：{modal_example}，"
            f"而非复合词 {non_modal_example} 中的 {src}）。\n\n"
            f"若存在，将第一处助动词「{src}」替换为「{dst}」，"
            f"输出替换后的完整文字（只替换一处，其余不变）。\n"
            f"若不存在助动词用法的「{src}」，只输出：NO_CHANGE\n\n"
            f"只输出结果文字或 NO_CHANGE，不要任何解释。"
        )

        result = (self._call_llm(prompt) or "").strip()
        if not result or result == "NO_CHANGE":
            return content, -1

        # 找第一个差异位置
        new_excerpt = result
        idx = -1
        for i in range(min(len(excerpt), len(new_excerpt))):
            if excerpt[i] != new_excerpt[i]:
                idx = i
                break

        new_content = new_excerpt + content[len(excerpt):]
        return new_content, idx if idx != -1 else 0

    def _simple_replace(
        self,
        sections: list[Section],
        src: str,
        dst: str,
        error_type: str,
        basis: str,
    ) -> list[CounterExample]:
        """通用单词替换框架：找候选节 → LLM 确认助动词 → 原地修改 → 打印 → 返回反例。
        对每个候选最多尝试 3 次，避免全部候选都是非助动词用法时无限重试。
        """
        candidates = [s for s in sections if s.content and src in s.content]
        if not candidates:
            return []

        # 随机打乱，逐个让 LLM 确认是否含助动词用法，最多尝试 3 个候选
        sample = random.sample(candidates, min(3, len(candidates)))
        target = None
        new_content = ""
        match_idx = -1

        for candidate in sample:
            nc, idx = self._replace_modal_llm(candidate.content, src, dst)
            if idx != -1:
                target = candidate
                new_content = nc
                match_idx = idx
                break

        if target is None:
            return []

        snap_content = target.content
        target.content = new_content   # 原地修改

        # 打印替换位置上下文（前后各 15 字）
        ctx_s = max(0, match_idx - 15)
        ctx_e = min(len(snap_content), match_idx + len(src) + 15)
        print(
            f"\n[{error_type}] 章节「{target.title}」助动词替换（LLM 确认）：\n"
            f"  替换：「{src}」→「{dst}」\n"
            f"  原文片段：...{snap_content[ctx_s:ctx_e]}...\n"
            f"  修改片段：...{new_content[ctx_s:ctx_e]}...\n"
        )

        return [CounterExample(
            dimension=self.dimension,
            error_type=error_type,
            original=Section(section_number=target.section_number,
                             title=target.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=target.section_number,
                        title=target.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, target.title)}」的正文中，"
                    f"助动词「{src}」被错误替换为「{dst}」；错误正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"将该处助动词「{dst}」恢复为「{src}」，"
                    f"原正文摘要为「{self._brief_content(snap_content)}」"
                ),
                basis=basis,
            ),
        )]

    # ── E-T-01：「宜」替换「应」（降强度）──────────────────────
    def _shall_to_should(self, sections: list[Section]) -> list[CounterExample]:
        """强制要求「应」误用推荐语气「宜」，降低条款约束力。"""
        return self._simple_replace(
            sections, "应", "宜", "E-T-01",
            "GB/T 1.1 中「应」用于表达要求，改为「宜」会把强制要求降为推荐性表述，降低条款约束力",
        )

    # ── E-T-02：「应」替换「宜」（升强度）──────────────────────
    def _should_to_shall(self, sections: list[Section]) -> list[CounterExample]:
        """推荐性建议「宜」误用强制语气「应」，过度约束。"""
        return self._simple_replace(
            sections, "宜", "应", "E-T-02",
            "GB/T 1.1 中「宜」用于表达推荐，改为「应」会把建议性内容提升为强制要求，造成过度约束",
        )

    # ── E-T-03：「可」替换「应」（降强度）──────────────────────
    def _shall_to_may(self, sections: list[Section]) -> list[CounterExample]:
        """强制要求「应」误用许可语气「可」，条款失去约束力。"""
        return self._simple_replace(
            sections, "应", "可", "E-T-03",
            "GB/T 1.1 中「应」用于表达要求，改为「可」会把强制要求降为许可性表述，使条款失去约束力",
        )

    # ── E-T-04：混用「应」与「须」────────────────────────────
    def _shall_to_must(self, sections: list[Section]) -> list[CounterExample]:
        """将「应」替换为「须」；GB/T 1.1 不采用「须」作为要求性助动词。"""
        return self._simple_replace(
            sections, "应", "须", "E-T-04",
            "GB/T 1.1 使用「应」表达要求，不采用「须」作为规范性要求助动词",
        )

    # ── E-T-05：使用模糊语气词 ───────────────────────────────
    _VAGUE_REPLACEMENTS: list[tuple[str, str]] = [
        ("应", "尽量"),
        ("应", "最好"),
        ("应", "一般"),
        ("宜", "尽量"),
        ("宜", "最好"),
        ("可", "一般"),
    ]

    def _vague_modal(self, sections: list[Section]) -> list[CounterExample]:
        """用「尽量」「最好」「一般」等模糊词替代规定助动词。"""
        valid: list[tuple[Section, str, str]] = []
        for src, dst in self._VAGUE_REPLACEMENTS:
            for s in sections:
                if s.content and src in s.content:
                    valid.append((s, src, dst))
        if not valid:
            return []

        target, src, dst = random.choice(valid)
        snap_content = target.content
        new_content, idx = self._replace_first(snap_content, src, dst)
        if idx == -1:
            return []

        target.content = new_content   # 原地修改

        ctx_s = max(0, idx - 15)
        ctx_e = min(len(snap_content), idx + len(src) + 15)
        print(
            f"\n[E-T-05] 章节「{target.title}」规定助动词被替换为模糊语气词：\n"
            f"  替换：「{src}」→「{dst}」\n"
            f"  原文片段：...{snap_content[ctx_s:ctx_e]}...\n"
            f"  修改片段：...{new_content[ctx_s:ctx_e]}...\n"
        )

        return [CounterExample(
            dimension=self.dimension,
            error_type="E-T-05",
            original=Section(section_number=target.section_number,
                             title=target.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=target.section_number,
                        title=target.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, target.title)}」的正文中，"
                    f"规定助动词「{src}」被替换为模糊语气词「{dst}」；"
                    f"错误正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"将模糊语气词「{dst}」恢复为规范助动词「{src}」，"
                    f"原正文摘要为「{self._brief_content(snap_content)}」"
                ),
                basis="GB/T 1.1 要求规范性条款使用规定的助动词表达要求、推荐或许可，不应使用「尽量」「最好」「一般」等模糊语气",
            ),
        )]

    # ── E-T-06：「不宜」替换「不应」（降强度）────────────────
    def _not_shall_to_not_should(self, sections: list[Section]) -> list[CounterExample]:
        """禁止性条款「不应」误用不推荐语气「不宜」，降低禁止强度。"""
        return self._simple_replace(
            sections, "不应", "不宜", "E-T-06",
            "GB/T 1.1 中「不应」用于表达禁止性要求，改为「不宜」会降低禁止强度，使禁止要求变成不推荐表述",
        )

    # ── E-T-07：助动词与条款内容矛盾（LLM）──────────────────
    _SAFETY_KEYWORDS = ["安全", "防护", "禁止", "危险", "保护", "故障", "泄漏", "爆炸"]

    def _modal_content_contradiction(self, sections: list[Section]) -> list[CounterExample]:
        """在含安全/强制要求的条款中将「应」改为「宜」或「可」，使语气与内容矛盾。"""
        # 优先选含安全关键词且含「应」的节
        candidates = [
            s for s in sections
            if s.content and "应" in s.content
            and any(kw in s.content for kw in self._SAFETY_KEYWORDS)
        ]
        # 降级：任意含「应」的节
        if not candidates:
            candidates = [s for s in sections if s.content and "应" in s.content]
        if not candidates:
            return []

        target = random.choice(candidates)
        snap_content = target.content

        prompt = f"""以下是 GB/T 标准某章节的内容：
【{target.section_number} {target.title}】
{target.content}

该章节包含强制性安全要求或技术规定。请将其中一处使用「应」的强制要求语句改为使用「宜」或「可」，使助动词的语气强度与条款内容（明显应当强制）产生矛盾。
只输出修改后的完整章节正文，不要解释。"""

        new_content = self._call_llm(prompt)
        if not new_content:
            # 降级：直接规则替换
            new_content, _idx = self._replace_first(snap_content, "应", "宜")
            if _idx == -1:
                return []

        target.content = new_content   # 原地修改
        print(
            f"\n[E-T-07] 章节「{target.title}」助动词与条款内容产生矛盾：\n"
            f"  修改前：{snap_content.strip()[:120]}\n"
            f"  修改后：{new_content.strip()[:120]}\n"
        )

        return [CounterExample(
            dimension=self.dimension,
            error_type="E-T-07",
            original=Section(section_number=target.section_number,
                             title=target.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=target.section_number,
                        title=target.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, target.title)}」中明显属于强制安全要求"
                    f"或技术规定的内容被改用「宜」或「可」；错误正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"将安全或强制性技术要求恢复为与条款性质一致的要求性助动词，"
                    f"原正文摘要为「{self._brief_content(snap_content)}」"
                ),
                basis="GB/T 1.1 要求根据条款性质准确选用助动词，强制性安全要求不应使用推荐或许可语气",
            ),
        )]


# ---------------------------------------------------------------------------
# C3.2  术语审查反例(完成)
# ---------------------------------------------------------------------------

class TerminologyErrorGenerator(BaseCounterExampleGenerator):
    """
    C3.2 条款级术语审查反例。

    错误类型：
      E-TM-01  使用未定义术语     : LLM 生成新术语插入正文，但「术语和定义」章节无对应定义
      E-TM-02  术语未在正文使用   : LLM 生成新术语条目插入「术语和定义」，但正文中从不出现
      E-TM-03  术语与定义不一致   : 正文中该术语出现处被 LLM 替换为含义偏移的表达
      E-TM-04  同一概念多术语     : LLM 在不同正文章节中将同一术语替换为近义词，造成表达不统一
      E-TM-05  术语编号不连续     : 在「术语和定义」章节中删除一个编号，制造编号跳号
    """

    dimension = "C3.2"
    error_types = ["E-TM-01", "E-TM-02", "E-TM-03", "E-TM-04", "E-TM-05"]

    # 术语条目行识别：以 3.N 或 3.N.M 形式编号开头
    _TERM_ENTRY_RE = re.compile(r'^\d+\.\d+')

    def generate(
        self,
        sections: list[Section],
        error_type: Optional[str] = None,
    ) -> list[CounterExample]:
        term_section = self._find_term_section(sections)
        error_type = self._resolve_error_type(error_type)
        if error_type is None:
            return []
        # error_type = "E-TM-01"
        dispatch = {
            "E-TM-01": lambda: self._undefined_term(sections, term_section),
            "E-TM-02": lambda: self._unused_term(term_section),
            "E-TM-03": lambda: self._term_mismatch(sections, term_section),
            "E-TM-04": lambda: self._multi_term(sections, term_section),
            "E-TM-05": lambda: self._nonsequential_numbering(term_section),
        }
        return dispatch[error_type]()

    # ── 辅助方法 ──────────────────────────────────────────────

    def _find_term_section(self, sections: list[Section]) -> Optional[Section]:
        """找到 title 含「术语」且 content 有实质内容的章节。"""
        for s in sections:
            if "术语" in s.title and s.content:
                if s.content.strip().strip("…．。 \t\n"):
                    return s
        return None

    def _body_sections(
        self, sections: list[Section], term_section: Optional[Section]
    ) -> list[Section]:
        """返回 title 和 content 均非空、且非术语章节的正文节列表。"""
        return [
            s for s in sections
            if s.title and s.content and len(s.content) > 30
            and s is not term_section
        ]

    def _term_content(self, term_section: Optional[Section]) -> str:
        return term_section.content if term_section else "（无术语和定义章节）"

    # ── E-TM-01：使用未定义术语 ──────────────────────────────
    def _undefined_term(
        self, sections: list[Section], term_section: Optional[Section]
    ) -> list[CounterExample]:
        """LLM 在随机一条正文章节中插入一个在「术语和定义」里没有定义的技术术语。"""
        candidates = self._body_sections(sections, term_section)
        if not candidates:
            return []
        target = random.choice(candidates)
        snap_content = target.content

        prompt = (
            f"以下是 GB/T 标准「术语和定义」章节的内容：\n"
            f"{self._term_content(term_section)}\n\n"
            f"以下是某正文章节的内容：\n"
            f"【{target.section_number} {target.title}】\n"
            f"{target.content}\n\n"
            f"请在正文中自然地插入或替换出一个技术术语，该术语在上述「术语和定义」章节中没有任何定义。"
            f"只输出修改后的完整正文，不要解释。"
        )
        new_content = self._call_llm(prompt)
        if not new_content:
            return []

        target.content = new_content
        print(
            f"\n[E-TM-01] 章节「{target.title}」出现了未在「术语和定义」中定义的术语：\n"
            f"  修改前：{snap_content[:100].strip()}...\n"
            f"  修改后：{new_content[:100].strip()}...\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-TM-01",
            original=Section(section_number=target.section_number,
                             title=target.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=target.section_number,
                        title=target.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, target.title)}」的正文中"
                    f"出现了未在「术语和定义」章节给出定义的技术术语；"
                    f"错误正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"删除该未定义术语，或在「术语和定义」章节补充对应术语条目和定义；"
                    f"原正文摘要为「{self._brief_content(snap_content)}」"
                ),
                basis="GB/T 1.1 要求标准中使用的关键术语应在「术语和定义」中给出定义，并保持先定义后使用",
            ),
        )]

    # ── E-TM-02：术语未在正文中使用 ─────────────────────────
    def _unused_term(self, term_section: Optional[Section]) -> list[CounterExample]:
        """LLM 在「术语和定义」章节末尾新增一条术语条目，但该术语在正文中从不出现。"""
        if term_section is None:
            return []
        snap_content = term_section.content

        # 取现有术语条目确定领域和编号格式
        term_lines = [l for l in snap_content.splitlines() if self._TERM_ENTRY_RE.match(l.strip())]
        sample_terms = "\n".join(term_lines[-3:]) if term_lines else snap_content[:300]

        # 计算下一个编号
        last_num_match = None
        for l in reversed(term_lines):
            m = re.match(r'^(\d+\.\d+)', l.strip())
            if m:
                last_num_match = m.group(1)
                break
        if last_num_match:
            parts = last_num_match.split(".")
            next_num = f"{parts[0]}.{int(parts[1]) + 1}"
        else:
            next_num = "3.99"

        prompt = (
            f"以下是 GB/T 标准「术语和定义」章节中已有的部分术语条目：\n"
            f"{sample_terms}\n\n"
            f"请仿照上述格式，生成一条编号为 {next_num} 的新术语条目（含术语名称和定义），"
            f"该术语与本领域相关但在实际正文中从未被使用。"
            f"只输出该条目（编号、术语名称、定义），不要解释。"
        )
        new_entry = (self._call_llm(prompt) or "").strip()
        if not new_entry:
            return []

        new_content = snap_content.rstrip() + f"\n{new_entry}"
        term_section.content = new_content

        print(
            f"\n[E-TM-02] 「术语和定义」章节新增了一条正文中从未使用的术语：\n"
            f"  新增条目：{new_entry[:120].strip()}\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-TM-02",
            original=Section(section_number=term_section.section_number,
                             title=term_section.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=term_section.section_number,
                        title=term_section.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「术语和定义」章节新增了正文中从未使用的术语条目「{self._brief_content(new_entry, 80)}」；"
                    f"错误术语章节摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"删除该未使用的术语条目；若该术语确需保留，应在正文中使用并保证定义与用法一致"
                ),
                basis="GB/T 1.1 要求术语和定义服务于正文实际使用的概念，不应设置正文没有使用的冗余术语条目",
            ),
        )]

    # ── E-TM-03：术语与定义不一致使用 ───────────────────────
    def _term_mismatch(
        self, sections: list[Section], term_section: Optional[Section]
    ) -> list[CounterExample]:
        """从术语章节提取一个术语，在正文中找到该术语出现的章节，
        让 LLM 将该处的使用方式改写成与定义含义不符的表达。"""
        if term_section is None:
            return []

        # 从术语章节提取术语名列表
        term_lines = [l.strip() for l in term_section.content.splitlines()
                      if self._TERM_ENTRY_RE.match(l.strip())]
        if not term_lines:
            return []

        # 让 LLM 从术语条目中提取术语名
        sample = "\n".join(term_lines[:10])
        pick_prompt = (
            f"以下是「术语和定义」章节的部分条目：\n{sample}\n\n"
            f"请从中提取 1 个最具代表性的术语名称（仅输出术语词本身，不含编号和定义）。"
        )
        term_name = (self._call_llm(pick_prompt) or "").strip()
        if not term_name:
            return []

        # 找正文中含该术语的章节
        candidates = [
            s for s in self._body_sections(sections, term_section)
            if term_name in s.content
        ]
        if not candidates:
            return []

        target = random.choice(candidates)
        snap_content = target.content

        prompt = (
            f"以下是 GB/T 标准「术语和定义」章节的内容：\n"
            f"{self._term_content(term_section)}\n\n"
            f"以下是某正文章节的内容：\n"
            f"【{target.section_number} {target.title}】\n"
            f"{target.content}\n\n"
            f"术语「{term_name}」在上述正文中出现。请修改正文中该术语的使用方式，"
            f"使其描述的概念内涵与「术语和定义」中的定义不符（如扩大或缩小适用范围、混淆上下位概念等）。"
            f"只输出修改后的完整正文，不要解释。"
        )
        new_content = self._call_llm(prompt)
        if not new_content:
            return []

        target.content = new_content
        print(
            f"\n[E-TM-03] 章节「{target.title}」中术语「{term_name}」的使用与定义不一致：\n"
            f"  修改前：{snap_content[:100].strip()}...\n"
            f"  修改后：{new_content[:100].strip()}...\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-TM-03",
            original=Section(section_number=target.section_number,
                             title=target.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=target.section_number,
                        title=target.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, target.title)}」中术语「{term_name}」"
                    f"的使用方式与「术语和定义」中的定义含义不符；"
                    f"错误正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"按「术语和定义」中对「{term_name}」的定义修正文中用法，"
                    f"或恢复原正文「{self._brief_content(snap_content)}」"
                ),
                basis="GB/T 1.1 要求同一术语的定义和正文使用保持一致，不应扩大、缩小或偏移已定义概念的内涵",
            ),
        )]

    # ── E-TM-04：同一概念使用多个术语 ──────────────────────
    def _multi_term(
        self, sections: list[Section], term_section: Optional[Section]
    ) -> list[CounterExample]:
        """从术语章节提取一个术语，在正文另一章节中将该术语替换为近义词，
        造成同一概念在不同章节用不同词语表达的不一致。"""
        if term_section is None:
            return []

        term_lines = [l.strip() for l in term_section.content.splitlines()
                      if self._TERM_ENTRY_RE.match(l.strip())]
        if not term_lines:
            return []

        sample = "\n".join(term_lines[:10])
        pick_prompt = (
            f"以下是「术语和定义」章节的部分条目：\n{sample}\n\n"
            f"请从中提取 1 个最具代表性的术语名称（仅输出术语词本身，不含编号和定义）。"
        )
        term_name = (self._call_llm(pick_prompt) or "").strip()
        if not term_name:
            return []

        candidates = [
            s for s in self._body_sections(sections, term_section)
            if term_name in s.content
        ]
        if not candidates:
            return []

        target = random.choice(candidates)
        snap_content = target.content

        prompt = (
            f"以下是 GB/T 标准某正文章节的内容：\n"
            f"【{target.section_number} {target.title}】\n"
            f"{target.content}\n\n"
            f"术语「{term_name}」在正文中出现。请将该章节中「{term_name}」的某一处"
            f"替换为一个近义词或同义表达（但不是标准术语），"
            f"造成同一概念在文件不同章节使用了不同术语的不一致问题。"
            f"只输出修改后的完整正文，不要解释。"
        )
        new_content = self._call_llm(prompt)
        if not new_content:
            return []

        target.content = new_content
        print(
            f"\n[E-TM-04] 章节「{target.title}」中「{term_name}」被替换为近义词，"
            f"造成同一概念在不同章节表达不一致：\n"
            f"  修改前：{snap_content[:100].strip()}...\n"
            f"  修改后：{new_content[:100].strip()}...\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-TM-04",
            original=Section(section_number=target.section_number,
                             title=target.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=target.section_number,
                        title=target.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"章节「{self._section_label(target.section_number, target.title)}」中标准术语「{term_name}」"
                    f"被替换为近义词或非标准表达，导致同一概念在不同章节使用不同术语；"
                    f"错误正文摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"将非标准近义表达统一恢复为术语「{term_name}」，"
                    f"原正文摘要为「{self._brief_content(snap_content)}」"
                ),
                basis="GB/T 1.1 要求同一概念在标准全文中使用同一术语，不应混用近义词或非标准表达",
            ),
        )]

    # ── E-TM-05：术语编号不连续 ─────────────────────────────
    def _nonsequential_numbering(
        self, term_section: Optional[Section]
    ) -> list[CounterExample]:
        """在「术语和定义」章节中删除一个编号居中的术语条目，制造编号跳号错误。"""
        if term_section is None:
            return []

        lines = term_section.content.splitlines()
        # 找出所有术语条目行的行索引
        entry_indices = [
            i for i, l in enumerate(lines)
            if self._TERM_ENTRY_RE.match(l.strip())
        ]
        # 至少需要 3 条才能删中间那条并产生跳号
        if len(entry_indices) < 3:
            return []

        # 随机选一个非首尾的条目删除，保证产生跳号
        del_pos = random.choice(entry_indices[1:-1])
        removed_line = lines[del_pos]

        new_lines = lines[:del_pos] + lines[del_pos + 1:]
        new_content = "\n".join(new_lines)
        snap_content = term_section.content
        term_section.content = new_content

        print(
            f"\n[E-TM-05] 「术语和定义」章节删除了一个编号条目，造成编号跳号：\n"
            f"  删除条目：{removed_line.strip()}\n"
            f"  剩余术语条目数：{len(entry_indices) - 1}\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-TM-05",
            original=Section(section_number=term_section.section_number,
                             title=term_section.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=term_section.section_number,
                        title=term_section.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「术语和定义」章节删除了编号居中的术语条目「{self._brief_content(removed_line, 80)}」，"
                    f"导致剩余术语编号出现跳号；错误术语章节摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"恢复被删除的术语条目「{self._brief_content(removed_line, 80)}」，"
                    f"或重新调整后续术语编号使其连续"
                ),
                basis="GB/T 1.1 要求术语条目编号按层次连续排列，删除中间条目后不应保留跳号",
            ),
        )]


# ---------------------------------------------------------------------------
# C3.3  引用审查反例(完成)
# ---------------------------------------------------------------------------

class ReferenceErrorGenerator(BaseCounterExampleGenerator):
    """
    C3.3 条款级引用审查反例。

    错误类型：
      E-R-01  漏列引用文件       : 正文规范性引用了某标准，但清单未列出
      E-R-02  冗余引用文件       : 清单列出了某标准，但正文从未引用
      E-R-03  资料性引用混入清单 : 仅在"注"或"示例"中提及的标准被列入规范性清单
    #   E-R-000  引用版本错误       : 引用了已作废的旧版本
    #   E-R-000  注日期引用格式错误 : 注日期引用缺少年份或格式不规范
      E-R-04  引用文件排列顺序错误: 清单未按标准编号顺序排列
    """

    dimension = "C3.3"
    error_types = ["E-R-01", "E-R-02", "E-R-03", "E-R-04"]

    # 标准编号正则：匹配编号主体（含可选年份），不含标题文字
    _STD_RE = re.compile(
        r'(?:GB(?:/T)?|IEC|ISO|YD/T|JG/T|HJ|QB|SJ|JB|NB)\s*\d+'
        r'(?:[./]\d+)*(?:\s*[—\-]\s*\d{4})?'
    )

    # 虚构冗余标准，用于注入（E-R-02/E-R-03 降级时使用）
    #TODO 
    #后续让大模型虚假随机生成冗余标准，以实现反例的随机性
    _FAKE_STANDARDS = [
        "GB/T 99999—2020 电工产品虚拟测试方法",
        "GB/T 88888—2019 高原设备通用规范",
        "IEC 00001:2021 Dummy standard for testing",
        "GB/T 77777—2022 通用电气装置安全规范",
        "GB 66666—2018 工业用电设备安装规程",
    ]

    def generate(
        self,
        sections: list[Section],
        error_type: Optional[str] = None,
    ) -> list[CounterExample]:
        ref_section = self._find_ref_section(sections)
        if ref_section is None:
            logger.warning("[C3.3] 未找到「规范性引用文件」章节，跳过")
            return []

        error_type = self._resolve_error_type(error_type)
        if error_type is None:
            return []
        # error_type = 'E-R-04'
        dispatch = {
            "E-R-01": lambda: self._missing_reference(sections, ref_section),
            "E-R-02": lambda: self._redundant_reference(ref_section),
            "E-R-03": lambda: self._informative_in_normative(sections, ref_section),
            # "E-R-000": lambda: self._wrong_version(ref_section),
            # "E-R-000": lambda: self._bad_date_format(ref_section),
            "E-R-04": lambda: self._wrong_order(ref_section),
        }
        return dispatch[error_type]()

    def _find_ref_section(self, sections: list[Section]) -> Optional[Section]:
        for s in sections:
            if "规范性引用" in s.title and s.content:
                # 去除省略号、空白后仍有实质内容才认为有效
                real_text = s.content.strip().strip("…．。 \t\n")
                if real_text:
                    return s
        return None

    # 标准条目行的识别正则：以 GB、IEC、ISO、YD/T、JG/T 等开头
    _STD_LINE_RE = re.compile(
        r'^(?:GB(?:/T)?|IEC|ISO|YD/T|JG/T|HJ|QB|SJ|JB|NB)\s*\d+'
    )

    @classmethod
    def _parse_ref_lines(cls, ref_section: Section) -> list[str]:
        """返回引用清单解析结果：
          - index 0：非标准条目行合并为一条（前言/说明文字）
          - index 1+：每条标准编号单独一行（GB/T… / IEC… 等开头的行）
        """
        all_lines = [l for l in ref_section.content.splitlines() if l.strip()]
        intro_parts: list[str] = []
        std_lines: list[str] = []
        for l in all_lines:
            if cls._STD_LINE_RE.match(l.strip()):
                std_lines.append(l)
            else:
                intro_parts.append(l.strip())
        intro = "".join(intro_parts)   # 合并前言文字（去掉换行）
        return [intro] + std_lines

    @classmethod
    def _extract_stds(cls, text: str) -> set[str]:
        """从任意文本中提取所有标准编号集合（去掉空格，规范化破折号）。"""
        return {cls._normalize_std(m.group()) for m in cls._STD_RE.finditer(text)}

    @staticmethod
    def _normalize_std(s: str) -> str:
        """规范化标准编号：去除多余空格，将连字符统一为破折号，去掉年份。"""
        s = re.sub(r'\s+', '', s)          # 去除空格
        s = re.sub(r'[—\-]\d{4}$', '', s)  # 去掉末尾年份，只保留编号主体
        return s.upper()

    # ── E-R-01：漏列引用文件 ─────────────────────────────────
    def _missing_reference(
        self, sections: list[Section], ref_section: Section
    ) -> list[CounterExample]:
        """从清单中删除一条正文实际引用的标准，制造"正文引用了但清单漏列"的错误。

        策略：
          1. 扫描所有正文 sections（title 和 content 均非空，且非引用清单节）提取标准编号集合
          2. 遍历引用清单各行，找出在正文中确实出现过的标准（即真正被引用的）
          3. 从中随机删除一条 → 清单漏列
          4. 找不到交集时降级：随机删除清单中任意一条
        """
        body_stds: set[str] = set()
        for s in sections:
            if s.title and s.content and s is not ref_section:
                body_stds |= self._extract_stds(s.content)

        lines = self._parse_ref_lines(ref_section)
        if len(lines) < 2:
            return []

        # 找到正文中实际引用过的清单行
        cited_idx = [
            i for i, l in enumerate(lines)
            if self._extract_stds(l) & body_stds
        ]
        # 降级：正文中找不到对应引用时，随机选一条清单行删除
        target_idx = random.choice(cited_idx) if cited_idx else random.randrange(len(lines))
        removed = lines[target_idx]
        new_lines = lines[:target_idx] + lines[target_idx + 1:]
        new_content = "\n".join(new_lines)

        mode = "正文有引用" if cited_idx else "降级（正文未检测到对应引用）"
        snap_content = ref_section.content
        ref_section.content = new_content

        print(
            f"\n[E-R-01] 「规范性引用文件」漏列了以下标准（{mode}）：\n"
            f"  删除条目：{removed.strip()}\n"
            f"  清单剩余 {len(new_lines)} 条\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-R-01",
            original=Section(section_number=ref_section.section_number,
                             title=ref_section.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=ref_section.section_number,
                        title=ref_section.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「规范性引用文件」清单删除了条目「{removed.strip()}」，"
                    f"导致正文中仍被规范性引用的文件未在清单中列出；"
                    f"错误清单摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"将漏删的引用条目「{removed.strip()}」恢复到「规范性引用文件」清单中，"
                    f"并保持清单与正文引用一致"
                ),
                basis="GB/T 1.1 要求正文中规范性引用的文件应完整列入「规范性引用文件」章节",
            ),
        )]

    # ── E-R-02：冗余引用文件 ─────────────────────────────────
    def _redundant_reference(self, ref_section: Section) -> list[CounterExample]:
        """用 LLM 生成 1~3 条与本文件领域相关但正文从未引用的虚构标准条目，注入规范性清单。

        策略：
          1. 将引用清单现有条目发给 LLM，让其仿照格式生成 1~3 条同领域但实际不存在的标准
          2. 将生成条目逐行追加到 ref_section.content 末尾（原地修改）
          3. LLM 不可用时降级使用 _FAKE_STANDARDS 列表
        """
        lines = self._parse_ref_lines(ref_section)
        # lines[0] 是前言文字，lines[1:] 是标准条目
        std_sample = "\n".join(lines[1:min(6, len(lines))])  # 取前几条作示例
        count = random.randint(1, 3)

        new_entries: list[str] = []
        if self._llm is not None:
            prompt = (
                f"以下是某 GB/T 标准「规范性引用文件」章节中已有的引用条目示例：\n"
                f"{std_sample}\n\n"
                f"请仿照上述格式，生成 {count} 条与该领域相关但实际上并不存在（虚构）的标准条目，"
                f"每条单独一行，格式为：标准编号 标准名称（如 GB/T12345 设备安全通用要求）。\n"
                f"只输出条目行，不要编号、解释或多余文字。"
            )
            result = (self._call_llm(prompt) or "").strip()
            if result:
                new_entries = [l.strip() for l in result.splitlines() if l.strip()]

        # LLM 未返回有效内容时降级
        if not new_entries:
            new_entries = random.sample(self._FAKE_STANDARDS, min(count, len(self._FAKE_STANDARDS)))

        snap_content = ref_section.content
        new_content = snap_content.rstrip() + "\n" + "\n".join(new_entries)
        ref_section.content = new_content

        entries_str = "\n".join(f"    {e}" for e in new_entries)
        source = "LLM 生成" if self._llm is not None and new_entries else "降级（固定列表）"
        print(
            f"\n[E-R-02] 「规范性引用文件」注入了 {len(new_entries)} 条冗余引用（{source}）：\n"
            f"{entries_str}\n"
            f"  以上条目在正文中均无对应引用\n"
        )

        return [CounterExample(
            dimension=self.dimension,
            error_type="E-R-02",
            original=Section(section_number=ref_section.section_number,
                             title=ref_section.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=ref_section.section_number,
                        title=ref_section.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「规范性引用文件」清单新增了 {len(new_entries)} 条正文从未引用的标准："
                    f"{'、'.join(new_entries)}；错误清单摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    f"删除这些正文没有规范性引用依据的条目；若确需保留，应在正文中设置对应的规范性引用"
                ),
                basis="GB/T 1.1 要求「规范性引用文件」只列出正文规范性引用且必不可少的文件，不应列入冗余标准",
            ),
        )]

    # ── E-R-03：资料性引用混入规范性清单 ─────────────────────

    # 「参考文献」条目行识别：以 [N] 开头，或以标准编号前缀开头
    _INFO_ENTRY_RE = re.compile(
        r'^\[\d+\]|^(?:GB(?:/T)?|IEC|ISO|YD/T|JG/T|HJ|QB|SJ|JB|NB)\s*\d+'
    )

    def _find_informative_section(self, sections: list[Section]) -> Optional[Section]:
        """找到 title 含「参考文献」且 content 有实质内容的章节。"""
        for s in sections:
            if "参考文献" in s.title and s.content:
                if s.content.strip().strip("…．。 \t\n"):
                    return s
        return None

    def _informative_in_normative(
        self, sections: list[Section], ref_section: Section
    ) -> list[CounterExample]:
        """从「参考文献」章节随机取一条条目，将其从「参考文献」中删除，
        再追加到「规范性引用文件」末尾，制造资料性引用混入规范性清单的错误。

        降级策略：找不到「参考文献」章节时，通过 LLM 或固定列表生成一条虚构条目注入。
        """
        info_section = self._find_informative_section(sections)
        info_lines: list[str] = []
        entry_lines: list[str] = []

        if info_section is not None:
            info_lines = [l for l in info_section.content.splitlines() if l.strip()]
            entry_lines = [l for l in info_lines if self._INFO_ENTRY_RE.match(l.strip())]
            if not entry_lines:
                info_section = None  # 没有可识别条目，降级

        if info_section is not None:
            # ── 正常路径：从「参考文献」取一条并删除 ──────────────
            picked_raw = random.choice(entry_lines)
            # 去掉 [N] 编号前缀，保留标准编号+标准名称
            injected = re.sub(r'^\[\d+\]\s*', '', picked_raw).strip()

            new_info_lines = [l for l in info_lines if l != picked_raw]
            info_section.content = "\n".join(new_info_lines)
            source_desc = f"从「{info_section.title}」提取"
        else:
            # ── 降级路径：LLM 生成或固定列表 ────────────────────
            picked_raw = None
            if self._llm is not None:
                ref_sample = "\n".join(self._parse_ref_lines(ref_section)[1:5])
                prompt = (
                    f"以下是某 GB/T 标准「规范性引用文件」中已有的条目：\n{ref_sample}\n\n"
                    f"请仿照格式生成 1 条与该领域相关、属于资料性参考（非规范性引用）的标准条目，"
                    f"格式：标准编号 标准名称。只输出该一条，不要解释。"
                )
                result = (self._call_llm(prompt) or "").strip()
                injected = result if result else random.choice(self._FAKE_STANDARDS)
            else:
                injected = random.choice(self._FAKE_STANDARDS)
            source_desc = "虚构（降级，未找到「参考文献」章节）"

        # 追加到「规范性引用文件」末尾
        ref_snap = ref_section.content
        new_ref_content = ref_snap.rstrip() + f"\n{injected}"
        ref_section.content = new_ref_content

        # 打印变更详情
        if info_section is not None:
            print(
                f"\n[E-R-03] 资料性引用被混入规范性清单（{source_desc}）：\n"
                f"  来源章节：[{info_section.section_number}] {info_section.title}\n"
                f"  原始条目：{picked_raw.strip()}\n"
                f"  ↳ 已从「{info_section.title}」中删除该行\n"
                f"  ↳ 已追加到「规范性引用文件」末尾：{injected}\n"
            )
        else:
            print(
                f"\n[E-R-03] 资料性引用被混入规范性清单（{source_desc}）：\n"
                f"  注入条目：{injected}\n"
                f"  已追加到「规范性引用文件」末尾\n"
            )

        return [CounterExample(
            dimension=self.dimension,
            error_type="E-R-03",
            original=Section(section_number=ref_section.section_number,
                             title=ref_section.title, content=ref_snap),
            corrupted=self._make_corrupted(
                Section(section_number=ref_section.section_number,
                        title=ref_section.title, content=ref_snap),
                new_ref_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「规范性引用文件」清单末尾混入资料性引用条目「{injected}」；"
                    f"错误清单摘要为「{self._brief_content(new_ref_content)}」"
                ),
                fix=(
                    f"从「规范性引用文件」清单删除「{injected}」，"
                    f"并将其放回或保留在「参考文献」等资料性引用位置"
                ),
                basis="GB/T 1.1 要求规范性引用文件与资料性参考文献分列，资料性引用不应混入规范性引用清单",
            ),
        )]

    # ── E-R-0000：引用版本错误 ──────────────────────────────────
    def _wrong_version(self, ref_section: Section) -> list[CounterExample]:
        """将清单中某条标准的年份倒退 5~15 年，模拟引用已作废旧版本。"""
        lines = self._parse_ref_lines(ref_section)
        candidates = [
            (i, l) for i, l in enumerate(lines)
            if re.search(r'[—\-]\s*(\d{4})', l)
        ]
        if not candidates:
            return []

        idx, line = random.choice(candidates)
        m = re.search(r'([—\-]\s*)(\d{4})', line)
        old_year = int(m.group(2))
        new_year = str(old_year - random.randint(5, 15))
        new_line = line[:m.start(2)] + new_year + line[m.start(2) + 4:]
        lines[idx] = new_line
        new_content = "\n".join(lines)

        snap_content = ref_section.content
        ref_section.content = new_content

        print(
            f"\n[E-R-04] 「规范性引用文件」中引用了已作废版本：\n"
            f"  原条目：{line.strip()}\n"
            f"  错误条目：{new_line.strip()}\n"
            f"  年份 {old_year} → {new_year}（模拟引用旧版本）\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-R-04",
            original=Section(section_number=ref_section.section_number,
                             title=ref_section.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=ref_section.section_number,
                        title=ref_section.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「规范性引用文件」中的条目「{line.strip()}」被错误改为旧版本「{new_line.strip()}」，"
                    f"引用年份由 {old_year} 倒退为 {new_year}"
                ),
                fix=(
                    f"将引用条目恢复为当前应引用版本「{line.strip()}」，"
                    f"或根据正文需要核对并更新为有效版本"
                ),
                basis="GB/T 1.1 要求规范性引用文件的版本信息准确，注日期引用应指向明确且有效的文件版本",
            ),
        )]

    # ── E-R-0000：注日期引用格式错误 ───────────────────────────
    def _bad_date_format(self, ref_section: Section) -> list[CounterExample]:
        """破坏注日期引用的年份格式：删除年份或将破折号改为连字符。"""
        lines = self._parse_ref_lines(ref_section)
        candidates = [
            (i, l) for i, l in enumerate(lines)
            if re.search(r'—\s*\d{4}', l)
        ]
        if not candidates:
            return []

        idx, line = random.choice(candidates)
        mode = random.choice(["drop_year", "wrong_dash"])

        if mode == "drop_year":
            new_line = re.sub(r'—\s*\d{4}', '—', line)
            desc = "删除年份"
        else:
            new_line = re.sub(r'—', '-', line)
            desc = "破折号「—」改为连字符「-」"

        lines[idx] = new_line
        new_content = "\n".join(lines)

        snap_content = ref_section.content
        ref_section.content = new_content

        print(
            f"\n[E-R-05] 注日期引用格式错误（{desc}）：\n"
            f"  原条目：{line.strip()}\n"
            f"  错误条目：{new_line.strip()}\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-R-05",
            original=Section(section_number=ref_section.section_number,
                             title=ref_section.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=ref_section.section_number,
                        title=ref_section.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「规范性引用文件」中的注日期引用条目「{line.strip()}」被改为错误格式"
                    f"「{new_line.strip()}」（{desc}）"
                ),
                fix=(
                    f"将该引用条目恢复为规范注日期格式「{line.strip()}」"
                ),
                basis="GB/T 1.1 要求注日期引用使用规范的标准编号和年份标注格式，年份或连接符不应缺失或误用",
            ),
        )]

    # ── E-R-04：引用文件排列顺序错误 ────────────────────────
    def _wrong_order(self, ref_section: Section) -> list[CounterExample]:
        """打乱清单顺序，使其不再按标准编号升序排列。"""
        lines = self._parse_ref_lines(ref_section)
        if len(lines) < 3:
            return []

        shuffled = lines[:]
        for _ in range(10):
            random.shuffle(shuffled)
            if shuffled != lines:
                break
        if shuffled == lines:
            shuffled = lines[::-1]

        new_content = "\n".join(shuffled)
        snap_content = ref_section.content
        ref_section.content = new_content
        first_diff = next(
            (i for i, (old, new) in enumerate(zip(lines, shuffled)) if old != new),
            0,
        )

        print(
            f"\n[E-R-04] 「规范性引用文件」排列顺序被打乱：\n"
            f"  原顺序首条：{lines[0].strip()}\n"
            f"  打乱后首条：{shuffled[0].strip()}\n"
            f"  共 {len(lines)} 条标准被重新排列\n"
        )
        return [CounterExample(
            dimension=self.dimension,
            error_type="E-R-04",
            original=Section(section_number=ref_section.section_number,
                             title=ref_section.title, content=snap_content),
            corrupted=self._make_corrupted(
                Section(section_number=ref_section.section_number,
                        title=ref_section.title, content=snap_content),
                new_content,
            ),
            error_description=self._format_description(
                error=(
                    f"「规范性引用文件」清单的 {len(lines)} 条内容被打乱顺序，"
                    f"第 {first_diff + 1} 条由「{lines[first_diff].strip()}」"
                    f"错误变为「{shuffled[first_diff].strip()}」；"
                    f"错误清单摘要为「{self._brief_content(new_content)}」"
                ),
                fix=(
                    "按标准编号或文件要求的顺序重新排列「规范性引用文件」清单，恢复引用条目的规范顺序"
                ),
                basis="GB/T 1.1 要求规范性引用文件清单按规定顺序排列，不能随意打乱引用文件顺序",
            ),
        )]


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

_GENERATOR_REGISTRY: dict[str, type[BaseCounterExampleGenerator]] = {
    "C2.1": StructureErrorGenerator,
    "C2.2": ScopeErrorGenerator,
    "C3.1": ModalErrorGenerator,
    "C3.2": TerminologyErrorGenerator,
    "C3.3": ReferenceErrorGenerator,
}


def get_counter_example_error_types(
    dimensions: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """Return supported error types by dimension."""
    dims = list(_GENERATOR_REGISTRY.keys()) if dimensions is None else dimensions
    result: dict[str, list[str]] = {}
    for dim in dims:
        cls = _GENERATOR_REGISTRY.get(dim)
        if cls is not None:
            result[dim] = list(cls.error_types)
    return result


class CounterExamplePipeline:
    """
    反例构造流水线。

    用法::

        pipeline = CounterExamplePipeline(llm_client=llm)
        doc = pipeline.run(all_sections, dimensions=["C2.1", "C3.1", "C3.3"])

        print(doc.to_text())        # 注入错误后的完整文档
        print(doc.to_annotation())  # 错误标注清单
    """

    def __init__(self, llm_client=None):
        self._llm = llm_client

    @staticmethod
    def _ordered_error_types(
        error_types: list[str],
        error_type_counts: dict[str, int],
    ) -> list[str]:
        ordered = list(error_types)
        random.shuffle(ordered)
        return sorted(ordered, key=lambda item: error_type_counts.get(item, 0))

    def run(
        self,
        sections: list[Section],
        dimensions: Optional[list[str]] = None,
        error_counts: Optional[dict[str, int]] = None,
        error_type_counts: Optional[dict[str, int]] = None,
    ) -> CounterExampleDocument:
        """
        Args:
            sections   : 原始正确 Section 列表
            dimensions : 要生成的维度列表，默认全部维度
            error_counts: 每个维度需要生成的反例数量；未指定时每个维度 1 条

        Returns:
            CounterExampleDocument
        """
        dims = list(_GENERATOR_REGISTRY.keys()) if dimensions is None else dimensions
        balance_counts = error_type_counts if error_type_counts is not None else {}
        doc = CounterExampleDocument(source_sections=sections)#构造了反例的all_sections

        for dim in dims:
            #注入错误类型
            cls = _GENERATOR_REGISTRY.get(dim)
            if cls is None:
                logger.warning("未知维度 %s，跳过", dim)
                continue
            generator = cls(llm_client=self._llm)
            target_count = 1
            if error_counts is not None:
                target_count = int(error_counts.get(dim, 1))
            if target_count < 1:
                logger.warning("%s 的目标反例数量小于 1，跳过", dim)
                continue

            logger.info("生成 %s 反例，目标 %d 次注入...", dim, target_count)
            dim_examples: list[CounterExample] = []
            successful_injections = 0
            attempts = 0
            max_attempts = max(target_count * len(generator.error_types) * 2, 12)
            for error_type in generator.error_types:
                balance_counts.setdefault(error_type, 0)

            while successful_injections < target_count and attempts < max_attempts:
                made_progress = False
                for target_error_type in self._ordered_error_types(
                    generator.error_types,
                    balance_counts,
                ):
                    if attempts >= max_attempts:
                        break
                    attempts += 1

                    # 在副本上试注入；一次 generate 返回的多条标注视为同一个错误事件。
                    candidate_sections = copy.deepcopy(sections)
                    examples = generator.generate(
                        candidate_sections,
                        error_type=target_error_type,
                    )
                    if not examples:
                        continue

                    sections[:] = candidate_sections
                    dim_examples.extend(examples)
                    actual_error_type = examples[0].error_type or target_error_type
                    balance_counts[actual_error_type] = balance_counts.get(actual_error_type, 0) + 1
                    successful_injections += 1
                    made_progress = True
                    break

                if not made_progress and attempts >= max_attempts:
                    break

            if successful_injections < target_count:
                logger.warning(
                    "%s 目标 %d 次注入，实际成功 %d 次；可能是可用候选章节不足",
                    dim,
                    target_count,
                    successful_injections,
                )

            doc.examples.extend(dim_examples)#存入文档，后续需要去除
            logger.info(
                "%s 生成完成，成功注入 %d 次，共 %d 条反例",
                dim,
                successful_injections,
                len(dim_examples),
            )

        return doc
