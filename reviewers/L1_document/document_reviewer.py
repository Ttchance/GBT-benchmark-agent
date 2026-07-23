from __future__ import annotations

# -*- coding: utf-8 -*-
"""
C1.1 结构审查（L1 文档级）

审查目标：
  - 章节结构是否符合 GB/T 1.1《标准化工作导则》的要求
  - 封面、目次、前言、正文、附录等结构是否完整
  - 章节编号、标题层级是否规范
  - 页边距、字体、字号、行间距等排版是否符合规定
"""

from core.base_reviewer import BaseReviewer, ReviewIssue, ReviewResult
from core.document_parser import ParsedDocument
from core.llm_client import BaseLLMClient


class StructureReviewer(BaseReviewer):
    """结构审查器 C1.1（文档级）"""

    reviewer_id = "C1.1-DOC"
    reviewer_name = "结构审查"
    level = "L1_document"

    # 国标规定的必须章节（可按需扩展）
    REQUIRED_SECTIONS = ["封面", "目次", "前言", "正文", "附录"]

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        # TODO: 检查必须章节完整性
        issues.extend(self._check_required_sections(document))

        # TODO: 检查章节编号规范性
        issues.extend(self._check_section_numbering(document))

        # TODO: 检查标题层级规范性
        issues.extend(self._check_heading_levels(document))

        return self._build_result(issues)

    def _check_required_sections(self, document: ParsedDocument) -> list[ReviewIssue]:
        """检查封面、目次、前言、正文、附录等是否完整。"""
        # TODO: 实现
        return []

    def _check_section_numbering(self, document: ParsedDocument) -> list[ReviewIssue]:
        """检查章节编号是否连续、规范。"""
        # TODO: 实现
        return []

    def _check_heading_levels(self, document: ParsedDocument) -> list[ReviewIssue]:
        """检查标题层级是否符合 GB/T 1.1 规定。"""
        # TODO: 实现
        return []


# ─────────────────────────────────────────────────────────────────────────────

class ScopeReviewer(BaseReviewer):
    """
    C1.2 范围审查（L1 文档级）

    审查目标：
      - 检查正文是否引入了范围之外的对象或概念
      - 术语是否与范围声明的领域相匹配
      - 技术指标是否超出范围边界
    """

    reviewer_id = "C1.2"
    reviewer_name = "范围审查"
    level = "L1_document"

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        # TODO: Step 1 - 提取第 1 章"范围"的声明内容
        # TODO: Step 2 - 基于 LLM 语义理解，判断各章节内容是否超出范围边界
        # TODO: Step 3 - 输出问题列表

        return self._build_result(issues)

    def _extract_scope_statement(self, document: ParsedDocument) -> str:
        """从文档中提取第 1 章"范围"的完整文本。"""
        # TODO: 实现
        raise NotImplementedError

    def _check_out_of_scope(self, scope_text: str, document: ParsedDocument) -> list[ReviewIssue]:
        """使用 LLM 判断正文内容是否超出范围声明。"""
        # TODO: 构造 Prompt → 调用 LLM → 解析结果
        raise NotImplementedError
