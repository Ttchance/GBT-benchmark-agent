from __future__ import annotations

# -*- coding: utf-8 -*-
"""
L4 多模态级审查器
  - C4.1 表格审查
  - C4.2 图文审查
"""

from core.base_reviewer import BaseReviewer, ReviewIssue, ReviewResult
from core.document_parser import ParsedDocument, Table, Figure
from core.llm_client import BaseLLMClient


# ═════════════════════════════════════════════════════════════════════════════
# C4.1 表格审查
# ═════════════════════════════════════════════════════════════════════════════

class TableReviewer(BaseReviewer):
    """
    C4.1 表格审查

    本质：判断表格内的数据、指标、描述，与正文条款中对应内容是否一致。

    流程：文档解析 → 表格提取 → 正文关联段落定位 → 多模态对比 → 问题检出 → 定位输出

    多模态一致性比对四角度：
      ① 表格数据与正文描述是否一致
      ② 正文指标是否在表格中完整体现
      ③ 表格是否存在正文未提及的数据项
      ④ 若不一致，指出具体位置和冲突内容

    评估指标：Detection-F1 + Location-Acc
    """

    reviewer_id = "C4.1"
    reviewer_name = "表格审查"
    level = "L4_multimodal"

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        for table in document.tables:
            # TODO: Step 1 - 结构化提取表格数据
            # table_data = self._extract_table_data(table)

            # TODO: 定位正文关联段落
            # related_paras = self._locate_related_paragraphs(document, table)

            # TODO: Step 2 - 多模态一致性比对（四角度）
            # raw = self.llm.chat(self._build_messages(table_data, related_paras))
            # parsed = self.llm.parse_json_response(raw)
            # issues.extend(self._parse_issues(parsed, table))
            pass

        return self._build_result(issues)

    def _extract_table_data(self, table: Table) -> dict:
        """结构化提取表格数据并规范化。"""
        raise NotImplementedError

    def _locate_related_paragraphs(self, document: ParsedDocument, table: Table) -> list:
        """定位与表格相关的正文段落。"""
        raise NotImplementedError

    def _build_messages(self, table_data: dict, related_paras: list) -> list[dict]:
        """构造四角度一致性比对的 Prompt。"""
        raise NotImplementedError

    def _parse_issues(self, llm_output: dict, table: Table) -> list[ReviewIssue]:
        """解析比对结果，同时给出表格侧和正文侧的精确定位。"""
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════════
# C4.2 图文审查
# ═════════════════════════════════════════════════════════════════════════════

class FigureReviewer(BaseReviewer):
    """
    C4.2 图文审查

    本质：同时理解图片内容和文本语义，判断两者是否匹配。

    流程：图片提取 → 标题提取 → 正文关联 → 多模态理解 → 一致性判断 → 诊断输出

    三维度一致性判断：
      ① 图片与标题一致性
      ② 图片与正文引用一致性
      ③ 技术参数一致性

    文档层面：图片序号与正文引用顺序是否一致

    评估指标：Detection-F1 + Diagnosis-Acc
    """

    reviewer_id = "C4.2"
    reviewer_name = "图文审查"
    level = "L4_multimodal"

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        for figure in document.figures:
            # TODO: Step 1 - 调用多模态模型生成图片描述
            # description = self._describe_figure(figure)

            # TODO: Step 2 - 提取标题并定位正文关联段落
            # related_paras = self._locate_related_paragraphs(document, figure)

            # TODO: Step 3 - 三维度一致性判断
            # raw = self.llm.chat_with_image(
            #     self._build_messages(description, figure.caption, related_paras),
            #     figure.image_data
            # )
            # parsed = self.llm.parse_json_response(raw)
            # issues.extend(self._parse_issues(parsed, figure))
            pass

        # TODO: Step 4 - 文档层面检查图片序号与正文引用顺序
        issues.extend(self._check_figure_order(document))

        return self._build_result(issues)

    def _describe_figure(self, figure: Figure) -> str:
        """调用多模态模型生成图片结构化描述。"""
        # 描述内容：图片类型、主体内容、关键元素、图例、坐标轴等
        raise NotImplementedError

    def _locate_related_paragraphs(self, document: ParsedDocument, figure: Figure) -> list:
        """定位正文中对该图片的引用段落。"""
        raise NotImplementedError

    def _build_messages(self, description: str, caption: str, related_paras: list) -> list[dict]:
        """构造三维度一致性判断的 Prompt。"""
        raise NotImplementedError

    def _parse_issues(self, llm_output: dict, figure: Figure) -> list[ReviewIssue]:
        """解析诊断结果，分类输出。"""
        raise NotImplementedError

    def _check_figure_order(self, document: ParsedDocument) -> list[ReviewIssue]:
        """检查图片序号与正文引用顺序是否一致。"""
        return []
