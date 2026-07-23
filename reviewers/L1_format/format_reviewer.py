from __future__ import annotations

# -*- coding: utf-8 -*-
"""
C1.1 格式审查（L1 格式级）

审查目标：
  - 将文档中每个可检查的格式元素与 GB/T 1.1 第10章规定的格式规则逐条比对。
  - 问题类型固定、规则明确，是整个审查体系中最适合规则化/自动化的部分。

实施步骤（详见能力体系文档）：
  1. 将输入文档解析为结构化的段落序列（含位置、样式、字体属性）
  2. 对每个有效段落构造 Prompt → LLM 输出结构化 JSON 审查结果
  3. 去重处理，同类错误合并
  4. 生成摘要 + 明细报告
"""

from core.base_reviewer import BaseReviewer, ReviewIssue, ReviewResult
from core.document_parser import ParsedDocument
from core.llm_client import BaseLLMClient


class FormatReviewer(BaseReviewer):
    """格式审查器 C1.1"""

    reviewer_id = "C1.1"
    reviewer_name = "格式审查"
    level = "L1_format"

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def review(self, document: ParsedDocument) -> ReviewResult:
        """
        对文档执行格式审查。

        Args:
            document: 已解析的 ParsedDocument

        Returns:
            ReviewResult
        """
        issues: list[ReviewIssue] = []

        # Step 1: 遍历段落
        for para in document.paragraphs:
            # TODO: Step 2 - 构造 Prompt 并调用 LLM
            # raw_result = self.llm.chat(self._build_messages(para))
            # parsed = self.llm.parse_json_response(raw_result)
            # issues.extend(self._parse_issues(parsed, para))
            pass

        # TODO: Step 3 - 去重 & 合并
        issues = self._deduplicate(issues)

        # Step 4: 构建结果
        return self._build_result(issues)

    # ------------------------------------------------------------------
    # 私有方法（待实现）
    # ------------------------------------------------------------------

    def _build_messages(self, para) -> list[dict]:
        """构造发送给 LLM 的消息列表。"""
        # TODO: 拼装 system prompt（规则库）+ user message（段落信息）
        raise NotImplementedError

    def _parse_issues(self, llm_output: dict, para) -> list[ReviewIssue]:
        """将 LLM 返回的 JSON 解析为 ReviewIssue 列表。"""
        # TODO: 从 llm_output 中提取 error_type / description / suggestion 等字段
        raise NotImplementedError

    def _deduplicate(self, issues: list[ReviewIssue]) -> list[ReviewIssue]:
        """同类错误合并，记录总出现次数和所有位置。"""
        # TODO: 按 issue_type 聚合，更新 extra 字段记录所有位置
        return issues
