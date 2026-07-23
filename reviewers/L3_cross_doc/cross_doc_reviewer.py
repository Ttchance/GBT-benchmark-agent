from __future__ import annotations

# -*- coding: utf-8 -*-
"""
C3.2 内容一致性审查 —— 跨文档术语一致性审查（L3 跨文档级）

与 C2.2 的区别：
  - C2.2 的比对基准是本文档第 3 章的术语定义
  - C3.2 的比对基准是被引用的外部标准，需同时理解两份文档的语义

流程：术语对齐 → 被引标准获取 → 定义提取 → 跨文档语义比对 → 问题检出 → 报告输出
评估指标：Detection-F1
"""

from core.base_reviewer import BaseReviewer, ReviewIssue, ReviewResult
from core.document_parser import ParsedDocument
from core.llm_client import BaseLLMClient


class CrossDocTermReviewer(BaseReviewer):
    """跨文档术语一致性审查器 C3.2"""

    reviewer_id = "C3.2"
    reviewer_name = "跨文档术语一致性审查"
    level = "L3_cross_doc"

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client
        # 外部标准数据库（term → definition）
        self._external_term_db: dict[str, dict] = {}

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        # TODO: Step 1 - 构建当前文档术语使用清单（位置 + 上下文）
        # term_usage_list = self._build_term_usage_list(document)

        # TODO: Step 2 - 获取被引标准内容，提取术语定义 → 存入 _external_term_db
        # self._load_external_standards(document)

        # TODO: Step 3 - 术语对齐（当前文档术语 ↔ 被引标准术语）
        # alignment = self._align_terms(term_usage_list)

        # TODO: Step 4 - 跨文档语义比对（注意多版本问题，避免版本混用误判）
        # for aligned_pair in alignment:
        #     result = self.llm.chat(self._build_messages(aligned_pair))
        #     issues.extend(self._parse_issues(result, aligned_pair))

        # TODO: Step 5 - 输出检测报告

        return self._build_result(issues)

    def _build_term_usage_list(self, document: ParsedDocument) -> list[dict]:
        """构建当前文档的术语使用清单。"""
        # 返回格式：[{term, location, context}, ...]
        raise NotImplementedError

    def _load_external_standards(self, document: ParsedDocument):
        """获取被引标准内容，用 LLM 自动提取术语定义并存入数据库。"""
        # TODO: 从文档的引用清单中获取标准ID，调用外部API或本地数据库
        raise NotImplementedError

    def _align_terms(self, term_usage_list: list[dict]) -> list[dict]:
        """建立当前文档术语与被引标准术语的对应关系。"""
        raise NotImplementedError

    def _build_messages(self, aligned_pair: dict) -> list[dict]:
        """构造跨文档语义比对的 Prompt。"""
        raise NotImplementedError

    def _parse_issues(self, llm_output: dict, aligned_pair: dict) -> list[ReviewIssue]:
        """解析 LLM 返回的比对结果。"""
        raise NotImplementedError
