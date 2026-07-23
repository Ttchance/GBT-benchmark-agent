from __future__ import annotations

# -*- coding: utf-8 -*-
"""
审查流水线（Pipeline）
负责按能力体系框架编排并调度各审查器，汇总审查结果。

整体研究流程（参照能力体系文档）：
  第一步: 文档解析
  第二步: 各层审查器依次/并行执行
  第三步: 汇总结果 → 生成报告
"""

import logging
from dataclasses import dataclass, field

from core.base_reviewer import ReviewResult
from core.document_parser import BaseDocumentParser, ParsedDocument
from core.llm_client import BaseLLMClient

# L1 格式级
from reviewers.L1_format.format_reviewer import FormatReviewer
# L1 文档级
from reviewers.L1_document.document_reviewer import StructureReviewer, ScopeReviewer
# L2 条款级
from reviewers.L2_clause.clause_reviewer import ToneReviewer, TermReviewer, ReferenceReviewer
# L3 跨文档级
from reviewers.L3_cross_doc.cross_doc_reviewer import CrossDocTermReviewer
# L4 多模态级
from reviewers.L4_multimodal.multimodal_reviewer import TableReviewer, FigureReviewer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """整个流水线的汇总结果"""
    document_path: str
    results: list[ReviewResult] = field(default_factory=list)

    @property
    def total_issues(self) -> int:
        return sum(len(r.issues) for r in self.results)

    @property
    def summary_by_reviewer(self) -> dict:
        return {
            r.reviewer_id: r.summary for r in self.results
        }


class ReviewPipeline:
    """
    审查流水线

    使用方式：
        pipeline = ReviewPipeline(parser, llm_client)
        result = pipeline.run("/path/to/document.docx")
    """

    def __init__(
        self,
        parser: BaseDocumentParser,
        llm_client: BaseLLMClient,
        enabled_reviewers: list[str] | None = None,
    ):
        """
        Args:
            parser: 文档解析器
            llm_client: LLM 客户端
            enabled_reviewers: 要启用的审查器 ID 列表（None 表示全部启用）
                               例如 ["C1.1", "C2.1", "C2.2"]
        """
        self.parser = parser
        self.llm = llm_client
        self.enabled_reviewers = enabled_reviewers

        # 按层级注册所有审查器
        self._all_reviewers = self._build_reviewers()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def run(self, file_path: str) -> PipelineResult:
        """
        对指定文档运行完整审查流水线。

        Args:
            file_path: 文档路径

        Returns:
            PipelineResult
        """
        logger.info(f"开始审查文档: {file_path}")

        # Step 1: 文档解析，得到pdf文件内容的结构化表示（段落、表格、图片等）
        document = self._parse(file_path)

        # Step 2: 过滤启用的审查器
        reviewers = self._filter_reviewers()

        # Step 3: 依次执行各审查器
        results: list[ReviewResult] = []
        for reviewer in reviewers:
            logger.info(f"[{reviewer.reviewer_id}] {reviewer.reviewer_name} 开始审查...")
            try:
                result = reviewer.review(document)
                results.append(result)
                logger.info(
                    f"[{reviewer.reviewer_id}] 完成，发现 {len(result.issues)} 个问题"
                )
            except Exception as e:
                logger.error(f"[{reviewer.reviewer_id}] 审查失败: {e}", exc_info=True)

        return PipelineResult(document_path=file_path, results=results)

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _parse(self, file_path: str) -> ParsedDocument:
        """调用文档解析器。"""
        logger.info("文档解析中...")
        document = self.parser.parse(file_path)#得到解析后的所有内容，之后开始多个维度进行分析
        logger.info(
            f"解析完成: {len(document.paragraphs)} 段落 / "
            f"{len(document.tables)} 表格 / "
            f"{len(document.figures)} 图片"
        )
        return document

    def _build_reviewers(self) -> list:
        """实例化所有审查器（按层级顺序）。"""
        return [
            # L1 格式级
            FormatReviewer(self.llm),
            # L1 文档级
            StructureReviewer(self.llm),
            ScopeReviewer(self.llm),
            # L2 条款级
            ToneReviewer(self.llm),
            TermReviewer(self.llm),
            ReferenceReviewer(self.llm),
            # L3 跨文档级
            CrossDocTermReviewer(self.llm),
            # L4 多模态级
            # TableReviewer(self.llm),
            # FigureReviewer(self.llm),
        ]

    def _filter_reviewers(self) -> list:
        """根据 enabled_reviewers 过滤审查器。"""
        if self.enabled_reviewers is None:
            return self._all_reviewers
        return [
            r for r in self._all_reviewers
            if r.reviewer_id in self.enabled_reviewers
        ]
