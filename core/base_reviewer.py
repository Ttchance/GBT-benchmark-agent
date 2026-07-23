from __future__ import annotations

# -*- coding: utf-8 -*-
"""
审查器基类
所有具体的审查器（C1.1 ~ C4.2）均继承此基类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReviewIssue:
    """单条审查问题的结构化表示"""
    reviewer_id: str            # 审查器编号，例如 "C1.1"
    location: str               # 问题位置（章节/段落）
    issue_type: str             # 问题类型
    description: str            # 问题描述
    suggestion: str             # 修改建议
    confidence: float = 1.0    # 置信度 [0, 1]
    extra: dict = field(default_factory=dict)   # 扩展字段（按需使用）


@dataclass
class ReviewResult:
    """审查器的整体输出结果"""
    reviewer_id: str
    reviewer_name: str
    level: str                          # "L1_format" / "L1_document" / "L2_clause" / "L3_cross" / "L4_multimodal"
    issues: list[ReviewIssue] = field(default_factory=list)
    summary: dict = field(default_factory=dict)   # 统计摘要

    @property
    def total_issues(self) -> int:
        return len(self.issues)


class BaseReviewer(ABC):
    """
    审查器抽象基类。

    子类必须实现：
        - reviewer_id   (类属性) 例如 "C1.1"
        - reviewer_name (类属性) 例如 "格式审查"
        - level         (类属性) 例如 "L1_format"
        - review(document) → ReviewResult
    """

    reviewer_id: str = ""
    reviewer_name: str = ""
    level: str = ""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._validate_class_attrs()
        logger.info(f"[{self.reviewer_id}] {self.reviewer_name} 初始化完成")

    def _validate_class_attrs(self):
        for attr in ("reviewer_id", "reviewer_name", "level"):
            if not getattr(self, attr):
                raise NotImplementedError(f"子类必须定义类属性 `{attr}`")

    @abstractmethod
    def review(self, document: Any) -> ReviewResult:
        """
        对传入的文档对象执行审查。

        Args:
            document: 已解析的文档对象（由 DocumentParser 返回）

        Returns:
            ReviewResult
        """

    def _build_result(self, issues: list[ReviewIssue]) -> ReviewResult:
        """快捷方法：将问题列表包装成 ReviewResult 并生成统计摘要。"""
        summary = {
            "total_issues": len(issues),
            "by_type": {},
        }
        for issue in issues:
            summary["by_type"].setdefault(issue.issue_type, 0)
            summary["by_type"][issue.issue_type] += 1

        return ReviewResult(
            reviewer_id=self.reviewer_id,
            reviewer_name=self.reviewer_name,
            level=self.level,
            issues=issues,
            summary=summary,
        )
