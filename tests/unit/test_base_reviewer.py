# -*- coding: utf-8 -*-
"""
单元测试示例：BaseReviewer
"""
import pytest
from core.base_reviewer import BaseReviewer, ReviewIssue, ReviewResult


class _DummyReviewer(BaseReviewer):
    """用于测试的最简实现"""
    reviewer_id = "TEST"
    reviewer_name = "测试审查器"
    level = "L1_format"

    def review(self, document):
        return self._build_result([])


def test_build_result_empty():
    reviewer = _DummyReviewer()
    result = reviewer._build_result([])
    assert isinstance(result, ReviewResult)
    assert result.total_issues == 0


def test_build_result_with_issues():
    reviewer = _DummyReviewer()
    issues = [
        ReviewIssue("TEST", "第1章", "字体错误", "字体不符合规定", "请改为宋体"),
        ReviewIssue("TEST", "第2章", "字体错误", "字体不符合规定", "请改为宋体"),
        ReviewIssue("TEST", "第3章", "字号错误", "字号不符合规定", "请改为小四"),
    ]
    result = reviewer._build_result(issues)
    assert result.summary["total_issues"] == 3
    assert result.summary["by_type"]["字体错误"] == 2
    assert result.summary["by_type"]["字号错误"] == 1
