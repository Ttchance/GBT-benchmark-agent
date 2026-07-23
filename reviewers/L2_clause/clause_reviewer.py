from __future__ import annotations

# -*- coding: utf-8 -*-
"""
L2 条款级审查器
  - C2.1 语气审查
  - C2.2 术语审查
  - C2.3 引用审查
"""

from core.base_reviewer import BaseReviewer, ReviewIssue, ReviewResult
from core.document_parser import ParsedDocument
from core.llm_client import BaseLLMClient


# ═════════════════════════════════════════════════════════════════════════════
# C2.1 语气审查
# ═════════════════════════════════════════════════════════════════════════════

class ToneReviewer(BaseReviewer):
    """
    C2.1 语气审查

    流程：整段文本输入 → 模型一次性完成全链路分析 → 结果聚合 → 报告输出

    全链路分析包括：
      ① 条款识别
      ② 助动词定位（应/宜/可/不应/不宜 等）
      ③ 语义意图判断（强制/推荐/允许）
      ④ 匹配判断与错误诊断
    """

    reviewer_id = "C2.1"
    reviewer_name = "语气审查"
    level = "L2_clause"

    # GB/T 1.1 助动词体系
    MODAL_WORDS = {
        "强制": ["应", "不应", "必须", "禁止"],
        "推荐": ["宜", "不宜"],
        "允许": ["可", "不必"],
    }

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        for para in document.paragraphs:
            # TODO: Step 1 - 整段提取（含章节编号、标题、完整段落文本）
            # TODO: Step 2 - 整段送入 LLM 进行全链路分析
            # raw = self.llm.chat(self._build_messages(para))
            # parsed = self.llm.parse_json_response(raw)
            # TODO: Step 3 - 结果解析与聚合（容错处理）
            # issues.extend(self._parse_issues(parsed, para))
            pass

        issues = self._aggregate(issues)
        return self._build_result(issues)

    def _build_messages(self, para) -> list[dict]:
        """构造整段送审的 Prompt。"""
        # TODO: 实现
        raise NotImplementedError

    def _parse_issues(self, llm_output: dict, para) -> list[ReviewIssue]:
        """解析 LLM 返回的 JSON 结果为 ReviewIssue 列表。"""
        # TODO: 按 clause_id / modal_word / intent / is_correct 等字段解析
        raise NotImplementedError

    def _aggregate(self, issues: list[ReviewIssue]) -> list[ReviewIssue]:
        """按错误类型统计问题频次（错误聚合）。"""
        # TODO: 实现
        return issues


# ═════════════════════════════════════════════════════════════════════════════
# C2.2 术语审查
# ═════════════════════════════════════════════════════════════════════════════

class TermReviewer(BaseReviewer):
    """
    C2.2 术语审查

    流程：术语定义提取 → 术语出现位置识别 → 使用语境分析 → 一致性判断 → 定位输出

    问题类型：外延扩大 / 外延缩小 / 概念偷换 / 混用近义词 / 否定误用
    评测指标：Detection-F1 + Location-Acc
    """

    reviewer_id = "C2.2"
    reviewer_name = "术语审查"
    level = "L2_clause"

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        # TODO: Step 1 - 从第 3 章解析结构化术语定义表
        # terms = self._extract_term_definitions(document)

        # TODO: Step 2 - 遍历正文，识别每处术语出现位置（含上下文窗口 前后各2句）
        # occurrences = self._find_term_occurrences(document, terms)

        # TODO: Step 3 - 对每处术语进行语义分析
        # for occ in occurrences:
        #     result = self.llm.chat(self._build_messages(occ))
        #     issues.extend(self._parse_issues(result, occ))

        # TODO: Step 4 - 合并处理，按问题类型聚合

        return self._build_result(issues)

    def _extract_term_definitions(self, document: ParsedDocument) -> list[dict]:
        """从第 3 章提取结构化术语定义表。"""
        # 返回格式：[{term, en_term, definition, constraints}, ...]
        raise NotImplementedError

    def _find_term_occurrences(self, document: ParsedDocument, terms: list[dict]) -> list[dict]:
        """遍历正文，识别每处术语出现位置，记录上下文窗口。"""
        raise NotImplementedError

    def _build_messages(self, occurrence: dict) -> list[dict]:
        """为单次术语使用构造审查 Prompt。"""
        raise NotImplementedError

    def _parse_issues(self, llm_output: dict, occurrence: dict) -> list[ReviewIssue]:
        """解析 LLM 返回结果。"""
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════════════════
# C2.3 引用审查
# ═════════════════════════════════════════════════════════════════════════════

class ReferenceReviewer(BaseReviewer):
    """
    C2.3 引用审查

    流程：清单提取 → 正文引用扫描 → 双向比对 → 有效性核查 → 诊断输出

    问题分类：漏列 / 冗余 / 版本过期
    评测指标：Detection-F1 + Diagnosis-Acc + Location-Acc
    """

    reviewer_id = "C2.3"
    reviewer_name = "引用审查"
    level = "L2_clause"

    def __init__(self, llm_client: BaseLLMClient, config: dict | None = None):
        super().__init__(config)
        self.llm = llm_client

    def review(self, document: ParsedDocument) -> ReviewResult:
        issues: list[ReviewIssue] = []

        # TODO: Step 1 - 从第 2 章解析结构化引用清单
        # ref_list = self._extract_reference_list(document)

        # TODO: Step 2 - 遍历正文，识别每一处标准引用行为
        # body_refs = self._scan_body_references(document)

        # TODO: Step 3 - 双向比对（漏列 & 冗余）
        # issues.extend(self._cross_check(ref_list, body_refs, document))

        # TODO: Step 4 - 核查现行状态（有效/已废止/已被替代）
        # issues.extend(self._check_validity(ref_list))

        return self._build_result(issues)

    def _extract_reference_list(self, document: ParsedDocument) -> list[dict]:
        """从第 2 章提取引用清单。"""
        # 返回格式：[{standard_id, year, title, is_dated}, ...]
        raise NotImplementedError

    def _scan_body_references(self, document: ParsedDocument) -> list[dict]:
        """遍历正文识别引用行为。"""
        raise NotImplementedError

    def _cross_check(self, ref_list: list[dict], body_refs: list[dict], document: ParsedDocument) -> list[ReviewIssue]:
        """双向比对，检出漏列和冗余。"""
        raise NotImplementedError

    def _check_validity(self, ref_list: list[dict]) -> list[ReviewIssue]:
        """核查每条引用的现行状态。"""
        raise NotImplementedError
