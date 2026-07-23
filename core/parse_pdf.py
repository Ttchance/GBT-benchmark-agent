# -*- coding: utf-8 -*-
"""
GB/T 推荐性国家标准 PDF 解析器

核心策略：
  1. 检测 PDF 文字层是否可正常解码
  2. 可解码 → 直接提取文本 → LLM 文本结构化
  3. 不可解码（方正书版私有字体等）→ 渲染为高清图片 → GPT-4o 视觉直读
  4. 未配置 LLM → 纯正则降级解析

输出统一为 Section(section_number, title, content) 列表。
"""

from __future__ import annotations

import base64
import io
import re
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """一个独立的章节单元"""
    section_number: Optional[str]   # "1"  "4.1.2"  None(前言等无编号章节)
    title: str                       # "范围"  "规范性引用文件"
    content: str                     # 原文正文，不加工

    def to_dict(self) -> dict:
        return {
            "section_number": self.section_number,
            "title": self.title,
            "content": self.content,
        }


@dataclass
class ParsedDocument:
    """解析结果"""
    file_path: str
    title: str
    sections: list[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    mode: str = ""    # "vision" / "text_llm" / "regex"

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.sections]


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一个专业的文档结构化分析与数据抽取专家，擅长 GB/T 国家推荐性标准文档。

任务：识别章节层级结构，抽取为 JSON 数组，每个元素包含：
- "section_number": 章节编号（如 "1", "4.1.2"），无编号设 null
- "title": 章节完整标题
- "content": 该章节正文原文

严格规则：
1. content 必须原文一字不差，不得润色、缩写或概括
2. 忽略页码、页眉（如"GB/T 1.1—2020"）、页脚
3. 章节边界：持续到下一个同级或上级标题为止
4. 直接输出 JSON 数组，禁止输出任何思考过程、解释或额外文字"""

_VISION_USER_PROMPT = """\
以下是 GB/T 标准文档的页面图片。请直接阅读图片中的文字，按章节结构抽取为 JSON 数组。

注意：
- 如果某个章节从本页开始但未在本页结束（跨页），请在 content 末尾标注 "…[续]"
- 如果某个章节从上一页延续到本页（无新标题开头），请创建一个 section_number 和 title 与上页相同的条目，content 只包含本页的延续内容，并在开头标注 "[续]…"
- 不要遗漏任何章节或段落内容

请输出 JSON 数组："""

_TEXT_USER_PROMPT = """\
以下是从 GB/T 标准 PDF 提取的原始文本。请按章节结构抽取为 JSON 数组。

注意：
- 如果某个章节在文本末尾被截断（跨块），请在 content 末尾标注 "…[续]"
- 如果文本开头是上一块的延续内容（无标题），请创建一个 section_number 为 null、title 为 "__continued__" 的条目

原始文本：
---
{raw_text}
---

请输出 JSON 数组："""

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

_MUST_HAVE_WORDS = ["前言", "引言", "范围", "规范性", "标准", "要求", "术语"]

_NUMBERED_HEADING_RE = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2})*)\s+([\u4e00-\u9fffA-Za-z].{0,60})$"
)
_UNNUMBERED_HEADING_RE = re.compile(
    r"^(前\s*言|引\s*言|参\s*考\s*文\s*献|索\s*引|附\s*录\s*[A-Z].{0,30})$"
)
_TOC_LINE_RE = re.compile(r"[…\.]{3,}\s*\d+\s*$")
_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")
_HEADER_FOOTER_RE = re.compile(
    r"^(?:GB/?T\s*\d+(?:\.\d+)?(?:[-—]\d{4})?|ICS\s+\S+|CCS\s+.+)\s*$|版权|©|All rights reserved"
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
_BRACKET_HEADING_RE = re.compile(r"^\[(?P<num>[^\]]+)\]\s*(?P<title>.+?)\s*$")
_PLAIN_SECTION_HEADING_RE = re.compile(
    r"^(?P<num>\d{1,2}(?:\.\d{1,2})*)\s+(?P<title>[\u4e00-\u9fffA-Za-z（(].{0,80})$"
)
_APPENDIX_HEADING_RE = re.compile(r"^(?P<num>附录\s*[A-Z])\s*(?P<title>.*)$")


def _is_garbled(text: str) -> bool:
    """检测 PDF 文字层是否乱码（方正书版私有字体等）。"""
    return not any(w in text for w in _MUST_HAVE_WORDS)


def _render_page_to_png(page, dpi: int = 200) -> bytes:
    """将 fitz.Page 渲染为 PNG 字节。"""
    import fitz
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


def _stitch_images_vertical(images_bytes: list[bytes], gap: int = 20) -> bytes:
    """将多张 PNG 图片纵向拼接为一张，减少 API 调用次数。"""
    from PIL import Image

    imgs = [Image.open(io.BytesIO(b)) for b in images_bytes]
    max_w = max(img.width for img in imgs)
    total_h = sum(img.height for img in imgs) + gap * (len(imgs) - 1)

    canvas = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for img in imgs:
        canvas.paste(img, (0, y))
        y += img.height + gap

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if _PAGE_NUMBER_RE.match(s):
        return True
    if _HEADER_FOOTER_RE.search(s):
        return True
    return False


def _fix_broken_section_numbers(text: str) -> str:
    """将 PDF 提取时被换行切断的章节编号拼接回来。
    例如：'5.\n1 标题' → '5.1 标题'
         '5.\n1.\n2 标题' → '5.1.2 标题'
    """
    pattern = re.compile(r'(\d+\.)\n(\d)')
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(r'\1\2', text)
    return text


def _preprocess_text(text: str) -> str:
    """轻量去噪：修复断行编号、移除页码、页眉/页脚行。"""
    text = _fix_broken_section_numbers(text)
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if not _is_noise_line(l)]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主解析器
# ---------------------------------------------------------------------------

class GBTPdfParser:
    """
    GB/T 标准 PDF 解析器。

    用法::

        from config.settings import LLM_CONFIG
        from core.llm_client import OpenAILLMClient

        llm = OpenAILLMClient(LLM_CONFIG)
        parser = GBTPdfParser(llm_client=llm)
        doc = parser.parse("data/data_pdf/GBT+1.1-2020.pdf")

        for s in doc.sections:
            print(f"[{s.section_number}] {s.title}")
            print(s.content[:200])
    """

    def __init__(
        self,
        llm_client=None,
        vision_dpi: int = 300,
        pages_per_chunk: int = 1,
    ):
        """
        Args:
            llm_client      : BaseLLMClient 实例（支持 chat / chat_with_image）
            vision_dpi      : 视觉模式渲染分辨率
            pages_per_chunk : 每次送入 LLM 的页数（视觉模式拼图 / 文本模式分块）
        """
        self._llm = llm_client
        self._dpi = vision_dpi
        self._chunk_size = pages_per_chunk

    def parse(self, file_path: Union[str, Path]) -> ParsedDocument:
        """解析 GB/T PDF，返回 ParsedDocument。"""
        import fitz

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        doc = fitz.open(str(path))
        metadata = {
            "title": (doc.metadata or {}).get("title", ""),
            "author": (doc.metadata or {}).get("author", ""),
            "page_count": doc.page_count,
        }

        # 判断文字层是否可用
        sample = "".join(page.get_text("text") for page in list(doc)[:5])
        text_ok = not _is_garbled(sample)

        # 选择解析路径
        if self._llm is not None:
            if text_ok:
                mode = "text_llm"
                sections = self._parse_text_llm(doc)
            else:
                mode = "vision"
                sections = self._parse_vision(doc)
        else:
            mode = "regex"
            logger.warning("未配置 LLM，降级正则解析（精度有限）")
            sections = self._parse_regex(doc, text_ok)

        doc.close()

        return ParsedDocument(
            file_path=str(path),
            title=metadata["title"] or path.stem,
            sections=sections,
            metadata=metadata,
            mode=mode,
        )

    def parse_docling(self, file_path: Union[str, Path]) -> ParsedDocument:
        """使用 Docling 解析 PDF，并转换为项目统一的 Section 列表。"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")

        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ImportError(
                "当前 Python 环境未能导入 docling，无法使用 Docling 解析路径。"
                f"当前解释器: {sys.executable}。"
                "请在同一个环境中运行: python3 -m pip install docling。"
                f"原始错误: {exc}"
            ) from exc

        logger.info("使用 Docling 解析 PDF: %s", path)
        converter = DocumentConverter()
        result = converter.convert(str(path))
        markdown = result.document.export_to_markdown()
        sections = self._parse_docling_markdown(markdown)

        return ParsedDocument(
            file_path=str(path),
            title=path.stem,
            sections=sections,
            metadata={"parser": "docling"},
            mode="docling",
        )

    # ------------------------------------------------------------------
    # 路径 A：视觉模式（乱码 PDF 专用，GPT-4o 直读图片）
    # ------------------------------------------------------------------

    def _parse_vision(self, doc) -> list[Section]:
        """渲染页面为图片 → 拼接 → GPT-4o 视觉直读 → 结构化。"""
        page_count = doc.page_count
        all_sections: list[Section] = []

        # 按 chunk_size 分组
        for start in range(0, page_count, self._chunk_size):
            end = min(start + self._chunk_size, page_count)
            page_range = f"{start + 1}-{end}"
            logger.info("视觉解析 第 %s 页（共 %d 页）", page_range, page_count)

            # 渲染各页
            page_pngs = []
            for i in range(start, end):
                page_pngs.append(_render_page_to_png(doc[i], self._dpi))

            # 拼接为一张图
            if len(page_pngs) == 1:
                composite = page_pngs[0]
            else:
                composite = _stitch_images_vertical(page_pngs)

            # 调用视觉模型
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _VISION_USER_PROMPT},
            ]
            try:
                response = self._llm.chat_with_image(
                    messages, composite, max_tokens=16384
                )
                raw = self._llm.parse_json_response(response)
            except Exception as exc:
                logger.error("视觉 LLM 调用失败 (第 %s 页): %s", page_range, exc)
                continue

            chunk_sections = self._parse_section_list(raw)
            self._merge_continued(all_sections, chunk_sections)

        return all_sections

    # ------------------------------------------------------------------
    # 路径 B：文本 LLM（字体正常时，直接提取文字 → LLM 结构化）
    # ------------------------------------------------------------------

    def _parse_text_llm(self, doc) -> list[Section]:
        """直接提取文字 → 分块 → LLM 文本结构化。"""
        page_count = doc.page_count
        all_sections: list[Section] = []

        for start in range(0, page_count, self._chunk_size):
            end = min(start + self._chunk_size, page_count)
            page_range = f"{start + 1}-{end}"
            logger.info("文本 LLM 结构化 第 %s 页（共 %d 页）", page_range, page_count)

            chunk_text = ""
            for i in range(start, end):
                page_text = _preprocess_text(doc[i].get_text("text"))
                chunk_text += page_text + "\n\n"

            if not chunk_text.strip():
                continue

            prompt = _TEXT_USER_PROMPT.format(raw_text=chunk_text)
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ]
            try:
                response = self._llm.chat(messages, temperature=0.0, max_tokens=16384)
                raw = self._llm.parse_json_response(response)
            except Exception as exc:
                logger.error("文本 LLM 调用失败 (第 %s 页): %s", page_range, exc)
                continue

            chunk_sections = self._parse_section_list(raw)
            
            self._merge_continued(all_sections, chunk_sections)
        full_text = sections_to_text(all_sections)
        return all_sections

    # ------------------------------------------------------------------
    # 路径 C：纯正则降级
    # ------------------------------------------------------------------

    def _parse_regex(self, doc, text_ok: bool) -> list[Section]:
        """无 LLM 时纯正则解析，精度有限。"""
        sections: list[Section] = []
        current: Optional[Section] = None
        content_lines: list[str] = []
        in_toc = False

        def _flush():
            nonlocal content_lines
            if current is not None:
                current.content = "\n".join(content_lines).strip()
            content_lines = []

        for page in doc:
            if text_ok:
                text = page.get_text("text")
            else:
                # 无 LLM + 乱码 → 尝试 OCR
                try:
                    text = self._ocr_page(page)
                except Exception:
                    text = page.get_text("text")

            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line or _is_noise_line(line):
                    continue
                if re.match(r"^目\s*次\s*$", line):
                    in_toc = True
                    continue
                if _TOC_LINE_RE.search(line):
                    continue

                nm = _NUMBERED_HEADING_RE.match(line)
                um = _UNNUMBERED_HEADING_RE.match(line)
                is_heading = bool(nm or um)

                if in_toc:
                    if is_heading:
                        in_toc = False
                    else:
                        continue

                if is_heading:
                    _flush()
                    num = nm.group(1) if nm else None
                    title = nm.group(2).strip() if nm else line
                    current = Section(section_number=num, title=title, content="")
                    sections.append(current)
                else:
                    content_lines.append(line)

        _flush()
        return sections

    @staticmethod
    def _ocr_page(page, lang="chi_sim+eng", dpi=200) -> str:
        import pytesseract
        from PIL import Image
        import fitz

        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        return pytesseract.image_to_string(img, lang=lang, config="--psm 3")

    # ------------------------------------------------------------------
    # 公共辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_section_list(raw) -> list[Section]:
        """将 LLM 返回的 JSON 解析为 Section 列表。"""
        if not isinstance(raw, list):
            logger.warning("LLM 返回非列表: %s", type(raw))
            return []

        sections = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            num = item.get("section_number")
            title = str(item.get("title", "")).strip()
            content = str(item.get("content", "")).strip()
            if not title:
                continue
            sections.append(Section(
                section_number=num if num else None,
                title=title,
                content=content,
            ))
        return GBTPdfParser._split_nested_sections(sections)

    @staticmethod
    def _parse_docling_markdown(markdown: str) -> list[Section]:
        """将 Docling 导出的 Markdown 切分为 Section 列表。"""
        sections: list[Section] = []
        current: Optional[Section] = None
        content_lines: list[str] = []

        def _flush() -> None:
            nonlocal current, content_lines
            if current is not None:
                current.content = "\n".join(content_lines).strip()
                sections.append(current)
            content_lines = []

        def _clean_heading(text: str) -> str:
            text = re.sub(r"\s+", " ", text).strip()
            text = text.strip("#").strip()
            return text

        def _heading_to_section(text: str) -> Optional[Section]:
            heading = _clean_heading(text)
            if not heading:
                return None

            bracket_match = _BRACKET_HEADING_RE.match(heading)
            if bracket_match:
                num = bracket_match.group("num").strip()
                title = bracket_match.group("title").strip()
                return Section(section_number=num or None, title=title, content="")

            plain_match = _PLAIN_SECTION_HEADING_RE.match(heading)
            if plain_match:
                return Section(
                    section_number=plain_match.group("num").strip(),
                    title=plain_match.group("title").strip(),
                    content="",
                )

            appendix_match = _APPENDIX_HEADING_RE.match(heading)
            if appendix_match:
                title = appendix_match.group("title").strip() or heading
                return Section(
                    section_number=appendix_match.group("num").replace(" ", ""),
                    title=title,
                    content="",
                )

            if re.fullmatch(r"前\s*言|引\s*言|目\s*次|参考文献|索\s*引", heading):
                return Section(section_number=None, title=re.sub(r"\s+", "", heading), content="")

            return None

        for raw_line in markdown.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                if current is not None and content_lines and content_lines[-1] != "":
                    content_lines.append("")
                continue

            md_match = _MARKDOWN_HEADING_RE.match(stripped)
            candidate = _heading_to_section(md_match.group(1) if md_match else stripped)
            if candidate is not None:
                _flush()
                current = candidate
                continue

            if current is None:
                current = Section(section_number=None, title="文档标题", content="")
            content_lines.append(line)

        _flush()
        return GBTPdfParser._split_nested_sections(sections)

    @staticmethod
    def _split_nested_sections(sections: list[Section]) -> list[Section]:
        """
        后处理：将 LLM 未拆分的嵌套子节从 content 中提取出来，
        生成独立的 Section 并插入到列表中。

        例如 section_number='5' 的 content 中包含：
            '5.1 试验要求\n内容...\n5.2 试品状态\n内容...'
        将被拆分为独立的 Section(5.1)、Section(5.2)，
        原 Section(5) 的 content 设为空。
        """
        # 匹配形如 "5.1 标题" 或 "5.1.1 标题" 的行
        _SUB_HEADING = re.compile(
            r'^(\d{1,2}(?:\.\d{1,2})+)\s+([\u4e00-\u9fffA-Za-z（(].{0,80})$'
        )

        result = []
        for section in sections:
            if not section.content:
                result.append(section)
                continue

            lines = section.content.splitlines()
            # 找出 content 中所有子节标题的行索引
            heading_positions = []
            for i, line in enumerate(lines):
                m = _SUB_HEADING.match(line.strip())
                if m:
                    num = m.group(1)
                    # 只拆分比当前 section 编号层级更深的子节
                    parent_num = section.section_number or ""
                    if parent_num and not num.startswith(parent_num + "."):
                        continue
                    heading_positions.append((i, num, m.group(2).strip()))

            if not heading_positions:
                result.append(section)
                continue

            # 父节 content 取第一个子节标题之前的内容
            parent_content = "\n".join(lines[:heading_positions[0][0]]).strip()
            result.append(Section(
                section_number=section.section_number,
                title=section.title,
                content=parent_content,
            ))

            # 逐段切出每个子节
            for idx, (pos, num, title) in enumerate(heading_positions):
                end = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
                sub_content = "\n".join(lines[pos + 1:end]).strip()
                result.append(Section(
                    section_number=num,
                    title=title,
                    content=sub_content,
                ))

        return result

    @staticmethod
    def _merge_continued(existing: list[Section], new_chunk: list[Section]):
        """
        跨块章节合并：

        1. new_chunk 首条是 __continued__ → 追加到 existing 末尾 section 的 content
        2. new_chunk 首条与 existing 末条 number+title 相同 → 合并 content
        3. content 中的 "…[续]" / "[续]…" 标记自动清理
        """
        if not new_chunk:
            return

        if existing:
            last = existing[-1]
            first = new_chunk[0]

            should_merge = False
            if first.title == "__continued__":
                should_merge = True
            elif (first.section_number == last.section_number
                  and first.title == last.title):
                should_merge = True

            if should_merge:
                # 清理续接标记
                last_content = re.sub(r"\s*…\[续\]\s*$", "", last.content)
                first_content = re.sub(r"^\s*\[续\]…\s*", "", first.content)
                last.content = (last_content + "\n" + first_content).strip()
                new_chunk = new_chunk[1:]

        existing.extend(new_chunk)


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

def parse_gbt_pdf(
    file_path: Union[str, Path],
    llm_client=None,
    vision_dpi: int = 300,
    pages_per_chunk: int = 1,
    Docling_Is_true: bool = False,
) -> ParsedDocument:
    """
    解析 GB/T 标准 PDF，返回结构化章节列表。

    Args:
        file_path       : PDF 路径
        llm_client      : BaseLLMClient 实例（推荐，不传则降级正则）
        vision_dpi      : 视觉模式渲染 DPI
        pages_per_chunk : 每次送入 LLM 的页数
        Docling_Is_true : True 时使用 Docling 新解析路径，False 时使用原解析路径

    Returns:
        ParsedDocument

    Example::

        from config.settings import LLM_CONFIG
        from core.llm_client import OpenAILLMClient
        from core.parse_pdf import parse_gbt_pdf

        llm = OpenAILLMClient(LLM_CONFIG)
        doc = parse_gbt_pdf("GBT_1.1-2020.pdf", llm_client=llm)

        for s in doc.sections:
            print(f"[{s.section_number}] {s.title}")
            print(s.content[:200])
    """
    parser = GBTPdfParser(
        llm_client=llm_client,
        vision_dpi=vision_dpi,
        pages_per_chunk=pages_per_chunk,
    )
    if Docling_Is_true:
        return parser.parse_docling(file_path)
    return parser.parse(file_path)


def sections_to_text(sections: list[Section]) -> str:
    """
    将 Section 列表拼接为一个字符串，各章节之间用空行隔开。

    Args:
        sections : Section 列表（ParsedDocument.sections）

    Returns:
        包含所有章节文本的字符串
    """
    lines = []
    for section in sections:
        heading = f"[{section.section_number}] {section.title}" if section.section_number else section.title
        lines.append(heading)
        if section.content:
            lines.append(section.content)
        lines.append("")  # 章节间空行
    return "\n".join(lines)


def save_sections_to_txt(sections: list[Section], output_path: Union[str, Path]) -> None:
    """
    将 Section 列表保存为纯文本文件，各章节之间用空行隔开。

    Args:
        sections    : Section 列表（ParsedDocument.sections）
        output_path : 输出文件路径
    """
    lines = []
    for section in sections:
        heading = f"[{section.section_number}] {section.title}" if section.section_number else section.title
        lines.append(heading)
        if section.content:
            lines.append(section.content)
        lines.append("")  # 章节间空行

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
