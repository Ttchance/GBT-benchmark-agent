from __future__ import annotations

# -*- coding: utf-8 -*-
"""
工具函数模块
"""

import re

from core.document_parser import AutoDocumentParser


def load_document(file_path: str, llm_client=None):
    """
    根据文件后缀自动选择解析器，返回 ParsedDocument。

    Args:
        file_path: 文档路径（.docx 或 .pdf）
        llm_client: BaseLLMClient 实例（PDF 解析时用于 LLM/视觉结构化）

    Returns:
        ParsedDocument
    """
    parser = AutoDocumentParser(llm_client=llm_client)
    return parser.parse(file_path)


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """
    初始化日志配置。

    Args:
        level: 日志级别（"DEBUG" / "INFO" / "WARNING" / "ERROR"）
        log_file: 日志文件路径（None 表示仅输出到控制台）
    """
    import logging
    import sys

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=handlers,
    )


def chunk_text(text: str, max_length: int = 2000) -> list[str]:
    """
    将长文本按句子边界切分为若干块，避免超过 LLM 上下文限制。

    Args:
        text: 原始文本
        max_length: 每块最大字符数

    Returns:
        文本块列表
    """
    if len(text) <= max_length:
        return [text]

    sentences = re.split(r"(?<=[。！？.!?；;])", text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) <= max_length:
            current += sentence
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_length:
            current = sentence
            continue
        chunks.extend(
            sentence[i:i + max_length]
            for i in range(0, len(sentence), max_length)
        )
        current = ""

    if current:
        chunks.append(current)

    return chunks
