# -*- coding: utf-8 -*-
"""
反例构造独立运行脚本

用法：
    # 单文件模式
    python run_counter_example.py --file data/data_pdf/GBT+47024.2-2026.pdf
    python run_counter_example.py --file data/data_pdf/GBT+47024.2-2026.pdf --dims C2.1 C3.1 C3.3

    # 批量目录模式
    python run_counter_example.py --dir data/data_pdf/
    python run_counter_example.py --dir data/data_pdf/ --dims C2.1 C3.1 C3.2 C3.3

    # 公共参数
    --output / -o  输出目录（默认 data/data_test）
    --backend / -b LLM 后端：proxy 或 azure（默认 azure）
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable, Optional

# 将项目根目录加入 sys.path，确保各模块可正常导入
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import LLM_CONFIG, AZURE_LLM_CONFIG, Docling_Is_true
from core.llm_client import OpenAILLMClient, AzureLLMClient
from core.parse_pdf import parse_gbt_pdf, sections_to_text
from core.counter_example import CounterExamplePipeline, get_counter_example_error_types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 默认值
DEFAULT_PDF = "data/data_pdf/GBT+47024.2-2026.pdf"
DEFAULT_OUTPUT = "data/data_test"
DEFAULT_SOURCE_TEXT_OUTPUT = "data/source_text"
DEFAULT_PARSED_SOURCE_JSON = "parsed_source_text.json"
SOURCE_TEXT_JSON_NAME = "GBT_source_text.json"
ALL_DIMENSIONS = ["C2.1", "C2.2", "C3.1", "C3.2", "C3.3"]


def parse_args():
    parser = argparse.ArgumentParser(description="GB/T 标准反例构造脚本")

    # --file 与 --dir 互斥，至少指定一个
    src_group = parser.add_mutually_exclusive_group()
    src_group.add_argument("--file", "-f", default=DEFAULT_PDF, help="待处理的单个 PDF 路径")
    src_group.add_argument("--dir", "-D", default=None, help="批量处理：包含多个 PDF 的文件夹路径")

    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help="输出目录（默认 data/data_test）")
    parser.add_argument(
        "--parsed-source-json",
        default=None,
        help=(
            "解析正文汇总 JSON 输出路径；默认写到输入 PDF 目录下的 "
            f"{DEFAULT_PARSED_SOURCE_JSON}"
        ),
    )
    parser.add_argument(
        "--dims", "-d",
        nargs="*",
        default=ALL_DIMENSIONS,
        choices=ALL_DIMENSIONS,
        help=f"要生成的反例维度（默认全部）：{ALL_DIMENSIONS}",
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["proxy", "azure"],
        default="azure",
        help="LLM 后端：proxy（代理 OpenAI）或 azure（Azure OpenAI），默认 azure",
    )
    parser.add_argument(
        "--num", "-n",
        type=int,
        default=None,
        help="每个 PDF 要注入的反例总数；需大于等于 --dims 指定维度数量，默认等于本次选中维度数量",
    )
    parser.add_argument(
        "--docling",
        dest="Docling_Is_true",
        action="store_true",
        default=Docling_Is_true,
        help="使用 Docling 新解析路径解析 PDF",
    )
    parser.add_argument(
        "--no-docling",
        dest="Docling_Is_true",
        action="store_false",
        help="使用原有 PyMuPDF/LLM/regex 解析路径解析 PDF",
    )
    args = parser.parse_args()
    if not args.dims:
        parser.error("--dims 至少需要指定一个维度")
    if args.num is not None and args.num < len(args.dims):
        parser.error("--num 必须大于或等于 --dims 指定的维度数量")
    return args


def _select_dimensions_for_run(dims: list[str]) -> list[str]:
    """Use all requested dimensions so balancing can cover every enabled error type."""
    return list(dims)


def _allocate_error_counts(
    dims: list[str],
    num: int,
    balance_counts: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    """Allocate dimension counts by the least-used concrete error types."""
    if num < len(dims):
        raise ValueError(f"num={num} 小于维度数量 {len(dims)}")

    catalog = get_counter_example_error_types(dims)
    flat_error_types = [
        (dim, error_type)
        for dim, error_types in catalog.items()
        for error_type in error_types
    ]
    if not flat_error_types:
        return {dim: 0 for dim in dims}

    counts = {dim: 0 for dim in dims}
    local_counts = {error_type: 0 for _, error_type in flat_error_types}
    balance_counts = balance_counts or {}
    for _ in range(num):
        candidates = list(flat_error_types)
        random.shuffle(candidates)
        dim, error_type = min(
            candidates,
            key=lambda item: (
                balance_counts.get(item[1], 0) + local_counts.get(item[1], 0),
                counts.get(item[0], 0),
            ),
        )
        counts[dim] += 1
        local_counts[error_type] += 1
    return counts


_TOKEN_USAGE_KEYS = ("requests", "prompt_tokens", "completion_tokens", "total_tokens")


def _llm_usage_snapshot(llm) -> dict[str, int]:
    getter = getattr(llm, "get_usage_snapshot", None)
    if not callable(getter):
        return {key: 0 for key in _TOKEN_USAGE_KEYS}
    usage = getter() or {}
    return {key: int(usage.get(key, 0) or 0) for key in _TOKEN_USAGE_KEYS}


def _llm_usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: max(0, int(after.get(key, 0) or 0) - int(before.get(key, 0) or 0))
        for key in _TOKEN_USAGE_KEYS
    }


def build_source_text_record(
    doc,
    pdf_path: Path,
    record_index: int,
    token_usage: Optional[dict[str, int]] = None,
) -> dict:
    """将 ParsedDocument.sections 转为可写入 source_text JSON 的记录。"""
    record = {
        "record_index": record_index,
        "file_name": pdf_path.name,
        "file_stem": pdf_path.stem,
        "file_path": str(pdf_path),
        "parse_mode": doc.mode,
        "section_count": len(doc.sections),
        "source_text": sections_to_text(doc.sections),
    }
    if token_usage is not None:
        record["token_usage"] = token_usage
    return record


def write_source_text_record(f, record: dict, written: int) -> None:
    """向 GBT_source_text.json 的 documents 数组流式追加一条解析文本记录。"""
    if written > 0:
        f.write(",\n")
    record_str = json.dumps(record, ensure_ascii=False, indent=2)
    indented = "\n".join("    " + line for line in record_str.splitlines())
    f.write(indented)
    f.flush()


def process_one_pdf(
    pdf_path: Path,
    llm,
    dims: list[str],
    num: Optional[int] = None,
    Docling_Is_true: bool = False,
    parsed_source_writer: Optional[Callable[[dict], None]] = None,
    balance_error_type_counts: Optional[dict[str, int]] = None,
) -> dict:
    """处理单个 PDF：解析 → 构造反例 → 返回文档记录字典（不写文件）。"""
    stem = pdf_path.stem

    # ── 1. 解析 PDF ───────────────────────────────────────────────
    logger.info("[%s] 开始解析 PDF: %s", stem, pdf_path)
    parse_usage_before = _llm_usage_snapshot(llm)
    doc = parse_gbt_pdf(str(pdf_path), llm_client=llm, Docling_Is_true=Docling_Is_true)
    parse_usage_after = _llm_usage_snapshot(llm)
    parse_token_usage = _llm_usage_delta(parse_usage_before, parse_usage_after)
    logger.info("[%s] 解析完成，解析模式 %s，共 %d 个章节", stem, doc.mode, len(doc.sections))
    logger.info("[%s] 解析 token 消耗: %s", stem, parse_token_usage)
    parsed_source_text_record = build_source_text_record(
        doc,
        pdf_path,
        0,
        token_usage=parse_token_usage,
    )

    if parsed_source_writer is not None:
        parsed_source_writer(dict(parsed_source_text_record))

    # ── 2. 构造反例 ───────────────────────────────────────────────
    selected_dims = _select_dimensions_for_run(dims)
    extra_errors = random.randint(0, 3)
    target_num = (num if num is not None else len(selected_dims)) + extra_errors
    error_counts = _allocate_error_counts(
        selected_dims,
        target_num,
        balance_error_type_counts,
    )
    logger.info("[%s] 开始构造反例，传入维度: %s", stem, dims)
    logger.info("[%s] 本次启用的构造维度: %s", stem, selected_dims)
    logger.info("[%s] 本次计划注入 %d 处错误（随机追加 %d 处），各维度分配: %s", stem, target_num, extra_errors, error_counts)
    pipeline = CounterExamplePipeline(llm_client=llm)
    ce_doc = pipeline.run(
        doc.sections,
        dimensions=selected_dims,
        error_counts=error_counts,
        error_type_counts=balance_error_type_counts,
    )
    logger.info("[%s] 反例构造完成，共注入 %d 处错误", stem, len(ce_doc.examples))

    # ── 3. 打印标注摘要 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"[{stem}] 反例标注摘要")
    print(ce_doc.to_annotation())
    print("=" * 60)

    # ── 4. 构造并返回文档记录 ─────────────────────────────────────
    return {
        "file_name": pdf_path.name,
        "file_stem": stem,
        "parse_token_usage": parse_token_usage,
        "parsed_source_text_record": parsed_source_text_record,
        "source_text": sections_to_text(ce_doc.source_sections),
        "examples": [
            {
                "dimension": ex.dimension,
                "error_type": ex.error_type,
                "original": {
                    "section_number": ex.original.section_number,
                    "title": ex.original.title,
                    "content": ex.original.content,
                },
                "corrupted": {
                    "section_number": ex.corrupted.section_number,
                    "title": ex.corrupted.title,
                    "content": ex.corrupted.content,
                } if ex.corrupted is not None else None,
                "error_description": ex.error_description,
            }
            for ex in ce_doc.examples
        ],
    }


def parse_docling_source_text_record(pdf_path: Path, record_index: int) -> dict:
    """使用 Docling 解析 PDF，并返回 source_text JSON 记录。"""
    stem = pdf_path.stem
    logger.info("[%s] 开始使用 Docling 解析 PDF: %s", stem, pdf_path)
    doc = parse_gbt_pdf(str(pdf_path), llm_client=None, Docling_Is_true=True)
    logger.info(
        "[%s] Docling 解析完成，共 %d 个章节",
        stem,
        len(doc.sections),
    )
    return build_source_text_record(doc, pdf_path, record_index)


def add_record_stats(record: dict, record_index: int) -> dict:
    """给单条输出记录补充写入序号和反例条数。"""
    examples = record.get("examples") or []
    updated = {
        "record_index": record_index,
        "example_count": len(examples),
    }
    for key, value in record.items():
        if key not in updated:
            updated[key] = value
    return updated


def count_error_types(record: dict) -> dict[str, int]:
    """统计单条记录中每种 error_type 的数量。"""
    counts: dict[str, int] = {}
    for example in record.get("examples") or []:
        error_type = example.get("error_type") or "UNKNOWN"
        counts[error_type] = counts.get(error_type, 0) + 1
    return counts


def _default_parsed_source_json_path(args, pdf_files: list[Path]) -> Path:
    """解析正文汇总 JSON 默认写入输入 PDF 所在目录。"""
    if args.parsed_source_json:
        return Path(args.parsed_source_json)
    if args.dir:
        return Path(args.dir) / DEFAULT_PARSED_SOURCE_JSON
    return pdf_files[0].parent / DEFAULT_PARSED_SOURCE_JSON


def main():
    args = parse_args()
    logger.info("PDF 解析路径: %s", "Docling" if args.Docling_Is_true else "原有 PyMuPDF/LLM/regex")

    # ── 收集待处理的 PDF 列表 ──────────────────────────────────────
    if args.dir:
        pdf_dir = Path(args.dir)
        if not pdf_dir.is_dir():
            logger.error("目录不存在: %s", pdf_dir)
            sys.exit(1)
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        if not pdf_files:
            logger.error("目录下未找到任何 PDF 文件: %s", pdf_dir)
            sys.exit(1)
        logger.info("批量模式：共找到 %d 个 PDF 文件", len(pdf_files))
    else:
        pdf_path = Path(args.file)
        if not pdf_path.exists():
            logger.error("文件不存在: %s", pdf_path)
            sys.exit(1)
        pdf_files = [pdf_path]

    if args.Docling_Is_true:
        source_text_dir = Path(DEFAULT_SOURCE_TEXT_OUTPUT)
        source_text_dir.mkdir(parents=True, exist_ok=True)
        source_text_json_path = source_text_dir / SOURCE_TEXT_JSON_NAME
        written_source_text = 0
        with source_text_json_path.open("w", encoding="utf-8") as f:
            f.write(f'{{\n  "total_files": {len(pdf_files)},\n  "documents": [\n')
            for idx, pdf_path in enumerate(pdf_files, 1):
                logger.info("── Docling 解析 [%d/%d]: %s ──", idx, len(pdf_files), pdf_path.name)
                try:
                    record = parse_docling_source_text_record(pdf_path, written_source_text + 1)
                    write_source_text_record(f, record, written_source_text)
                    written_source_text += 1
                    logger.info("[%d/%d] 已写入第 %d 条解析文本", idx, len(pdf_files), written_source_text)
                except Exception as exc:
                    logger.error("[%d/%d] Docling 解析失败 %s: %s", idx, len(pdf_files), pdf_path.name, exc)
            f.write(f'\n  ],\n  "written": {written_source_text}\n}}\n')
        logger.info(
            "Docling 解析完成，共写入 %d/%d 条 source_text 记录，输出文件: %s",
            written_source_text,
            len(pdf_files),
            source_text_json_path,
        )
        return

    # ── 初始化 LLM 客户端 ─────────────────────────────────────────
    if args.backend == "azure":
        logger.info("使用 Azure OpenAI 后端")
        llm = AzureLLMClient(AZURE_LLM_CONFIG)
    else:
        logger.info("使用代理 OpenAI 后端")
        llm = OpenAILLMClient(LLM_CONFIG)

    out_dir = Path(args.output)
    parsed_source_json_path = _default_parsed_source_json_path(args, pdf_files)

    # ── 流式写入唯一输出文件 GBT_test.json ───────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed_source_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "GBT_test.json"
    source_text_dir = Path(DEFAULT_SOURCE_TEXT_OUTPUT)
    source_text_dir.mkdir(parents=True, exist_ok=True)
    source_text_json_path = source_text_dir / SOURCE_TEXT_JSON_NAME
    total = len(pdf_files)
    written = 0  # 成功写入条数
    source_written = 0
    source_text_written = 0
    total_examples = 0
    error_type_counts: dict[str, int] = {}
    balance_error_type_counts: dict[str, int] = {}

    with (
        output_path.open("w", encoding="utf-8") as f,
        parsed_source_json_path.open("w", encoding="utf-8") as parsed_source_f,
        source_text_json_path.open("w", encoding="utf-8") as source_text_f,
    ):
        # 写入文件头：total_files 用计划处理数量，documents 数组开始
        f.write(f'{{\n  "total_files": {total},\n  "documents": [\n')
        parsed_source_f.write(f'{{\n  "total_files": {total},\n  "documents": [\n')
        source_text_f.write(f'{{\n  "total_files": {total},\n  "documents": [\n')

        def write_parsed_source(source_record: dict) -> None:
            nonlocal source_written
            if source_written > 0:
                parsed_source_f.write(",\n")
            record_str = json.dumps(source_record, ensure_ascii=False, indent=2)
            indented = "\n".join("    " + line for line in record_str.splitlines())
            parsed_source_f.write(indented)
            parsed_source_f.flush()
            source_written += 1
            logger.info(
                "[%s] 解析正文已流式写入: %s",
                source_record.get("file_stem", ""),
                parsed_source_json_path,
            )

        for idx, pdf_path in enumerate(pdf_files, 1):
            logger.info("── 处理 [%d/%d]: %s ──", idx, total, pdf_path.name)
            try:
                record = process_one_pdf(
                    pdf_path,
                    llm,
                    args.dims,
                    args.num,
                    Docling_Is_true=args.Docling_Is_true,
                    parsed_source_writer=write_parsed_source,
                    balance_error_type_counts=balance_error_type_counts,
                )
                record = add_record_stats(record, written + 1)
                source_record = record.pop("parsed_source_text_record", None)
                if source_record is None:
                    source_record = {
                        "record_index": source_text_written + 1,
                        "file_name": record["file_name"],
                        "file_stem": record["file_stem"],
                        "parse_mode": "unknown",
                        "section_count": None,
                        "source_text": record["source_text"],
                    }
                else:
                    source_record["record_index"] = source_text_written + 1
                write_source_text_record(source_text_f, source_record, source_text_written)
                source_text_written += 1
                if written > 0:
                    f.write(",\n")
                # 将记录缩进两格后写入，与外层结构对齐
                record_str = json.dumps(record, ensure_ascii=False, indent=2)
                indented = "\n".join("    " + line for line in record_str.splitlines())
                f.write(indented)
                f.flush()  # 立即落盘，防止 OOM 时丢数据

                record_error_type_counts = count_error_types(record)
                for error_type, count in record_error_type_counts.items():
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + count
                total_examples += record["example_count"]
                written += 1
                logger.info(
                    "[%d/%d] 已写入第 %d 条数据，反例 %d 条（累计 %d 条），错误类型统计: %s",
                    idx,
                    total,
                    record["record_index"],
                    record["example_count"],
                    written,
                    record_error_type_counts,
                )
            except Exception as exc:
                logger.error("[%d/%d] 处理失败 %s: %s", idx, total, pdf_path.name, exc)

        # 写入文件尾
        f.write(f'\n  ],\n  "total_examples": {total_examples},\n')
        counts_str = json.dumps(
            dict(sorted(error_type_counts.items())),
            ensure_ascii=False,
            indent=2,
        )
        f.write('  "error_type_counts": ')
        f.write(counts_str.replace("\n", "\n  "))
        f.write("\n}\n")
        parsed_source_f.write(f'\n  ],\n  "total_documents": {source_written}\n')
        parsed_source_f.write("}\n")
        source_text_f.write(f'\n  ],\n  "written": {source_text_written}\n}}\n')

    logger.info(
        "全部完成，共写入 %d/%d 条，反例总数 %d，错误类型统计: %s，输出文件: %s，解析文本文件: %s",
        written,
        total,
        total_examples,
        dict(sorted(error_type_counts.items())),
        output_path,
        source_text_json_path,
    )
    logger.info("解析正文汇总 JSON 已写入: %s，共 %d 条", parsed_source_json_path, source_written)


if __name__ == "__main__":
    main()
