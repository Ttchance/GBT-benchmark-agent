from pathlib import Path

from core.parse_pdf import GBTPdfParser, GBTParsedDocument


def test_gbt_pdf_parser_parses_sample_pdf():
    sample_pdf = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "data_pdf"
        / "GBT+1.1-2020.pdf"
    )

    parser = GBTPdfParser(ocr_fallback=False)
    document = parser.parse(sample_pdf)

    assert isinstance(document, GBTParsedDocument)
    assert document.file_path == str(sample_pdf)
    assert document.metadata["page_count"] > 0
    assert document.used_ocr is False
    assert document.sections

    paragraphs = document.to_paragraphs()
    assert paragraphs
    assert all("text" in row and row["text"].strip() for row in paragraphs)
    assert any(row["section_title"] == "范围" for row in paragraphs)
