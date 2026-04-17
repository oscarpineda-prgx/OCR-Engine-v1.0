from pathlib import Path

from openpyxl import Workbook

from app.services.extraction.structured_documents import (
    _extract_text_from_openpyxl_workbook,
    extract_text_from_structured_document,
)


def test_extract_text_from_xlsx_flattens_sheets_and_rows(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Proveedores"
    sheet.append(["Razón Social", "RFC"])
    sheet.append(["VICTOR MANUEL BACA APODACA", "BAAV7822025P7"])

    second = workbook.create_sheet("Resumen")
    second.append(["Tipo de Documento", "Convenio Entrega Local"])

    file_path = tmp_path / "proveedores.xlsx"
    workbook.save(file_path)

    extracted = _extract_text_from_openpyxl_workbook(str(file_path))

    assert "Hoja: Proveedores" in extracted
    assert "Razón Social | RFC" in extracted
    assert "VICTOR MANUEL BACA APODACA | BAAV7822025P7" in extracted
    assert "Hoja: Resumen" in extracted
    assert "Tipo de Documento | Convenio Entrega Local" in extracted


def test_dispatches_docx_to_docx_extractor(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "app.services.extraction.structured_documents._extract_text_from_docx",
        lambda _: "docx text",
    )

    assert extract_text_from_structured_document(str(file_path)) == "docx text"


def test_dispatches_doc_to_word_com_extractor(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.doc"
    file_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "app.services.extraction.structured_documents._extract_text_from_doc_via_word",
        lambda _: "doc text",
    )

    assert extract_text_from_structured_document(str(file_path)) == "doc text"


def test_dispatches_xlsb_to_tabular_engine(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.xlsb"
    file_path.write_bytes(b"placeholder")

    observed: list[tuple[str, str]] = []

    def _fake_tabular(path: str, engine: str) -> str:
        observed.append((path, engine))
        return "xlsb text"

    monkeypatch.setattr(
        "app.services.extraction.structured_documents._extract_text_from_tabular_workbook",
        _fake_tabular,
    )

    result = extract_text_from_structured_document(str(file_path))

    assert result == "xlsb text"
    assert observed == [(str(file_path), "pyxlsb")]


def test_dispatches_msg_to_msg_extractor(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.msg"
    file_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "app.services.extraction.structured_documents._extract_text_from_msg",
        lambda _: "mail body",
    )

    assert extract_text_from_structured_document(str(file_path)) == "mail body"


def test_returns_empty_string_when_structured_extraction_fails(monkeypatch, tmp_path):
    file_path = tmp_path / "sample.docx"
    file_path.write_bytes(b"placeholder")

    def _explode(_: str) -> str:
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.services.extraction.structured_documents._extract_text_from_docx",
        _explode,
    )

    assert extract_text_from_structured_document(str(file_path)) == ""


def test_unsupported_structured_extension_raises_value_error(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_bytes(b"placeholder")

    try:
        extract_text_from_structured_document(str(file_path))
    except ValueError as exc:
        assert "Unsupported structured source type" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("ValueError was expected for unsupported extension")
