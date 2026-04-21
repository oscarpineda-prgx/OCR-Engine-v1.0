from app.db.models import Batch, Document
from app.services.exports import batch_export
from app.services.exports.batch_export import (
    EXPORT_HEADERS,
    build_batch_export_rows,
    excel_hyperlink_formula,
)


def test_excel_hyperlink_formula_escapes_path_quotes():
    assert (
        excel_hyperlink_formula(r'\\amer.prgx.com\images\OxxoMex\Link 2026\A "B"\file.pdf')
        == r'=HYPERLINK("\\amer.prgx.com\images\OxxoMex\Link 2026\A ""B""\file.pdf")'
    )


def test_build_batch_export_rows_adds_link_before_filename(monkeypatch):
    monkeypatch.setattr(
        batch_export,
        "load_link2026_source_paths",
        lambda batch_key: {10: r"\\amer.prgx.com\images\OxxoMex\Link 2026\Proveedor\archivo.pdf"},
    )

    batch = Batch(id=1, batch_key="BATCH-20260420-120000-abcdef", status="completed")
    document = Document(
        id=10,
        batch_id=1,
        filename="archivo.pdf",
        source_type="pdf",
        route="ocr_image",
        processing_route="ocr",
        status="processed",
        file_path=r"data\incoming\BATCH-20260420-120000-abcdef\uuid_archivo.pdf",
    )

    rows = build_batch_export_rows(batch, [document])

    assert EXPORT_HEADERS.index("link") < EXPORT_HEADERS.index("filename")
    assert list(rows[0].keys()) == EXPORT_HEADERS
    assert rows[0]["link"] == (
        r'=HYPERLINK("\\amer.prgx.com\images\OxxoMex\Link 2026\Proveedor\archivo.pdf")'
    )
