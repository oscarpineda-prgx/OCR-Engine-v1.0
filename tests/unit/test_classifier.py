"""Unit tests for app.services.extraction.classifier."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.extraction.classifier import (
    _count_word_tokens,
    _text_quality_ratio,
    classify_and_extract,
    classify_document_route,
    get_source_type,
    is_pdf_digital,
)
from app.services.extraction.file_types import ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# get_source_type
# ---------------------------------------------------------------------------
# NOTE: get_source_type returns the *lowercase extension* (e.g. "pdf", "jpg"),
# not a category like "image".  Tests reflect actual behaviour.


class TestGetSourceType:
    """Determine file extension from path."""

    @pytest.mark.parametrize("path,expected", [
        ("invoice.pdf", "pdf"),
        ("SCAN.PDF", "pdf"),
        ("/tmp/docs/report.PDF", "pdf"),
    ])
    def test_pdf_files(self, path, expected):
        assert get_source_type(path) == expected

    @pytest.mark.parametrize("path,expected", [
        ("photo.jpg", "jpg"),
        ("photo.jpeg", "jpeg"),
        ("scan.png", "png"),
        ("archive.tiff", "tiff"),
        ("old.bmp", "bmp"),
        ("PHOTO.JPG", "jpg"),
    ])
    def test_image_files(self, path, expected):
        assert get_source_type(path) == expected

    @pytest.mark.parametrize("path,expected", [
        ("report.docx", "docx"),
        ("notes.txt", "txt"),
        ("data.xlsx", "xlsx"),
    ])
    def test_other_extensions(self, path, expected):
        assert get_source_type(path) == expected

    def test_no_extension(self):
        assert get_source_type("README") == ""

    def test_empty_string(self):
        assert get_source_type("") == ""

    def test_none_input(self):
        # Should not crash; returns empty string.
        assert get_source_type(None) == ""


# ---------------------------------------------------------------------------
# _text_quality_ratio
# ---------------------------------------------------------------------------


class TestTextQualityRatio:
    """Ratio of alphabetic chars to all non-space chars."""

    def test_normal_text_high_ratio(self):
        text = "This is a perfectly normal English sentence with words."
        ratio = _text_quality_ratio(text)
        assert ratio > 0.5

    def test_pure_alpha_returns_one(self):
        assert _text_quality_ratio("abcdef") == 1.0

    def test_garbled_text_low_ratio(self):
        text = "@#$%^&*()!<>{}[]|\\~`12345+=-_;:',./?"
        ratio = _text_quality_ratio(text)
        assert ratio < 0.1

    def test_mixed_content(self):
        # Half alpha, half digit -> ratio ~ 0.5
        text = "abcd1234"
        ratio = _text_quality_ratio(text)
        assert 0.4 <= ratio <= 0.6

    def test_empty_string(self):
        assert _text_quality_ratio("") == 0.0

    def test_only_whitespace(self):
        assert _text_quality_ratio("   \t\n  ") == 0.0


# ---------------------------------------------------------------------------
# _count_word_tokens
# ---------------------------------------------------------------------------


class TestCountWordTokens:
    """Count tokens with 3+ alphabetic characters."""

    def test_normal_sentence(self):
        text = "The quick brown fox jumps over the lazy dog"
        # "The" (3), "quick" (5), "brown" (5), "fox" (3), "jumps" (5),
        # "over" (4), "the" (3), "lazy" (4), "dog" (3) -> 9
        assert _count_word_tokens(text) == 9

    def test_short_words_ignored(self):
        # All words < 3 letters -> 0 tokens
        text = "I am at it on"
        assert _count_word_tokens(text) == 0

    def test_empty_string(self):
        assert _count_word_tokens("") == 0

    def test_only_spaces_and_punctuation(self):
        assert _count_word_tokens("   ... !! ?? ") == 0

    def test_digits_not_counted(self):
        text = "123 456 7890"
        assert _count_word_tokens(text) == 0

    def test_mixed_alpha_digit_tokens(self):
        # "abc" inside "abc123" still matches as a 3-letter run
        assert _count_word_tokens("abc123") >= 1

    def test_accented_characters_counted(self):
        # The regex includes Unicode Latin Extended (\u00C0-\u024F)
        text = "comunicacion resolucion informacion"
        assert _count_word_tokens(text) == 3


# ---------------------------------------------------------------------------
# is_pdf_digital  (mocked — no real PDF files)
# ---------------------------------------------------------------------------


def _make_mock_page(text):
    """Return a mock pdfplumber page whose extract_text() returns *text*."""
    page = MagicMock()
    page.extract_text.return_value = text
    return page


def _good_text():
    """Return a text string that passes every threshold in is_pdf_digital."""
    return (
        "Este documento contiene informacion importante sobre el convenio "
        "de entrega establecido entre las partes involucradas en la operacion "
        "comercial del presente ejercicio fiscal correspondiente al periodo "
        "vigente de la empresa registrada ante las autoridades competentes."
    )


class TestIsPdfDigital:
    """is_pdf_digital with pdfplumber mocked out."""

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_pdf_with_sufficient_text_returns_true(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        good = _good_text()
        pages = [_make_mock_page(good), _make_mock_page(good)]
        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        assert is_pdf_digital("dummy.pdf") is True

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_pdf_with_little_text_returns_false(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        pages = [_make_mock_page("hi"), _make_mock_page("ok")]
        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        assert is_pdf_digital("dummy.pdf") is False

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_pdf_with_no_pages_returns_false(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        assert is_pdf_digital("dummy.pdf") is False

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_corrupt_pdf_returns_false(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        # Build a real exception class and wire it into the mock so the
        # ``except pdfplumber.pdfminer.pdfparser.PDFSyntaxError`` clause works.
        _PDFSyntaxError = type("PDFSyntaxError", (Exception,), {})
        mock_plumber.pdfminer.pdfparser.PDFSyntaxError = _PDFSyntaxError

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=_PDFSyntaxError("corrupt"))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_plumber.open.return_value = mock_ctx

        assert is_pdf_digital("dummy.pdf") is False

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_generic_exception_returns_false(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        # Wire a real exception class for PDFSyntaxError so the except
        # clause doesn't choke, then raise a *different* exception so
        # the generic ``except Exception`` handler is exercised.
        _PDFSyntaxError = type("PDFSyntaxError", (Exception,), {})
        mock_plumber.pdfminer.pdfparser.PDFSyntaxError = _PDFSyntaxError

        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(side_effect=RuntimeError("unexpected"))
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_plumber.open.return_value = mock_ctx

        assert is_pdf_digital("dummy.pdf") is False

    @patch("app.services.extraction.classifier.Path")
    def test_nonexistent_file_returns_false(self, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = False

        assert is_pdf_digital("missing.pdf") is False

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_pages_with_none_text_returns_false(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        # extract_text() returns None (scanned image pages)
        pages = [_make_mock_page(None), _make_mock_page(None)]
        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        assert is_pdf_digital("dummy.pdf") is False

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_garbled_text_fails_quality_check(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        # Lots of characters but mostly non-alpha garbage
        garbled = "!@#$%^&*(){}[]|/\\<>~`" * 30
        pages = [_make_mock_page(garbled), _make_mock_page(garbled)]
        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        assert is_pdf_digital("dummy.pdf") is False


class TestClassifyAndExtract:
    """classify_and_extract should return the processing route."""

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_digital_pdf_returns_digital_route_and_text(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        good = _good_text()
        pages = [_make_mock_page(good), _make_mock_page(good)]
        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        route, text = classify_and_extract("dummy.pdf")
        assert route == "digital_pdf"
        assert good in text

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_scanned_pdf_returns_ocr_route(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        pages = [_make_mock_page(None), _make_mock_page(None)]
        mock_pdf = MagicMock()
        mock_pdf.pages = pages
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        route, text = classify_and_extract("dummy.pdf")
        assert route == "ocr_image"
        assert text == ""

    @patch("app.services.extraction.classifier.Path")
    @patch("app.services.extraction.classifier.pdfplumber")
    def test_zero_page_pdf_falls_back_to_ocr_route(self, mock_plumber, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = True
        mock_pdf = MagicMock()
        mock_pdf.pages = []
        mock_plumber.open.return_value.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber.open.return_value.__exit__ = MagicMock(return_value=False)

        route, text = classify_and_extract("dummy.pdf")

        assert route == "ocr_image"
        assert text == ""


class TestClassifyDocumentRoute:
    @pytest.mark.parametrize("source_type", ["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
    def test_images_use_ocr_route(self, source_type):
        assert classify_document_route(source_type, "dummy.file") == "ocr_image"

    @pytest.mark.parametrize("source_type", ["doc", "docx", "xls", "xlsx", "xlsm", "xlsb", "msg"])
    def test_structured_documents_use_structured_route(self, source_type):
        assert classify_document_route(source_type, "dummy.file") == "structured_document"

    def test_unknown_extension_returns_unknown(self):
        assert classify_document_route("zip", "dummy.zip") == "unknown"


def test_allowed_extensions_include_new_document_types():
    for extension in {".bmp", ".doc", ".docx", ".xls", ".xlsx", ".xlsm", ".xlsb", ".msg"}:
        assert extension in ALLOWED_EXTENSIONS
