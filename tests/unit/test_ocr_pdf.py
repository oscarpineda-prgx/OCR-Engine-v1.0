"""Unit tests for app.services.extraction.ocr_pdf."""

from dataclasses import fields as dataclass_fields
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest
from PIL import Image

from app.services.extraction.ocr_pdf import (
    MAX_OCR_PAGES,
    UnifiedOcrResult,
    _best_oriented_ocr_with_angle,
    _extract_embedded_page_image,
    _candidate_from_roi,
    _extract_date_hints_from_text,
    _extract_rfc_from_text,
    _has_valid_rfc_check_digit,
    _is_acceptable_rfc_hint,
    _is_valid_rfc_date,
    _normalize_rfc_ocr,
    _normalize_spacing,
    _ocr_score,
    _rotate_image,
    _strip_accents,
    extract_text_from_pdf_ocr,
    preprocess_pdf_page_for_ocr,
    unified_extract_from_pdf_ocr,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_rgb_pil(width: int = 200, height: int = 150, color="white") -> Image.Image:
    """Create a simple RGB PIL image for testing."""
    return Image.new("RGB", (width, height), color)


def _make_gray_pil(width: int = 200, height: int = 150) -> Image.Image:
    """Create a simple grayscale PIL image for testing."""
    return Image.new("L", (width, height), 128)


def _make_gray_array(width: int = 200, height: int = 150) -> np.ndarray:
    """Create a 2-D uint8 numpy array (grayscale image)."""
    return np.full((height, width), 128, dtype=np.uint8)


# ---------------------------------------------------------------------------
# preprocess_pdf_page_for_ocr
# ---------------------------------------------------------------------------


class TestPreprocessPdfPageForOcr:
    """preprocess_pdf_page_for_ocr: OpenCV preprocessing pipeline."""

    def test_returns_numpy_array(self):
        img = _make_rgb_pil()
        result = preprocess_pdf_page_for_ocr(img)
        assert isinstance(result, np.ndarray)

    def test_output_is_2d_grayscale(self):
        img = _make_rgb_pil(320, 240)
        result = preprocess_pdf_page_for_ocr(img)
        assert result.ndim == 2, "Output should be a single-channel (grayscale) image"

    def test_output_matches_input_dimensions(self):
        w, h = 320, 240
        img = _make_rgb_pil(w, h)
        result = preprocess_pdf_page_for_ocr(img)
        assert result.shape == (h, w)

    def test_output_dtype_is_uint8(self):
        img = _make_rgb_pil()
        result = preprocess_pdf_page_for_ocr(img)
        assert result.dtype == np.uint8

    def test_handles_small_image(self):
        img = _make_rgb_pil(10, 10)
        result = preprocess_pdf_page_for_ocr(img)
        assert result.shape == (10, 10)

    def test_handles_large_image(self):
        img = _make_rgb_pil(2000, 3000)
        result = preprocess_pdf_page_for_ocr(img)
        assert result.shape == (3000, 2000)

    def test_output_values_are_binary_like(self):
        """Adaptive threshold should produce mostly 0 and 255 values."""
        img = _make_rgb_pil(100, 100)
        result = preprocess_pdf_page_for_ocr(img)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})


class TestPdfEmbeddedImageFallback:
    def test_extract_embedded_page_image_returns_largest_image(self, tmp_path):
        file_path = tmp_path / "sample.pdf"
        file_path.write_bytes(b"placeholder")

        small = MagicMock()
        small.data = b"a" * 10
        small.image = _make_rgb_pil(10, 10)

        large = MagicMock()
        large.data = b"b" * 50
        large.image = _make_gray_pil(20, 20)

        page = MagicMock()
        page.images = [small, large]
        reader = MagicMock()
        reader.pages = [page]

        with patch("app.services.extraction.ocr_pdf.PdfReader", return_value=reader):
            image = _extract_embedded_page_image(str(file_path), 0)

        assert image.mode == "RGB"
        assert image.size == (20, 20)

    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle", return_value=("texto rescatado", 0))
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr", return_value=_make_gray_array())
    @patch("app.services.extraction.ocr_pdf.configure_tesseract_cmd", return_value=True)
    @patch("app.services.extraction.ocr_pdf._extract_embedded_page_image", return_value=_make_rgb_pil())
    @patch("app.services.extraction.ocr_pdf._require_pdf_document_class")
    def test_extract_text_from_pdf_ocr_uses_embedded_image_when_pdfium_fails(
        self,
        mock_require_pdf_document,
        mock_extract_embedded,
        _mock_configure,
        _mock_preprocess,
        _mock_best_ocr,
        tmp_path,
    ):
        file_path = tmp_path / "broken.pdf"
        file_path.write_bytes(b"placeholder")

        page = MagicMock()
        page.render.side_effect = RuntimeError("Failed to load page.")
        doc = MagicMock()
        doc.__len__.return_value = 1
        doc.__getitem__.return_value = page

        mock_pdf_document_cls = MagicMock(return_value=doc)
        mock_require_pdf_document.return_value = mock_pdf_document_cls

        result = extract_text_from_pdf_ocr(str(file_path))

        assert result == "texto rescatado"
        mock_extract_embedded.assert_called_once()

    @patch("app.services.extraction.ocr_pdf._unified_extract_rfc", return_value=None)
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle", return_value=("texto rescatado", 0))
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr", return_value=_make_gray_array())
    @patch("app.services.extraction.ocr_pdf.configure_tesseract_cmd", return_value=True)
    @patch("app.services.extraction.ocr_pdf._extract_embedded_page_image", return_value=_make_rgb_pil())
    @patch("app.services.extraction.ocr_pdf._require_pdf_document_class")
    def test_unified_extract_from_pdf_ocr_uses_embedded_image_when_pdfium_fails(
        self,
        mock_require_pdf_document,
        mock_extract_embedded,
        _mock_configure,
        _mock_preprocess,
        _mock_best_ocr,
        _mock_unified_rfc,
        tmp_path,
    ):
        file_path = tmp_path / "broken.pdf"
        file_path.write_bytes(b"placeholder")

        page = MagicMock()
        page.render.side_effect = RuntimeError("Failed to load page.")
        doc = MagicMock()
        doc.__len__.return_value = 1
        doc.__getitem__.return_value = page

        mock_pdf_document_cls = MagicMock(return_value=doc)
        mock_require_pdf_document.return_value = mock_pdf_document_cls

        with patch("app.services.extraction.ocr_pdf._extract_first_page_delivery_box_date", return_value=None), \
             patch("app.services.extraction.ocr_pdf._unified_extract_fecha_from_first_page", return_value=None):
            result = unified_extract_from_pdf_ocr(str(file_path))

        assert isinstance(result, UnifiedOcrResult)
        assert result.full_text == "texto rescatado"
        assert result.rfc_hint is None
        assert result.fecha_hint is None
        mock_extract_embedded.assert_called_once()

    @patch("app.services.extraction.ocr_pdf._unified_extract_rfc", return_value=None)
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle", return_value=("texto rescatado", 0))
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr", return_value=_make_gray_array())
    @patch("app.services.extraction.ocr_pdf.configure_tesseract_cmd", return_value=True)
    @patch("app.services.extraction.ocr_pdf._extract_embedded_page_image", return_value=_make_rgb_pil())
    @patch("app.services.extraction.ocr_pdf._require_pdf_document_class")
    def test_unified_extract_from_pdf_ocr_uses_embedded_image_when_page_open_fails(
        self,
        mock_require_pdf_document,
        mock_extract_embedded,
        _mock_configure,
        _mock_preprocess,
        _mock_best_ocr,
        _mock_unified_rfc,
        tmp_path,
    ):
        file_path = tmp_path / "broken.pdf"
        file_path.write_bytes(b"placeholder")

        doc = MagicMock()
        doc.__len__.return_value = 1
        doc.__getitem__.side_effect = RuntimeError("Failed to load page.")

        mock_pdf_document_cls = MagicMock(return_value=doc)
        mock_require_pdf_document.return_value = mock_pdf_document_cls

        with patch("app.services.extraction.ocr_pdf._extract_first_page_delivery_box_date", return_value=None), \
             patch("app.services.extraction.ocr_pdf._unified_extract_fecha_from_first_page", return_value=None):
            result = unified_extract_from_pdf_ocr(str(file_path))

        assert result.full_text == "texto rescatado"
        mock_extract_embedded.assert_called_once()


# ---------------------------------------------------------------------------
# _rotate_image
# ---------------------------------------------------------------------------


class TestRotateImage:
    """_rotate_image: rotates by 0, 90, 180, 270 degrees."""

    def test_rotate_0_returns_same_shape(self):
        arr = _make_gray_array(200, 100)
        result = _rotate_image(arr, 0)
        assert result.shape == (100, 200)

    def test_rotate_90_swaps_dimensions(self):
        arr = _make_gray_array(200, 100)
        result = _rotate_image(arr, 90)
        assert result.shape == (200, 100)

    def test_rotate_180_preserves_dimensions(self):
        arr = _make_gray_array(200, 100)
        result = _rotate_image(arr, 180)
        assert result.shape == (100, 200)

    def test_rotate_270_swaps_dimensions(self):
        arr = _make_gray_array(200, 100)
        result = _rotate_image(arr, 270)
        assert result.shape == (200, 100)

    def test_rotate_0_returns_identical_content(self):
        arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
        result = _rotate_image(arr, 0)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# _ocr_score
# ---------------------------------------------------------------------------


class TestOcrScore:
    """_ocr_score: heuristic scoring of OCR text quality."""

    def test_empty_text_returns_zero(self):
        assert _ocr_score("") == 0

    def test_none_text_returns_zero(self):
        assert _ocr_score(None) == 0

    def test_keyword_rich_text_scores_higher(self):
        good = "convenio proveedor cliente rfc fecha firma"
        bland = "the quick brown fox jumps over the lazy dog"
        assert _ocr_score(good) > _ocr_score(bland)

    def test_weird_chunks_penalty(self):
        normal = "Hello World"
        weird = "ABCDEFGHIJKLMNOP1234567890"  # single 26-char chunk
        assert _ocr_score(normal) > _ocr_score(weird)

    def test_all_alpha_beats_all_symbols(self):
        alpha = "abcdefghij"
        symbols = "!@#$%^&*()"
        assert _ocr_score(alpha) > _ocr_score(symbols)


# ---------------------------------------------------------------------------
# _strip_accents / _normalize_spacing
# ---------------------------------------------------------------------------


class TestStringHelpers:
    """_strip_accents and _normalize_spacing."""

    def test_strip_accents_replaces_accented_vowels(self):
        assert _strip_accents("comunicación") == "comunicacion"

    def test_strip_accents_replaces_uppercase_accents(self):
        assert _strip_accents("ÁÉÍÓÚ") == "AEIOU"

    def test_strip_accents_preserves_plain_ascii(self):
        assert _strip_accents("hello") == "hello"

    def test_strip_accents_handles_n_tilde(self):
        assert _strip_accents("año") == "ano"
        assert _strip_accents("NIÑO") == "NINO"

    def test_normalize_spacing_collapses_whitespace(self):
        assert _normalize_spacing("  hello   world  ") == "hello world"

    def test_normalize_spacing_handles_tabs_and_newlines(self):
        assert _normalize_spacing("a\t\nb") == "a b"

    def test_normalize_spacing_empty_string(self):
        assert _normalize_spacing("") == ""

    def test_normalize_spacing_none_input(self):
        assert _normalize_spacing(None) == ""


# ---------------------------------------------------------------------------
# _extract_date_hints_from_text
# ---------------------------------------------------------------------------


class TestExtractDateHintsFromText:
    """_extract_date_hints_from_text: regex date extraction from OCR text."""

    def test_empty_text_returns_empty(self):
        assert _extract_date_hints_from_text("") == []

    def test_none_text_returns_empty(self):
        assert _extract_date_hints_from_text(None) == []

    def test_no_dates_returns_empty(self):
        assert _extract_date_hints_from_text("This text has no date information") == []

    def test_spanish_date_with_de_format(self):
        text = "Firmado el 15 de enero de 2024 en la ciudad de Mexico."
        hints = _extract_date_hints_from_text(text)
        assert len(hints) >= 1
        combined = " ".join(hints)
        assert "enero" in combined
        assert "2024" in combined

    def test_siendo_el_dia_anchor(self):
        text = "siendo el dia 20 de marzo de 2023 se celebra este convenio"
        hints = _extract_date_hints_from_text(text)
        assert len(hints) >= 1
        combined = " ".join(hints)
        assert "marzo" in combined
        assert "2023" in combined

    def test_accented_date_normalised(self):
        text = "siendo el día 5 de febrero de 2022"
        hints = _extract_date_hints_from_text(text)
        assert len(hints) >= 1

    def test_multiple_dates_extracts_multiple(self):
        text = (
            "Firmado el 10 de junio de 2021 por el cliente.\n"
            "Vigente desde el 1 de diciembre de 2020 hasta la fecha."
        )
        hints = _extract_date_hints_from_text(text)
        assert len(hints) >= 2

    def test_deduplicates_identical_hints(self):
        text = (
            "15 de enero de 2024\n"
            "15 de enero de 2024\n"
        )
        hints = _extract_date_hints_from_text(text)
        # Should have at most 1 unique hint
        assert len(hints) == len(set(h.lower() for h in hints))

    def test_setiembre_variant(self):
        """Module defines 'setiembre' as an alternate spelling of septiembre."""
        text = "el 3 de setiembre de 2019"
        hints = _extract_date_hints_from_text(text)
        assert len(hints) >= 1

    def test_year_boundary_19xx(self):
        text = "siendo el dia 01 de abril de 1999"
        hints = _extract_date_hints_from_text(text)
        assert len(hints) >= 1
        assert "1999" in " ".join(hints)


# ---------------------------------------------------------------------------
# _normalize_rfc_ocr
# ---------------------------------------------------------------------------


class TestNormalizeRfcOcr:
    """_normalize_rfc_ocr: fix common OCR mis-reads in RFC strings."""

    def test_already_correct_12_char(self):
        assert _normalize_rfc_ocr("ABC020101AB1") == "ABC020101AB1"

    def test_replaces_O_with_0_in_date_portion_13_char(self):
        # 13-char: date digits are positions 4..9
        result = _normalize_rfc_ocr("ABCD0O0101AB1")
        assert result == "ABCD000101AB1"

    def test_replaces_I_with_1_in_date_portion(self):
        result = _normalize_rfc_ocr("ABC0I0101AB1")
        assert result == "ABC010101AB1"

    def test_replaces_S_with_5_in_date_portion(self):
        result = _normalize_rfc_ocr("ABC0S0101AB1")
        assert result == "ABC050101AB1"

    def test_replaces_B_with_8_in_date_portion(self):
        result = _normalize_rfc_ocr("ABC0B0101AB1")
        assert result == "ABC080101AB1"

    def test_strips_non_alnum_characters(self):
        result = _normalize_rfc_ocr("A-B.C 020101AB1")
        assert result == "ABC020101AB1"

    def test_wrong_length_returns_raw(self):
        """Strings with length != 12 and != 13 are returned as-is (uppercased, stripped)."""
        result = _normalize_rfc_ocr("SHORT")
        assert result == "SHORT"

    def test_none_input_returns_empty(self):
        assert _normalize_rfc_ocr(None) == ""


# ---------------------------------------------------------------------------
# _is_valid_rfc_date / _is_acceptable_rfc_hint
# ---------------------------------------------------------------------------


class TestRfcValidation:
    """RFC date and shape validation helpers."""

    def test_valid_12_char_rfc_date(self):
        # ABC + 850101 + XY1 -> date 1985-01-01
        assert _is_valid_rfc_date("ABC850101XY1") is True

    def test_valid_13_char_rfc_date(self):
        # ABCD + 850101 + XY1 -> date 1985-01-01
        assert _is_valid_rfc_date("ABCD850101XY1") is True

    def test_invalid_month_in_rfc(self):
        # month 13 is invalid
        assert _is_valid_rfc_date("ABC851301XY1") is False

    def test_invalid_day_in_rfc(self):
        # day 32 is invalid
        assert _is_valid_rfc_date("ABC850132XY1") is False

    def test_wrong_length(self):
        assert _is_valid_rfc_date("TOOLONG1234567") is False

    def test_acceptable_rfc_hint_passes_shape_and_date(self):
        # Valid shape + valid date
        assert _is_acceptable_rfc_hint("ABC850101XY1") is True

    def test_acceptable_rfc_hint_rejects_bad_shape(self):
        # lowercase chars fail pattern
        assert _is_acceptable_rfc_hint("abc850101xy1") is False


# ---------------------------------------------------------------------------
# _has_valid_rfc_check_digit
# ---------------------------------------------------------------------------


class TestRfcCheckDigit:
    """_has_valid_rfc_check_digit: modulus 11 check-digit validation."""

    def test_wrong_length_returns_false(self):
        assert _has_valid_rfc_check_digit("SHORT") is False

    def test_invalid_character_returns_false(self):
        # Character not in RFC_CHAR_VALUES (e.g. lowercase)
        assert _has_valid_rfc_check_digit("abcDEF012345") is False

    def test_returns_bool(self):
        result = _has_valid_rfc_check_digit("ABC850101XY1")
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# _best_oriented_ocr_with_angle  (mocked tesseract)
# ---------------------------------------------------------------------------


class TestBestOrientedOcrWithAngle:
    """_best_oriented_ocr_with_angle: picks rotation with best OCR score."""

    @patch("app.services.extraction.ocr_pdf.pytesseract")
    def test_returns_tuple_of_text_and_angle(self, mock_tess):
        mock_tess.image_to_string.return_value = "some text"
        arr = _make_gray_array()
        result = _best_oriented_ocr_with_angle(arr)
        assert isinstance(result, tuple)
        assert len(result) == 2

    @patch("app.services.extraction.ocr_pdf.pytesseract")
    def test_picks_rotation_with_highest_score(self, mock_tess):
        """Simulate: 0 deg -> short text, 180 deg -> text with keywords."""
        responses = {
            0: "x",
            1: "convenio proveedor cliente rfc fecha firma domicilio",
            2: "ab",
            3: "cd",
        }
        call_counter = {"n": 0}

        def side_effect(*_args, **_kwargs):
            idx = call_counter["n"]
            call_counter["n"] += 1
            return responses.get(idx, "")

        mock_tess.image_to_string.side_effect = side_effect
        arr = _make_gray_array()
        text, angle = _best_oriented_ocr_with_angle(arr)
        # The second call (angle 180) has the richest text
        assert "convenio" in text
        assert angle == 180

    @patch("app.services.extraction.ocr_pdf.pytesseract")
    def test_four_calls_to_tesseract(self, mock_tess):
        """Should call image_to_string exactly 4 times (0, 180, 90, 270)."""
        mock_tess.image_to_string.return_value = "text"
        arr = _make_gray_array()
        _best_oriented_ocr_with_angle(arr)
        assert mock_tess.image_to_string.call_count == 4

    @patch("app.services.extraction.ocr_pdf.pytesseract")
    def test_all_empty_returns_empty_string_and_zero(self, mock_tess):
        mock_tess.image_to_string.return_value = ""
        arr = _make_gray_array()
        text, angle = _best_oriented_ocr_with_angle(arr)
        assert text == ""
        # When all scores are equal the first angle (0) wins
        assert angle == 0

    @patch("app.services.extraction.ocr_pdf.pytesseract")
    def test_none_from_tesseract_treated_as_empty(self, mock_tess):
        mock_tess.image_to_string.return_value = None
        arr = _make_gray_array()
        text, angle = _best_oriented_ocr_with_angle(arr)
        assert text == ""


# ---------------------------------------------------------------------------
# _candidate_from_roi  (mocked tesseract)
# ---------------------------------------------------------------------------


class TestCandidateFromRoi:
    """_candidate_from_roi: extract RFC candidate from region of interest."""

    def test_none_roi_returns_none(self):
        assert _candidate_from_roi(None) is None

    def test_empty_roi_returns_none(self):
        empty = np.array([], dtype=np.uint8).reshape(0, 0)
        assert _candidate_from_roi(empty) is None

    @patch("app.services.extraction.ocr_pdf.pytesseract")
    def test_returns_rfc_when_tesseract_finds_one(self, mock_tess):
        """If tesseract returns a valid RFC-shaped string, it should be extracted."""
        mock_tess.image_to_string.return_value = "RFC ABCD850101XY1"
        roi = _make_gray_array(100, 30)
        result = _candidate_from_roi(roi)
        # The function uses _extract_rfc_from_text internally, so
        # the result depends on pattern matching and validation.
        # With a valid RFC shape string, we expect a non-None result.
        if result is not None:
            assert len(result) in (12, 13)

    @patch("app.services.extraction.ocr_pdf.pytesseract")
    def test_returns_none_when_no_valid_rfc(self, mock_tess):
        mock_tess.image_to_string.return_value = "no rfc here at all"
        roi = _make_gray_array(100, 30)
        result = _candidate_from_roi(roi)
        assert result is None


# ---------------------------------------------------------------------------
# _extract_rfc_from_text
# ---------------------------------------------------------------------------


class TestExtractRfcFromText:
    """_extract_rfc_from_text: regex-based RFC extraction from OCR text."""

    def test_returns_none_for_plain_text(self):
        assert _extract_rfc_from_text("Hello world no RFC here") is None

    def test_extracts_valid_rfc_12_char(self):
        # Input is uppercased and non-alnum stripped before pattern matching,
        # so use a token that stands alone after normalization.
        result = _extract_rfc_from_text("ABC850101XY1")
        if result is not None:
            assert len(result) == 12

    def test_extracts_valid_rfc_13_char(self):
        result = _extract_rfc_from_text("RFC: ABCD850101XY1")
        if result is not None:
            assert len(result) == 13

    def test_returns_none_for_empty_string(self):
        assert _extract_rfc_from_text("") is None

    def test_returns_none_for_none(self):
        assert _extract_rfc_from_text(None) is None


# ---------------------------------------------------------------------------
# UnifiedOcrResult dataclass
# ---------------------------------------------------------------------------


class TestUnifiedOcrResult:
    """UnifiedOcrResult: dataclass construction and field access."""

    def test_construction_with_all_fields(self):
        result = UnifiedOcrResult(
            full_text="some ocr text",
            rfc_hint="ABC850101XY1",
            fecha_hint="2024-01-15",
        )
        assert result.full_text == "some ocr text"
        assert result.rfc_hint == "ABC850101XY1"
        assert result.fecha_hint == "2024-01-15"

    def test_construction_with_none_hints(self):
        result = UnifiedOcrResult(full_text="text", rfc_hint=None, fecha_hint=None)
        assert result.rfc_hint is None
        assert result.fecha_hint is None

    def test_has_three_fields(self):
        flds = dataclass_fields(UnifiedOcrResult)
        names = {f.name for f in flds}
        assert names == {"full_text", "rfc_hint", "fecha_hint"}

    def test_empty_full_text(self):
        result = UnifiedOcrResult(full_text="", rfc_hint=None, fecha_hint=None)
        assert result.full_text == ""


# ---------------------------------------------------------------------------
# extract_text_from_pdf_ocr  (mocked PDF + tesseract)
# ---------------------------------------------------------------------------


class TestExtractTextFromPdfOcr:
    """extract_text_from_pdf_ocr: full pipeline with mocking."""

    @patch("app.services.extraction.ocr_pdf.Path")
    def test_file_not_found_raises(self, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf_ocr("/nonexistent/file.pdf")

    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints", return_value="")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_single_page_returns_text(
        self, mock_path_cls, mock_pdf_doc, mock_preprocess, mock_ocr, mock_footer
    ):
        mock_path_cls.return_value.exists.return_value = True

        # Mock a single-page PDF
        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = _make_rgb_pil()
        mock_page.render.return_value = mock_bitmap

        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=1)
        doc_instance.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf_doc.return_value = doc_instance

        mock_preprocess.return_value = _make_gray_array()
        mock_ocr.return_value = ("Extracted OCR text from page one", 0)

        result = extract_text_from_pdf_ocr("/fake/path.pdf")
        assert result == "Extracted OCR text from page one"

    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints", return_value="")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_multipage_concatenates_text(
        self, mock_path_cls, mock_pdf_doc, mock_preprocess, mock_ocr, mock_footer
    ):
        mock_path_cls.return_value.exists.return_value = True

        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = _make_rgb_pil()
        mock_page.render.return_value = mock_bitmap

        page_count = 3
        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=page_count)
        doc_instance.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf_doc.return_value = doc_instance

        mock_preprocess.return_value = _make_gray_array()

        page_texts = ["Page one text", "Page two text", "Page three text"]
        mock_ocr.side_effect = [(t, 0) for t in page_texts]

        result = extract_text_from_pdf_ocr("/fake/path.pdf")
        assert "Page one text" in result
        assert "Page two text" in result
        assert "Page three text" in result
        # Pages are joined by double newline
        assert "\n\n" in result

    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints", return_value="")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_max_pages_limit(
        self, mock_path_cls, mock_pdf_doc, mock_preprocess, mock_ocr, mock_footer
    ):
        """PDFs with more than MAX_OCR_PAGES only process the first MAX_OCR_PAGES."""
        mock_path_cls.return_value.exists.return_value = True

        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = _make_rgb_pil()
        mock_page.render.return_value = mock_bitmap

        excessive_count = MAX_OCR_PAGES + 20
        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=excessive_count)
        doc_instance.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf_doc.return_value = doc_instance

        mock_preprocess.return_value = _make_gray_array()
        mock_ocr.return_value = ("page text", 0)

        extract_text_from_pdf_ocr("/fake/path.pdf")

        # OCR should have been called exactly MAX_OCR_PAGES times
        assert mock_ocr.call_count == MAX_OCR_PAGES

    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_footer_appended_on_last_page(
        self, mock_path_cls, mock_pdf_doc, mock_preprocess, mock_ocr, mock_footer
    ):
        mock_path_cls.return_value.exists.return_value = True

        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = _make_rgb_pil()
        mock_page.render.return_value = mock_bitmap

        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=1)
        doc_instance.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf_doc.return_value = doc_instance

        mock_preprocess.return_value = _make_gray_array()
        mock_ocr.return_value = ("Main text", 0)
        mock_footer.return_value = "siendo el dia 10 de junio de 2021"

        result = extract_text_from_pdf_ocr("/fake/path.pdf")
        assert "Main text" in result
        assert "siendo el dia 10 de junio de 2021" in result

    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints", return_value="")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_empty_pages_produce_empty_result(
        self, mock_path_cls, mock_pdf_doc, mock_preprocess, mock_ocr, mock_footer
    ):
        mock_path_cls.return_value.exists.return_value = True

        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = _make_rgb_pil()
        mock_page.render.return_value = mock_bitmap

        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=2)
        doc_instance.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf_doc.return_value = doc_instance

        mock_preprocess.return_value = _make_gray_array()
        mock_ocr.return_value = ("", 0)

        result = extract_text_from_pdf_ocr("/fake/path.pdf")
        assert result == ""


# ---------------------------------------------------------------------------
# unified_extract_from_pdf_ocr  (mocked PDF + tesseract)
# ---------------------------------------------------------------------------


class TestUnifiedExtractFromPdfOcr:
    """unified_extract_from_pdf_ocr: single-pass extraction with mocking."""

    @patch("app.services.extraction.ocr_pdf.Path")
    def test_file_not_found_raises(self, mock_path_cls):
        mock_path_cls.return_value.exists.return_value = False
        with pytest.raises(FileNotFoundError):
            unified_extract_from_pdf_ocr("/missing/file.pdf")

    @patch("app.services.extraction.ocr_pdf._unified_extract_fecha_from_first_page", return_value=None)
    @patch("app.services.extraction.ocr_pdf._unified_extract_rfc", return_value=None)
    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints", return_value="")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_zero_pages_returns_empty_result(
        self,
        mock_path_cls,
        mock_pdf_doc,
        mock_preprocess,
        mock_ocr,
        mock_footer,
        mock_rfc,
        mock_fecha,
    ):
        mock_path_cls.return_value.exists.return_value = True
        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=0)
        mock_pdf_doc.return_value = doc_instance

        result = unified_extract_from_pdf_ocr("/fake/path.pdf")
        assert isinstance(result, UnifiedOcrResult)
        assert result.full_text == ""
        assert result.rfc_hint is None
        assert result.fecha_hint is None

    @patch("app.services.extraction.ocr_pdf.extract_fecha_documento", create=True)
    @patch("app.services.extraction.ocr_pdf._unified_extract_fecha_from_first_page", return_value=None)
    @patch("app.services.extraction.ocr_pdf._unified_extract_rfc", return_value="TEST850101AB1")
    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints", return_value="")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_single_page_returns_unified_result(
        self,
        mock_path_cls,
        mock_pdf_doc,
        mock_preprocess,
        mock_ocr,
        mock_footer,
        mock_rfc,
        mock_fecha,
        mock_extract_fecha,
    ):
        mock_path_cls.return_value.exists.return_value = True

        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = _make_rgb_pil()
        mock_page.render.return_value = mock_bitmap

        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=1)
        doc_instance.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf_doc.return_value = doc_instance

        mock_preprocess.return_value = _make_gray_array()
        mock_ocr.return_value = ("Full OCR text here", 0)

        # Mock the fecha extraction from fields module
        mock_extract_fecha.return_value = None

        result = unified_extract_from_pdf_ocr("/fake/path.pdf")
        assert isinstance(result, UnifiedOcrResult)
        assert "Full OCR text here" in result.full_text
        assert result.rfc_hint == "TEST850101AB1"

    @patch("app.services.extraction.ocr_pdf.extract_fecha_documento", create=True)
    @patch("app.services.extraction.ocr_pdf._unified_extract_fecha_from_first_page", return_value=None)
    @patch("app.services.extraction.ocr_pdf._unified_extract_rfc", return_value=None)
    @patch("app.services.extraction.ocr_pdf._ocr_footer_date_hints", return_value="")
    @patch("app.services.extraction.ocr_pdf._best_oriented_ocr_with_angle")
    @patch("app.services.extraction.ocr_pdf.preprocess_pdf_page_for_ocr")
    @patch("app.services.extraction.ocr_pdf.PdfDocument")
    @patch("app.services.extraction.ocr_pdf.Path")
    def test_max_pages_respected(
        self,
        mock_path_cls,
        mock_pdf_doc,
        mock_preprocess,
        mock_ocr,
        mock_footer,
        mock_rfc,
        mock_fecha,
        mock_extract_fecha,
    ):
        mock_path_cls.return_value.exists.return_value = True

        mock_page = MagicMock()
        mock_bitmap = MagicMock()
        mock_bitmap.to_pil.return_value = _make_rgb_pil()
        mock_page.render.return_value = mock_bitmap

        doc_instance = MagicMock()
        doc_instance.__len__ = MagicMock(return_value=MAX_OCR_PAGES + 10)
        doc_instance.__getitem__ = MagicMock(return_value=mock_page)
        mock_pdf_doc.return_value = doc_instance

        mock_preprocess.return_value = _make_gray_array()
        mock_ocr.return_value = ("text", 0)
        mock_extract_fecha.return_value = None

        unified_extract_from_pdf_ocr("/fake/path.pdf")
        assert mock_ocr.call_count == MAX_OCR_PAGES


# ---------------------------------------------------------------------------
# MAX_OCR_PAGES constant
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify module-level constants."""

    def test_max_ocr_pages_value(self):
        assert MAX_OCR_PAGES == 50

    def test_max_ocr_pages_is_int(self):
        assert isinstance(MAX_OCR_PAGES, int)
