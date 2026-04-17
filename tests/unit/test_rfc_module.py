"""Tests for app.core.rfc — the unified RFC normalization and validation module."""

import pytest

from app.core.rfc import (
    canonicalize_rfc,
    derive_persona_fisica_rfc_prefix,
    has_valid_check_digit,
    is_acceptable_rfc,
    is_valid_rfc_date,
    normalize_rfc_for_match,
    normalize_rfc_ocr,
    pick_best_rfc,
    repair_persona_fisica_rfc_ocr,
)

# ---------------------------------------------------------------------------
# normalize_rfc_ocr
# ---------------------------------------------------------------------------


class TestNormalizeRfcOcr:
    def test_basic_cleaning_strips_special_chars_and_uppercases(self):
        assert normalize_rfc_ocr("abc-850101.xy9") == "ABC850101XY9"

    def test_o_to_0_correction(self):
        assert normalize_rfc_ocr("ABCO21001AB1") == "ABC021001AB1"

    def test_i_to_1_correction(self):
        assert normalize_rfc_ocr("ABCI21001AB1") == "ABC121001AB1"

    def test_s_to_5_correction(self):
        assert normalize_rfc_ocr("ABCS21001AB1") == "ABC521001AB1"

    def test_b_to_8_correction(self):
        assert normalize_rfc_ocr("ABCB21001AB1") == "ABC821001AB1"

    def test_g_to_6_correction(self):
        assert normalize_rfc_ocr("ABCG21001AB1") == "ABC621001AB1"

    def test_z_to_2_correction(self):
        assert normalize_rfc_ocr("ABCZ21001AB1") == "ABC221001AB1"

    def test_d_to_0_correction(self):
        assert normalize_rfc_ocr("ABCD21001AB1") == "ABC021001AB1"

    def test_lowercase_l_to_1_correction(self):
        assert normalize_rfc_ocr("ABCl21001AB1") == "ABC121001AB1"

    def test_alpha_portion_not_corrected_12char(self):
        # First 3 chars are the alpha prefix; letters there must NOT be replaced.
        result = normalize_rfc_ocr("OIS850101XY9")
        assert result[:3] == "OIS"

    def test_alpha_portion_not_corrected_13char(self):
        # First 4 chars are the alpha prefix for persona fisica RFCs.
        result = normalize_rfc_ocr("OISB850101XY99")
        assert result[:4] == "OISB"

    def test_handles_13char_rfc(self):
        # D in position 4 (start of date portion) should be corrected to 0.
        assert normalize_rfc_ocr("ABCDD21001AB1") == "ABCD021001AB1"

    def test_nonrfc_length_returns_cleaned_string(self):
        assert normalize_rfc_ocr("abc") == "ABC"
        assert normalize_rfc_ocr("too-short!!") == "TOOSHORT"


# ---------------------------------------------------------------------------
# normalize_rfc_for_match
# ---------------------------------------------------------------------------


class TestNormalizeRfcForMatch:
    def test_strips_nonalphanumeric_and_uppercases(self):
        assert normalize_rfc_for_match("abc-850101.xy9") == "ABC850101XY9"

    def test_returns_empty_for_none(self):
        assert normalize_rfc_for_match(None) == ""


# ---------------------------------------------------------------------------
# canonicalize_rfc
# ---------------------------------------------------------------------------


class TestCanonicalizeRfc:
    def test_foreign_tax_id_9_digits(self):
        assert canonicalize_rfc("123456789") == "12-3456789"

    def test_normal_rfc_unchanged(self):
        assert canonicalize_rfc("ABC850101XY9") == "ABC850101XY9"

    def test_none_returns_none(self):
        assert canonicalize_rfc(None) is None

    def test_whitespace_only_returns_none(self):
        assert canonicalize_rfc("   ") is None


# ---------------------------------------------------------------------------
# is_valid_rfc_date
# ---------------------------------------------------------------------------


class TestIsValidRfcDate:
    def test_valid_date_12char(self):
        assert is_valid_rfc_date("ABC850630MN3") is True  # June 30, 1985

    def test_invalid_month(self):
        assert is_valid_rfc_date("ABC851300MN3") is False

    def test_invalid_day(self):
        assert is_valid_rfc_date("ABC850632MN3") is False

    def test_valid_date_13char(self):
        assert is_valid_rfc_date("ABCD850630MN3") is True

    def test_wrong_length(self):
        assert is_valid_rfc_date("ABC85063") is False


# ---------------------------------------------------------------------------
# is_acceptable_rfc
# ---------------------------------------------------------------------------


class TestIsAcceptableRfc:
    def test_valid_rfc_with_valid_date(self):
        assert is_acceptable_rfc("ABC850630MN3") is True

    def test_bad_shape(self):
        assert is_acceptable_rfc("12345") is False

    def test_good_shape_bad_date(self):
        # Month 13 is invalid, but shape still matches the regex.
        assert is_acceptable_rfc("ABC851300MN3") is False

    def test_none_input(self):
        assert is_acceptable_rfc(None) is False


# ---------------------------------------------------------------------------
# has_valid_check_digit
# ---------------------------------------------------------------------------


class TestHasValidCheckDigit:
    def _compute_check_digit(self, base: str) -> str:
        """Helper: compute the expected check digit for a base string."""
        from app.core.rfc import RFC_CHAR_VALUES

        start_weight = 13 if (len(base) + 1) == 13 else 12
        total = sum(
            RFC_CHAR_VALUES[ch] * (start_weight - idx)
            for idx, ch in enumerate(base)
        )
        mod = total % 11
        calc = 11 - mod
        if calc == 11:
            return "0"
        if calc == 10:
            return "A"
        return str(calc)

    def test_known_valid_rfc(self):
        # Build an RFC whose check digit we compute ourselves.
        base = "CCO860523IN"  # 11 chars -> 12-char RFC
        digit = self._compute_check_digit(base)
        rfc = base + digit
        assert has_valid_check_digit(rfc) is True

    def test_wrong_last_char(self):
        base = "CCO860523IN"
        digit = self._compute_check_digit(base)
        # Flip the digit to something else.
        wrong = "X" if digit != "X" else "Y"
        assert has_valid_check_digit(base + wrong) is False

    def test_too_short(self):
        assert has_valid_check_digit("ABC") is False

    def test_none_input(self):
        assert has_valid_check_digit(None) is False


# ---------------------------------------------------------------------------
# pick_best_rfc
# ---------------------------------------------------------------------------


class TestPickBestRfc:
    def test_single_candidate(self):
        result = pick_best_rfc(["ABC850101XY9"])
        assert result is not None

    def test_empty_list(self):
        assert pick_best_rfc([]) is None

    def test_prefers_valid_check_digit(self):
        # Build one candidate with a valid check digit.
        from app.core.rfc import RFC_CHAR_VALUES

        base = "CCO860523IN"
        start_weight = 12
        total = sum(
            RFC_CHAR_VALUES[ch] * (start_weight - idx)
            for idx, ch in enumerate(base)
        )
        mod = total % 11
        calc = 11 - mod
        if calc == 11:
            digit = "0"
        elif calc == 10:
            digit = "A"
        else:
            digit = str(calc)

        good = base + digit
        bad_date = "ABC991301XY9"  # month 13 = invalid date
        result = pick_best_rfc([bad_date, good])
        assert result == good

    def test_prefers_valid_date_over_no_date(self):
        valid_date = "ABC850101XY9"  # Jan 1, 1985
        bad_date = "ABC991301XY9"  # month 13
        result = pick_best_rfc([bad_date, valid_date])
        assert result == valid_date

    def test_returns_first_nonempty_when_none_acceptable(self):
        result = pick_best_rfc(["SHORTX", "SHORTY"])
        # Neither matches RFC shape, so first non-empty cleaned value wins.
        assert result is not None


# ---------------------------------------------------------------------------
# Persona-fisica RFC repair helpers
# ---------------------------------------------------------------------------


class TestPersonaFisicaHelpers:
    def test_derive_persona_fisica_rfc_prefix(self):
        assert derive_persona_fisica_rfc_prefix("Felipe Velazquez Santiago") == "VESF"

    def test_derive_persona_fisica_rfc_prefix_skips_common_given_name(self):
        assert derive_persona_fisica_rfc_prefix("Maria Fernanda Lopez Perez") == "LOPF"

    def test_derive_persona_fisica_rfc_prefix_returns_none_for_legal_entity(self):
        assert derive_persona_fisica_rfc_prefix("Ferrero de Mexico S.A. de C.V.") is None

    def test_repair_persona_fisica_rfc_ocr(self):
        repaired = repair_persona_fisica_rfc_ocr("VESE9405249TB", "Felipe Velazquez Santiago")
        assert repaired == "VESF940524LT8"

    def test_repair_persona_fisica_rfc_ocr_from_labeled_text_scope(self):
        repaired = repair_persona_fisica_rfc_ocr(
            "Razón Social: Felipe Velazquez Santiago\nRFC: NESE 940524) TS\nDomicilio Fiscal:",
            "Felipe Velazquez Santiago",
        )
        assert repaired == "VESF940524LT8"

    def test_repair_persona_fisica_rfc_ocr_returns_none_without_viable_candidate(self):
        repaired = repair_persona_fisica_rfc_ocr("RFC DAÑADO", "Felipe Velazquez Santiago")
        assert repaired is None
