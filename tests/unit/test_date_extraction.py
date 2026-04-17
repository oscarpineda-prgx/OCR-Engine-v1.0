from app.services.parsing.fields import extract_fecha_documento, _match_month_name


def test_match_month_exact():
    assert _match_month_name("enero") == 1
    assert _match_month_name("diciembre") == 12
    assert _match_month_name("setiembre") == 9


def test_match_month_prefix_truncated():
    assert _match_month_name("sept") == 9
    assert _match_month_name("oct") == 10
    assert _match_month_name("dic") == 12
    assert _match_month_name("ene") == 1


def test_match_month_rejects_non_months():
    assert _match_month_name("junto") is None
    assert _match_month_name("ab") is None
    assert _match_month_name("") is None


def test_match_month_case_insensitive():
    assert _match_month_name("ENERO") == 1
    assert _match_month_name("Marzo") == 3


def test_match_month_with_accents():
    assert _match_month_name("Septiembre") == 9


def test_match_month_with_single_ocr_error():
    assert _match_month_name("MOy") == 5


def test_numeric_date_extraction():
    assert extract_fecha_documento("Fecha: 15/06/2024") == "2024-06-15"
    assert extract_fecha_documento("2024-03-01 contrato") == "2024-03-01"


def test_textual_date_extraction():
    text = "siendo el dia 15 de junio de 2024 se firma"
    result = extract_fecha_documento(text)
    assert result == "2024-06-15"


def test_fecha_de_documento_context_scores_higher():
    text = (
        "Fecha de emision: 01/03/2024\n"
        "Firma: 15/06/2020"
    )
    result = extract_fecha_documento(text)
    assert result == "2024-03-01"


def test_ocr_truncated_month():
    text = "15 de sept de 2024"
    result = extract_fecha_documento(text)
    assert result == "2024-09-15"


def test_slash_textual_month_with_two_digit_year():
    text = "Fecha de Entrega: 24/May/14"
    result = extract_fecha_documento(text)
    assert result == "2014-05-24"


def test_ocr_slash_textual_month_with_noise():
    text = "Fecha de Entrega: [24 MOy1 14 ]"
    result = extract_fecha_documento(text)
    assert result == "2014-05-24"


def test_month_first_textual_date_with_comma():
    text = "Fecha: Agosto 26, 2016"
    result = extract_fecha_documento(text)
    assert result == "2016-08-26"
