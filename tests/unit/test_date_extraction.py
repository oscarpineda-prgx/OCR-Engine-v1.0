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


def test_omits_historical_legal_public_deed_date_and_uses_signature_date():
    text = (
        "Ser una persona moral constituida conforme a las leyes vigentes en los "
        "Estados Unidos Mexicanos, segun se acredita con la escritura publica "
        "numero 47,162 de fecha 9 mayo 1996, pasada ante la fe del Notario "
        "Publico numero 103 e inscrita en el Registro Publico de la Propiedad "
        "y del Comercio.\n"
        "Habiendo sido leido el presente Contrato por las partes y enteradas "
        "del contenido y alcance legal de cada una de sus estipulaciones, lo "
        "firman en presencia de dos testigos que lo suscriben en la Ciudad de "
        "Mexico a los 20 dias del mes de SEPTIEMBRE del 2022."
    )
    result = extract_fecha_documento(
        text,
        tipo_documento="Suministro de Productos de Linea en CEDIS",
    )
    assert result == "2022-09-20"


def test_omits_dates_before_2000():
    assert extract_fecha_documento("Fecha: 01/06/1980") is None


def test_noisy_signature_date_with_split_year():
    text = (
        'Todo lo anteriormente expuesto es voluntad de ambas partes y acuerdan '
        'firmar el presente convenio, siendo el día 28 | de Febrero! de 20/22 |'
    )
    result = extract_fecha_documento(text, tipo_documento="Convenio Entrega Local")
    assert result == "2022-02-28"


def test_noisy_split_year_outside_range_is_rejected():
    text = "firmar el presente convenio, siendo el día 28 de Febrero de 21/01"
    assert extract_fecha_documento(text, tipo_documento="Convenio Entrega Local") is None
