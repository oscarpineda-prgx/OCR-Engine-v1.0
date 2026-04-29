"""Comprehensive tests for app/services/parsing/fields.py.

Covers all four main extraction functions plus the normalize_company_name helper.
"""

import pytest

from app.services.parsing.fields import (
    extract_fecha_documento,
    extract_nombre_proveedor,
    extract_rfc,
    extract_tipo_documento,
    normalize_company_name,
)


# =========================================================================
# 1. extract_rfc
# =========================================================================


class TestExtractRfc:
    """Tests for Mexican RFC and foreign tax-ID extraction."""

    def test_standard_13_char_rfc(self):
        text = "RFC: XAXX010101000"
        result = extract_rfc(text)
        assert result is not None
        assert len(result) == 13

    def test_12_char_rfc_persona_moral(self):
        text = "RFC del contribuyente: AAA010101XX3"
        result = extract_rfc(text)
        assert result is not None

    def test_rfc_with_surrounding_context(self):
        text = "El proveedor con RFC GARC850101AB3 firmo el contrato ayer."
        result = extract_rfc(text)
        assert result == "GARC850101AB3"

    def test_foreign_tax_id_extraction(self):
        text = "Tax ID: 12-1234567"
        result = extract_rfc(text)
        assert result == "12-1234567"

    def test_foreign_tax_id_with_ein_label(self):
        text = "EIN 98-7654321 is the company identifier."
        result = extract_rfc(text)
        assert result == "98-7654321"

    def test_multiple_rfcs_picks_best_scored(self):
        text = (
            "EL CLIENTE CADENA COMERCIAL OXXO con RFC CCO8605231N4.\n"
            "y PROVEEDOR ABCD900101XY3 con RFC ABCD900101XY3."
        )
        result = extract_rfc(text)
        # The client RFC CCO8605231N4 is in the ignored set and near CLIENTE
        # context, so the proveedor RFC should win.
        assert result == "ABCD900101XY3"

    def test_no_rfc_returns_none(self):
        text = "Este documento no contiene informacion fiscal."
        assert extract_rfc(text) is None

    def test_empty_text_returns_none(self):
        assert extract_rfc("") is None
        assert extract_rfc(None) is None

    def test_rfc_near_cliente_context_deprioritized(self):
        # The 90-char scoring window must not overlap, so we pad
        # with enough filler between the two lines.
        text = (
            "EL CLIENTE con RFC XYZZ010101AB3 firma. "
            + "x" * 200 + "\n"
            + "EL PROVEEDOR con RFC ABCD900101XY3 firma."
        )
        result = extract_rfc(text)
        # PROVEEDOR context (+30) beats CLIENTE context (-20).
        assert result == "ABCD900101XY3"

    def test_rfc_with_rfc_label_scores_higher(self):
        text = (
            "GARC850101AB3\n"
            "RFC: XYZA900515MN2\n"
        )
        result = extract_rfc(text)
        # The one preceded by the "RFC:" label gets a base score of 70
        # vs 35, so it should be preferred.
        assert result == "XYZA900515MN2"

    def test_rfc_with_ocr_spaces_grouped_pattern(self):
        # Grouped pattern handles separators between RFC parts.
        text = "RFC: GARC 85 01 01 AB3"
        result = extract_rfc(text)
        assert result is not None
        assert result == "GARC850101AB3"

    def test_rfc_with_dashes_grouped_pattern(self):
        text = "RFC: AGD-811217-HS7"
        result = extract_rfc(text)
        assert result == "AGD811217HS7"

    def test_curp_line_is_skipped(self):
        # Lines containing "CURP" should be skipped entirely.
        text = "CURP: GARC850101HDFRRL09 RFC extra"
        result = extract_rfc(text)
        assert result is None

    def test_rfc_ocr_digit_correction_O_to_0(self):
        # 'O' in the date portion should be corrected to '0'.
        text = "RFC: GARC85O1O1AB3"
        result = extract_rfc(text)
        assert result is not None
        assert "O" not in result[4:10]

    def test_rfc_invalid_date_rejected(self):
        # An RFC whose date portion is invalid (month 99) should not match.
        text = "RFC: XAXX999999ZZZ"
        result = extract_rfc(text)
        assert result is None

    def test_ignored_client_rfc_excluded(self):
        text = "RFC: CCO8605231N4"
        result = extract_rfc(text)
        assert result is None

    def test_soft_labeled_rfc_fallback_for_supplier_form(self):
        text = (
            "Razón Social: VICTOR MANUEL BACA APODACA\n"
            "RFC: BAAV7822025P7\n"
            "Domicilio Fiscal: CONSTITUCION #206\n"
        )
        result = extract_rfc(text)
        assert result == "BAAV7822025P7"

    def test_rfc_with_registro_federal_label(self):
        text = "REGISTRO FEDERAL DE CONTRIBUYENTES ABCD900101XY3."
        result = extract_rfc(text)
        assert result == "ABCD900101XY3"

    def test_foreign_tax_id_with_federal_label(self):
        text = "US FEDERAL TAX ID 123456789"
        result = extract_rfc(text)
        assert result == "12-3456789"


# =========================================================================
# 2. extract_fecha_documento
# =========================================================================


class TestExtractFechaDocumento:
    """Tests for document date extraction."""

    def test_iso_format(self):
        text = "Fecha: 2024-01-15"
        assert extract_fecha_documento(text) == "2024-01-15"

    def test_slash_format_dd_mm_yyyy(self):
        text = "15/01/2024"
        assert extract_fecha_documento(text) == "2024-01-15"

    def test_dash_format_dd_mm_yyyy(self):
        text = "15-01-2024"
        assert extract_fecha_documento(text) == "2024-01-15"

    def test_dot_format_dd_mm_yyyy(self):
        text = "15.01.2024"
        assert extract_fecha_documento(text) == "2024-01-15"

    def test_textual_spanish_date(self):
        text = "15 de enero de 2024"
        assert extract_fecha_documento(text) == "2024-01-15"

    def test_textual_date_with_marzo(self):
        text = "1 de marzo de 2023"
        assert extract_fecha_documento(text) == "2023-03-01"

    def test_textual_date_diciembre(self):
        text = "25 de diciembre de 2023"
        assert extract_fecha_documento(text) == "2023-12-25"

    def test_fecha_del_documento_context_prioritized(self):
        # "Fecha del documento" label must be within the 60-char context
        # window of its date.  Put it directly before the date.
        text = "Fecha del documento: 15/03/2024"
        result = extract_fecha_documento(text)
        assert result == "2024-03-15"

    def test_fecha_de_emision_context_prioritized(self):
        # "Fecha de emision" boosts the score by +6 +2.  Place it directly
        # before its date so the context window captures the label.
        text = "Fecha de emision: 20/07/2024"
        result = extract_fecha_documento(text)
        assert result == "2024-07-20"

    def test_multiple_dates_picks_best_context(self):
        text = (
            "Fecha de emision: 01/03/2024\n"
            "Firma: 15/06/2020"
        )
        result = extract_fecha_documento(text)
        assert result == "2024-03-01"

    def test_no_date_returns_none(self):
        text = "Este documento no tiene ninguna fecha."
        assert extract_fecha_documento(text) is None

    def test_empty_text_returns_none(self):
        assert extract_fecha_documento("") is None
        assert extract_fecha_documento(None) is None

    def test_future_date_still_extracted(self):
        text = "Fecha: 2030-12-31"
        assert extract_fecha_documento(text) == "2030-12-31"

    def test_old_date_is_ignored(self):
        text = "Fecha: 01/06/1980"
        assert extract_fecha_documento(text) is None

    def test_textual_date_with_siendo_el_dia(self):
        text = "siendo el dia 10 de abril de 2024 se firma el contrato"
        assert extract_fecha_documento(text) == "2024-04-10"

    def test_textual_date_ocr_truncated_month(self):
        text = "15 de sept de 2024"
        assert extract_fecha_documento(text) == "2024-09-15"

    def test_numeric_textual_date_de_del(self):
        text = "firmado el 15 de 06 del 2024 en la ciudad"
        assert extract_fecha_documento(text) == "2024-06-15"

    def test_firma_context_boosts_score(self):
        # "firma" in the 60-char context window awards +3. Put it
        # directly before its date so it is captured.
        text = "firma del contrato: 25/12/2023"
        result = extract_fecha_documento(text)
        assert result == "2023-12-25"

    def test_month_first_textual_date_is_supported(self):
        text = "BACARDI\nCARTA SOPORTE\nFecha: Agosto 26, 2016\nFemsa Comercio"
        result = extract_fecha_documento(text)
        assert result == "2016-08-26"


# =========================================================================
# 3. extract_tipo_documento
# =========================================================================


class TestExtractTipoDocumento:
    """Tests for document type classification."""

    def test_portal_de_proveedores(self):
        text = "CONVENIO PARA ACCESO A PORTAL DE PROVEEDORES entre las partes."
        assert extract_tipo_documento(text) == "Portal de Proveedores"

    def test_convenio_entrega_cedis(self):
        text = "CONVENIO ENTREGA EN CEDIS para proveedores nacionales."
        assert extract_tipo_documento(text) == "Convenio Entrega Cedis"

    def test_convenio_modificatorio_cedis(self):
        text = "Convenio Modificatorio al contrato de suministro en CEDIS."
        assert extract_tipo_documento(text) == "Convenio Modificatorio Cedis"

    def test_carta_soporte_descuentos(self):
        text = (
            "Carta Soporte para ajustes comerciales.\n"
            "Este documento contiene descuentos por categoria."
        )
        assert extract_tipo_documento(text) == "Carta Soporte Descuentos"

    def test_carta_soporte_costos_excludes_descuento(self):
        text = (
            "Carta Soporte de costos logisticos.\n"
            "Incremento de costo por transporte."
        )
        assert extract_tipo_documento(text) == "Carta Soporte Costos"

    def test_convenio_factoraje(self):
        text = "Convenio relativo a la mecanica operativa del sistema de factoraje."
        assert extract_tipo_documento(text) == "Convenio Factoraje"

    def test_adendums_soporte(self):
        text = "Se anexa el adendum correspondiente al convenio vigente."
        assert extract_tipo_documento(text) == "Adendums Soporte"

    def test_product_supply_agreement(self):
        text = "This is a Product Supply Agreement between the parties."
        assert extract_tipo_documento(text) == "Product Supply Agreement"

    def test_comida_rapida(self):
        text = "Contrato comida rapida para servicio en tiendas."
        assert extract_tipo_documento(text) == "Comida Rapida"

    def test_generic_text_returns_documentos_varios(self):
        text = "Documento interno de revision operativa sin clasificacion estandar."
        assert extract_tipo_documento(text) == "Documentos Varios"

    def test_empty_text_returns_none(self):
        assert extract_tipo_documento("") is None
        assert extract_tipo_documento(None) is None

    def test_convenio_logistico(self):
        text = "Convenio logistico de distribucion entre las partes."
        assert extract_tipo_documento(text) == "Convenio Logistico"

    def test_convenio_condiciones_comerciales(self):
        text = "Convenio condiciones comerciales para el periodo 2024."
        assert extract_tipo_documento(text) == "Convenio Condiciones Comerciales"

    def test_carta_intercedis(self):
        text = "Carta intercedis para transferencia de producto."
        assert extract_tipo_documento(text) == "Carta Intercedis"

    def test_solicitud_cambio_descuento(self):
        text = "Solicitud de cambio en el descuento aplicado al proveedor."
        assert extract_tipo_documento(text) == "Solicitud cambio Descuento"

    def test_none_terms_exclusion_prevents_wrong_match(self):
        # "Carta Soporte Costos" requires none_terms ("descuento", "descuentos")
        # to be absent.  When descuento is present, it should NOT match Costos.
        text = (
            "Carta Soporte de costos y descuento combinados."
        )
        result = extract_tipo_documento(text)
        assert result != "Carta Soporte Costos"

    def test_convenio_entrega_cedis_excluded_when_carta_anexo_present(self):
        # "Convenio Entrega Cedis" has none_terms ("carta anexo",).
        text = "Carta Anexo convenio entrega en cedis para entrega."
        result = extract_tipo_documento(text)
        assert result != "Convenio Entrega Cedis"

    def test_encoding_noise_cleaned_before_classification(self):
        # Mojibake-style characters should be cleaned up.
        text = "ContrataciÃƒÂ³n de servicios logisticos convenio logistico."
        assert extract_tipo_documento(text) == "Convenio Logistico"


# =========================================================================
# 4. extract_nombre_proveedor
# =========================================================================


class TestExtractNombreProveedor:
    """Tests for supplier/vendor name extraction."""

    def test_proveedor_label_line(self):
        text = "PROVEEDOR: ACME INDUSTRIES S.A. DE C.V."
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "ACME" in result

    def test_emisor_label_line(self):
        text = "EMISOR: DISTRIBUIDORA NACIONAL S.A. DE C.V."
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "DISTRIBUIDORA" in result

    def test_y_con_rfc_pattern(self):
        text = (
            "EL CLIENTE CADENA COMERCIAL OXXO S.A. DE C.V.\n"
            "y PRODUCTOS DEL CAMPO S.A. DE C.V. con RFC PDCA900101AB3\n"
            'denominado "EL PROVEEDOR".'
        )
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "PRODUCTOS DEL CAMPO" in result

    def test_confirma_acuerdo_con_pattern(self):
        text = "confirma su acuerdo con ALIMENTOS PREMIUM S.A. DE C.V. a quien en lo sucesivo se le denomine"
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "ALIMENTOS PREMIUM" in result

    def test_sociedad_mercantil_pattern(self):
        text = "por otra parte la sociedad mercantil GLOBAL TECH S.A. DE C.V. representada por Juan Perez"
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "GLOBAL TECH" in result

    def test_legal_entity_sa_de_cv_validates(self):
        text = "PROVEEDOR: TECNOLOGIA AVANZADA S.A. DE C.V."
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "S" in result and "A" in result

    def test_short_name_rejected(self):
        text = "PROVEEDOR: AB"
        result = extract_nombre_proveedor(text)
        assert result is None

    def test_blocked_name_oxxo_rejected(self):
        text = "PROVEEDOR: CADENA COMERCIAL OXXO S.A. DE C.V."
        result = extract_nombre_proveedor(text)
        assert result is None

    def test_blocked_name_femsa_rejected(self):
        text = "PROVEEDOR: FEMSA LOGISTICA S.A. DE C.V."
        result = extract_nombre_proveedor(text)
        assert result is None

    def test_blocked_name_el_cliente_rejected(self):
        text = "PROVEEDOR: EL CLIENTE DE SERVICIOS"
        result = extract_nombre_proveedor(text)
        assert result is None

    def test_no_supplier_pattern_returns_none(self):
        text = "Documento sin informacion de proveedor."
        assert extract_nombre_proveedor(text) is None

    def test_empty_text_returns_none(self):
        assert extract_nombre_proveedor("") is None
        assert extract_nombre_proveedor(None) is None

    def test_razon_social_label(self):
        text = "RAZON SOCIAL: SERVICIOS INDUSTRIALES S.A. DE C.V."
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "SERVICIOS INDUSTRIALES" in result

    def test_razon_social_proveedor_label(self):
        text = "Razón Social Proveedor: Ferrero de México S.A. de C.V."
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "Ferrero de" in result
        assert "S.A. de C.V" in result

    def test_razon_social_proveedor_label_without_colon(self):
        text = "Raz\u00f3n Social Proveedor COMPA\u00d1\u00cdA COMERCIAL HERDEZ, S.A de C.V"
        result = extract_nombre_proveedor(text)
        assert result == "COMPA\u00d1\u00cdA COMERCIAL HERDEZ, S.A de C.V"

    def test_razon_social_beats_noisy_context_ocr(self):
        text = (
            "confirma su acuerdo con YZTOR MANUEL PACA APODACA a quien en lo sucesivo se le denominará EL PROVEEDOR\n"
            "Razón Social: VICTOR MANUEL BACA APODACA\n"
            "RFC: BAAV7822025P7\n"
        )
        result = extract_nombre_proveedor(text)
        assert result == "VICTOR MANUEL BACA APODACA"

    def test_razon_social_person_name_beats_template_placeholder(self):
        text = (
            'confirma su acuerdo con NOMBRE DE RAZON SOCIAL, a quien en lo sucesivo se le denominara "EL PROVEEDOR"\n'
            "RazÃ³n Social: Felipe Velazquez Santiago\n"
            "RFC: VESF940524LT8\n"
        )
        result = extract_nombre_proveedor(text)
        assert result == "Felipe Velazquez Santiago"

    def test_razon_social_ocr_fragments_are_preferred_over_placeholder(self):
        text = (
            'confirma su acuerdo con NOMBRE DE RAZON SOCIAL, a quien en lo sucesivo se le denominara "EL PROVEEDOR"\n'
            "Raz?n Social: Feli pe Vela? quez Santiago\n"
            "RFC: NESE 940524 TS\n"
        )
        result = extract_nombre_proveedor(text)
        assert result == "Felipe Velaquez Santiago"

    def test_confirma_acuerdo_con_line_pattern(self):
        # The "confirma su acuerdo con ... a quien en lo sucesivo" pattern
        # is the context pattern that captures the ACUERDO CON case.
        text = "confirma su acuerdo con DISTRIBUIDORA NORTE S.A. DE C.V. a quien en lo sucesivo se le denomine"
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "DISTRIBUIDORA NORTE" in result

    def test_trailing_noise_stripped_from_name(self):
        text = "PROVEEDOR: INDUSTRIAS UNIDAS S.A. DE C.V. NO. CUENTA 12345"
        result = extract_nombre_proveedor(text)
        assert result is not None
        assert "CUENTA" not in result
        assert "12345" not in result

    def test_uppercase_ratio_validation(self):
        # All uppercase with >= 2 words and no legal suffix should still
        # pass the 55% uppercase-ratio check.
        text = "PROVEEDOR: MEGA DISTRIBUCIONES INTERNACIONALES"
        result = extract_nombre_proveedor(text)
        assert result is not None


# =========================================================================
# 5. normalize_company_name
# =========================================================================


class TestNormalizeCompanyName:
    """Tests for the normalize_company_name helper."""

    def test_strips_surrounding_whitespace(self):
        # The trailing dot of "C.V." is stripped by the strip(" .:-|") call.
        result = normalize_company_name("  ACME S.A. DE C.V.  ")
        assert result == "ACME S.A. DE C.V"

    def test_strips_leading_y(self):
        result = normalize_company_name("y PRODUCTOS DEL CAMPO S.A. DE C.V.")
        assert "PRODUCTOS DEL CAMPO" in result
        assert not result.startswith("y ")

    def test_strips_con_rfc_suffix(self):
        result = normalize_company_name("ACME S.A. DE C.V. CON RFC XAXX010101000")
        assert "RFC" not in result
        assert "XAXX" not in result

    def test_strips_a_quien_suffix(self):
        result = normalize_company_name("ACME S.A. DE C.V. a quien en lo sucesivo")
        assert "quien" not in result

    def test_strips_representada_por_suffix(self):
        result = normalize_company_name("GLOBAL TECH S.A. DE C.V. representada por Juan Perez")
        assert "representada" not in result
        assert "Juan" not in result

    def test_strips_trailing_account_noise(self):
        result = normalize_company_name("INDUSTRIAS S.A. DE C.V. NUM. CUENTA 999")
        assert "CUENTA" not in result

    def test_strips_acuerdo_con_prefix(self):
        result = normalize_company_name("ACUERDO CON DISTRIBUIDORA S.A. DE C.V.")
        assert "ACUERDO" not in result
        assert "DISTRIBUIDORA" in result

    def test_multiple_spaces_collapsed(self):
        result = normalize_company_name("ACME   CORP   S.A.   DE   C.V.")
        assert "  " not in result

    def test_strips_punctuation_edges(self):
        # strip(" .:-|") removes leading and trailing dots.
        result = normalize_company_name("...ACME S.A. DE C.V....")
        assert not result.startswith(".")
        assert "ACME" in result
        assert result.endswith("C.V")

    def test_extracts_sa_de_cv_ending(self):
        result = normalize_company_name("ACME CORP S.A. DE C.V. y demas clausulas")
        # COMPANY_END_PATTERN extracts up to S.A. DE C.V., then
        # strip(" .:-|") removes the trailing dot.
        assert result.endswith("C.V")
        assert "clausulas" not in result
