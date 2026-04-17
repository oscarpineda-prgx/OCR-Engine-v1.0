from app.services.quality.scoring import score_document_fields


def test_all_valid_fields_score_100():
    result = score_document_fields(
        rfc="ABC850101XY9",
        fecha_documento="2024-06-15",
        tipo_documento="Convenio Entrega Cedis",
        nombre_proveedor="GRUPO CATANA SA DE CV",
    )
    assert result["score"] == 100
    assert result["traffic_light"] == "verde"
    assert result["reasons"] == []


def test_missing_all_fields_score_zero():
    result = score_document_fields(
        rfc=None, fecha_documento=None, tipo_documento=None, nombre_proveedor=None
    )
    assert result["score"] == 0
    assert result["traffic_light"] == "rojo"
    assert "missing_rfc" in result["reasons"]
    assert "missing_fecha_documento" in result["reasons"]
    assert "missing_tipo_documento" in result["reasons"]
    assert "missing_nombre_proveedor" in result["reasons"]


def test_invalid_rfc_format_reduces_score():
    result = score_document_fields(
        rfc="0000000000000",
        fecha_documento="2024-06-15",
        tipo_documento="Convenio Entrega Cedis",
        nombre_proveedor="GRUPO CATANA SA DE CV",
    )
    assert result["score"] < 100
    assert "invalid_rfc_format" in result["reasons"]


def test_invalid_date_format_reduces_score():
    result = score_document_fields(
        rfc="ABC850101XY9",
        fecha_documento="99/99/2020",
        tipo_documento="Convenio Entrega Cedis",
        nombre_proveedor="GRUPO CATANA SA DE CV",
    )
    assert result["score"] < 100
    assert "invalid_fecha_format" in result["reasons"]


def test_future_date_reduces_score():
    result = score_document_fields(
        rfc="ABC850101XY9",
        fecha_documento="2099-01-01",
        tipo_documento="Convenio Entrega Cedis",
        nombre_proveedor="GRUPO CATANA SA DE CV",
    )
    assert result["score"] < 100
    assert "fecha_future" in result["reasons"]


def test_documentos_varios_fallback_reduces_score():
    result = score_document_fields(
        rfc="ABC850101XY9",
        fecha_documento="2024-06-15",
        tipo_documento="Documentos Varios",
        nombre_proveedor="GRUPO CATANA SA DE CV",
    )
    assert result["score"] < 100
    assert "tipo_documento_fallback" in result["reasons"]


def test_short_nombre_reduces_score():
    result = score_document_fields(
        rfc="ABC850101XY9",
        fecha_documento="2024-06-15",
        tipo_documento="Convenio Entrega Cedis",
        nombre_proveedor="AB",
    )
    assert result["score"] < 100
    assert "nombre_proveedor_too_short" in result["reasons"]


def test_foreign_tax_id_accepted():
    result = score_document_fields(
        rfc="75-2218815",
        fecha_documento="2024-06-15",
        tipo_documento="Convenio Entrega Cedis",
        nombre_proveedor="ACME INC",
    )
    assert result["field_confidence"]["rfc"] == 0.9
    assert result["score"] >= 80


def test_field_confidence_returned():
    result = score_document_fields(
        rfc="ABC850101XY9",
        fecha_documento="2024-06-15",
        tipo_documento="Convenio Entrega Cedis",
        nombre_proveedor="GRUPO CATANA SA DE CV",
    )
    assert "field_confidence" in result
    assert result["field_confidence"]["rfc"] == 1.0
    assert result["field_confidence"]["fecha_documento"] == 1.0
