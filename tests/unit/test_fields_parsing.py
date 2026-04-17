from app.services.parsing.fields import extract_rfc, extract_tipo_documento


def test_extract_rfc_with_hyphenated_format():
    text = "RFC: AGD-811217-HS7"
    assert extract_rfc(text) == "AGD811217HS7"


def test_tipo_documento_portal_proveedores_detected_on_later_page():
    text = (
        "Texto preliminar sin titulo claro en primera pagina.\n\n"
        "CONVENIO PARA ACCESO A PORTAL DE PROVEEDORES\n"
        "Las partes acuerdan..."
    )
    assert extract_tipo_documento(text) == "Portal de Proveedores"


def test_tipo_documento_carta_soporte_descuentos():
    text = (
        "Carta Soporte para ajustes comerciales.\n"
        "Este documento contiene descuentos por categoria."
    )
    assert extract_tipo_documento(text) == "Carta Soporte Descuentos"


def test_tipo_documento_carta_soporte_costos():
    text = (
        "Carta Soporte de costos logisticos.\n"
        "Incremento de costo por transporte."
    )
    assert extract_tipo_documento(text) == "Carta Soporte Costos"


def test_tipo_documento_convenio_modificatorio_cedis():
    text = "Convenio Modificatorio al contrato de suministro de productos de linea en CEDIS."
    assert extract_tipo_documento(text) == "Convenio Modificatorio Cedis"


def test_tipo_documento_convenio_entrega_cedis_not_carta_anexo():
    text = "CONVENIO ENTREGA EN CEDIS para proveedores nacionales."
    assert extract_tipo_documento(text) == "Convenio Entrega Cedis"


def test_tipo_documento_documentos_varios_when_no_rule_matches():
    text = "Documento interno de revision operativa sin clasificacion estandar."
    assert extract_tipo_documento(text) == "Documentos Varios"
