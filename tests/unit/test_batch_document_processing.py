from app.api.routes.batches import _process_single_document


def test_process_single_structured_document_extracts_required_fields(monkeypatch, tmp_path):
    file_path = tmp_path / "proveedor.docx"
    file_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "app.api.routes.batches.classify_document_route",
        lambda source_type, file_path: "structured_document",
    )
    monkeypatch.setattr(
        "app.api.routes.batches.extract_text_from_structured_document",
        lambda _: "Razón Social: VICTOR MANUEL BACA APODACA\nRFC: BAAV7822025P7",
    )
    monkeypatch.setattr("app.api.routes.batches.extract_rfc", lambda _: "BAAV7822025P7")
    monkeypatch.setattr("app.api.routes.batches.extract_fecha_documento", lambda _: "2014-05-24")
    monkeypatch.setattr("app.api.routes.batches.extract_tipo_documento", lambda _: "Convenio Entrega Local")
    monkeypatch.setattr(
        "app.api.routes.batches.extract_nombre_proveedor",
        lambda _: "VICTOR MANUEL BACA APODACA",
    )

    result = _process_single_document(7, str(file_path), "docx")

    assert result.doc_id == 7
    assert result.route == "structured_document"
    assert result.processing_route == "structured"
    assert result.status == "processed"
    assert result.rfc == "BAAV7822025P7"
    assert result.fecha_documento == "2014-05-24"
    assert result.tipo_documento == "Convenio Entrega Local"
    assert result.nombre_proveedor == "VICTOR MANUEL BACA APODACA"


def test_process_single_structured_document_fails_when_no_text_is_extracted(monkeypatch, tmp_path):
    file_path = tmp_path / "correo.msg"
    file_path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "app.api.routes.batches.classify_document_route",
        lambda source_type, file_path: "structured_document",
    )
    monkeypatch.setattr(
        "app.api.routes.batches.extract_text_from_structured_document",
        lambda _: "",
    )

    result = _process_single_document(8, str(file_path), "msg")

    assert result.route == "structured_document"
    assert result.processing_route == "structured"
    assert result.status == "failed"
    assert result.error_category == "extraction_empty"
