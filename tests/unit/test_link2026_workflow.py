import pandas as pd
from pathlib import Path

from app.services.link2026_workflow import (
    apply_finalized_batch,
    apply_prepared_batch,
    merge_manifest_with_index,
    normalize_manifest_dataframe,
    release_batch,
    select_next_batch,
)


def test_normalize_manifest_dataframe_builds_tracking_keys():
    frame = pd.DataFrame(
        [
            {
                "Concatenado": "archivo.pdf100",
                "Name": "Archivo.PDF",
                "Extension": ".PDF",
                "Attributes.Size": 100,
                "Size KB": 0.1,
                "Bandera": "Procesar",
            }
        ]
    )

    normalized = normalize_manifest_dataframe(frame)

    assert normalized.loc[0, "manifest_row"] == 2
    assert normalized.loc[0, "match_name"] == "archivo.pdf"
    assert normalized.loc[0, "match_size"] == 100
    assert normalized.loc[0, "extension_normalized"] == "pdf"
    assert bool(normalized.loc[0, "is_supported"]) is True
    assert normalized.loc[0, "control_key"] == "archivo.pdf|100|0"


def test_merge_manifest_with_index_assigns_duplicate_paths_by_sequence():
    manifest = normalize_manifest_dataframe(
        pd.DataFrame(
            [
                {
                    "Concatenado": "repetido.pdf10",
                    "Name": "repetido.pdf",
                    "Extension": ".pdf",
                    "Attributes.Size": 10,
                    "Size KB": 0.01,
                    "Bandera": "Procesar",
                },
                {
                    "Concatenado": "repetido.pdf10b",
                    "Name": "repetido.pdf",
                    "Extension": ".pdf",
                    "Attributes.Size": 10,
                    "Size KB": 0.01,
                    "Bandera": "Procesar",
                },
            ]
        )
    )
    file_index = pd.DataFrame(
        [
            {
                "match_name": "repetido.pdf",
                "match_size": 10,
                "source_path": r"\\root\\A\\repetido.pdf",
                "source_folder": "A",
                "source_extension": "pdf",
                "match_seq": 0,
            },
            {
                "match_name": "repetido.pdf",
                "match_size": 10,
                "source_path": r"\\root\\B\\repetido.pdf",
                "source_folder": "B",
                "source_extension": "pdf",
                "match_seq": 1,
            },
        ]
    )

    merged = merge_manifest_with_index(manifest, file_index)

    assert merged.loc[0, "source_folder"] == "A"
    assert merged.loc[1, "source_folder"] == "B"
    assert merged["tracking_status"].tolist() == ["pending", "pending"]


def test_select_next_batch_only_returns_pending_rows():
    control = pd.DataFrame(
        [
            {"manifest_row": 2, "name": "a.pdf", "tracking_status": "pending"},
            {"manifest_row": 3, "name": "b.pdf", "tracking_status": "uploaded"},
            {"manifest_row": 4, "name": "c.pdf", "tracking_status": "pending"},
        ]
    )

    selected = select_next_batch(control, 10)

    assert selected["name"].tolist() == ["a.pdf", "c.pdf"]


def test_apply_prepared_batch_updates_selected_rows():
    control = pd.DataFrame(
        [
            {"control_key": "a|1|0", "tracking_status": "pending"},
            {"control_key": "b|2|0", "tracking_status": "pending"},
        ]
    )
    selected = pd.DataFrame(
        [
            {"control_key": "a|1|0"},
            {"control_key": "b|2|0"},
        ]
    )
    upload_payload = {
        "batch_key": "BATCH-20260407-120000-abcdef",
        "saved_files": [
            {
                "document_id": 11,
                "stored_filename": "uuid_a.pdf",
                "saved_path": "data/incoming/BATCH-.../uuid_a.pdf",
            },
            {
                "document_id": 12,
                "stored_filename": "uuid_b.pdf",
                "saved_path": "data/incoming/BATCH-.../uuid_b.pdf",
            },
        ],
    }

    updated = apply_prepared_batch(control, selected, upload_payload)

    assert updated["tracking_status"].tolist() == ["uploaded", "uploaded"]
    assert updated["batch_key"].tolist() == [upload_payload["batch_key"], upload_payload["batch_key"]]
    assert updated["batch_document_id"].tolist() == [11, 12]


def test_apply_finalized_batch_updates_document_results():
    control = pd.DataFrame(
        [
            {
                "control_key": "a|1|0",
                "batch_key": "BATCH-20260407-120000-abcdef",
                "batch_document_id": 11,
                "tracking_status": "uploaded",
            }
        ]
    )
    batch_detail = {
        "batch_key": "BATCH-20260407-120000-abcdef",
        "batch_status": "completed",
        "documents": [
            {
                "document_id": 11,
                "status": "processed",
                "route": "structured_document",
                "processing_route": "structured",
                "rfc": "BAAV7822025P7",
                "fecha_documento": "2014-05-24",
                "tipo_documento": "Convenio Entrega Local",
                "nombre_proveedor": "VICTOR MANUEL BACA APODACA",
                "quality_score": 100,
                "quality_traffic_light": "verde",
                "quality_reasons": None,
                "error_message": None,
            }
        ],
    }

    updated = apply_finalized_batch(control, batch_detail, export_path=Path("data/exports/fake.xlsx"))

    assert updated.loc[0, "tracking_status"] == "processed"
    assert updated.loc[0, "batch_status"] == "completed"
    assert updated.loc[0, "route"] == "structured_document"
    assert updated.loc[0, "processing_route"] == "structured"
    assert updated.loc[0, "rfc"] == "BAAV7822025P7"


def test_release_batch_returns_uploaded_rows_to_pending():
    control = pd.DataFrame(
        [
            {
                "control_key": "a|1|0",
                "tracking_status": "uploaded",
                "batch_key": "BATCH-20260409-074053-53cceb",
                "batch_document_id": 1301,
                "saved_path": "data/incoming/batch/a.pdf",
                "prepared_at": "2026-04-09T12:40:53+00:00",
                "batch_status": "pending",
            },
            {
                "control_key": "b|2|0",
                "tracking_status": "processed",
                "batch_key": "BATCH-OLD",
                "batch_document_id": 99,
            },
        ]
    )

    updated = release_batch(control, "BATCH-20260409-074053-53cceb")

    assert updated.loc[0, "tracking_status"] == "pending"
    assert pd.isna(updated.loc[0, "batch_key"])
    assert pd.isna(updated.loc[0, "batch_document_id"])
    assert pd.isna(updated.loc[0, "saved_path"])
    assert updated.loc[1, "tracking_status"] == "processed"
    assert updated.loc[1, "batch_key"] == "BATCH-OLD"
