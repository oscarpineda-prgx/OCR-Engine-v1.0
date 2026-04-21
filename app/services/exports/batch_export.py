from __future__ import annotations

import logging
from typing import Iterable

from app.db.models import Batch, Document
from app.services.link2026_control import load_link2026_source_paths

logger = logging.getLogger(__name__)

EXPORT_HEADERS = [
    "batch_key", "document_id", "link", "filename", "source_type", "file_size_bytes",
    "file_size_kb", "file_path", "mime_type", "route", "processing_route", "status",
    "rfc", "fecha_documento", "tipo_documento", "nombre_proveedor",
    "quality_score", "quality_traffic_light", "quality_reasons", "error_message",
]


def excel_hyperlink_formula(target_path: str | None) -> str:
    if not target_path:
        return ""
    escaped_path = str(target_path).replace('"', '""')
    return f'=HYPERLINK("{escaped_path}")'


def build_batch_export_rows(batch: Batch, documents: Iterable[Document]) -> list[dict]:
    source_paths = load_link2026_source_paths(batch.batch_key)

    rows: list[dict] = []
    for doc in documents:
        rows.append({
            "batch_key": batch.batch_key,
            "document_id": doc.id,
            "link": excel_hyperlink_formula(source_paths.get(doc.id)),
            "filename": doc.filename,
            "source_type": doc.source_type,
            "file_size_bytes": doc.file_size,
            "file_size_kb": round((doc.file_size or 0) / 1024, 2),
            "file_path": doc.file_path,
            "mime_type": doc.mime_type,
            "route": doc.route,
            "processing_route": doc.processing_route,
            "status": doc.status,
            "rfc": doc.rfc,
            "fecha_documento": doc.fecha_documento,
            "tipo_documento": doc.tipo_documento,
            "nombre_proveedor": doc.nombre_proveedor,
            "quality_score": doc.quality_score,
            "quality_traffic_light": doc.quality_traffic_light,
            "quality_reasons": doc.quality_reasons,
            "error_message": doc.error_message,
        })

    return rows
