import asyncio
import csv
import hashlib
import json
import logging
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import asc, case, desc, func
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Batch, Document, DocumentProcessingLog
from app.db.session import get_db
from app.services.extraction.classifier import classify_document_route
from app.services.extraction.digital_pdf import extract_text_from_digital_pdf
from app.services.extraction.file_types import ALLOWED_EXTENSIONS
from app.services.extraction.ocr_image import extract_text_from_image_ocr
from app.services.extraction.ocr_pdf import extract_text_from_pdf_ocr
from app.services.extraction.structured_documents import extract_text_from_structured_document
from app.services.parsing.fields import (
    extract_fecha_documento,
    extract_nombre_proveedor,
    extract_rfc,
    extract_tipo_documento,
)
from app.domain.schemas import (
    BatchDeleteResponse,
    BatchDetailResponse,
    BatchListResponse,
    BatchRetryResponse,
    BatchUploadResponse,
    ProcessBatchResponse,
)
from app.services.quality.scoring import score_document_fields
from app.services.vendor_master.matcher import VendorMasterResolver

# Unified single-pass OCR (new)
try:
    from app.services.extraction.ocr_pdf import UnifiedOcrResult, unified_extract_from_pdf_ocr
    _HAS_UNIFIED_OCR = True
except ImportError:
    _HAS_UNIFIED_OCR = False

# RFC validation (new shared module)
try:
    from app.core.rfc import (
        has_valid_check_digit,
        is_acceptable_rfc,
        pick_best_rfc,
        repair_persona_fisica_rfc_ocr,
    )
    _HAS_RFC_VALIDATOR = True
except ImportError:
    _HAS_RFC_VALIDATOR = False

# Classify-and-extract in one pass (avoids opening PDF twice)
try:
    from app.services.extraction.classifier import classify_and_extract
    _HAS_CLASSIFY_EXTRACT = True
except ImportError:
    _HAS_CLASSIFY_EXTRACT = False

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/batches")
BATCH_KEY_PATTERN = re.compile(r"^BATCH-\d{8}-\d{6}-[a-f0-9]{6}$")
MAX_UPLOAD_SIZE_BYTES = settings.max_upload_size_mb * 1024 * 1024


def _validate_batch_key(batch_key: str) -> None:
    if not BATCH_KEY_PATTERN.match(batch_key):
        raise HTTPException(status_code=400, detail="Invalid batch_key format")


def _file_hash(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_batch_status(documents: list[Document]) -> str:
    if not documents:
        return "failed"
    statuses = {doc.status for doc in documents}
    if statuses == {"processed"}:
        return "completed"
    if statuses == {"failed"}:
        return "failed"
    if "processing" in statuses:
        return "processing"
    if "failed" in statuses and "processed" in statuses:
        return "partial"
    return "pending"


# ---------------------------------------------------------------------------
# Document processing — pure function safe for parallel execution
# ---------------------------------------------------------------------------


@dataclass
class DocumentResult:
    doc_id: int
    route: str
    raw_text: str
    rfc: str | None
    rfc_hint: str | None
    fecha_documento: str | None
    fecha_hint: str | None
    tipo_documento: str | None
    nombre_proveedor: str | None
    status: str
    error_message: str | None
    error_category: str | None = None  # "file_not_found", "extraction_empty", "unsupported_type", "processing_error"
    processing_route: str | None = None  # "digital" | "ocr" | "structured"


def _process_single_document(
    doc_id: int,
    file_path: str,
    source_type: str,
) -> DocumentResult:
    """Process one document. No DB access — safe for parallel execution."""
    if not Path(file_path).exists():
        return DocumentResult(
            doc_id=doc_id, route="pending", raw_text="",
            rfc=None, rfc_hint=None, fecha_documento=None, fecha_hint=None,
            tipo_documento=None, nombre_proveedor=None,
            status="failed", error_message="file not found in storage",
            error_category="file_not_found",
        )

    try:
        if _HAS_CLASSIFY_EXTRACT and source_type == "pdf":
            route, classified_text = classify_and_extract(file_path)
        else:
            route = classify_document_route(source_type=source_type, file_path=file_path)
            classified_text = ""

        extracted_text = ""
        rfc_hint = None
        fecha_hint = None

        # --- Text extraction ---
        if route == "digital_pdf":
            # Use text from classify_and_extract if available
            if classified_text and len(classified_text.strip()) >= 300:
                extracted_text = classified_text
            else:
                if not classified_text:
                    extracted_text = extract_text_from_digital_pdf(file_path)
                else:
                    extracted_text = classified_text
                if len((extracted_text or "").strip()) < 300:
                    if _HAS_UNIFIED_OCR:
                        ocr_result = unified_extract_from_pdf_ocr(file_path)
                        extracted_text = ocr_result.full_text
                        rfc_hint = ocr_result.rfc_hint
                        fecha_hint = ocr_result.fecha_hint
                    else:
                        extracted_text = extract_text_from_pdf_ocr(file_path)

        elif route == "ocr_image":
            if source_type == "pdf":
                if _HAS_UNIFIED_OCR:
                    ocr_result = unified_extract_from_pdf_ocr(file_path)
                    extracted_text = ocr_result.full_text
                    rfc_hint = ocr_result.rfc_hint
                    fecha_hint = ocr_result.fecha_hint
                else:
                    extracted_text = extract_text_from_pdf_ocr(file_path)
            else:
                extracted_text = extract_text_from_image_ocr(file_path)

        elif route == "structured_document":
            extracted_text = extract_text_from_structured_document(file_path)

        proc_route_map = {
            "digital_pdf": "digital",
            "ocr_image": "ocr",
            "structured_document": "structured",
        }
        proc_route = proc_route_map.get(route, route)

        if route in {"digital_pdf", "ocr_image", "structured_document"} and not (extracted_text or "").strip():
            return DocumentResult(
                doc_id=doc_id, route=route, raw_text="",
                rfc=None, rfc_hint=None, fecha_documento=None, fecha_hint=None,
                tipo_documento=None, nombre_proveedor=None,
                status="failed", error_message=f"{route} without extractable text",
                error_category="extraction_empty", processing_route=proc_route,
            )

        # --- Field extraction ---
        rfc = None
        fecha_documento = None
        tipo_documento = None
        nombre_proveedor = None

        if route in {"digital_pdf", "ocr_image", "structured_document"}:
            fecha_documento = extract_fecha_documento(extracted_text or "")
            tipo_documento = extract_tipo_documento(extracted_text or "")
            nombre_proveedor = extract_nombre_proveedor(extracted_text or "")
            rfc = extract_rfc(extracted_text or "")

            repaired_rfc = None
            if _HAS_RFC_VALIDATOR and not rfc and nombre_proveedor:
                repaired_rfc = repair_persona_fisica_rfc_ocr(extracted_text or "", nombre_proveedor)
                if repaired_rfc is None and rfc_hint:
                    repaired_rfc = repair_persona_fisica_rfc_ocr(rfc_hint, nombre_proveedor)

            # --- RFC: pick best between text extraction and OCR hint ---
            if _HAS_RFC_VALIDATOR and rfc_hint:
                candidates = [c for c in [rfc, repaired_rfc, rfc_hint] if c]
                if candidates:
                    rfc = pick_best_rfc(candidates)
            elif not rfc and rfc_hint:
                rfc = rfc_hint

            # Fecha focused on a dedicated OCR ROI is more reliable than a
            # generic body-date match from the full text.
            if fecha_hint:
                fecha_documento = fecha_hint

        if route == "unknown":
            return DocumentResult(
                doc_id=doc_id, route=route, raw_text=extracted_text,
                rfc=None, rfc_hint=None, fecha_documento=None, fecha_hint=None,
                tipo_documento=None, nombre_proveedor=None,
                status="failed", error_message=f"unsupported source_type '{source_type}'",
                error_category="unsupported_type",
            )

        return DocumentResult(
            doc_id=doc_id, route=route, raw_text=extracted_text,
            rfc=rfc, rfc_hint=rfc_hint,
            fecha_documento=fecha_documento, fecha_hint=fecha_hint,
            tipo_documento=tipo_documento, nombre_proveedor=nombre_proveedor,
            status="processed", error_message=None,
            processing_route=proc_route,
        )

    except Exception as exc:
        logger.exception("Processing error for doc %d (%s)", doc_id, file_path)
        return DocumentResult(
            doc_id=doc_id, route="pending", raw_text="",
            rfc=None, rfc_hint=None, fecha_documento=None, fecha_hint=None,
            tipo_documento=None, nombre_proveedor=None,
            status="failed", error_message=f"processing error: {str(exc)}",
            error_category="processing_error",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/ping")
def ping_batches() -> dict:
    return {"module": "batches", "status": "ok"}


@router.post("/upload", response_model=BatchUploadResponse)
async def upload_batch(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> dict:
    accepted_files: list[str] = []
    rejected_files: list[dict] = []
    batch_key = f"BATCH-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
    batch_record = Batch(batch_key=batch_key, status="pending")
    db.add(batch_record)
    db.commit()
    db.refresh(batch_record)
    batch_dir = Path("data/incoming") / batch_key
    batch_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[dict] = []

    for file in files:
        original_name = file.filename or ""
        safe_name = Path(original_name).name
        extension = Path(safe_name).suffix.lower()

        if not safe_name:
            rejected_files.append({"filename": None, "reason": "missing filename"})
            continue

        if extension not in ALLOWED_EXTENSIONS:
            rejected_files.append(
                {"filename": safe_name, "reason": f"unsupported extension '{extension}'"}
            )
            continue

        accepted_files.append(original_name)
        stored_name = f"{uuid4().hex}_{safe_name}"
        target_path = batch_dir / stored_name

        with target_path.open("wb") as buffer:
            await asyncio.to_thread(shutil.copyfileobj, file.file, buffer)

        file_size = target_path.stat().st_size
        if file_size > MAX_UPLOAD_SIZE_BYTES:
            target_path.unlink(missing_ok=True)
            rejected_files.append(
                {"filename": safe_name, "reason": f"file exceeds {settings.max_upload_size_mb}MB limit"}
            )
            continue

        computed_hash = await asyncio.to_thread(_file_hash, str(target_path))

        document_record = Document(
            batch_id=batch_record.id,
            filename=original_name,
            source_type=extension.replace(".", ""),
            route="pending",
            status="pending",
            file_path=str(target_path),
            file_size=file_size,
            file_hash=computed_hash,
            mime_type=file.content_type,
        )
        db.add(document_record)
        db.flush()

        entry: dict = {
            "document_id": document_record.id,
            "original_filename": original_name,
            "stored_filename": stored_name,
            "saved_path": str(target_path),
            "file_hash": computed_hash,
        }

        existing = (
            db.query(Document)
            .filter(Document.file_hash == computed_hash, Document.id != document_record.id)
            .first()
        )
        if existing:
            existing_batch = db.query(Batch).filter(Batch.id == existing.batch_id).first()
            entry["duplicate_of"] = existing.filename
            entry["duplicate_batch"] = existing_batch.batch_key if existing_batch else None

        saved_files.append(entry)

    batch_record.status = "pending" if accepted_files else "failed"
    db.commit()
    db.refresh(batch_record)

    return {
        "message": "batch received",
        "batch_key": batch_key,
        "batch_dir": str(batch_dir),
        "total_files": len(files),
        "accepted_files": accepted_files,
        "rejected_files": rejected_files,
        "saved_files": saved_files,
        "batch_db_id": batch_record.id,
        "upload_mode": "single_or_multiple",
    }


@router.get("/{batch_key}", response_model=BatchDetailResponse)
def get_batch(batch_key: str, db: Session = Depends(get_db)) -> dict:
    _validate_batch_key(batch_key)
    batch = db.query(Batch).filter(Batch.batch_key == batch_key).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    documents = db.query(Document).filter(Document.batch_id == batch.id).all()

    computed_status = compute_batch_status(documents)
    if batch.status != computed_status:
        batch.status = computed_status
        db.commit()
        db.refresh(batch)

    counts = {"pending": 0, "processed": 0, "failed": 0}
    for doc in documents:
        counts[doc.status] = counts.get(doc.status, 0) + 1

    return {
        "batch_key": batch.batch_key,
        "batch_db_id": batch.id,
        "batch_status": batch.status,
        "created_at": batch.created_at,
        "processing_started_at": batch.processing_started_at,
        "processing_finished_at": batch.processing_finished_at,
        "processing_seconds": batch.processing_seconds,
        "total_documents": len(documents),
        "status_counts": counts,
        "documents": [
            {
                "document_id": doc.id,
                "filename": doc.filename,
                "source_type": doc.source_type,
                "route": doc.route,
                "file_path": doc.file_path,
                "status": doc.status,
                "raw_text": doc.raw_text,
                "rfc": doc.rfc,
                "fecha_documento": doc.fecha_documento,
                "tipo_documento": doc.tipo_documento,
                "nombre_proveedor": doc.nombre_proveedor,
                "quality_score": doc.quality_score,
                "quality_traffic_light": doc.quality_traffic_light,
                "quality_reasons": doc.quality_reasons,
                "error_message": doc.error_message,
                "processing_route": doc.processing_route,
                "error_category": doc.error_category,
                "vendor_match_score": doc.vendor_match_score,
                "field_confidence": json.loads(doc.field_confidence_json) if doc.field_confidence_json else None,
            }
            for doc in documents
        ],
    }


@router.get("", response_model=BatchListResponse)
def list_batches(
    limit: int = 50,
    offset: int = 0,
    status: str | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    db: Session = Depends(get_db),
) -> dict:
    # Validate sort_by against allowed columns
    allowed_sort_fields = {"created_at", "batch_key", "status"}
    if sort_by not in allowed_sort_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sort_by field. Allowed: {', '.join(sorted(allowed_sort_fields))}",
        )
    if sort_order not in {"asc", "desc"}:
        raise HTTPException(status_code=400, detail="sort_order must be 'asc' or 'desc'")

    # Validate status filter
    allowed_statuses = {"pending", "processing", "completed", "failed"}
    if status is not None and status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status filter. Allowed: {', '.join(sorted(allowed_statuses))}",
        )

    # Total unfiltered count
    total = db.query(func.count(Batch.id)).scalar() or 0

    # Build query
    query = (
        db.query(Batch, func.count(Document.id).label("total_documents"))
        .outerjoin(Document, Document.batch_id == Batch.id)
        .group_by(Batch.id)
    )

    if status is not None:
        query = query.filter(Batch.status == status)

    sort_column = getattr(Batch, sort_by)
    order_func = asc if sort_order == "asc" else desc
    query = query.order_by(order_func(sort_column))

    records = query.offset(offset).limit(limit).all()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "batch_key": b.batch_key,
                "batch_status": b.status,
                "created_at": b.created_at,
                "processing_started_at": b.processing_started_at,
                "processing_finished_at": b.processing_finished_at,
                "processing_seconds": b.processing_seconds,
                "total_documents": int(total_documents or 0),
            }
            for b, total_documents in records
        ],
    }


@router.post("/{batch_key}/process", response_model=ProcessBatchResponse)
def process_batch(batch_key: str, db: Session = Depends(get_db)) -> dict:
    _validate_batch_key(batch_key)
    batch = db.query(Batch).filter(Batch.batch_key == batch_key).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    documents = db.query(Document).filter(Document.batch_id == batch.id).all()
    if not documents:
        raise HTTPException(status_code=400, detail="Batch has no documents")

    if batch.status == "processing":
        raise HTTPException(status_code=409, detail="Batch is already processing")

    pending_docs = [doc for doc in documents if doc.status == "pending"]
    if not pending_docs:
        raise HTTPException(status_code=400, detail="Batch has no pending documents")

    start_clock = time.perf_counter()
    batch.status = "processing"
    batch.processing_started_at = datetime.now(timezone.utc)
    batch.processing_finished_at = None
    batch.processing_seconds = None
    db.commit()
    db.refresh(batch)

    # Prepare: load vendor master once, build doc-id map
    vendor_master_resolver = VendorMasterResolver.from_db(db)
    doc_map: dict[int, Document] = {}
    tasks: list[tuple[int, str, str]] = []

    for doc in pending_docs:
        doc.status = "processing"
        doc_map[doc.id] = doc
        tasks.append((doc.id, doc.file_path, doc.source_type))
    db.commit()

    processed_count = 0
    failed_count = 0
    max_workers = min(settings.max_workers, len(tasks)) or 1

    logger.info(
        "Processing batch %s: %d documents with %d workers",
        batch_key, len(tasks), max_workers,
    )

    # --- Phase 1: Parallel OCR + field extraction (no DB access) ---
    results: list[DocumentResult] = []

    if len(tasks) == 1:
        # Single document — no thread overhead
        t = tasks[0]
        results.append(_process_single_document(t[0], t[1], t[2]))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_single_document,
                    doc_id, file_path, source_type,
                ): doc_id
                for doc_id, file_path, source_type in tasks
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    doc_id = futures[future]
                    logger.exception("Unexpected error processing doc %d", doc_id)
                    results.append(DocumentResult(
                        doc_id=doc_id, route="pending", raw_text="",
                        rfc=None, rfc_hint=None, fecha_documento=None, fecha_hint=None,
                        tipo_documento=None, nombre_proveedor=None,
                        status="failed", error_message=f"thread error: {str(exc)}",
                    ))

    # --- Phase 2: Sequential DB update + vendor master fill + quality scoring ---
    for result in results:
        doc = doc_map.get(result.doc_id)
        if doc is None:
            continue

        doc.route = result.route
        doc.raw_text = result.raw_text
        doc.status = result.status
        doc.error_message = result.error_message
        doc.error_category = result.error_category
        doc.processing_route = result.processing_route

        if result.status == "failed":
            doc.rfc = None
            doc.fecha_documento = None
            doc.tipo_documento = None
            doc.nombre_proveedor = None
            doc.quality_score = None
            doc.quality_traffic_light = None
            doc.quality_reasons = None
            doc.field_confidence_json = None
            failed_count += 1
            # Log failed processing
            db.add(DocumentProcessingLog(
                document_id=doc.id, action="processed", status="failed",
                error_category=result.error_category,
                error_message=result.error_message,
                details_json=json.dumps({"route": result.route, "processing_route": result.processing_route}),
            ))
            continue

        doc.rfc = result.rfc
        doc.fecha_documento = result.fecha_documento
        doc.tipo_documento = result.tipo_documento
        doc.nombre_proveedor = result.nombre_proveedor

        # Vendor master fill (needs resolver — runs sequentially)
        doc.rfc, doc.nombre_proveedor = vendor_master_resolver.fill_missing_fields(
            rfc=doc.rfc,
            nombre_proveedor=doc.nombre_proveedor,
        )

        # Quality scoring
        quality = score_document_fields(
            rfc=doc.rfc,
            fecha_documento=doc.fecha_documento,
            tipo_documento=doc.tipo_documento,
            nombre_proveedor=doc.nombre_proveedor,
        )
        doc.quality_score = quality["score"]
        doc.quality_traffic_light = quality["traffic_light"]
        doc.quality_reasons = ",".join(quality["reasons"]) if quality["reasons"] else None
        doc.field_confidence_json = json.dumps(quality["field_confidence"])

        # Audit log
        db.add(DocumentProcessingLog(
            document_id=doc.id, action="processed", status="success",
            details_json=json.dumps({
                "route": result.route, "processing_route": result.processing_route,
                "rfc_hint": result.rfc_hint, "fecha_hint": result.fecha_hint,
                "field_confidence": quality["field_confidence"],
            }),
        ))

        processed_count += 1

    db.flush()

    refreshed_docs = db.query(Document).filter(Document.batch_id == batch.id).all()
    batch.status = compute_batch_status(refreshed_docs)
    batch.processing_finished_at = datetime.now(timezone.utc)
    batch.processing_seconds = round(time.perf_counter() - start_clock, 3)

    db.commit()
    db.refresh(batch)

    logger.info(
        "Batch %s done: %d processed, %d failed in %.1fs",
        batch_key, processed_count, failed_count, batch.processing_seconds,
    )

    return {
        "message": "batch processed",
        "batch_key": batch.batch_key,
        "batch_status": batch.status,
        "total_documents": len(documents),
        "processed_count": processed_count,
        "failed_count": failed_count,
        "processing_seconds": batch.processing_seconds,
    }


@router.get("/{batch_key}/export/csv")
def export_batch_csv(batch_key: str, db: Session = Depends(get_db)) -> FileResponse:
    _validate_batch_key(batch_key)
    batch = db.query(Batch).filter(Batch.batch_key == batch_key).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    documents = db.query(Document).filter(Document.batch_id == batch.id).all()
    if not documents:
        raise HTTPException(status_code=400, detail="Batch has no documents")

    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{batch_key}.csv"

    headers = [
        "batch_key", "document_id", "filename", "source_type", "file_size_bytes",
        "file_size_kb", "file_path", "mime_type", "route", "processing_route", "status",
        "rfc", "fecha_documento", "tipo_documento", "nombre_proveedor",
        "quality_score", "quality_traffic_light", "quality_reasons", "error_message",
    ]

    with export_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for doc in documents:
            writer.writerow({
                "batch_key": batch.batch_key,
                "document_id": doc.id,
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

    return FileResponse(path=str(export_path), media_type="text/csv", filename=export_path.name)


@router.get("/{batch_key}/export/xlsx")
def export_batch_xlsx(batch_key: str, db: Session = Depends(get_db)) -> FileResponse:
    _validate_batch_key(batch_key)
    batch = db.query(Batch).filter(Batch.batch_key == batch_key).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    documents = db.query(Document).filter(Document.batch_id == batch.id).all()
    if not documents:
        raise HTTPException(status_code=400, detail="Batch has no documents")

    export_dir = Path("data/exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    export_path = export_dir / f"{batch_key}.xlsx"

    rows = [
        {
            "batch_key": batch.batch_key,
            "document_id": doc.id,
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
        }
        for doc in documents
    ]

    df = pd.DataFrame(rows)
    df.to_excel(export_path, index=False)

    return FileResponse(
        path=str(export_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=export_path.name,
    )


@router.get("/{batch_key}/metrics")
def get_batch_metrics(batch_key: str, db: Session = Depends(get_db)) -> dict:
    _validate_batch_key(batch_key)
    batch = db.query(Batch).filter(Batch.batch_key == batch_key).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    total = db.query(func.count(Document.id)).filter(Document.batch_id == batch.id).scalar() or 0
    if not total:
        raise HTTPException(status_code=400, detail="Batch has no documents")

    def pct(value: int) -> float:
        return round((value / total) * 100, 2) if total else 0.0

    # Status counts via SQL GROUP BY
    status_counts = dict(
        db.query(Document.status, func.count())
        .filter(Document.batch_id == batch.id)
        .group_by(Document.status).all()
    )

    # Quality traffic light counts
    quality_counts = dict(
        db.query(Document.quality_traffic_light, func.count())
        .filter(Document.batch_id == batch.id, Document.quality_traffic_light.isnot(None))
        .group_by(Document.quality_traffic_light).all()
    )

    # Aggregates: field coverage, avg quality, total size
    agg = db.query(
        func.count(Document.rfc),
        func.count(Document.fecha_documento),
        func.count(Document.tipo_documento),
        func.count(Document.nombre_proveedor),
        func.coalesce(func.avg(Document.quality_score), 0),
        func.coalesce(func.sum(Document.file_size), 0),
    ).filter(Document.batch_id == batch.id).first()
    with_rfc, with_fecha, with_tipo, with_proveedor, avg_quality, total_size_bytes = agg

    # All required fields count
    all_required = db.query(func.count()).filter(
        Document.batch_id == batch.id,
        Document.rfc.isnot(None), Document.fecha_documento.isnot(None),
        Document.tipo_documento.isnot(None), Document.nombre_proveedor.isnot(None),
    ).scalar() or 0

    # Grouped counts
    source_type_counts = dict(
        db.query(Document.source_type, func.count())
        .filter(Document.batch_id == batch.id).group_by(Document.source_type).all()
    )
    route_counts = dict(
        db.query(Document.route, func.count())
        .filter(Document.batch_id == batch.id).group_by(Document.route).all()
    )
    tipo_counts = dict(
        db.query(Document.tipo_documento, func.count())
        .filter(Document.batch_id == batch.id).group_by(Document.tipo_documento).all()
    )
    error_cat_counts = dict(
        db.query(Document.error_category, func.count())
        .filter(Document.batch_id == batch.id, Document.error_category.isnot(None))
        .group_by(Document.error_category).all()
    )

    processed = status_counts.get("processed", 0)
    failed = status_counts.get("failed", 0)
    pending = status_counts.get("pending", 0)
    processing = status_counts.get("processing", 0)
    total_size_bytes = int(total_size_bytes)

    return {
        "batch_key": batch.batch_key,
        "batch_status": batch.status,
        "processing_seconds": batch.processing_seconds,
        "total_documents": total,
        "total_size_bytes": total_size_bytes,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "status_counts": {"processed": processed, "failed": failed, "pending": pending, "processing": processing},
        "status_percentages": {
            "processed_pct": pct(processed), "failed_pct": pct(failed),
            "pending_pct": pct(pending), "processing_pct": pct(processing),
        },
        "quality_counts": {
            "verde": quality_counts.get("verde", 0),
            "amarillo": quality_counts.get("amarillo", 0),
            "rojo": quality_counts.get("rojo", 0),
        },
        "quality_percentages": {
            "verde_pct": pct(quality_counts.get("verde", 0)),
            "amarillo_pct": pct(quality_counts.get("amarillo", 0)),
            "rojo_pct": pct(quality_counts.get("rojo", 0)),
        },
        "field_coverage_counts": {
            "rfc": with_rfc, "fecha_documento": with_fecha,
            "tipo_documento": with_tipo, "nombre_proveedor": with_proveedor,
        },
        "field_coverage_percentages": {
            "rfc_pct": pct(with_rfc), "fecha_documento_pct": pct(with_fecha),
            "tipo_documento_pct": pct(with_tipo), "nombre_proveedor_pct": pct(with_proveedor),
        },
        "all_required_fields_count": all_required,
        "all_required_fields_pct": pct(all_required),
        "source_type_counts": source_type_counts,
        "route_counts": route_counts,
        "tipo_documento_counts": tipo_counts,
        "error_category_counts": error_cat_counts,
        "error_rate_pct": pct(failed),
        "average_quality_score": round(float(avg_quality), 2),
    }


@router.delete("/{batch_key}", response_model=BatchDeleteResponse)
def delete_batch(batch_key: str, db: Session = Depends(get_db)) -> dict:
    _validate_batch_key(batch_key)
    batch = db.query(Batch).filter(Batch.batch_key == batch_key).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    documents = db.query(Document).filter(Document.batch_id == batch.id).all()
    documents_deleted = len(documents)

    # Delete associated files on disk
    for doc in documents:
        path = Path(doc.file_path)
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Could not delete file %s: %s", doc.file_path, exc)

    # Delete the batch (cascade removes documents)
    db.delete(batch)
    db.commit()

    logger.info("Deleted batch %s with %d documents", batch_key, documents_deleted)

    return {
        "message": "Batch deleted",
        "batch_key": batch_key,
        "documents_deleted": documents_deleted,
    }


@router.post("/{batch_key}/retry", response_model=BatchRetryResponse)
def retry_failed_documents(batch_key: str, db: Session = Depends(get_db)) -> dict:
    _validate_batch_key(batch_key)
    batch = db.query(Batch).filter(Batch.batch_key == batch_key).first()
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    failed_docs = (
        db.query(Document)
        .filter(Document.batch_id == batch.id, Document.status == "failed")
        .all()
    )

    if not failed_docs:
        return {
            "message": "No failed documents to retry",
            "batch_key": batch_key,
            "retried_count": 0,
            "success_count": 0,
            "failed_count": 0,
        }

    # Reset failed documents to pending
    for doc in failed_docs:
        doc.status = "pending"
        doc.error_message = None
    db.commit()

    retried_count = len(failed_docs)

    # Process each document reusing _process_single_document
    results: list[DocumentResult] = []
    for doc in failed_docs:
        result = _process_single_document(doc.id, doc.file_path, doc.source_type)
        results.append(result)

    # Apply results back to DB
    success_count = 0
    failed_count = 0
    doc_map = {doc.id: doc for doc in failed_docs}

    for result in results:
        doc = doc_map.get(result.doc_id)
        if doc is None:
            continue

        doc.route = result.route
        doc.raw_text = result.raw_text
        doc.status = result.status
        doc.error_message = result.error_message
        doc.error_category = result.error_category
        doc.processing_route = result.processing_route
        doc.rfc = result.rfc
        doc.fecha_documento = result.fecha_documento
        doc.tipo_documento = result.tipo_documento
        doc.nombre_proveedor = result.nombre_proveedor

        if result.status == "processed":
            success_count += 1
        else:
            failed_count += 1

        # Audit log for retry
        db.add(DocumentProcessingLog(
            document_id=doc.id, action="retried", status=result.status,
            error_category=result.error_category,
            error_message=result.error_message,
            details_json=json.dumps({"route": result.route, "processing_route": result.processing_route}),
        ))

    # Recompute batch status
    all_docs = db.query(Document).filter(Document.batch_id == batch.id).all()
    batch.status = compute_batch_status(all_docs)
    db.commit()

    logger.info(
        "Retry batch %s: %d retried, %d succeeded, %d failed",
        batch_key, retried_count, success_count, failed_count,
    )

    return {
        "message": "Retry complete",
        "batch_key": batch_key,
        "retried_count": retried_count,
        "success_count": success_count,
        "failed_count": failed_count,
    }
