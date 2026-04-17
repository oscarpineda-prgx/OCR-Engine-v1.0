from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    app_name: str
    version: str
    status: str
    timestamp: datetime
    checks: dict[str, bool] | None = None


# ---------------------------------------------------------------------------
# Batch upload
# ---------------------------------------------------------------------------


class BatchUploadResponse(BaseModel):
    message: str
    batch_key: str
    batch_dir: str
    total_files: int
    accepted_files: list[str]
    rejected_files: list[dict]
    saved_files: list[dict]
    batch_db_id: int
    upload_mode: str


# ---------------------------------------------------------------------------
# Batch detail
# ---------------------------------------------------------------------------


class DocumentDetail(BaseModel):
    document_id: int
    filename: str
    source_type: str
    route: str
    file_path: str
    status: str
    raw_text: str | None = None
    rfc: str | None = None
    fecha_documento: str | None = None
    tipo_documento: str | None = None
    nombre_proveedor: str | None = None
    quality_score: int | None = None
    quality_traffic_light: str | None = None
    quality_reasons: str | None = None
    error_message: str | None = None
    processing_route: str | None = None
    error_category: str | None = None
    vendor_match_score: float | None = None
    field_confidence: dict | None = None


class BatchDetailResponse(BaseModel):
    batch_key: str
    batch_db_id: int
    batch_status: str
    created_at: datetime
    processing_started_at: datetime | None = None
    processing_finished_at: datetime | None = None
    processing_seconds: float | None = None
    total_documents: int
    status_counts: dict[str, int]
    documents: list[DocumentDetail]


# ---------------------------------------------------------------------------
# Batch list
# ---------------------------------------------------------------------------


class BatchListItem(BaseModel):
    batch_key: str
    batch_status: str
    created_at: datetime
    processing_started_at: datetime | None = None
    processing_finished_at: datetime | None = None
    processing_seconds: float | None = None
    total_documents: int


class BatchListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[BatchListItem]


# ---------------------------------------------------------------------------
# Process batch
# ---------------------------------------------------------------------------


class ProcessBatchResponse(BaseModel):
    message: str
    batch_key: str
    batch_status: str
    total_documents: int
    processed_count: int
    failed_count: int
    processing_seconds: float | None = None


# ---------------------------------------------------------------------------
# Delete batch
# ---------------------------------------------------------------------------


class BatchDeleteResponse(BaseModel):
    message: str
    batch_key: str
    documents_deleted: int


# ---------------------------------------------------------------------------
# Retry failed documents
# ---------------------------------------------------------------------------


class BatchRetryResponse(BaseModel):
    message: str
    batch_key: str
    retried_count: int
    success_count: int
    failed_count: int


# ---------------------------------------------------------------------------
# Processing logs
# ---------------------------------------------------------------------------


class ProcessingLogEntry(BaseModel):
    id: int
    action: str
    status: str
    error_category: str | None = None
    error_message: str | None = None
    details: dict | None = None
    created_at: datetime


class DocumentDetailWithLogs(DocumentDetail):
    processing_logs: list[ProcessingLogEntry] = []


# ---------------------------------------------------------------------------
# Batch metrics
# ---------------------------------------------------------------------------


class BatchMetricsResponse(BaseModel):
    batch_key: str
    batch_status: str
    processing_seconds: float | None = None
    total_documents: int
    total_size_bytes: int
    total_size_mb: float
    status_counts: dict[str, int]
    status_percentages: dict[str, float]
    quality_counts: dict[str, int]
    quality_percentages: dict[str, float]
    field_coverage_counts: dict[str, int]
    field_coverage_percentages: dict[str, float]
    all_required_fields_count: int
    all_required_fields_pct: float
    source_type_counts: dict[str, int]
    route_counts: dict[str, int]
    tipo_documento_counts: dict[str, int]
    error_category_counts: dict[str, int]
    error_rate_pct: float
    average_quality_score: float
