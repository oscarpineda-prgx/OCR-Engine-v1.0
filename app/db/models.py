from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Batch(Base):
    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    batch_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    processing_started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    processing_finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    processing_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    documents = relationship("Document", back_populates="batch", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id"), index=True)
    filename: Mapped[str] = mapped_column(String(255), index=True)
    source_type: Mapped[str] = mapped_column(String(20), default="unknown")
    route: Mapped[str] = mapped_column(String(30), default="pending")
    processing_route: Mapped[str | None] = mapped_column(String(30), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_category: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    rfc: Mapped[str | None] = mapped_column(String(20), nullable=True)
    fecha_documento: Mapped[str | None] = mapped_column(String(10), nullable=True)
    tipo_documento: Mapped[str | None] = mapped_column(String(100), nullable=True)
    nombre_proveedor: Mapped[str | None] = mapped_column(String(255), nullable=True)
    vendor_match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, onupdate=lambda: datetime.now(timezone.utc))
    file_path: Mapped[str] = mapped_column(String(500))
    file_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    file_hash: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    quality_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_traffic_light: Mapped[str | None] = mapped_column(String(20), nullable=True)
    quality_reasons: Mapped[str | None] = mapped_column(Text, nullable=True)
    field_confidence_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    batch = relationship("Batch", back_populates="documents")
    processing_logs = relationship("DocumentProcessingLog", back_populates="document", cascade="all, delete-orphan")


class VendorMaster(Base):
    __tablename__ = "vendor_master"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    vendor_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    rfc: Mapped[str | None] = mapped_column(String(20), nullable=True)
    vendor_name_normalized: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    vendor_name_core: Mapped[str | None] = mapped_column(String(255), index=True, nullable=True)
    rfc_normalized: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class DocumentProcessingLog(Base):
    __tablename__ = "document_processing_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    action: Mapped[str] = mapped_column(String(50))  # "processed", "retried", "reprocessed"
    status: Mapped[str] = mapped_column(String(20))  # "success", "failed"
    error_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON with extraction details
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    document = relationship("Document", back_populates="processing_logs")
