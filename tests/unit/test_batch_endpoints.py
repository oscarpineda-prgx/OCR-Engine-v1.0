"""Integration tests for the batch API endpoints in app/api/routes/batches.py."""

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Batch, Document
from app.db.session import Base, get_db
from app.main import app

# ---------------------------------------------------------------------------
# Test database setup
# ---------------------------------------------------------------------------

TEST_DB_PATH = "test_batches.db"
TEST_DB_URL = f"sqlite:///{TEST_DB_PATH}"

engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    future=True,
)
TestSession = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)

# A valid batch_key that satisfies BATCH_KEY_PATTERN = r"^BATCH-\d{8}-\d{6}-[a-f0-9]{6}$"
VALID_BATCH_KEY = "BATCH-20250101-120000-abcdef"
VALID_BATCH_KEY_2 = "BATCH-20250102-130000-abc123"
VALID_BATCH_KEY_3 = "BATCH-20250103-140000-def456"
NONEXISTENT_BATCH_KEY = "BATCH-29990101-000000-ffffff"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def setup_module():
    """Create all tables before any test in this module runs."""
    Base.metadata.create_all(bind=engine)


def teardown_module():
    """Drop all tables and remove the test database file after all tests."""
    Base.metadata.drop_all(bind=engine)
    engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture(autouse=True)
def _clean_tables():
    """Truncate all rows between tests so each test starts with a clean slate."""
    yield
    db = TestSession()
    try:
        db.query(Document).delete()
        db.query(Batch).delete()
        db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _insert_batch(batch_key: str, status: str = "pending") -> Batch:
    """Insert a Batch row directly via the test session and return it."""
    db = TestSession()
    try:
        batch = Batch(
            batch_key=batch_key,
            status=status,
            created_at=datetime.now(timezone.utc),
        )
        db.add(batch)
        db.commit()
        db.refresh(batch)
        return batch
    finally:
        db.close()


def _insert_document(batch_id: int, **overrides) -> Document:
    """Insert a Document row directly via the test session and return it."""
    defaults = {
        "batch_id": batch_id,
        "filename": "test.pdf",
        "source_type": "pdf",
        "route": "pending",
        "status": "pending",
        "file_path": "/tmp/fake/test.pdf",
    }
    defaults.update(overrides)
    db = TestSession()
    try:
        doc = Document(**defaults)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc
    finally:
        db.close()


# ===========================================================================
# 1. GET /api/v1/batches  (list_batches)
# ===========================================================================


class TestListBatches:
    """Tests for the list_batches endpoint."""

    def test_empty_list(self):
        """Empty database returns zero items with default pagination."""
        resp = client.get("/api/v1/batches")
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"total": 0, "items": [], "limit": 50, "offset": 0}

    def test_returns_inserted_batches(self):
        """Inserted batches appear in the response."""
        _insert_batch(VALID_BATCH_KEY)
        _insert_batch(VALID_BATCH_KEY_2)

        resp = client.get("/api/v1/batches")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 2
        assert len(body["items"]) == 2

    def test_pagination_limit(self):
        """Limit restricts the number of returned items."""
        _insert_batch(VALID_BATCH_KEY)
        _insert_batch(VALID_BATCH_KEY_2)
        _insert_batch(VALID_BATCH_KEY_3)

        resp = client.get("/api/v1/batches", params={"limit": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["limit"] == 2
        assert len(body["items"]) == 2

    def test_pagination_offset(self):
        """Offset skips the first N results."""
        _insert_batch(VALID_BATCH_KEY)
        _insert_batch(VALID_BATCH_KEY_2)
        _insert_batch(VALID_BATCH_KEY_3)

        resp = client.get("/api/v1/batches", params={"offset": 2, "limit": 50})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["offset"] == 2
        assert len(body["items"]) == 1

    def test_pagination_limit_and_offset_combined(self):
        """Limit and offset work together."""
        _insert_batch(VALID_BATCH_KEY)
        _insert_batch(VALID_BATCH_KEY_2)
        _insert_batch(VALID_BATCH_KEY_3)

        resp = client.get("/api/v1/batches", params={"limit": 1, "offset": 1})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["limit"] == 1
        assert body["offset"] == 1
        assert len(body["items"]) == 1

    def test_status_filter_valid(self):
        """Filtering by a valid status returns only matching batches."""
        _insert_batch(VALID_BATCH_KEY, status="pending")
        _insert_batch(VALID_BATCH_KEY_2, status="completed")

        resp = client.get("/api/v1/batches", params={"status": "completed"})
        assert resp.status_code == 200
        body = resp.json()
        # total is the unfiltered count
        assert body["total"] == 2
        # items should only contain the completed batch
        assert len(body["items"]) == 1
        assert body["items"][0]["batch_status"] == "completed"

    def test_status_filter_invalid_returns_400(self):
        """An invalid status filter returns HTTP 400."""
        resp = client.get("/api/v1/batches", params={"status": "bogus"})
        assert resp.status_code == 400
        assert "Invalid status filter" in resp.json()["detail"]

    def test_sort_by_invalid_returns_400(self):
        """An invalid sort_by field returns HTTP 400."""
        resp = client.get("/api/v1/batches", params={"sort_by": "nonexistent_column"})
        assert resp.status_code == 400
        assert "Invalid sort_by field" in resp.json()["detail"]

    def test_sort_order_invalid_returns_400(self):
        """An invalid sort_order returns HTTP 400."""
        resp = client.get("/api/v1/batches", params={"sort_order": "upside_down"})
        assert resp.status_code == 400
        assert "sort_order must be" in resp.json()["detail"]

    def test_sort_by_batch_key_asc(self):
        """Sorting by batch_key ascending returns items in correct order."""
        _insert_batch(VALID_BATCH_KEY_3)
        _insert_batch(VALID_BATCH_KEY)
        _insert_batch(VALID_BATCH_KEY_2)

        resp = client.get(
            "/api/v1/batches",
            params={"sort_by": "batch_key", "sort_order": "asc"},
        )
        assert resp.status_code == 200
        keys = [item["batch_key"] for item in resp.json()["items"]]
        assert keys == sorted(keys)

    def test_total_documents_count(self):
        """Each batch item includes the correct total_documents count."""
        batch = _insert_batch(VALID_BATCH_KEY)
        _insert_document(batch.id, filename="a.pdf")
        _insert_document(batch.id, filename="b.pdf")

        resp = client.get("/api/v1/batches")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert len(items) == 1
        assert items[0]["total_documents"] == 2


# ===========================================================================
# 2. GET /api/v1/batches/{batch_key}  (get_batch)
# ===========================================================================


class TestGetBatch:
    """Tests for the get_batch endpoint."""

    def test_nonexistent_batch_returns_404(self):
        """Requesting a batch that does not exist returns HTTP 404."""
        resp = client.get(f"/api/v1/batches/{NONEXISTENT_BATCH_KEY}")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Batch not found"

    def test_invalid_batch_key_format_returns_400(self):
        """A batch_key that does not match the expected pattern returns 400."""
        resp = client.get("/api/v1/batches/NOT-A-VALID-KEY")
        assert resp.status_code == 400
        assert "Invalid batch_key format" in resp.json()["detail"]

    def test_existing_batch_returns_correct_details(self):
        """An existing batch returns its details and documents."""
        batch = _insert_batch(VALID_BATCH_KEY, status="pending")
        doc = _insert_document(
            batch.id,
            filename="invoice.pdf",
            source_type="pdf",
            status="pending",
            route="pending",
        )

        resp = client.get(f"/api/v1/batches/{VALID_BATCH_KEY}")
        assert resp.status_code == 200
        body = resp.json()

        assert body["batch_key"] == VALID_BATCH_KEY
        assert body["batch_db_id"] == batch.id
        assert body["total_documents"] == 1
        assert body["status_counts"]["pending"] == 1
        assert body["status_counts"]["processed"] == 0
        assert body["status_counts"]["failed"] == 0

        docs = body["documents"]
        assert len(docs) == 1
        assert docs[0]["document_id"] == doc.id
        assert docs[0]["filename"] == "invoice.pdf"
        assert docs[0]["source_type"] == "pdf"
        assert docs[0]["status"] == "pending"

    def test_batch_with_no_documents_shows_failed_status(self):
        """A batch with zero documents gets its status recomputed to 'failed'."""
        _insert_batch(VALID_BATCH_KEY, status="pending")

        resp = client.get(f"/api/v1/batches/{VALID_BATCH_KEY}")
        assert resp.status_code == 200
        body = resp.json()
        # compute_batch_status returns "failed" for an empty document list
        assert body["batch_status"] == "failed"
        assert body["total_documents"] == 0

    def test_batch_status_recomputed_to_completed(self):
        """When all documents are processed, batch status is recomputed to completed."""
        batch = _insert_batch(VALID_BATCH_KEY, status="pending")
        _insert_document(batch.id, filename="a.pdf", status="processed")
        _insert_document(batch.id, filename="b.pdf", status="processed")

        resp = client.get(f"/api/v1/batches/{VALID_BATCH_KEY}")
        assert resp.status_code == 200
        assert resp.json()["batch_status"] == "completed"

    def test_batch_status_recomputed_to_partial(self):
        """Mixed processed/failed documents result in 'partial' status."""
        batch = _insert_batch(VALID_BATCH_KEY, status="pending")
        _insert_document(batch.id, filename="a.pdf", status="processed")
        _insert_document(batch.id, filename="b.pdf", status="failed")

        resp = client.get(f"/api/v1/batches/{VALID_BATCH_KEY}")
        assert resp.status_code == 200
        assert resp.json()["batch_status"] == "partial"


# ===========================================================================
# 3. POST /api/v1/batches/{batch_key}/process  (process_batch)
# ===========================================================================


class TestProcessBatch:
    """Tests for the process_batch endpoint."""

    def test_nonexistent_batch_returns_404(self):
        """Requesting to process a non-existent batch returns HTTP 404."""
        resp = client.post(f"/api/v1/batches/{NONEXISTENT_BATCH_KEY}/process")
        assert resp.status_code == 404
        assert resp.json()["detail"] == "Batch not found"

    def test_invalid_batch_key_format_returns_400(self):
        """An invalid batch_key format returns HTTP 400."""
        resp = client.post("/api/v1/batches/INVALID/process")
        assert resp.status_code == 400
        assert "Invalid batch_key format" in resp.json()["detail"]

    def test_batch_with_no_documents_returns_400(self):
        """A batch that exists but has no documents returns HTTP 400."""
        _insert_batch(VALID_BATCH_KEY, status="pending")

        resp = client.post(f"/api/v1/batches/{VALID_BATCH_KEY}/process")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Batch has no documents"

    def test_batch_already_processing_returns_409(self):
        """A batch already in processing state cannot be started again."""
        batch = _insert_batch(VALID_BATCH_KEY, status="processing")
        _insert_document(
            batch.id,
            filename="still-processing.pdf",
            source_type="pdf",
            status="processing",
            file_path="/tmp/fake/still-processing.pdf",
        )

        resp = client.post(f"/api/v1/batches/{VALID_BATCH_KEY}/process")
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Batch is already processing"

    def test_batch_with_no_pending_documents_returns_400(self):
        """A batch with only processed documents cannot be started again."""
        batch = _insert_batch(VALID_BATCH_KEY, status="completed")
        _insert_document(
            batch.id,
            filename="done.pdf",
            source_type="pdf",
            status="processed",
            file_path="/tmp/fake/done.pdf",
        )

        resp = client.post(f"/api/v1/batches/{VALID_BATCH_KEY}/process")
        assert resp.status_code == 400
        assert resp.json()["detail"] == "Batch has no pending documents"

    def test_process_batch_with_missing_file(self):
        """Processing a batch whose document file does not exist marks it failed."""
        batch = _insert_batch(VALID_BATCH_KEY, status="pending")
        _insert_document(
            batch.id,
            filename="missing.pdf",
            source_type="pdf",
            status="pending",
            file_path="/tmp/this/path/does/not/exist/missing.pdf",
        )

        resp = client.post(f"/api/v1/batches/{VALID_BATCH_KEY}/process")
        assert resp.status_code == 200
        body = resp.json()

        assert body["batch_key"] == VALID_BATCH_KEY
        assert body["message"] == "batch processed"
        assert body["total_documents"] == 1
        assert body["failed_count"] == 1
        assert body["processed_count"] == 0
        assert body["batch_status"] == "failed"


class TestExportBatchXlsx:
    """Tests for the XLSX export endpoint."""

    def test_export_xlsx_includes_file_identity_columns(self):
        batch = _insert_batch(VALID_BATCH_KEY, status="completed")
        _insert_document(
            batch.id,
            filename="same-name.pdf",
            source_type="pdf",
            status="processed",
            route="digital_pdf",
            processing_route="digital",
            file_path=r"data\\incoming\\BATCH-20250101-120000-abcdef\\uuid_same-name.pdf",
            file_size=188766,
            mime_type="application/pdf",
            rfc="ABC123456T12",
            nombre_proveedor="Proveedor Demo",
        )

        export_path = Path("data/exports") / f"{VALID_BATCH_KEY}.xlsx"
        if export_path.exists():
            export_path.unlink()

        resp = client.get(f"/api/v1/batches/{VALID_BATCH_KEY}/export/xlsx")
        assert resp.status_code == 200
        assert export_path.exists()

        frame = pd.read_excel(export_path)
        row = frame.iloc[0].to_dict()

        assert row["document_id"] > 0
        assert row["file_size_bytes"] == 188766
        assert row["file_size_kb"] == round(188766 / 1024, 2)
        assert row["file_path"] == r"data\\incoming\\BATCH-20250101-120000-abcdef\\uuid_same-name.pdf"
        assert row["mime_type"] == "application/pdf"
        assert row["processing_route"] == "digital"
