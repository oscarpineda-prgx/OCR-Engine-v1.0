import importlib.util
import os
from datetime import datetime, timezone

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.domain.schemas import HealthResponse
from app.services.extraction.tesseract_runtime import is_tesseract_available

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()

    # --- Health checks ---
    # 1. Database connectivity
    db_ok = False
    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
            db_ok = True
        finally:
            db.close()
    except Exception:
        db_ok = False

    # 2. Tesseract availability
    tesseract_ok = is_tesseract_available()

    # 3. Upload directory writable
    upload_dir = settings.data_dir / "incoming"
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_dir_ok = os.access(upload_dir, os.W_OK)
    except Exception:
        upload_dir_ok = False

    pytesseract_ok = importlib.util.find_spec("pytesseract") is not None
    pypdfium2_ok = importlib.util.find_spec("pypdfium2") is not None
    python_docx_ok = importlib.util.find_spec("docx") is not None
    extract_msg_ok = importlib.util.find_spec("extract_msg") is not None
    xlrd_ok = importlib.util.find_spec("xlrd") is not None
    pyxlsb_ok = importlib.util.find_spec("pyxlsb") is not None
    pywin32_ok = (
        importlib.util.find_spec("pythoncom") is not None
        and importlib.util.find_spec("win32com") is not None
    )

    checks = {
        "database": db_ok,
        "tesseract": tesseract_ok,
        "upload_dir_writable": upload_dir_ok,
        "pytesseract_package": pytesseract_ok,
        "pypdfium2_package": pypdfium2_ok,
        "python_docx_package": python_docx_ok,
        "extract_msg_package": extract_msg_ok,
        "xlrd_package": xlrd_ok,
        "pyxlsb_package": pyxlsb_ok,
        "pywin32_package": pywin32_ok,
    }

    overall_status = "ok" if all(
        [
            db_ok,
            tesseract_ok,
            upload_dir_ok,
            pytesseract_ok,
            pypdfium2_ok,
            python_docx_ok,
            extract_msg_ok,
            xlrd_ok,
            pyxlsb_ok,
            pywin32_ok,
        ]
    ) else "degraded"

    return HealthResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        checks=checks,
    )
