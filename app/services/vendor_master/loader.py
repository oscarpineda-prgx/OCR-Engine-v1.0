from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.core.config import get_settings
from app.db.models import VendorMaster
from app.services.vendor_master.matcher import (
    canonicalize_rfc,
    core_vendor_name,
    normalize_rfc_for_match,
    normalize_vendor_name_for_match,
)


DEFAULT_VENDOR_MASTER_DIR = Path("data/reference/vendor_master")
PREFERRED_VENDOR_MASTER_FILENAMES = (
    "Vendor Master BD.xlsx",
)


def _normalize_col_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").strip().lower())


def _detect_column(columns: Iterable[str], candidates: set[str]) -> str | None:
    normalized_to_original = {_normalize_col_name(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized_to_original:
            return normalized_to_original[candidate]
    for normalized, original in normalized_to_original.items():
        if any(candidate in normalized for candidate in candidates):
            return original
    return None


def _best_vendor_master_file() -> Path | None:
    settings = get_settings()
    directory = settings.data_dir / "reference" / "vendor_master"
    if not directory.exists():
        return None

    for preferred_name in PREFERRED_VENDOR_MASTER_FILENAMES:
        preferred_path = directory / preferred_name
        if preferred_path.exists() and preferred_path.is_file():
            return preferred_path

    candidates = sorted(directory.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def refresh_vendor_master_from_excel(
    db: Session,
    excel_path: str | Path,
    *,
    replace: bool = True,
) -> dict:
    path = Path(excel_path)
    logger.info("Starting vendor master refresh from %s", path)
    if not path.exists():
        logger.error("Excel file not found: %s", path)
        raise FileNotFoundError(f"Excel not found: {path}")

    try:
        frame = pd.read_excel(path, dtype=str)
    except (ValueError, KeyError) as exc:
        logger.warning("Excel file has an invalid format %s: %s", path, exc)
        raise
    except Exception as exc:
        logger.warning("Cannot read Excel file %s (corrupt or unsupported): %s", path, exc)
        raise

    if frame.empty:
        raise ValueError("Vendor master excel is empty")

    vendor_col = _detect_column(
        frame.columns,
        {
            "vendorname",
            "suppliername",
            "nombreproveedor",
            "nombrevendor",
            "proveedor",
            "razonsocial",
        },
    )
    rfc_col = _detect_column(
        frame.columns,
        {
            "rfc",
            "taxid",
            "federaltaxid",
            "taxidnumber",
            "registrofederaldecontribuyentes",
        },
    )

    if not vendor_col and not rfc_col:
        raise ValueError("Could not detect vendor name/RFC columns in vendor master excel")

    inserted = 0
    skipped = 0
    prepared: list[VendorMaster] = []
    seen: set[tuple[str, str]] = set()

    for _, row in frame.iterrows():
        raw_name = (str(row.get(vendor_col, "")) if vendor_col else "").strip()
        raw_rfc = (str(row.get(rfc_col, "")) if rfc_col else "").strip()

        if raw_name.lower() == "nan":
            raw_name = ""
        if raw_rfc.lower() == "nan":
            raw_rfc = ""

        normalized_name = normalize_vendor_name_for_match(raw_name) if raw_name else ""
        core_name = core_vendor_name(normalized_name) if normalized_name else ""
        canonical_rfc = canonicalize_rfc(raw_rfc)
        normalized_rfc = normalize_rfc_for_match(canonical_rfc)

        if not normalized_name and not normalized_rfc:
            skipped += 1
            continue

        dedup_key = (normalized_name, normalized_rfc)
        if dedup_key in seen:
            skipped += 1
            continue
        seen.add(dedup_key)

        prepared.append(
            VendorMaster(
                vendor_name=raw_name or None,
                rfc=canonical_rfc,
                vendor_name_normalized=normalized_name or None,
                vendor_name_core=core_name or None,
                rfc_normalized=normalized_rfc or None,
                source_file=path.name,
            )
        )
        inserted += 1

    try:
        if replace:
            db.query(VendorMaster).delete(synchronize_session=False)
            db.flush()

        if prepared:
            db.bulk_save_objects(prepared)

        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning(
            "Database transaction failed while loading vendor master from %s: %s",
            path, exc,
        )
        raise
    except Exception as exc:
        db.rollback()
        logger.warning(
            "Unexpected error during vendor master DB update from %s: %s",
            path, exc,
        )
        raise

    logger.info(
        "Vendor master refresh complete: source=%s, rows_read=%d, inserted=%d, skipped=%d",
        path.name, len(frame), inserted, skipped,
    )

    return {
        "source_file": path.name,
        "replace": replace,
        "rows_read": int(len(frame)),
        "rows_inserted": inserted,
        "rows_skipped": skipped,
        "vendor_col": vendor_col,
        "rfc_col": rfc_col,
    }


def bootstrap_vendor_master_if_empty(db: Session) -> dict:
    existing = db.query(VendorMaster.id).first()
    if existing:
        return {"loaded": False, "reason": "already_loaded"}

    best_file = _best_vendor_master_file()
    if not best_file:
        return {"loaded": False, "reason": "file_not_found"}

    stats = refresh_vendor_master_from_excel(db, best_file, replace=True)
    return {"loaded": True, **stats}
