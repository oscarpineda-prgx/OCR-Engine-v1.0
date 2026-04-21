from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

LINK2026_CONTROL_DIR = Path("data/control/link2026")
LINK2026_CONTROL_PARQUET = LINK2026_CONTROL_DIR / "link2026_control.parquet"
LINK2026_CONTROL_XLSX = LINK2026_CONTROL_DIR / "link2026_control.xlsx"


def coerce_document_id(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            numeric_value = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if not numeric_value.is_integer():
            return None
        return int(numeric_value)


def load_link2026_control_frame() -> pd.DataFrame | None:
    if LINK2026_CONTROL_PARQUET.exists():
        try:
            return pd.read_parquet(LINK2026_CONTROL_PARQUET)
        except Exception as exc:
            logger.warning("Could not read %s: %s", LINK2026_CONTROL_PARQUET, exc)

    if LINK2026_CONTROL_XLSX.exists():
        try:
            return pd.read_excel(LINK2026_CONTROL_XLSX, sheet_name="control")
        except Exception as exc:
            logger.warning("Could not read %s: %s", LINK2026_CONTROL_XLSX, exc)

    return None


def load_link2026_source_paths(batch_key: str) -> dict[int, str]:
    frame = load_link2026_control_frame()
    if frame is None or frame.empty:
        return {}

    required_columns = {"batch_key", "batch_document_id", "source_path"}
    if not required_columns.issubset(frame.columns):
        return {}

    batch_rows = frame[frame["batch_key"].astype(str) == batch_key]
    if batch_rows.empty:
        return {}

    source_paths: dict[int, str] = {}
    for row in batch_rows[["batch_document_id", "source_path"]].itertuples(index=False):
        document_id = coerce_document_id(row.batch_document_id)
        source_path = "" if pd.isna(row.source_path) else str(row.source_path).strip()
        if document_id is None or not source_path:
            continue
        source_paths[document_id] = source_path

    return source_paths
