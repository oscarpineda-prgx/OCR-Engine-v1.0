from __future__ import annotations

import mimetypes
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd

from app.services.extraction.file_types import ALLOWED_SOURCE_TYPES


TRACKING_COLUMNS = [
    "tracking_status",
    "batch_key",
    "batch_document_id",
    "stored_filename",
    "saved_path",
    "prepared_at",
    "batch_status",
    "document_status",
    "finalized_at",
    "route",
    "processing_route",
    "rfc",
    "fecha_documento",
    "tipo_documento",
    "nombre_proveedor",
    "quality_score",
    "quality_traffic_light",
    "quality_reasons",
    "error_message",
    "export_xlsx_path",
]


@dataclass(frozen=True)
class Link2026Paths:
    manifest_path: Path
    source_root: Path
    control_dir: Path
    api_base: str

    @property
    def file_index_path(self) -> Path:
        return self.control_dir / "link2026_file_index.parquet"

    @property
    def control_parquet_path(self) -> Path:
        return self.control_dir / "link2026_control.parquet"

    @property
    def control_excel_path(self) -> Path:
        return self.control_dir / "link2026_control.xlsx"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_manifest_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.columns = [str(column).strip() for column in df.columns]

    required = [
        "Concatenado",
        "Name",
        "Extension",
        "Attributes.Size",
        "Size KB",
        "Bandera",
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    df = df[required].copy()
    df.rename(
        columns={
            "Concatenado": "concatenado",
            "Name": "name",
            "Extension": "extension",
            "Attributes.Size": "attributes_size",
            "Size KB": "size_kb",
            "Bandera": "bandera_source",
        },
        inplace=True,
    )

    df.insert(0, "manifest_row", range(2, len(df) + 2))
    df["name"] = df["name"].map(_normalize_text)
    df["extension"] = df["extension"].map(_normalize_text)
    df["bandera_source"] = df["bandera_source"].map(_normalize_text).str.lower()
    df["extension_normalized"] = df["extension"].str.lower().str.lstrip(".")
    df["attributes_size"] = (
        pd.to_numeric(df["attributes_size"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    df["size_kb"] = pd.to_numeric(df["size_kb"], errors="coerce")
    df["match_name"] = df["name"].str.lower()
    df["match_size"] = df["attributes_size"]
    df["is_supported"] = df["extension_normalized"].isin(ALLOWED_SOURCE_TYPES)
    df["match_seq"] = df.groupby(["match_name", "match_size"]).cumcount()
    df["control_key"] = (
        df["match_name"]
        + "|"
        + df["match_size"].astype(str)
        + "|"
        + df["match_seq"].astype(str)
    )
    return df


def load_manifest_dataframe(manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_excel(manifest_path)
    return normalize_manifest_dataframe(frame)


def build_file_index(source_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for current_root, _, files in os.walk(source_root):
        for file_name in files:
            full_path = Path(current_root) / file_name
            try:
                size = full_path.stat().st_size
            except OSError:
                continue

            rows.append(
                {
                    "match_name": file_name.strip().lower(),
                    "match_size": int(size),
                    "source_path": str(full_path),
                    "source_folder": full_path.parent.name,
                    "source_extension": full_path.suffix.lower().lstrip("."),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "match_name",
                "match_size",
                "source_path",
                "source_folder",
                "source_extension",
                "match_seq",
            ]
        )

    index_df = pd.DataFrame(rows)
    index_df.sort_values(
        by=["match_name", "match_size", "source_path"],
        inplace=True,
        kind="stable",
    )
    index_df["match_seq"] = index_df.groupby(["match_name", "match_size"]).cumcount()
    index_df.reset_index(drop=True, inplace=True)
    return index_df


def merge_manifest_with_index(
    manifest_df: pd.DataFrame,
    file_index_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = manifest_df.merge(
        file_index_df,
        how="left",
        on=["match_name", "match_size", "match_seq"],
    )

    merged["path_status"] = merged["source_path"].notna().map(
        lambda matched: "matched" if matched else "missing_source"
    )
    merged["tracking_status"] = "pending"
    merged.loc[merged["bandera_source"] != "procesar", "tracking_status"] = "skipped_manifest_processed"
    merged.loc[~merged["is_supported"], "tracking_status"] = "unsupported_extension"
    merged.loc[merged["source_path"].isna(), "tracking_status"] = "missing_source"

    for column in TRACKING_COLUMNS:
        if column not in merged.columns:
            merged[column] = None

    return merged


def merge_existing_tracking(
    base_df: pd.DataFrame,
    existing_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if existing_df is None or existing_df.empty:
        return base_df

    tracking = existing_df[["control_key", *[c for c in TRACKING_COLUMNS if c in existing_df.columns]]].copy()
    merged = base_df.merge(tracking, how="left", on="control_key", suffixes=("", "_old"))

    for column in TRACKING_COLUMNS:
        old_column = f"{column}_old"
        if old_column not in merged.columns:
            continue
        merged[column] = merged[old_column].combine_first(merged[column])
        merged.drop(columns=[old_column], inplace=True)

    return merged


def select_next_batch(control_df: pd.DataFrame, limit: int) -> pd.DataFrame:
    pending = control_df[control_df["tracking_status"] == "pending"].copy()
    pending.sort_values(by=["manifest_row", "name"], inplace=True, kind="stable")
    return pending.head(limit).copy()


def build_batch_summary(control_df: pd.DataFrame) -> pd.DataFrame:
    subset = control_df[control_df["batch_key"].notna()].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "batch_key",
                "files_in_batch",
                "prepared_at",
                "batch_status",
                "processed_documents",
                "failed_documents",
                "export_xlsx_path",
            ]
        )

    summary = (
        subset.groupby("batch_key", dropna=False)
        .agg(
            files_in_batch=("control_key", "count"),
            prepared_at=("prepared_at", "first"),
            batch_status=("batch_status", "last"),
            processed_documents=("document_status", lambda values: sum(value == "processed" for value in values)),
            failed_documents=("document_status", lambda values: sum(value == "failed" for value in values)),
            export_xlsx_path=("export_xlsx_path", "last"),
        )
        .reset_index()
        .sort_values(by=["prepared_at", "batch_key"], ascending=[False, False], kind="stable")
    )
    return summary


def save_control_outputs(control_df: pd.DataFrame, parquet_path: Path, excel_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    control_df.to_parquet(parquet_path, index=False)

    summary_df = build_batch_summary(control_df)
    pending_preview = select_next_batch(control_df, 300)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        control_df.to_excel(writer, index=False, sheet_name="control")
        summary_df.to_excel(writer, index=False, sheet_name="batch_summary")
        pending_preview.to_excel(writer, index=False, sheet_name="next_300_preview")


def load_existing_control(parquet_path: Path) -> pd.DataFrame | None:
    if not parquet_path.exists():
        return None
    return pd.read_parquet(parquet_path)


def load_or_build_control(paths: Link2026Paths, *, refresh_index: bool = False) -> pd.DataFrame:
    paths.control_dir.mkdir(parents=True, exist_ok=True)

    if refresh_index or not paths.file_index_path.exists():
        file_index_df = build_file_index(paths.source_root)
        file_index_df.to_parquet(paths.file_index_path, index=False)
    else:
        file_index_df = pd.read_parquet(paths.file_index_path)

    manifest_df = load_manifest_dataframe(paths.manifest_path)
    base_df = merge_manifest_with_index(manifest_df, file_index_df)
    existing_df = load_existing_control(paths.control_parquet_path)
    return merge_existing_tracking(base_df, existing_df)


def _guess_content_type(path: Path) -> str:
    content_type, _ = mimetypes.guess_type(str(path))
    return content_type or "application/octet-stream"


def upload_selected_batch(
    selected_df: pd.DataFrame,
    api_base: str,
    timeout_seconds: int = 1800,
) -> dict:
    url = f"{api_base.rstrip('/')}/batches/upload"
    files: list[tuple[str, tuple[str, object, str]]] = []
    handles: list[object] = []

    try:
        for row in selected_df.itertuples(index=False):
            source_path = Path(row.source_path)
            handle = source_path.open("rb")
            handles.append(handle)
            files.append(
                (
                    "files",
                    (row.name, handle, _guess_content_type(source_path)),
                )
            )

        with httpx.Client(timeout=timeout_seconds) as client:
            response = client.post(url, files=files)
            response.raise_for_status()
            return response.json()
    finally:
        for handle in handles:
            handle.close()


def apply_prepared_batch(
    control_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    upload_payload: dict,
) -> pd.DataFrame:
    saved_files = upload_payload.get("saved_files") or []
    batch_key = upload_payload.get("batch_key")

    if len(saved_files) != len(selected_df):
        raise ValueError(
            f"Upload payload size mismatch: expected {len(selected_df)} saved files, got {len(saved_files)}"
        )

    prepared_at = _utc_now_iso()
    updated = control_df.copy()

    for control_key, saved in zip(selected_df["control_key"].tolist(), saved_files, strict=True):
        mask = updated["control_key"] == control_key
        updated.loc[mask, "tracking_status"] = "uploaded"
        updated.loc[mask, "batch_key"] = batch_key
        updated.loc[mask, "batch_document_id"] = saved.get("document_id")
        updated.loc[mask, "stored_filename"] = saved.get("stored_filename")
        updated.loc[mask, "saved_path"] = saved.get("saved_path")
        updated.loc[mask, "prepared_at"] = prepared_at
        updated.loc[mask, "batch_status"] = "pending"

    return updated


def download_batch_export(batch_key: str, api_base: str, export_path: Path) -> Path:
    url = f"{api_base.rstrip('/')}/batches/{batch_key}/export/xlsx"
    export_path.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=1800) as client:
        response = client.get(url)
        response.raise_for_status()
        export_path.write_bytes(response.content)

    return export_path


def fetch_batch_detail(batch_key: str, api_base: str) -> dict:
    url = f"{api_base.rstrip('/')}/batches/{batch_key}"
    with httpx.Client(timeout=1800) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def delete_batch(batch_key: str, api_base: str) -> dict:
    url = f"{api_base.rstrip('/')}/batches/{batch_key}"
    with httpx.Client(timeout=1800) as client:
        response = client.delete(url)
        response.raise_for_status()
        return response.json()


def apply_finalized_batch(
    control_df: pd.DataFrame,
    batch_detail: dict,
    export_path: Path,
) -> pd.DataFrame:
    batch_key = batch_detail["batch_key"]
    batch_status = batch_detail["batch_status"]
    finalized_at = _utc_now_iso()
    updated = control_df.copy()

    documents = batch_detail.get("documents") or []
    for document in documents:
        doc_id = document.get("document_id")
        mask = (
            (updated["batch_key"] == batch_key)
            & (updated["batch_document_id"] == doc_id)
        )
        if not mask.any():
            continue

        doc_status = document.get("status")
        tracking_status = (
            "processed"
            if doc_status == "processed"
            else "failed"
            if doc_status == "failed"
            else "uploaded"
        )

        updated.loc[mask, "tracking_status"] = tracking_status
        updated.loc[mask, "batch_status"] = batch_status
        updated.loc[mask, "document_status"] = doc_status
        updated.loc[mask, "finalized_at"] = finalized_at
        updated.loc[mask, "route"] = document.get("route")
        updated.loc[mask, "processing_route"] = document.get("processing_route")
        updated.loc[mask, "rfc"] = document.get("rfc")
        updated.loc[mask, "fecha_documento"] = document.get("fecha_documento")
        updated.loc[mask, "tipo_documento"] = document.get("tipo_documento")
        updated.loc[mask, "nombre_proveedor"] = document.get("nombre_proveedor")
        updated.loc[mask, "quality_score"] = document.get("quality_score")
        updated.loc[mask, "quality_traffic_light"] = document.get("quality_traffic_light")
        updated.loc[mask, "quality_reasons"] = document.get("quality_reasons")
        updated.loc[mask, "error_message"] = document.get("error_message")
        updated.loc[mask, "export_xlsx_path"] = str(export_path)

    return updated


def release_batch(control_df: pd.DataFrame, batch_key: str) -> pd.DataFrame:
    updated = control_df.copy()
    mask = updated["batch_key"] == batch_key
    if not mask.any():
        return updated

    releasable_statuses = {"uploaded", "pending", "processing"}
    release_mask = mask & updated["tracking_status"].isin(releasable_statuses)
    if not release_mask.any():
        return updated

    for column in TRACKING_COLUMNS:
        updated.loc[release_mask, column] = None

    updated.loc[release_mask, "tracking_status"] = "pending"
    return updated


def summarize_tracking(control_df: pd.DataFrame) -> dict[str, int]:
    counts = control_df["tracking_status"].fillna("unknown").value_counts().to_dict()
    return {str(key): int(value) for key, value in counts.items()}
