from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.api.routes.batches import (
    _apply_document_result,
    _process_document_isolated,
    compute_batch_status,
)
from app.db.models import Batch, Document
from app.db.session import SessionLocal
from app.services.link2026_workflow import save_control_outputs
from app.services.vendor_master.matcher import VendorMasterResolver


DEFAULT_CONTROL_DIR = Path("data/control/link2026")
DEFAULT_REPORT_DIR = Path("data/exports/retry_reports")
RETRYABLE_CATEGORIES = {"native_processing_crash", "processing_timeout"}


@dataclass(frozen=True)
class RetryCandidate:
    control_index: int
    document_id: int
    batch_key: str
    filename: str
    file_path: str
    source_type: str
    source_path: str | None
    error_category: str | None
    error_message: str | None


def _control_paths(control_dir: Path) -> tuple[Path, Path]:
    return control_dir / "link2026_control.parquet", control_dir / "link2026_control.xlsx"


def _load_control(control_dir: Path) -> pd.DataFrame:
    parquet_path, _ = _control_paths(control_dir)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Control parquet not found: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _coerce_document_id(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_retryable(error_category: str | None, error_message: str | None) -> bool:
    message = error_message or ""
    if error_category == "processing_timeout":
        return True
    if error_category == "native_processing_crash":
        return "UnicodeEncodeError" in message or "charmap codec" in message
    return False


def _fetch_failed_documents(control_df: pd.DataFrame, *, retryable_only: bool) -> list[RetryCandidate]:
    failed_rows = control_df[control_df["tracking_status"].eq("failed")].copy()
    if failed_rows.empty:
        return []

    doc_ids = [
        _coerce_document_id(value)
        for value in failed_rows["batch_document_id"].tolist()
    ]
    wanted_doc_ids = {doc_id for doc_id in doc_ids if doc_id is not None}
    if not wanted_doc_ids:
        return []

    with SessionLocal() as db:
        documents = (
            db.query(Document, Batch)
            .join(Batch, Document.batch_id == Batch.id)
            .filter(Document.id.in_(wanted_doc_ids))
            .all()
        )
        by_doc_id = {document.id: (document, batch) for document, batch in documents}

    candidates: list[RetryCandidate] = []
    for row in failed_rows.itertuples():
        doc_id = _coerce_document_id(row.batch_document_id)
        if doc_id is None or doc_id not in by_doc_id:
            continue

        document, batch = by_doc_id[doc_id]
        if document.status != "failed":
            continue
        if retryable_only and not _is_retryable(document.error_category, document.error_message):
            continue

        source_path = None if pd.isna(row.source_path) else str(row.source_path)
        candidates.append(
            RetryCandidate(
                control_index=int(row.Index),
                document_id=document.id,
                batch_key=batch.batch_key,
                filename=document.filename,
                file_path=document.file_path,
                source_type=document.source_type,
                source_path=source_path,
                error_category=document.error_category,
                error_message=document.error_message,
            )
        )

    return candidates


def _write_analysis(control_df: pd.DataFrame, report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    failed_rows = control_df[control_df["tracking_status"].eq("failed")].copy()
    failed_rows["batch_document_id_int"] = failed_rows["batch_document_id"].map(_coerce_document_id)
    wanted_doc_ids = [doc_id for doc_id in failed_rows["batch_document_id_int"].tolist() if doc_id is not None]

    with SessionLocal() as db:
        documents = (
            db.query(Document, Batch)
            .join(Batch, Document.batch_id == Batch.id)
            .filter(Document.id.in_(wanted_doc_ids))
            .all()
        )
        rows = []
        for document, batch in documents:
            rows.append(
                {
                    "document_id": document.id,
                    "batch_key": batch.batch_key,
                    "filename": document.filename,
                    "source_type": document.source_type,
                    "route": document.route,
                    "status": document.status,
                    "error_category": document.error_category,
                    "error_message": document.error_message,
                    "retryable": _is_retryable(document.error_category, document.error_message),
                    "file_size": document.file_size,
                    "file_path": document.file_path,
                }
            )

    report = pd.DataFrame(rows).sort_values(
        by=["retryable", "error_category", "batch_key", "document_id"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    output = report_dir / f"failed_analysis_{datetime.now():%Y%m%d-%H%M%S}.csv"
    report.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def _update_control_rows(control_df: pd.DataFrame, doc_ids: set[int]) -> pd.DataFrame:
    updated = control_df.copy()
    with SessionLocal() as db:
        records = (
            db.query(Document, Batch)
            .join(Batch, Document.batch_id == Batch.id)
            .filter(Document.id.in_(doc_ids))
            .all()
        )

        affected_batch_statuses: dict[str, str] = {}
        for document, batch in records:
            affected_batch_statuses[batch.batch_key] = batch.status
            mask = (
                updated["batch_key"].astype(str).eq(batch.batch_key)
                & updated["batch_document_id"].map(_coerce_document_id).eq(document.id)
            )
            if not mask.any():
                continue

            tracking_status = "processed" if document.status == "processed" else "failed"
            updated.loc[mask, "tracking_status"] = tracking_status
            updated.loc[mask, "batch_status"] = batch.status
            updated.loc[mask, "document_status"] = document.status
            updated.loc[mask, "finalized_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            updated.loc[mask, "route"] = document.route
            updated.loc[mask, "processing_route"] = document.processing_route
            updated.loc[mask, "rfc"] = document.rfc
            updated.loc[mask, "fecha_documento"] = document.fecha_documento
            updated.loc[mask, "tipo_documento"] = document.tipo_documento
            updated.loc[mask, "nombre_proveedor"] = document.nombre_proveedor
            updated.loc[mask, "quality_score"] = document.quality_score
            updated.loc[mask, "quality_traffic_light"] = document.quality_traffic_light
            updated.loc[mask, "quality_reasons"] = document.quality_reasons
            updated.loc[mask, "error_message"] = document.error_message

        for batch_key, batch_status in affected_batch_statuses.items():
            updated.loc[updated["batch_key"].astype(str).eq(batch_key), "batch_status"] = batch_status

    return updated


def _run_retry(candidates: list[RetryCandidate], *, workers: int) -> dict:
    if not candidates:
        return {"retried": 0, "processed": 0, "failed": 0, "seconds": 0.0}

    start = time.perf_counter()
    processed_count = 0
    failed_count = 0
    affected_batch_ids: set[int] = set()

    with SessionLocal() as db:
        for candidate in candidates:
            doc = db.get(Document, candidate.document_id)
            if doc is None:
                continue
            doc.status = "processing"
            doc.error_message = None
            doc.error_category = None
            doc.updated_at = datetime.now(timezone.utc)
            affected_batch_ids.add(doc.batch_id)
        db.commit()

    vendor_master_resolver = None
    source_paths_by_doc_id = {
        candidate.document_id: candidate.source_path for candidate in candidates
    }
    pending_iter = iter(enumerate(candidates, start=1))
    futures: dict = {}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for _ in range(max(1, workers)):
            try:
                index, candidate = next(pending_iter)
            except StopIteration:
                break
            print(f"Procesando retry {index}/{len(candidates)}: {candidate.filename}")
            futures[
                executor.submit(
                    _process_document_isolated,
                    candidate.document_id,
                    candidate.file_path,
                    candidate.source_type,
                )
            ] = (index, candidate)

        while futures:
            done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                index, candidate = futures.pop(future)
                result = future.result()

                with SessionLocal() as db:
                    if vendor_master_resolver is None:
                        vendor_master_resolver = VendorMasterResolver.from_db(db)
                    doc = db.get(Document, candidate.document_id)
                    if doc is None:
                        continue
                    status = _apply_document_result(
                        db=db,
                        doc=doc,
                        result=result,
                        vendor_master_resolver=vendor_master_resolver,
                        source_path=source_paths_by_doc_id.get(doc.id),
                    )
                    if status == "processed":
                        processed_count += 1
                    else:
                        failed_count += 1
                    affected_batch_ids.add(doc.batch_id)
                    db.commit()

                print(f"Retry {index}/{len(candidates)} terminado: {candidate.filename} -> {result.status}")

                try:
                    next_index, next_candidate = next(pending_iter)
                except StopIteration:
                    continue
                print(f"Procesando retry {next_index}/{len(candidates)}: {next_candidate.filename}")
                futures[
                    executor.submit(
                        _process_document_isolated,
                        next_candidate.document_id,
                        next_candidate.file_path,
                        next_candidate.source_type,
                    )
                ] = (next_index, next_candidate)

    with SessionLocal() as db:
        for batch_id in affected_batch_ids:
            batch = db.get(Batch, batch_id)
            if batch is None:
                continue
            documents = db.query(Document).filter(Document.batch_id == batch.id).all()
            batch.status = compute_batch_status(documents)
        db.commit()

    return {
        "retried": len(candidates),
        "processed": processed_count,
        "failed": failed_count,
        "seconds": round(time.perf_counter() - start, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reintenta fallidos Link 2026 de forma aislada")
    parser.add_argument("command", choices=["analyze", "run"])
    parser.add_argument("--control-dir", default=str(DEFAULT_CONTROL_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all-failed", action="store_true", help="Incluye fallidos no reintentables")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    control_dir = Path(args.control_dir)
    report_dir = Path(args.report_dir)
    parquet_path, excel_path = _control_paths(control_dir)
    control_df = _load_control(control_dir)

    if args.command == "analyze":
        output = _write_analysis(control_df, report_dir)
        failed_count = int(control_df["tracking_status"].eq("failed").sum())
        retryable_count = len(_fetch_failed_documents(control_df, retryable_only=True))
        print(f"Fallidos en control: {failed_count}")
        print(f"Reintentables detectados: {retryable_count}")
        print(f"Reporte: {output}")
        return 0

    candidates = _fetch_failed_documents(control_df, retryable_only=not args.all_failed)
    if args.limit is not None:
        candidates = candidates[: args.limit]

    print(f"Candidatos a reintentar: {len(candidates)}")
    if not candidates:
        return 0

    if args.dry_run:
        for candidate in candidates[:20]:
            print(
                f" - {candidate.document_id} | {candidate.batch_key} | "
                f"{candidate.error_category} | {candidate.filename}"
            )
        return 0

    result = _run_retry(candidates, workers=args.workers)
    touched_doc_ids = {candidate.document_id for candidate in candidates}
    updated_df = _update_control_rows(control_df, touched_doc_ids)
    save_control_outputs(updated_df, parquet_path, excel_path)

    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / f"retry_result_{datetime.now():%Y%m%d-%H%M%S}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Resultado retry:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Control actualizado: {excel_path}")
    print(f"Reporte: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
