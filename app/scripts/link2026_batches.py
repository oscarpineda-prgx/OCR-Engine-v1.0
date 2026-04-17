from __future__ import annotations

import argparse
from pathlib import Path

from app.services.link2026_workflow import (
    Link2026Paths,
    apply_finalized_batch,
    apply_prepared_batch,
    delete_batch,
    download_batch_export,
    fetch_batch_detail,
    load_or_build_control,
    release_batch,
    save_control_outputs,
    select_next_batch,
    summarize_tracking,
    upload_selected_batch,
)

DEFAULT_MANIFEST = Path("data/Listado de Soportes 2026 Oxxo.xlsx")
DEFAULT_SOURCE_ROOT = Path(r"\\amer.prgx.com\images\OxxoMex\Link 2026")
DEFAULT_CONTROL_DIR = Path("data/control/link2026")
DEFAULT_EXPORT_DIR = Path("data/exports")
DEFAULT_API_BASE = "http://127.0.0.1:8000/api/v1"


def _build_paths(args: argparse.Namespace) -> Link2026Paths:
    return Link2026Paths(
        manifest_path=Path(args.manifest),
        source_root=Path(args.source_root),
        control_dir=Path(args.control_dir),
        api_base=args.api_base,
    )


def _cmd_prepare(args: argparse.Namespace) -> int:
    paths = _build_paths(args)
    control_df = load_or_build_control(paths, refresh_index=args.refresh_index)
    selected_df = select_next_batch(control_df, args.limit)

    if selected_df.empty:
        print("No hay archivos pendientes para preparar.")
        print("Resumen:", summarize_tracking(control_df))
        return 0

    upload_payload = upload_selected_batch(selected_df, paths.api_base)
    updated_df = apply_prepared_batch(control_df, selected_df, upload_payload)
    save_control_outputs(updated_df, paths.control_parquet_path, paths.control_excel_path)

    print(f"Batch preparado: {upload_payload['batch_key']}")
    print(f"Archivos cargados: {len(selected_df)}")
    print(f"Control actualizado: {paths.control_excel_path}")
    print("Primeros archivos del lote:")
    for row in selected_df.head(10).itertuples(index=False):
        print(f" - {row.name} | {row.source_folder}")
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    paths = _build_paths(args)
    control_df = load_or_build_control(paths, refresh_index=False)
    batch_detail = fetch_batch_detail(args.batch_key, paths.api_base)
    export_path = DEFAULT_EXPORT_DIR / f"{args.batch_key}.xlsx"
    download_batch_export(args.batch_key, paths.api_base, export_path)
    updated_df = apply_finalized_batch(control_df, batch_detail, export_path)
    save_control_outputs(updated_df, paths.control_parquet_path, paths.control_excel_path)

    processed = sum(doc.get("status") == "processed" for doc in batch_detail.get("documents") or [])
    failed = sum(doc.get("status") == "failed" for doc in batch_detail.get("documents") or [])

    print(f"Batch finalizado: {args.batch_key}")
    print(f"Procesados: {processed}")
    print(f"Fallidos: {failed}")
    print(f"Export: {export_path}")
    print(f"Control actualizado: {paths.control_excel_path}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    paths = _build_paths(args)
    control_df = load_or_build_control(paths, refresh_index=args.refresh_index)
    summary = summarize_tracking(control_df)
    print("Resumen tracking:")
    for key, value in sorted(summary.items()):
        print(f" - {key}: {value}")

    preview = select_next_batch(control_df, args.preview)
    if preview.empty:
        print("No hay pendientes.")
        return 0

    print(f"Siguiente preview ({len(preview)} archivos):")
    for row in preview.itertuples(index=False):
        print(f" - fila {row.manifest_row}: {row.name} | {row.source_folder}")
    return 0


def _cmd_cancel(args: argparse.Namespace) -> int:
    paths = _build_paths(args)
    control_df = load_or_build_control(paths, refresh_index=False)
    delete_payload = delete_batch(args.batch_key, paths.api_base)
    updated_df = release_batch(control_df, args.batch_key)
    save_control_outputs(updated_df, paths.control_parquet_path, paths.control_excel_path)

    released_rows = int(((control_df["batch_key"] == args.batch_key) & updated_df["batch_key"].isna()).sum())

    print(f"Batch cancelado: {args.batch_key}")
    print(f"Documentos eliminados API: {delete_payload.get('documents_deleted', 0)}")
    print(f"Filas liberadas en control: {released_rows}")
    print(f"Control actualizado: {paths.control_excel_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control operativo para lotes Link 2026")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--control-dir", default=str(DEFAULT_CONTROL_DIR))
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)

    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepara el siguiente lote")
    prepare.add_argument("--limit", type=int, default=300)
    prepare.add_argument("--refresh-index", action="store_true")
    prepare.set_defaults(func=_cmd_prepare)

    finalize = subparsers.add_parser("finalize", help="Actualiza control y exporta un lote ya procesado")
    finalize.add_argument("--batch-key", required=True)
    finalize.set_defaults(func=_cmd_finalize)

    status = subparsers.add_parser("status", help="Muestra resumen y preview del siguiente lote")
    status.add_argument("--preview", type=int, default=20)
    status.add_argument("--refresh-index", action="store_true")
    status.set_defaults(func=_cmd_status)

    cancel = subparsers.add_parser("cancel", help="Elimina un lote no finalizado y libera sus filas")
    cancel.add_argument("--batch-key", required=True)
    cancel.set_defaults(func=_cmd_cancel)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
