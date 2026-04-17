import io
from pathlib import Path
import sys
from typing import Any

import threading
import time

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.extraction.file_types import ALLOWED_EXTENSIONS

DEFAULT_API_BASE = "http://127.0.0.1:8000/api/v1"
DEFAULT_READ_TIMEOUT_SECONDS = 60
UPLOAD_TIMEOUT_SECONDS = 300
PROCESS_TIMEOUT_SECONDS = 1800

logo_path = Path(__file__).parent / "assets" / "Logo_OCR_3.svg"
prgx_logo_path = Path(__file__).parent / "assets" / "Prgx-logo.png"
css_path = Path(__file__).parent / "assets" / "style.css"

page_icon = str(logo_path) if logo_path.exists() else "📄"
st.set_page_config(page_title="OCR Local MVP", page_icon=page_icon, layout="wide")

if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def api_get(url: str) -> dict[str, Any]:
    response = requests.get(url, timeout=DEFAULT_READ_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def api_post(
    url: str,
    files: list[tuple[str, tuple[str, io.BytesIO, str]]] | None = None,
    timeout_seconds: int = UPLOAD_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    response = requests.post(url, files=files, timeout=timeout_seconds)
    response.raise_for_status()
    return response.json()


def _parse_path_lines(raw_text: str) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for line in (raw_text or "").splitlines():
        candidate = line.strip().strip('"').strip("'")
        if not candidate:
            continue
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        paths.append(Path(candidate))
    return paths


def _build_payload_files(
    uploaded_files: list[Any] | None,
    path_lines: str,
) -> tuple[list[tuple[str, tuple[str, io.BytesIO, str]]], list[str]]:
    payload_files: list[tuple[str, tuple[str, io.BytesIO, str]]] = []
    messages: list[str] = []
    allowed_ext = ALLOWED_EXTENSIONS

    for f in uploaded_files or []:
        file_bytes = io.BytesIO(f.getvalue())
        mime = f.type or "application/octet-stream"
        payload_files.append(("files", (f.name, file_bytes, mime)))

    for p in _parse_path_lines(path_lines):
        if not p.exists():
            messages.append(f"No existe: {p}")
            continue
        if not p.is_file():
            messages.append(f"No es archivo: {p}")
            continue
        if p.suffix.lower() not in allowed_ext:
            messages.append(f"Extension no permitida ({p.suffix}): {p}")
            continue
        try:
            content = p.read_bytes()
        except Exception as exc:
            messages.append(f"No se pudo leer: {p} ({exc})")
            continue
        payload_files.append(("files", (p.name, io.BytesIO(content), "application/octet-stream")))

    return payload_files, messages


def _status_badge(status: str) -> str:
    color_map = {
        "completed": "#00A65A",
        "processed": "#00A65A",
        "failed": "#D7263D",
        "error": "#D7263D",
        "pending": "#C98A00",
        "processing": "#2F7FB8",
    }
    color = color_map.get((status or "").lower(), "#6B7280")
    label = status or "unknown"
    return f"<span class='status-chip' style='background:{color};'>{label}</span>"


def _extract_batches(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ["batches", "items", "results", "data"]:
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _render_documents_table(documents: list[dict[str, Any]]) -> None:
    if not documents:
        st.info("Este lote aun no tiene documentos para mostrar")
        return

    rows: list[dict[str, Any]] = []
    for doc in documents:
        rows.append(
            {
                "document_id": doc.get("document_id"),
                "filename": doc.get("filename"),
                "route": doc.get("route"),
                "processing_route": doc.get("processing_route"),
                "status": doc.get("status"),
                "tipo_documento": doc.get("tipo_documento"),
                "rfc": doc.get("rfc"),
                "nombre_proveedor": doc.get("nombre_proveedor"),
                "fecha_documento": doc.get("fecha_documento"),
                "quality_score": doc.get("quality_score"),
                "traffic_light": doc.get("quality_traffic_light"),
                "error_category": doc.get("error_category"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_upload_summary(upload_payload: dict[str, Any]) -> None:
    accepted = upload_payload.get("accepted_files") or []
    saved = upload_payload.get("saved_files") or []
    rejected = upload_payload.get("rejected_files") or []

    if saved:
        rows = []
        for item in saved:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "filename": item.get("original_filename"),
                    "stored_filename": item.get("stored_filename"),
                    "document_id": item.get("document_id"),
                    "status": "accepted",
                    "saved_path": item.get("saved_path"),
                }
            )
        st.markdown("Archivos cargados")
        st.dataframe(rows, use_container_width=True, hide_index=True)
    elif accepted:
        rows = [{"filename": str(name), "status": "accepted"} for name in accepted]
        st.markdown("Archivos cargados")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if rejected:
        rows = []
        for item in rejected:
            if isinstance(item, str):
                rows.append({"filename": None, "reason": item, "status": "rejected"})
                continue
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "filename": item.get("original_filename") or item.get("filename"),
                    "reason": item.get("reason"),
                    "status": "rejected",
                }
            )
        st.markdown("Archivos rechazados")
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _compute_progress(status_payload: dict[str, Any]) -> tuple[float, str]:
    total_docs = int(status_payload.get("total_documents") or len(status_payload.get("documents") or []))
    counts = status_payload.get("status_counts") or {}
    processed = int(counts.get("processed") or 0)
    failed = int(counts.get("failed") or 0)
    done = processed + failed
    progress = (done / total_docs) if total_docs > 0 else 0.0
    text = f"{done}/{total_docs} documentos procesados ({int(progress * 100)}%)"
    return min(max(progress, 0.0), 1.0), text


def _format_seconds(total_seconds: float) -> str:
    secs = max(0, int(total_seconds))
    mins, sec = divmod(secs, 60)
    hrs, min_ = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs:02d}:{min_:02d}:{sec:02d}"
    return f"{min_:02d}:{sec:02d}"


def _build_metrics_kpis(metrics_payload: dict[str, Any]) -> dict[str, float]:
    total_docs = int(metrics_payload.get("total_documents") or 0)
    status_counts = metrics_payload.get("status_counts") or {}
    status_percentages = metrics_payload.get("status_percentages") or {}

    processed = int(status_counts.get("processed") or 0)
    processing_seconds = float(metrics_payload.get("processing_seconds") or 0)
    avg_quality = float(metrics_payload.get("average_quality_score") or 0)
    total_size_mb = float(metrics_payload.get("total_size_mb") or 0)

    success_rate = status_percentages.get("processed_pct")
    if success_rate is None:
        success_rate = round((processed / total_docs) * 100, 2) if total_docs else 0.0

    return {
        "processing_seconds": processing_seconds,
        "total_documents": total_docs,
        "avg_quality": avg_quality,
        "success_rate": float(success_rate),
        "total_size_mb": total_size_mb,
    }


def _render_metrics_sections(metrics_payload: dict[str, Any]) -> None:
    with st.expander("Composicion de archivos", expanded=False):
        source_counts = metrics_payload.get("source_type_counts") or {}
        source_pct = metrics_payload.get("source_type_percentages") or {}
        rows = [
            {
                "tipo_archivo": key,
                "cantidad": source_counts.get(key, 0),
                "porcentaje": source_pct.get(key, 0.0),
            }
            for key in sorted(source_counts.keys())
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Sin datos de composicion por tipo de archivo.")

    with st.expander("Rutas de procesamiento", expanded=False):
        route_counts = metrics_payload.get("route_counts") or {}
        route_pct = metrics_payload.get("route_percentages") or {}
        rows = [
            {
                "route": key,
                "cantidad": route_counts.get(key, 0),
                "porcentaje": route_pct.get(key, 0.0),
            }
            for key in sorted(route_counts.keys())
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Sin datos de rutas de procesamiento.")

    with st.expander("Tipos de documento", expanded=False):
        tipo_counts = metrics_payload.get("tipo_documento_counts") or {}
        tipo_pct = metrics_payload.get("tipo_documento_percentages") or {}
        rows = [
            {
                "tipo_documento": key,
                "cantidad": count,
                "porcentaje": tipo_pct.get(key, 0.0),
            }
            for key, count in sorted(tipo_counts.items(), key=lambda item: (-item[1], item[0]))
        ]
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Sin datos de tipos de documento.")


    with st.expander("Cobertura de campos", expanded=False):
        coverage_pct = metrics_payload.get("field_coverage_percentages") or {}
        all_required = float(metrics_payload.get("all_required_fields_pct") or 0.0)
        rows = [
            {"indicador": "rfc_pct", "valor": coverage_pct.get("rfc_pct", 0.0)},
            {"indicador": "fecha_documento_pct", "valor": coverage_pct.get("fecha_documento_pct", 0.0)},
            {"indicador": "tipo_documento_pct", "valor": coverage_pct.get("tipo_documento_pct", 0.0)},
            {"indicador": "nombre_proveedor_pct", "valor": coverage_pct.get("nombre_proveedor_pct", 0.0)},
            {"indicador": "all_required_fields_pct", "valor": all_required},
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Errores", expanded=False):
        error_rate = float(metrics_payload.get("error_rate_pct") or 0.0)
        error_cats = metrics_payload.get("error_category_counts") or {}
        st.metric("% error", error_rate)
        if error_cats:
            rows = [
                {"categoria": cat, "cantidad": count}
                for cat, count in sorted(error_cats.items(), key=lambda x: -x[1])
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("No hay errores en este lote.")

    with st.expander("Calidad", expanded=False):
        quality_counts = metrics_payload.get("quality_counts") or {}
        quality_pct = metrics_payload.get("quality_percentages") or {}
        rows = [
            {
                "semaforo": key,
                "cantidad": quality_counts.get(key, 0),
                "porcentaje": quality_pct.get(f"{key}_pct", 0.0),
            }
            for key in ["verde", "amarillo", "rojo"]
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("Categorias de error", expanded=False):
        error_cats = metrics_payload.get("error_category_counts") or {}
        if error_cats:
            rows = [{"categoria": k, "cantidad": v} for k, v in sorted(error_cats.items(), key=lambda x: -x[1])]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.caption("Sin errores categorizados.")


header_center, header_right = st.columns([7, 1])
with header_center:
    st.markdown(
        """
        <div class="hero-card">
          <div class="section-title">OCR Local MVP</div>
          <div>Carga uno o varios archivos, procesa por lote y exporta resultados.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    if prgx_logo_path.exists():
        st.image(str(prgx_logo_path), width=78)

st.markdown("#### Configuracion")
api_base = st.text_input("API base", value=DEFAULT_API_BASE, placeholder="http://127.0.0.1:8000/api/v1")
if "upload_widget_key" not in st.session_state:
    st.session_state["upload_widget_key"] = 0
if "path_lines_value" not in st.session_state:
    st.session_state["path_lines_value"] = ""

if st.button("Nuevo lote", use_container_width=False):
    keys_to_clear = [
        "last_batch_key",
        "last_batch_status",
        "last_metrics",
        "last_batches",
    ]
    for k in keys_to_clear:
        st.session_state.pop(k, None)
    st.session_state["path_lines_value"] = ""
    st.session_state["upload_widget_key"] += 1
    st.rerun()

with st.container(border=True):
    st.markdown("### 1) Cargar Archivos")
    uploaded_files = st.file_uploader(
        "Selecciona archivos PDF/TIFF/JPG/PNG",
        type=["pdf", "tif", "tiff", "jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"uploaded_files_{st.session_state['upload_widget_key']}",
    )
    path_lines = st.text_area(
        "O pega rutas de archivos (una por linea, acepta rutas UNC)",
        placeholder="\\\\servidor\\carpeta\\archivo1.pdf\n\\\\servidor\\carpeta\\archivo2.pdf",
        height=120,
        key="path_lines_value",
    )

    if st.button("Crear lote y subir", type="primary", use_container_width=True):
        payload_files, load_messages = _build_payload_files(uploaded_files, path_lines)
        if not payload_files:
            st.warning("Selecciona archivos o pega rutas validas")
        else:
            try:
                result = api_post(f"{api_base}/batches/upload", files=payload_files)
                st.success(f"Lote creado: {result['batch_key']}")
                st.session_state["last_batch_key"] = result["batch_key"]
                _render_upload_summary(result)
                if load_messages:
                    st.warning("Algunas rutas no se cargaron:\n- " + "\n- ".join(load_messages))
            except Exception as exc:
                st.error(f"Error al subir lote: {exc}")

with st.container(border=True):
    st.markdown("### 2) Procesar Lote")
    default_batch = st.session_state.get("last_batch_key", "")
    batch_key = st.text_input("Batch key", value=default_batch)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Procesar", use_container_width=True):
            if not batch_key:
                st.warning("Ingresa un batch_key")
            else:
                try:
                    process_error: dict[str, str] = {}

                    def _run_process() -> None:
                        try:
                            api_post(
                                f"{api_base}/batches/{batch_key}/process",
                                timeout_seconds=PROCESS_TIMEOUT_SECONDS,
                            )
                        except Exception as exc:
                            process_error["message"] = str(exc)

                    worker = threading.Thread(target=_run_process, daemon=True)
                    worker.start()
                    started_at = time.time()
                    eta_target_ts: float | None = None

                    progress_placeholder = st.empty()
                    status_placeholder = st.empty()

                    while worker.is_alive():
                        try:
                            current_status = api_get(f"{api_base}/batches/{batch_key}")
                            st.session_state["last_batch_status"] = current_status
                            progress, label = _compute_progress(current_status)
                            total_docs = int(
                                current_status.get("total_documents")
                                or len(current_status.get("documents") or [])
                            )
                            counts = current_status.get("status_counts") or {}
                            done_docs = int(counts.get("processed") or 0) + int(counts.get("failed") or 0)
                            elapsed = time.time() - started_at
                            eta_text = "ETA: calculando..."
                            if done_docs > 0 and total_docs > done_docs:
                                docs_left = total_docs - done_docs
                                eta_seconds = (elapsed / done_docs) * docs_left
                                candidate_target = time.time() + max(0.0, eta_seconds)
                                if eta_target_ts is None:
                                    eta_target_ts = candidate_target
                                else:
                                    # ETA estable: solo se permite ajustar hacia abajo.
                                    eta_target_ts = min(eta_target_ts, candidate_target)
                                remaining = max(0.0, eta_target_ts - time.time())
                                eta_text = f"ETA: {_format_seconds(remaining)}"
                            elif total_docs > 0 and done_docs >= total_docs:
                                eta_text = "ETA: 00:00"
                            progress_placeholder.progress(progress, text=label)
                            status_placeholder.caption(
                                f"Estado: {current_status.get('batch_status', 'unknown')} | {eta_text}"
                            )
                        except Exception:
                            pass  # polling failure during processing — non-critical, retry on next tick
                        time.sleep(1.5)

                    worker.join(timeout=1)

                    if process_error.get("message"):
                        st.error(f"Error al procesar: {process_error['message']}")
                    else:
                        st.session_state["last_batch_status"] = api_get(f"{api_base}/batches/{batch_key}")
                        final_status = st.session_state["last_batch_status"]
                        progress, label = _compute_progress(final_status)
                        progress_placeholder.progress(progress, text=label)
                        status_placeholder.caption(f"Estado: {final_status.get('batch_status', 'unknown')}")
                        st.success("Procesamiento completado")
                except Exception as exc:
                    st.error(f"Error al procesar: {exc}")

    with col2:
        if st.button("Refrescar estado", use_container_width=True):
            if not batch_key:
                st.warning("Ingresa un batch_key")
            else:
                try:
                    st.session_state["last_batch_status"] = api_get(f"{api_base}/batches/{batch_key}")
                except Exception as exc:
                    st.error(f"Error al consultar estado: {exc}")

    status_payload = st.session_state.get("last_batch_status")
    if isinstance(status_payload, dict):
        s1, s2, s3, s4 = st.columns(4)
        total_docs = status_payload.get("total_documents") or len(status_payload.get("documents") or [])
        counts = status_payload.get("status_counts") or {}
        with s1:
            st.metric("Documentos", total_docs)
        with s2:
            st.metric("Procesados", counts.get("processed", 0))
        with s3:
            st.metric("Fallidos", counts.get("failed", 0))
        with s4:
            st.markdown(_status_badge(str(status_payload.get("batch_status", "unknown"))), unsafe_allow_html=True)

        _render_documents_table(status_payload.get("documents") or [])

with st.container(border=True):
    st.markdown("### 3) Metricas y Exportacion")
    batch_key_export = st.text_input("Batch key para metricas/export", value=st.session_state.get("last_batch_key", ""))

    if st.button("Ver metricas"):
        if not batch_key_export:
            st.warning("Ingresa un batch_key")
        else:
            try:
                metrics = api_get(f"{api_base}/batches/{batch_key_export}/metrics")
                st.session_state["last_metrics"] = metrics
            except Exception as exc:
                st.error(f"Error al obtener metricas: {exc}")

    metrics_payload = st.session_state.get("last_metrics")
    if isinstance(metrics_payload, dict):
        kpis = _build_metrics_kpis(metrics_payload)
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Total docs", kpis["total_documents"])
        with m2:
            st.metric("Tiempo (s)", kpis["processing_seconds"])
        with m3:
            st.metric("% exito", kpis["success_rate"])
        with m4:
            st.metric("Calidad prom.", kpis["avg_quality"])
        with m5:
            st.metric("Tamano lote (MB)", kpis["total_size_mb"])

        _render_metrics_sections(metrics_payload)

    if batch_key_export:
        st.markdown("Descargas")
        st.markdown(f"- [CSV]({api_base}/batches/{batch_key_export}/export/csv)")
        st.markdown(f"- [XLSX]({api_base}/batches/{batch_key_export}/export/xlsx)")

with st.container(border=True):
    st.markdown("### 4) Ultimos lotes")
    if st.button("Listar lotes"):
        try:
            st.session_state["last_batches"] = api_get(f"{api_base}/batches")
        except Exception as exc:
            st.error(f"Error al listar lotes: {exc}")

    batches_payload = st.session_state.get("last_batches")
    batches = _extract_batches(batches_payload)
    if batches:
        k1, k2, k3, k4 = st.columns(4)
        total_batches = len(batches)
        completed = sum(1 for b in batches if str(b.get("batch_status", "")).lower() == "completed")
        failed = sum(1 for b in batches if str(b.get("batch_status", "")).lower() == "failed")
        total_docs = sum(int(b.get("total_documents") or 0) for b in batches)
        with k1:
            st.metric("Lotes", total_batches)
        with k2:
            st.metric("Completados", completed)
        with k3:
            st.metric("Fallidos", failed)
        with k4:
            st.metric("Documentos", total_docs)

        st.markdown("#### Historial")
        for b in batches:
            batch_key_value = b.get("batch_key", "-")
            created_at = b.get("created_at", "-")
            status = str(b.get("batch_status", "unknown"))
            docs = b.get("total_documents", 0)
            st.markdown(
                f"<div class='batch-row'><b>{batch_key_value}</b> | docs: {docs} | {created_at} | {_status_badge(status)}</div>",
                unsafe_allow_html=True,
            )
