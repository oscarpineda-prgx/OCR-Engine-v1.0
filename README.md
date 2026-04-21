# OCR Local MVP

Motor local para procesar lotes de documentos, extraer campos clave y exponer el resultado por API, UI operativa y exportables.

Campos objetivo:
- `tipo_documento`
- `rfc`
- `nombre_proveedor`
- `fecha_documento`

El proyecto ya incluye:
- API FastAPI para carga, procesamiento, consulta, metricas y exportacion.
- UI Streamlit para operacion local.
- Clasificacion automatica de documentos por ruta de procesamiento.
- Extraccion para PDF digital, PDF escaneado, imagenes y documentos estructurados.
- Scoring de calidad con semaforo.
- Enriquecimiento con `vendor_master`.
- Flujo operativo para `Link 2026`.

## Formatos soportados

- PDF: `pdf`
- Imagen: `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`
- Word: `doc`, `docx`
- Excel: `xls`, `xlsx`, `xlsm`, `xlsb`
- Correo: `msg`

## Arquitectura actual

Componentes principales:
- `FastAPI` para la API local.
- `SQLite` para batches, documentos, resultados, vendor master y auditoria.
- `Streamlit` para la interfaz operativa.
- `Alembic` para versionado de esquema.

Ruta de procesamiento:
1. Se crea un batch y se guardan archivos en `data/incoming/<batch_key>`.
2. Se registra metadata en SQLite (`batches`, `documents`).
3. Cada documento se clasifica en una de estas rutas:
   - `digital_pdf`
   - `ocr_image`
   - `structured_document`
4. Se extrae texto segun la ruta:
   - PDF digital con `pdfplumber` y fallback a `pypdf`
   - PDF escaneado con `pypdfium2` + `pytesseract`
   - Imagen con `opencv` + `pytesseract`
   - Excel / Word / MSG con extractores estructurados
5. Se parsean los 4 campos objetivo.
6. Si falta `rfc` o `nombre_proveedor`, se completa desde `vendor_master`.
7. Se calcula `quality_score`, semaforo y razones de calidad.
8. Se expone el resultado por detalle de batch, metricas y export CSV/XLSX.

## Estructura del proyecto

Rutas clave:
- `app/main.py`: arranque FastAPI y bootstrap inicial.
- `app/api/routes/health.py`: health check del entorno.
- `app/api/routes/batches.py`: upload, proceso, detalle, metricas, export, delete y retry.
- `app/services/extraction/`: clasificadores y extractores por tipo de archivo.
- `app/services/parsing/fields.py`: parsing de RFC, fecha, tipo documental y proveedor.
- `app/services/quality/scoring.py`: scoring y semaforo.
- `app/services/vendor_master/`: carga y matching RFC <-> proveedor.
- `app/services/link2026_workflow.py`: control operativo para Link 2026.
- `app/ui/streamlit_app.py`: UI local.
- `docs/rules/`: reglas YAML de clasificacion y scoring.
- `tests/unit/`: pruebas unitarias e integracion ligera.
- `data/`: base local, archivos cargados, exports, control operativo y referencias.

## Requisitos

Python:
- Recomendado: crear un entorno virtual limpio para este proyecto.

Dependencias del sistema en Windows:
- `Tesseract OCR` instalado fuera de `pip`.
- Microsoft Word instalado si se van a procesar archivos `.doc`.

Notas:
- Si `tesseract.exe` no esta en `PATH`, configura `TESSERACT_CMD` en `.env`.
- El proyecto usa `pypdfium2` para render de PDF escaneado.
- No requiere `Ghostscript`.
- No requiere `ocrmypdf` para el flujo principal.

## Configuracion

Archivo base:
- Copiar `.env.example` a `.env`

Variables mas relevantes:
- `APP_NAME`
- `APP_VERSION`
- `LOG_LEVEL`
- `MAX_WORKERS`
- `MAX_UPLOAD_SIZE_MB`
- `TESSERACT_CMD`
- `DB_PATH` opcional, por defecto `data/ocr_local.db`

## Levantar en local

Crear entorno e instalar dependencias:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Levantar API:

```powershell
uvicorn app.main:app --reload
```

Levantar UI:

```powershell
streamlit run app/ui/streamlit_app.py
```

Notas de arranque:
- En startup la API crea tablas si no existen.
- En startup tambien intenta cargar `vendor_master` si la tabla esta vacia.
- La UI usa por defecto `http://127.0.0.1:8000/api/v1`.

## Health check

Endpoint:

```text
GET /api/v1/health
```

Valida, entre otros:
- conectividad SQLite
- disponibilidad de Tesseract
- permisos de escritura en `data/incoming`
- presencia de paquetes criticos como `pytesseract`, `pypdfium2`, `python-docx`, `extract-msg`, `xlrd`, `pyxlsb` y `pywin32`

Si algun check sale en `false`, el tipo de archivo asociado puede fallar en tiempo de proceso.

## Endpoints principales

Base:

```text
/api/v1
```

Health:
- `GET /health`

Batches:
- `GET /batches/ping`
- `POST /batches/upload`
- `GET /batches`
- `GET /batches/{batch_key}`
- `POST /batches/{batch_key}/process`
- `GET /batches/{batch_key}/metrics`
- `GET /batches/{batch_key}/export/csv`
- `GET /batches/{batch_key}/export/xlsx`
- `DELETE /batches/{batch_key}`
- `POST /batches/{batch_key}/retry`

Comportamiento general:
- `upload` acepta uno o varios archivos.
- `process` ejecuta OCR / extraccion en paralelo segun `MAX_WORKERS`.
- `metrics` devuelve cobertura, calidad, errores y tamano del lote.
- `export` genera archivo en `data/exports`.
- `delete` elimina registros y archivos fisicos del lote.
- `retry` vuelve a intentar documentos fallidos del batch.

## Clasificacion y extraccion

Clasificacion:
- PDF con texto suficiente se enruta como `digital_pdf`.
- PDF escaneado o con texto insuficiente se enruta como `ocr_image`.
- Imagenes se enrutan a OCR.
- Excel, Word y MSG se enrutan como `structured_document`.

Detalles relevantes:
- La clasificacion de PDF usa muestreo de paginas.
- Para PDF escaneado existe una extraccion OCR unificada que intenta obtener:
  - texto completo
  - `rfc_hint`
  - `fecha_hint`
- El parser esta afinado para documentos en espanol y RFC mexicano.

## Parsing de campos

Se extraen:
- `rfc`
- `fecha_documento`
- `tipo_documento`
- `nombre_proveedor`

Fuentes de reglas:
- `docs/rules/document_type_rules.yaml`
- `docs/rules/quality_thresholds.yaml`
- `docs/rules/vendor_matching.yaml`

Reglas actuales:
- `tipo_documento` se clasifica por reglas YAML con fallback a `Documentos Varios`.
- `quality_score` usa pesos por campo y semaforo:
  - verde: `>= 85`
  - amarillo: `>= 65`
  - rojo: `< 65`
- `vendor_master` usa:
  - RFC exacto normalizado
  - nombre normalizado + similitud fuzzy

## Vendor Master

Tabla:
- `vendor_master`

Comportamiento:
- Si `vendor_master` esta vacia al arrancar la API, se intenta cargar automaticamente el Excel mas reciente en `data/reference/vendor_master/*.xlsx`.
- Si existe `Vendor Master BD.xlsx`, se prioriza ese nombre.
- Solo completa campos faltantes; no esta pensado para sobreescribir datos ya extraidos.

Carga manual:

```powershell
python -m app.scripts.load_vendor_master --excel "data/reference/vendor_master/Vendor Master BD.xlsx"
```

Append en lugar de replace:

```powershell
python -m app.scripts.load_vendor_master --excel "ruta\\archivo.xlsx" --append
```

## UI Streamlit

La UI local permite:
- cargar archivos desde selector local
- pegar rutas de archivos una por linea, incluyendo rutas UNC
- crear batch
- procesar batch mostrando progreso y ETA
- consultar estado y documentos del lote
- ver metricas del lote
- descargar CSV / XLSX
- listar lotes recientes

Punto de entrada:

```powershell
streamlit run app/ui/streamlit_app.py
```

## Operacion Link 2026

El proyecto incluye un flujo especifico para preparar, controlar y finalizar lotes a partir de un manifiesto y una carpeta fuente.

Entradas:
- manifiesto: `data/Listado de Soportes 2026 Oxxo.xlsx`
- source root tipico: `\\amer.prgx.com\images\OxxoMex\Link 2026`

Archivos generados:
- `data/control/link2026/link2026_file_index.parquet`
- `data/control/link2026/link2026_control.parquet`
- `data/control/link2026/link2026_control.xlsx`

Hojas del Excel de control:
- `control`
- `batch_summary`
- `next_300_preview`

Comandos:

```powershell
python -m app.scripts.link2026_batches --manifest "data/Listado de Soportes 2026 Oxxo.xlsx" --source-root "\\amer.prgx.com\images\OxxoMex\Link 2026" --control-dir "data/control/link2026" status --preview 20 --refresh-index
python -m app.scripts.link2026_batches --manifest "data/Listado de Soportes 2026 Oxxo.xlsx" --source-root "\\amer.prgx.com\images\OxxoMex\Link 2026" --control-dir "data/control/link2026" prepare --limit 300
python -m app.scripts.link2026_batches --manifest "data/Listado de Soportes 2026 Oxxo.xlsx" --source-root "\\amer.prgx.com\images\OxxoMex\Link 2026" --control-dir "data/control/link2026" finalize --batch-key BATCH-YYYYMMDD-HHMMSS-abcdef
python -m app.scripts.link2026_batches --manifest "data/Listado de Soportes 2026 Oxxo.xlsx" --source-root "\\amer.prgx.com\images\OxxoMex\Link 2026" --control-dir "data/control/link2026" cancel --batch-key BATCH-YYYYMMDD-HHMMSS-abcdef
```

## Base de datos y auditoria

Base por defecto:
- `data/ocr_local.db`

Tablas principales:
- `batches`
- `documents`
- `vendor_master`
- `document_processing_logs`

El proyecto incluye migraciones Alembic en `alembic/versions`, aunque el arranque normal tambien crea tablas faltantes con SQLAlchemy.

## Pruebas

Suite:
- `tests/unit/`

Ejecucion:

```powershell
python -m pytest
```

Cobertura funcional principal:
- health
- clasificador
- parsing de campos
- scoring
- OCR PDF
- documentos estructurados
- vendor master
- workflow Link 2026
- endpoints de batches

## Estado del README

Este README describe el estado actual del proyecto. Ya no se documenta la UI, el clasificador ni el scoring como trabajo futuro porque esas piezas ya estan implementadas y en uso.
