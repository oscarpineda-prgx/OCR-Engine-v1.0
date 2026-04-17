# OCR Local MVP

Proyecto local para procesar lotes de documentos y extraer 4 campos:
- `tipo_documento`
- `rfc`
- `nombre_proveedor`
- `fecha_documento`

Formatos soportados:
- `pdf`
- `png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`
- `doc`, `docx`
- `xls`, `xlsx`, `xlsm`, `xlsb`
- `msg`

## Arquitectura Fase 1
- FastAPI (`localhost`) para API.
- SQLite local para metadatos y resultados.
- Streamlit (fase siguiente) para UI operativa.
- Pipeline hibrido para PDF digital y OCR de imagen/escaneado.

## Ejecutar en local
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

Requisitos OCR en Windows:
- Instala Tesseract OCR aparte del `pip install`.
- Si `tesseract.exe` no esta en `PATH`, configura `TESSERACT_CMD` en `.env`.
- El OCR de PDF escaneado usa `pypdfium2` para renderizar paginas y `pytesseract` para reconocer texto.
- El proyecto no requiere `Ghostscript` ni `ocrmypdf`.
- Para archivos `.doc` se usa automatizacion de Microsoft Word en Windows, por lo que el equipo debe tener Word instalado ademas de `pywin32`.

Verificacion rapida del entorno:
- `GET http://127.0.0.1:8000/api/v1/health`
- Si algun `check` clave sale en `false`, el tipo de archivo asociado no va a procesarse correctamente.

Health check:
- `GET http://127.0.0.1:8000/api/v1/health`

UI local Streamlit:
```powershell
streamlit run app/ui/streamlit_app.py
```

Pruebas:
```powershell
python -m pytest
```

## Operacion por lotes Link 2026
Control operativo para preparar lotes de 300 archivos desde `\\amer.prgx.com\images\OxxoMex\Link 2026` usando el manifiesto `data/Listado de Soportes 2026 Oxxo.xlsx`.

Comandos:
```powershell
python -m app.scripts.link2026_batches --manifest "data/Listado de Soportes 2026 Oxxo.xlsx" --source-root "\\amer.prgx.com\images\OxxoMex\Link 2026" --control-dir "data/control/link2026" status --preview 20 --refresh-index
python -m app.scripts.link2026_batches --manifest "data/Listado de Soportes 2026 Oxxo.xlsx" --source-root "\\amer.prgx.com\images\OxxoMex\Link 2026" --control-dir "data/control/link2026" prepare --limit 300
python -m app.scripts.link2026_batches --manifest "data/Listado de Soportes 2026 Oxxo.xlsx" --source-root "\\amer.prgx.com\images\OxxoMex\Link 2026" --control-dir "data/control/link2026" finalize --batch-key BATCH-YYYYMMDD-HHMMSS-abcdef
```

Archivos generados:
- `data/control/link2026/link2026_file_index.parquet`
- `data/control/link2026/link2026_control.parquet`
- `data/control/link2026/link2026_control.xlsx`

El Excel de control incluye:
- `control`: maestro completo con manifiesto, path real, batch, estado y resultados extraidos.
- `batch_summary`: resumen por lote.
- `next_300_preview`: preview de la siguiente tanda disponible.

## Nuevos endpoints utiles
- `GET /api/v1/batches` listar lotes recientes
- `GET /api/v1/batches/{batch_key}` detalle + tiempos de ejecucion
- `POST /api/v1/batches/{batch_key}/process` procesa lote y guarda duracion
- `GET /api/v1/batches/{batch_key}/metrics` metricas de cobertura/calidad
- `GET /api/v1/batches/{batch_key}/export/csv`
- `GET /api/v1/batches/{batch_key}/export/xlsx`

## Vendor Master (RFC <-> Proveedor)
- Tabla SQLite: `vendor_master`
- Carga automatica en startup: si `vendor_master` esta vacia, se intenta cargar el Excel mas reciente en `data/reference/vendor_master/*.xlsx`.
- Cruce final en procesamiento: si falta `rfc` o `nombre_proveedor`, se completa con `vendor_master` usando:
  - RFC exacto normalizado
  - Nombre normalizado + similitud fuzzy (tolerante a variaciones como `SA` vs `SA DE CV`)

Carga manual (reemplaza contenido):
```bash
.\venv\Scripts\python -m app.scripts.load_vendor_master --excel "data/reference/vendor_master/Vendor Master BD.xlsx"
```

## Proximos pasos
1. Endpoint de ingesta masiva de archivos.
2. Clasificador de ruta (digital vs OCR).
3. Parser de campos y control de calidad.
4. UI Streamlit para lote, revision y exportacion.
