"""Ground-truth evaluation script for the OCR extraction pipeline.

Loads a ground-truth CSV, runs the full extraction pipeline on each
document whose source file exists, compares extracted values against
expected values, and prints a summary report plus a detailed CSV.

Usage
-----
    python -m scripts.evaluate
    python -m scripts.evaluate --ground-truth data/ground_truth.csv --data-dir data/
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root (one level above this script's directory)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from app.services.extraction.classifier import (  # noqa: E402
    classify_document_route,
    get_source_type,
)
from app.services.extraction.digital_pdf import (  # noqa: E402
    extract_text_from_digital_pdf,
)
from app.services.extraction.ocr_image import (  # noqa: E402
    extract_text_from_image_ocr,
)
from app.services.extraction.ocr_pdf import (  # noqa: E402
    extract_text_from_pdf_ocr,
)
from app.services.parsing.fields import (  # noqa: E402
    extract_fecha_documento,
    extract_nombre_proveedor,
    extract_rfc,
    extract_tipo_documento,
)

# ---------------------------------------------------------------------------
# Normalization helpers for comparison
# ---------------------------------------------------------------------------

_ACCENT_TABLE = str.maketrans(
    "áéíóúÁÉÍÓÚñÑàèìòùÀÈÌÒÙäëïöüÄËÏÖÜ",
    "aeiouAEIOUnNaeiouAEIOUaeiouAEIOU",
)


def _normalize_rfc_for_compare(value: str | None) -> str:
    """Uppercase, strip non-alphanumeric, for exact RFC comparison."""
    if not value:
        return ""
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _normalize_text_for_fuzzy(value: str | None) -> str:
    """Lowercase, strip accents, collapse whitespace, remove punctuation."""
    if not value:
        return ""
    v = value.translate(_ACCENT_TABLE)
    v = unicodedata.normalize("NFKD", v)
    v = "".join(c for c in v if not unicodedata.combining(c))
    v = v.lower()
    v = re.sub(r"[^a-z0-9\s]", " ", v)
    v = re.sub(r"\s+", " ", v).strip()
    return v


def _fuzzy_match(expected: str | None, extracted: str | None) -> bool:
    """Case-insensitive, accent-insensitive containment match.

    Returns True if one normalized string contains the other, or if they
    are identical after normalization.  This handles minor OCR variations
    while still requiring meaningful overlap.
    """
    a = _normalize_text_for_fuzzy(expected)
    b = _normalize_text_for_fuzzy(extracted)
    if not a or not b:
        return a == b  # both empty => match
    return a == b or a in b or b in a


# ---------------------------------------------------------------------------
# Locate source file
# ---------------------------------------------------------------------------

def _find_source_file(filename: str, data_dir: Path) -> Path | None:
    """Search for *filename* in data_dir and its subdirectories."""
    # Direct match
    candidate = data_dir / filename
    if candidate.is_file():
        return candidate

    # Recursive search in subdirectories
    for child in data_dir.rglob(filename):
        if child.is_file():
            return child

    return None


# ---------------------------------------------------------------------------
# Run extraction pipeline on a single file
# ---------------------------------------------------------------------------

def _run_pipeline(file_path: Path) -> dict[str, str | None]:
    """Return a dict with keys: rfc, fecha_documento, tipo_documento, nombre_proveedor."""
    fp = str(file_path)
    source_type = get_source_type(fp)
    route = classify_document_route(source_type, fp)

    # Extract text
    text = ""
    try:
        if route == "digital_pdf":
            text = extract_text_from_digital_pdf(fp)
        elif route == "ocr_image":
            if source_type == "pdf":
                text = extract_text_from_pdf_ocr(fp)
            else:
                text = extract_text_from_image_ocr(fp)
        else:
            logger.warning("Unknown route '%s' for %s — skipping extraction", route, fp)
    except Exception as exc:
        logger.error("Text extraction failed for %s: %s", fp, exc)

    # Extract fields
    rfc = None
    fecha = None
    tipo = None
    proveedor = None

    try:
        rfc = extract_rfc(text)
    except Exception as exc:
        logger.error("RFC extraction failed for %s: %s", fp, exc)

    try:
        tipo = extract_tipo_documento(text)
    except Exception as exc:
        logger.error("Tipo documento extraction failed for %s: %s", fp, exc)

    try:
        fecha = extract_fecha_documento(text, tipo_documento=tipo)
    except Exception as exc:
        logger.error("Fecha extraction failed for %s: %s", fp, exc)

    try:
        proveedor = extract_nombre_proveedor(text)
    except Exception as exc:
        logger.error("Nombre proveedor extraction failed for %s: %s", fp, exc)

    return {
        "rfc": rfc,
        "fecha_documento": fecha,
        "tipo_documento": tipo,
        "nombre_proveedor": proveedor,
        "route": route,
        "text_length": len(text),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_FIELD_SPECS: list[dict] = [
    {
        "label": "rfc",
        "gt_column": "rfc_expected",
        "extract_key": "rfc",
        "compare": lambda exp, ext: (
            _normalize_rfc_for_compare(exp) == _normalize_rfc_for_compare(ext)
        ),
    },
    {
        "label": "fecha",
        "gt_column": "fecha_documento_iso",
        "extract_key": "fecha_documento",
        "compare": lambda exp, ext: (exp or "").strip() == (ext or "").strip(),
    },
    {
        "label": "tipo_doc",
        "gt_column": "tipo_documento_expected",
        "extract_key": "tipo_documento",
        "compare": lambda exp, ext: (
            _normalize_text_for_fuzzy(exp) == _normalize_text_for_fuzzy(ext)
        ),
    },
    {
        "label": "proveedor",
        "gt_column": "nombre_proveedor_expected",
        "extract_key": "nombre_proveedor",
        "compare": _fuzzy_match,
    },
]


def evaluate(
    ground_truth_path: Path,
    data_dir: Path,
    output_path: Path,
) -> None:
    """Run evaluation and print report."""

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    if not ground_truth_path.is_file():
        logger.error("Ground-truth file not found: %s", ground_truth_path)
        sys.exit(1)

    with open(ground_truth_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        gt_rows = list(reader)

    if not gt_rows:
        logger.error("Ground-truth CSV is empty")
        sys.exit(1)

    logger.info("Loaded %d ground-truth rows from %s", len(gt_rows), ground_truth_path)

    # ------------------------------------------------------------------
    # Process each document
    # ------------------------------------------------------------------
    results: list[dict] = []
    counters: dict[str, dict[str, int]] = {
        spec["label"]: {"correct": 0, "total": 0} for spec in _FIELD_SPECS
    }

    for row in gt_rows:
        doc_id = row.get("doc_id", "?")
        filename = row.get("filename", "")

        if not filename:
            logger.warning("Row %s has no filename — skipping", doc_id)
            continue

        source_file = _find_source_file(filename, data_dir)
        if source_file is None:
            logger.warning(
                "Source file not found for %s (%s) — skipping", doc_id, filename
            )
            continue

        logger.info("Processing %s  ->  %s", doc_id, source_file)
        extracted = _run_pipeline(source_file)

        detail: dict[str, str] = {
            "doc_id": doc_id,
            "filename": filename,
            "route": extracted.get("route", ""),
            "text_length": str(extracted.get("text_length", 0)),
        }

        for spec in _FIELD_SPECS:
            expected_val = row.get(spec["gt_column"], "")
            extracted_val = extracted.get(spec["extract_key"])

            match = spec["compare"](expected_val, extracted_val)

            detail[f"{spec['label']}_expected"] = expected_val or ""
            detail[f"{spec['label']}_extracted"] = extracted_val or ""
            detail[f"{spec['label']}_match"] = "OK" if match else "FAIL"

            if expected_val:  # only count rows where we have a ground-truth value
                counters[spec["label"]]["total"] += 1
                if match:
                    counters[spec["label"]]["correct"] += 1

        results.append(detail)

    # ------------------------------------------------------------------
    # Print summary report
    # ------------------------------------------------------------------
    total_docs = len(results)
    print()
    print("=== OCR Extraction Evaluation ===")
    print(f"Documents evaluated: {total_docs}")
    print()
    print(f"{'Field':<12}| {'Correct':>7} | {'Total':>5} | {'Precision':>9}")
    print(f"{'-'*12}|{'-'*9}|{'-'*7}|{'-'*10}")

    total_correct = 0
    total_fields = 0
    for spec in _FIELD_SPECS:
        c = counters[spec["label"]]
        correct = c["correct"]
        total = c["total"]
        precision = (correct / total * 100) if total else 0.0
        total_correct += correct
        total_fields += total
        print(f"{spec['label']:<12}| {correct:>7} | {total:>5} | {precision:>8.1f}%")

    overall = (total_correct / total_fields * 100) if total_fields else 0.0
    print()
    print(f"Overall accuracy: {overall:.1f}%")
    print()

    # ------------------------------------------------------------------
    # Write detailed CSV
    # ------------------------------------------------------------------
    if not results:
        logger.warning("No documents were evaluated — skipping CSV output")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    logger.info("Detailed results written to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR extraction pipeline against ground-truth data.",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/ground_truth.csv",
        help="Path to ground-truth CSV (default: data/ground_truth.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation_results.csv",
        help="Path for detailed per-document results CSV (default: data/evaluation_results.csv)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/",
        help="Root directory to search for source files (default: data/)",
    )

    args = parser.parse_args()

    gt_path = Path(args.ground_truth)
    out_path = Path(args.output)
    d_dir = Path(args.data_dir)

    # Resolve relative paths from project root
    if not gt_path.is_absolute():
        gt_path = _PROJECT_ROOT / gt_path
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    if not d_dir.is_absolute():
        d_dir = _PROJECT_ROOT / d_dir

    evaluate(
        ground_truth_path=gt_path,
        data_dir=d_dir,
        output_path=out_path,
    )


if __name__ == "__main__":
    main()
