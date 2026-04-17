import logging
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


def _tables_to_text(tables: list) -> str:
    """Convert pdfplumber tables into readable text.

    Each table is a list of rows, each row a list of cell values (str or None).
    Rows where every cell is empty are skipped.  Cells are joined with " | "
    and each table is preceded by a [TABLE] marker.
    """
    parts: list[str] = []
    for table in tables:
        rows: list[str] = []
        for row in table:
            cells = [(cell if cell is not None else "") for cell in row]
            if not any(c.strip() for c in cells):
                continue
            rows.append(" | ".join(cells))
        if rows:
            parts.append("\n[TABLE]\n" + "\n".join(rows))
    return "\n".join(parts)


def extract_text_from_digital_pdf(file_path: str) -> str:
    logger.info("Starting digital PDF extraction: %s", file_path)
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    texts: list[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            logger.info("PDF has %d pages: %s", len(pdf.pages), file_path)
            for page_num, page in enumerate(pdf.pages, start=1):
                page_parts: list[str] = []

                try:
                    text = (page.extract_text() or "").strip()
                    if text:
                        page_parts.append(text)
                except Exception as exc:
                    logger.warning(
                        "pdfplumber.extract_text failed on page %d of %s: %s",
                        page_num, file_path, exc,
                    )

                try:
                    tables = page.extract_tables()
                    if tables:
                        table_text = _tables_to_text(tables)
                        if table_text:
                            page_parts.append(table_text)
                except Exception as exc:
                    logger.warning(
                        "pdfplumber.extract_tables failed on page %d of %s: %s",
                        page_num, file_path, exc,
                    )

                if page_parts:
                    texts.append("\n".join(page_parts))
    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as exc:
        logger.warning("pdfplumber could not open %s (corrupt PDF): %s", file_path, exc)
    except Exception as exc:
        logger.warning("pdfplumber failed for %s: %s", file_path, exc)

    if texts:
        return "\n\n".join(texts)

    # fallback si pdfplumber no devolvió texto
    logger.warning("pdfplumber returned no text, falling back to pypdf: %s", file_path)
    try:
        reader = PdfReader(str(path))
        fallback_texts: list[str] = []
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
                if text:
                    fallback_texts.append(text)
            except Exception as exc:
                logger.warning(
                    "PdfReader.extract_text failed on page %d of %s: %s",
                    page_num, file_path, exc,
                )

        if fallback_texts:
            return "\n\n".join(fallback_texts)
    except Exception as exc:
        logger.warning("PdfReader fallback failed for %s: %s", file_path, exc)

    logger.warning("All extraction methods failed for %s — returning empty string", file_path)
    return ""
