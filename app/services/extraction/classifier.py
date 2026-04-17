import logging
import re
from pathlib import Path

import pdfplumber

from app.services.extraction.file_types import (
    IMAGE_SOURCE_TYPES,
    STRUCTURED_TEXT_SOURCE_TYPES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification thresholds — tune these based on corpus evaluation
# ---------------------------------------------------------------------------
SAMPLE_MAX_PAGES = 5
DIGITAL_PAGE_RATIO = 0.6  # 60% of sampled pages must have text
MIN_TOTAL_CHARS = 300        # minimum chars across the sampled pages
MIN_CHARS_PER_PAGE = 150     # average chars per sampled page
MIN_TEXT_QUALITY = 0.5       # alpha chars / non-space chars
MIN_WORD_TOKENS = 10         # tokens with 3+ alphabetic characters

# Legacy constant kept for any external code that references it directly
MIN_CHARS_DIGITAL = 250

# Regex for "word-like" tokens: 3 or more consecutive letters
_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u024F]{3,}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_source_type(file_path: str) -> str:
    """Extract and normalise the file extension (without the leading dot)."""
    try:
        raw_path = str(file_path).strip()
        if "." in raw_path:
            return raw_path.rsplit(".", 1)[-1].lower()

        ext = Path(file_path).suffix.lower().lstrip(".")
        return str(ext)
    except (TypeError, ValueError) as exc:
        logger.warning("Could not determine source type for %r: %s", file_path, exc)
        return ""


def _text_quality_ratio(text: str) -> float:
    """Return the ratio of alphabetic characters to all non-space characters.

    A high ratio (>0.5) indicates natural-language text.  Garbage OCR
    metadata or encoded streams tend to have many symbols / digits.
    """
    non_space = text.replace(" ", "").replace("\t", "").replace("\n", "")
    if not non_space:
        return 0.0
    alpha_count = sum(1 for ch in non_space if ch.isalpha())
    return alpha_count / len(non_space)


def _count_word_tokens(text: str) -> int:
    """Count tokens that look like real words (3+ alphabetic characters)."""
    return len(_WORD_RE.findall(text))


def _get_sample_indices(total_pages: int, max_samples: int = 5) -> list[int]:
    """Return indices of pages to sample, spread across the document."""
    if total_pages <= max_samples:
        return list(range(total_pages))
    # Always include first and last page
    indices = [0, total_pages - 1]
    # Add evenly spaced pages in between
    step = total_pages / (max_samples - 1)
    for i in range(1, max_samples - 1):
        idx = int(i * step)
        if idx not in indices:
            indices.append(idx)
    return sorted(indices)[:max_samples]


def _page_is_digital(page_text: str) -> bool:
    """Decide whether a single page's extracted text looks like real digital text."""
    if len(page_text) < MIN_CHARS_PER_PAGE:
        return False
    if _text_quality_ratio(page_text) < MIN_TEXT_QUALITY:
        return False
    if _count_word_tokens(page_text) < MIN_WORD_TOKENS:
        return False
    return True


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def is_pdf_digital(file_path: str, *, min_chars: int = MIN_TOTAL_CHARS) -> bool:
    """Decide whether a PDF contains *real* digital text.

    Samples up to ``SAMPLE_MAX_PAGES`` pages spread across the document
    (first, last, and evenly spaced in between) and classifies as digital
    when >= ``DIGITAL_PAGE_RATIO`` (60 %) of sampled pages pass **all** of:

      1. Characters on the page >= ``MIN_CHARS_PER_PAGE`` (150).
      2. Text-quality ratio (alpha / non-space) >= ``MIN_TEXT_QUALITY`` (0.5).
      3. At least ``MIN_WORD_TOKENS`` (10) word-like tokens.

    A final sanity check requires the merged text from sampled pages to
    have at least ``min_chars`` total characters (default 300).
    """
    path = Path(file_path)
    if not path.exists():
        return False

    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                return False

            # Sample pages: first, last, and up to 3 evenly spaced in between
            sample_indices = _get_sample_indices(total_pages, max_samples=SAMPLE_MAX_PAGES)

            digital_count = 0
            extracted_parts: list[str] = []

            for idx in sample_indices:
                text = (pdf.pages[idx].extract_text() or "").strip()
                if text:
                    extracted_parts.append(text)
                if _page_is_digital(text):
                    digital_count += 1

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as exc:
        logger.warning("Corrupt PDF, cannot classify %s: %s", file_path, exc)
        return False
    except Exception as exc:
        logger.warning("pdfplumber failed to open %s, defaulting to OCR route: %s", file_path, exc)
        return False

    if not extracted_parts:
        return False

    pages_sampled = len(sample_indices)

    # Majority decision: >= 60% of sampled pages must be digital
    if digital_count / pages_sampled < DIGITAL_PAGE_RATIO:
        return False

    # Final sanity check on merged text volume
    merged = " ".join(extracted_parts)
    if len(merged) < min_chars:
        return False

    return True


def classify_and_extract(file_path: str) -> tuple[str, str]:
    """Classify document and return (route, extracted_text).

    For digital PDFs, returns ``("digital_pdf", full_text)``.
    For scanned/image PDFs, returns ``("ocr_image", sampled_text)``.
    For images, returns ``("ocr_image", "")`` so the caller can OCR them.
    For Office / email documents, returns ``("structured_document", "")``.
    Unsupported or missing files return ``("unknown", "")``.
    """
    source_type = get_source_type(file_path)

    if source_type != "pdf":
        return (classify_document_route(source_type, file_path), "")

    path = Path(file_path)
    if not path.exists():
        return ("unknown", "")

    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                logger.warning(
                    "pdfplumber reported zero pages for %s; falling back to OCR route",
                    file_path,
                )
                return ("ocr_image", "")

            # --- Phase 1: sample pages for classification ---------------
            sample_indices = _get_sample_indices(total_pages, max_samples=SAMPLE_MAX_PAGES)

            digital_count = 0
            sampled_texts: list[str] = []

            for idx in sample_indices:
                text = (pdf.pages[idx].extract_text() or "").strip()
                if text:
                    sampled_texts.append(text)
                if _page_is_digital(text):
                    digital_count += 1

            pages_sampled = len(sample_indices)
            ratio_ok = pages_sampled > 0 and (digital_count / pages_sampled) >= DIGITAL_PAGE_RATIO
            merged_sample = " ".join(sampled_texts)
            volume_ok = len(merged_sample) >= MIN_TOTAL_CHARS

            is_digital = ratio_ok and volume_ok and bool(sampled_texts)

            # --- Phase 2: extract text based on classification ----------
            if is_digital:
                # Extract ALL pages (not just sampled) for full text
                all_parts: list[str] = []
                for page in pdf.pages:
                    text = (page.extract_text() or "").strip()
                    if text:
                        all_parts.append(text)
                full_text = "\n".join(all_parts)
                return ("digital_pdf", full_text)
            else:
                # Return partial text from sampling; caller will OCR
                return ("ocr_image", merged_sample)

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as exc:
        logger.warning("Corrupt PDF, cannot classify %s: %s", file_path, exc)
        return ("ocr_image", "")
    except Exception as exc:
        logger.warning("pdfplumber failed to open %s, defaulting to OCR route: %s", file_path, exc)
        return ("ocr_image", "")


def classify_document_route(source_type: str, file_path: str) -> str:
    """Classify a document into the correct extraction pipeline.

    Returns
    -------
    ``"digital_pdf"`` | ``"ocr_image"`` | ``"structured_document"`` | ``"unknown"``
    """
    normalized = source_type.lower()

    if normalized == "pdf":
        return "digital_pdf" if is_pdf_digital(file_path) else "ocr_image"

    if normalized in IMAGE_SOURCE_TYPES:
        return "ocr_image"

    if normalized in STRUCTURED_TEXT_SOURCE_TYPES:
        return "structured_document"

    return "unknown"
