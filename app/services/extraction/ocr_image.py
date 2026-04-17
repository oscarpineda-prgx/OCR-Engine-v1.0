import logging
from pathlib import Path

import cv2
import pytesseract

from app.services.extraction.tesseract_runtime import configure_tesseract_cmd

logger = logging.getLogger(__name__)
configure_tesseract_cmd()


def preprocess_for_ocr(image_path: str) -> "cv2.typing.MatLike":
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce ruido sin borrar bordes de texto
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Aumenta contraste local en fondos no uniformes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # Binarización adaptativa (mejor para documentos con iluminación desigual)
    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    return thresh


def extract_text_from_image_ocr(file_path: str) -> str:
    logger.info("Starting OCR on image: %s", file_path)
    path = Path(file_path)
    if not path.exists():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    if not configure_tesseract_cmd():
        logger.warning(
            "Tesseract binary was not resolved for %s. Set TESSERACT_CMD in .env or add tesseract to PATH.",
            file_path,
        )

    try:
        processed = preprocess_for_ocr(str(path))
    except (ValueError, cv2.error) as exc:
        logger.warning("Image preprocessing failed for %s: %s", file_path, exc)
        return ""
    except Exception as exc:
        logger.warning("Unexpected error reading image %s: %s", file_path, exc)
        return ""

    try:
        text = pytesseract.image_to_string(
            processed,
            lang="spa+eng",
            config="--oem 3 --psm 6",
        )
    except pytesseract.TesseractNotFoundError as exc:
        logger.warning("Tesseract binary not found for %s: %s", file_path, exc)
        return ""
    except pytesseract.TesseractError as exc:
        logger.warning("Tesseract OCR error for %s: %s", file_path, exc)
        return ""
    except Exception as exc:
        logger.warning("Unexpected OCR failure for %s: %s", file_path, exc)
        return ""

    result = (text or "").strip()
    logger.info("OCR completed for %s — extracted %d chars", file_path, len(result))
    return result
