from dataclasses import dataclass
from pathlib import Path
import gc
import logging
import re
from datetime import datetime

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pypdf import PdfReader

from app.services.extraction.tesseract_runtime import configure_tesseract_cmd

try:
    from pypdfium2 import PdfDocument
    _PDFIUM_IMPORT_ERROR = None
except ImportError as exc:
    PdfDocument = None
    _PDFIUM_IMPORT_ERROR = exc

logger = logging.getLogger(__name__)
configure_tesseract_cmd()

MAX_OCR_PAGES = 50

RFC_SHAPE_PATTERN = re.compile(r"^[A-Z\u00D1&]{3,4}[0-9]{6}[A-Z0-9]{2,3}$")
RFC_INLINE_PATTERN = re.compile(r"[A-Z\u00D1&]{3,4}[0-9OISB]{6}[A-Z0-9]{2,3}")
MONTH_WORDS = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "setiembre",
    "octubre",
    "noviembre",
    "diciembre",
)
MONTH_REGEX = "|".join(MONTH_WORDS)
DATE_HINT_PATTERNS = [
    re.compile(
        rf"(?is)(siendo\s+el\s+d[ií]a[\s\._:-]*[0-9OILS]{{1,2}}[\s\._:-]*(?:de)?[\s\._:-]*(?:{MONTH_REGEX})[\s\._:-]*(?:de)?[\s\._:-]*(?:19|20)[0-9OILS]{{2}})"
    ),
    re.compile(
        rf"(?is)\b([0-9OILS]{{1,2}}[\s\._:-]*de[\s\._:-]*(?:{MONTH_REGEX})[\s\._:-]*de[\s\._:-]*(?:19|20)[0-9OILS]{{2}})\b"
    ),
]
RFC_CHAR_VALUES = {
    **{str(i): i for i in range(10)},
    "A": 10,
    "B": 11,
    "C": 12,
    "D": 13,
    "E": 14,
    "F": 15,
    "G": 16,
    "H": 17,
    "I": 18,
    "J": 19,
    "K": 20,
    "L": 21,
    "M": 22,
    "N": 23,
    "&": 24,
    "O": 25,
    "P": 26,
    "Q": 27,
    "R": 28,
    "S": 29,
    "T": 30,
    "U": 31,
    "V": 32,
    "W": 33,
    "X": 34,
    "Y": 35,
    "Z": 36,
    "\u00D1": 37,
}


def _require_pdf_document_class():
    if PdfDocument is None:
        raise RuntimeError(
            "pypdfium2 is not installed. Install project dependencies with `pip install -r requirements.txt`."
        ) from _PDFIUM_IMPORT_ERROR
    return PdfDocument


def _extract_embedded_page_image(
    file_path: str,
    page_index: int,
    reader_cache: dict | None = None,
):
    reader: PdfReader | None = None
    if reader_cache is not None:
        reader = reader_cache.get("reader")

    if reader is None:
        reader = PdfReader(file_path)
        if reader_cache is not None:
            reader_cache["reader"] = reader

    if page_index >= len(reader.pages):
        return None

    page = reader.pages[page_index]
    images = list(page.images)
    if not images:
        return None

    best_image = max(images, key=lambda image: len(getattr(image, "data", b"") or b""))
    pil_image = best_image.image
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def _render_page_to_pil(
    page,
    *,
    scale: float,
    file_path: str,
    page_index: int,
    reader_cache: dict | None = None,
):
    if page is None:
        fallback_image = _extract_embedded_page_image(file_path, page_index, reader_cache)
        if fallback_image is None:
            raise RuntimeError(f"Could not load page {page_index + 1} from {file_path}")
        logger.warning(
            "Using embedded image fallback for %s page %d because Pdfium could not open the page.",
            file_path,
            page_index + 1,
        )
        return fallback_image

    try:
        return page.render(scale=scale).to_pil()
    except Exception as exc:
        fallback_image = _extract_embedded_page_image(file_path, page_index, reader_cache)
        if fallback_image is None:
            raise
        logger.warning(
            "Pdfium render failed for %s page %d: %s. Using embedded image fallback.",
            file_path,
            page_index + 1,
            exc,
        )
        return fallback_image


def _get_pdf_page(doc, page_index: int, file_path: str):
    try:
        return doc[page_index]
    except Exception as exc:
        logger.warning(
            "Pdfium could not open %s page %d: %s. Trying embedded image fallback.",
            file_path,
            page_index + 1,
            exc,
        )
        return None


def preprocess_pdf_page_for_ocr(pil_image) -> "cv2.typing.MatLike":
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return thresh


def _rotate_image(img: "cv2.typing.MatLike", angle: int) -> "cv2.typing.MatLike":
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def _ocr_score(text: str) -> int:
    if not text:
        return 0

    low = text.lower()
    keywords = [
        "convenio",
        "proveedor",
        "cliente",
        "razon social",
        "rfc",
        "fecha",
        "firma",
        "domicilio",
        "femsa",
        "oxxo",
        "pagina",
    ]
    keyword_hits = sum(low.count(k) for k in keywords)

    letters = sum(ch.isalpha() for ch in text)
    alnum = sum(ch.isalnum() for ch in text)
    non_space = sum(not ch.isspace() for ch in text) or 1
    alpha_ratio = letters / non_space
    alnum_ratio = alnum / non_space

    weird_chunks = len(re.findall(r"[A-Z0-9]{10,}", text))

    return int(keyword_hits * 120 + alpha_ratio * 40 + alnum_ratio * 20 - weird_chunks * 10)


def _strip_accents(value: str) -> str:
    table = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
    return value.translate(table)


def _normalize_spacing(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _extract_date_hints_from_text(text: str) -> list[str]:
    if not text:
        return []

    normalized = _strip_accents(text.lower())
    hints: list[str] = []

    for pattern in DATE_HINT_PATTERNS:
        for match in pattern.finditer(normalized):
            snippet = _normalize_spacing(match.group(1))
            snippet = snippet.strip(" .,:;_-")
            if len(snippet) >= 12:
                hints.append(snippet)

    # Fallback por linea para casos con OCR fragmentado.
    for line in normalized.splitlines():
        clean = _normalize_spacing(line)
        if not clean:
            continue
        has_month = any(month in clean for month in MONTH_WORDS)
        has_year = bool(re.search(r"(19|20)[0-9oils]{2}", clean))
        has_anchor = "siendo el dia" in clean
        has_day = bool(re.search(r"\b[0-9oils]{1,2}\b", clean))
        if (has_month and has_year and has_day) or (has_anchor and (has_month or has_year)):
            hints.append(clean.strip(" .,:;_-"))

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in hints:
        key = hint.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hint)
    return deduped


def _iter_footer_variants(gray_footer: "cv2.typing.MatLike") -> list["cv2.typing.MatLike"]:
    variants: list["cv2.typing.MatLike"] = []
    if gray_footer is None or gray_footer.size == 0:
        return variants

    gray = gray_footer if len(gray_footer.shape) == 2 else cv2.cvtColor(gray_footer, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.resize(otsu, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
    return variants


def _ocr_footer_date_hints(page_gray: "cv2.typing.MatLike") -> str:
    if page_gray is None or page_gray.size == 0:
        return ""

    h = page_gray.shape[0]
    w = page_gray.shape[1]
    y0 = int(h * 0.72)
    x0 = int(w * 0.08)
    x1 = int(w * 0.92)
    footer = page_gray[y0:, x0:x1]
    if footer.size == 0:
        return ""

    if footer.shape[1] > 1400:
        ratio = 1400.0 / float(footer.shape[1])
        footer = cv2.resize(footer, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

    collected: list[str] = []
    for variant in _iter_footer_variants(footer):
        for psm in (6, 11):
            txt = pytesseract.image_to_string(variant, lang="spa+eng", config=f"--oem 3 --psm {psm}")
            hints = _extract_date_hints_from_text(txt)
            if hints:
                collected.extend(hints)
                strong = any(
                    any(month in hint for month in MONTH_WORDS) and re.search(r"(19|20)[0-9oils]{2}", hint)
                    for hint in hints
                )
                if strong:
                    break
            if len(collected) >= 2:
                break
        if collected:
            strong_collected = any(
                any(month in hint for month in MONTH_WORDS) and re.search(r"(19|20)[0-9oils]{2}", hint)
                for hint in collected
            )
            if strong_collected:
                break
        if len(collected) >= 2:
            break

    if not collected:
        return ""

    deduped: list[str] = []
    seen: set[str] = set()
    for hint in collected:
        key = hint.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(hint)

    # Limitar ruido: solo 2 pistas de fecha maximo.
    return "\n".join(deduped[:2])


def _best_oriented_ocr_with_angle(processed: "cv2.typing.MatLike") -> tuple[str, int]:
    best_text = ""
    best_score = -10**9
    best_angle = 0

    for angle in (0, 180, 90, 270):
        rotated = _rotate_image(processed, angle)
        candidate = pytesseract.image_to_string(rotated, lang="spa+eng", config="--oem 3 --psm 6")
        candidate = (candidate or "").strip()
        score = _ocr_score(candidate)
        if score > best_score:
            best_text = candidate
            best_score = score
            best_angle = angle
        if angle != 0:
            del rotated

    return best_text, best_angle


def _best_oriented_ocr(processed: "cv2.typing.MatLike") -> str:
    text, _ = _best_oriented_ocr_with_angle(processed)
    return text


def _normalize_rfc_ocr(token: str) -> str:
    raw = re.sub(r"[^A-Z0-9\u00D1&]", "", (token or "").upper())
    chars = list(raw)

    if len(chars) == 12:
        num_range = range(3, 9)
    elif len(chars) == 13:
        num_range = range(4, 10)
    else:
        return raw

    for i in num_range:
        if chars[i] == "O":
            chars[i] = "0"
        elif chars[i] == "I":
            chars[i] = "1"
        elif chars[i] == "S":
            chars[i] = "5"
        elif chars[i] == "B":
            chars[i] = "8"

    return "".join(chars)


def _is_valid_rfc_date(token: str) -> bool:
    if len(token) == 12:
        part = token[3:9]
    elif len(token) == 13:
        part = token[4:10]
    else:
        return False

    if not part.isdigit():
        return False

    yy, mm, dd = int(part[:2]), int(part[2:4]), int(part[4:6])
    year = 1900 + yy if yy >= 30 else 2000 + yy

    try:
        datetime(year=year, month=mm, day=dd)
        return True
    except ValueError:
        return False


def _is_acceptable_rfc_hint(token: str) -> bool:
    if not RFC_SHAPE_PATTERN.fullmatch(token):
        return False
    return _is_valid_rfc_date(token)


def _has_valid_rfc_check_digit(token: str) -> bool:
    up = token.upper()
    if len(up) not in (12, 13):
        return False

    base = up[:-1]
    expected = up[-1]

    try:
        values = [RFC_CHAR_VALUES[ch] for ch in base]
    except KeyError:
        return False

    start_weight = 13 if len(up) == 13 else 12
    total = 0
    for idx, value in enumerate(values):
        total += value * (start_weight - idx)

    mod = total % 11
    calc = 11 - mod
    if calc == 11:
        check = "0"
    elif calc == 10:
        check = "A"
    else:
        check = str(calc)

    return expected == check


def _extract_rfc_from_text(text: str) -> str | None:
    cleaned = _normalize_rfc_ocr(text)
    for m in RFC_INLINE_PATTERN.finditer(cleaned):
        candidate = _normalize_rfc_ocr(m.group(0))
        if _is_acceptable_rfc_hint(candidate):
            return candidate
    return None


def _extract_rfc_from_line_tokens(data: dict, anchor_idx: int) -> str | None:
    ax = data["left"][anchor_idx]
    ay = data["top"][anchor_idx]
    aw = data["width"][anchor_idx]
    ah = data["height"][anchor_idx]
    a_right = ax + aw
    a_y1 = ay
    a_y2 = ay + ah

    parts: list[tuple[int, str]] = []
    n = len(data["text"])
    for j in range(n):
        token = (data["text"][j] or "").strip()
        if not token:
            continue
        tx = data["left"][j]
        ty = data["top"][j]
        tw = data["width"][j]
        th = data["height"][j]
        if tx <= a_right:
            continue
        t_y1 = ty
        t_y2 = ty + th
        overlap = max(0, min(a_y2, t_y2) - max(a_y1, t_y1))
        base = max(1, min(ah, th))
        if overlap / base >= 0.45:
            parts.append((tx, token))

    if not parts:
        return None

    parts.sort(key=lambda it: it[0])
    line_text = " ".join(tok for _, tok in parts[:6])
    return _extract_rfc_from_text(line_text)


def _is_fecha_anchor_token(token: str) -> bool:
    if not token:
        return False
    clean = _strip_accents(token.upper())
    clean = re.sub(r"[^A-Z]", "", clean)
    if not clean:
        return False
    return clean.startswith("FECH") or clean in {"FECMA", "FECNA"}


def _extract_date_from_line_tokens(data: dict, anchor_idx: int) -> str | None:
    from app.services.parsing.fields import extract_fecha_documento

    ax = data["left"][anchor_idx]
    ay = data["top"][anchor_idx]
    aw = data["width"][anchor_idx]
    ah = data["height"][anchor_idx]
    a_right = ax + aw
    a_y1 = ay
    a_y2 = ay + ah

    parts: list[tuple[int, str]] = []
    n = len(data["text"])
    for j in range(n):
        token = (data["text"][j] or "").strip()
        if not token:
            continue
        tx = data["left"][j]
        ty = data["top"][j]
        tw = data["width"][j]
        th = data["height"][j]
        if tx <= a_right:
            continue
        t_y1 = ty
        t_y2 = ty + th
        overlap = max(0, min(a_y2, t_y2) - max(a_y1, t_y1))
        base = max(1, min(ah, th))
        if overlap / base >= 0.40:
            parts.append((tx, token))

    if not parts:
        return None

    parts.sort(key=lambda it: it[0])
    line_text = " ".join(tok for _, tok in parts[:10])
    return extract_fecha_documento(line_text)


def _iter_date_roi_variants(roi: "cv2.typing.MatLike") -> list["cv2.typing.MatLike"]:
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variants: list["cv2.typing.MatLike"] = [gray]

    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)

    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))

    scaled: list["cv2.typing.MatLike"] = []
    for var in variants:
        scaled.append(var)
        scaled.append(cv2.resize(var, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC))

    return scaled


def _extract_date_from_roi(roi: "cv2.typing.MatLike") -> str | None:
    if roi is None or roi.size == 0:
        return None

    from app.services.parsing.fields import extract_fecha_documento

    for variant in _iter_date_roi_variants(roi):
        for psm in (7, 6, 11, 13):
            txt = pytesseract.image_to_string(variant, lang="spa+eng", config=f"--oem 3 --psm {psm}")
            txt = (txt or "").strip()
            if not txt:
                continue

            candidate = extract_fecha_documento(txt)
            if candidate:
                return candidate  # early exit: valid date found

            for hint in _extract_date_hints_from_text(txt):
                candidate = extract_fecha_documento(hint)
                if candidate:
                    return candidate  # early exit: valid date found from hint

    return None


def _extract_first_page_delivery_box_date(page) -> str | None:
    """Target the top-right delivery-date box on page 1.

    The generic OCR over the full page often misses this handwritten/boxed
    field, so we re-render the first page at higher resolution and inspect
    only the top-right date area.
    """
    from app.services.parsing.fields import extract_fecha_documento

    pil_image = page.render(scale=8.0).to_pil()
    gray = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    rois = [
        gray[int(h * 0.033): int(h * 0.098), int(w * 0.751): int(w * 0.961)],
        gray[int(h * 0.037): int(h * 0.090), int(w * 0.770): int(w * 0.930)],
        gray[int(h * 0.030): int(h * 0.105), int(w * 0.620): int(w * 0.961)],
    ]

    for roi in rois:
        if roi.size == 0:
            continue
        for variant in _iter_date_roi_variants(roi):
            for psm in (7, 6):
                for extra in ("", " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/:.-"):
                    txt = pytesseract.image_to_string(
                        variant,
                        lang="spa+eng",
                        config=f"--oem 3 --psm {psm}{extra}",
                    )
                    txt = (txt or "").strip()
                    if not txt:
                        continue

                    candidate = extract_fecha_documento(txt)
                    if candidate:
                        return candidate

                    for hint in _extract_date_hints_from_text(txt):
                        candidate = extract_fecha_documento(hint)
                        if candidate:
                            return candidate

    return None


def _iter_roi_variants(roi: "cv2.typing.MatLike") -> list["cv2.typing.MatLike"]:
    gray = roi if len(roi.shape) == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    variants: list["cv2.typing.MatLike"] = []

    variants.append(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(cv2.bitwise_not(otsu))
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    variants.append(adaptive)
    variants.append(cv2.bitwise_not(adaptive))
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, blur_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(blur_otsu)

    scaled: list["cv2.typing.MatLike"] = []
    for var in variants:
        scaled.append(var)
        scaled.append(cv2.resize(var, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
        scaled.append(cv2.resize(var, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
        scaled.append(cv2.resize(var, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC))
    return scaled


def _candidate_from_roi(roi: "cv2.typing.MatLike") -> str | None:
    if roi is None or roi.size == 0:
        return None

    best: str | None = None
    psms = (7, 8, 13, 6)
    for variant in _iter_roi_variants(roi):
        for psm in psms:
            for extra in ("", " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789&"):
                cfg = f"--oem 3 --psm {psm}{extra}"
                text = pytesseract.image_to_string(variant, lang="spa+eng", config=cfg)
                candidate = _extract_rfc_from_text(text)
                if candidate:
                    if _has_valid_rfc_check_digit(candidate):
                        return candidate
                    if best is None:
                        best = candidate

    return best


def extract_rfc_hint_from_pdf_ocr(file_path: str) -> str | None:
    path = Path(file_path)
    if not path.exists():
        return None

    if not configure_tesseract_cmd():
        logger.warning(
            "Tesseract binary was not resolved for PDF OCR on %s. Set TESSERACT_CMD in .env or add tesseract to PATH.",
            file_path,
        )

    pdf_document_cls = _require_pdf_document_class()
    doc = pdf_document_cls(str(path))
    if len(doc) == 0:
        return None

    reader_cache: dict = {}
    page = _get_pdf_page(doc, 0, str(path))
    pil_image = _render_page_to_pil(
        page,
        scale=2.5,
        file_path=str(path),
        page_index=0,
        reader_cache=reader_cache,
    )
    processed = preprocess_pdf_page_for_ocr(pil_image)
    raw_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)

    best_text = ""
    best_score = -10**9
    best_candidate = None

    for base_img in (raw_gray, processed):
        for angle in (0, 180, 90, 270):
            img = _rotate_image(base_img, angle)
            text = pytesseract.image_to_string(img, lang="spa+eng", config="--oem 3 --psm 6")
            score = _ocr_score(text or "")
            if score > best_score:
                best_score = score
                best_text = text or ""

            data = pytesseract.image_to_data(img, lang="spa+eng", config="--oem 3 --psm 11", output_type=Output.DICT)
            n = len(data["text"])
            for i in range(n):
                token = (data["text"][i] or "").strip().upper()
                if not token:
                    continue
                if token in {"RFC", "RFC:", "R.F.C", "R.F.C."} or token.startswith("RFC"):
                    # 1) Intento por tokens de la misma linea (mas estable para formularios)
                    line_candidate = _extract_rfc_from_line_tokens(data, i)
                    if line_candidate:
                        best_candidate = line_candidate
                        break

                    # 2) Intento por ROI a la derecha del anchor RFC
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]

                    x1 = min(img.shape[1] - 1, x + w + 2)
                    x2 = min(img.shape[1], x + int(img.shape[1] * 0.60))
                    y1 = max(0, y - int(h * 1.0))
                    y2 = min(img.shape[0], y + int(h * 3.0))
                    roi = img[y1:y2, x1:x2]
                    candidate = _candidate_from_roi(roi)
                    if candidate:
                        best_candidate = candidate
                        break
            if best_candidate:
                break
        if best_candidate:
            break

    if not best_candidate:
        m_text = re.search(r"(?i)\bRFC\b[^A-Z0-9\u00D1&]{0,10}([A-Z\u00D1&]{3,4}[0-9OIS]{6}[A-Z0-9]{2,3})", best_text)
        if m_text:
            normalized = _normalize_rfc_ocr(m_text.group(1))
            if _is_acceptable_rfc_hint(normalized):
                best_candidate = normalized

    return best_candidate


def extract_fecha_hint_from_pdf_ocr(file_path: str) -> str | None:
    path = Path(file_path)
    if not path.exists():
        return None

    if not configure_tesseract_cmd():
        logger.warning(
            "Tesseract binary was not resolved for PDF OCR on %s. Set TESSERACT_CMD in .env or add tesseract to PATH.",
            file_path,
        )

    pdf_document_cls = _require_pdf_document_class()
    doc = pdf_document_cls(str(path))
    if len(doc) == 0:
        return None

    from app.services.parsing.fields import extract_fecha_documento
    reader_cache: dict = {}

    # 1) Prioridad principal: fecha de firma al pie de la ultima pagina.
    last_index = len(doc) - 1
    page = _get_pdf_page(doc, last_index, str(path))
    pil_image = _render_page_to_pil(
        page,
        scale=2.0,
        file_path=str(path),
        page_index=last_index,
        reader_cache=reader_cache,
    )
    raw_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    processed = preprocess_pdf_page_for_ocr(pil_image)

    base_text, best_angle = _best_oriented_ocr_with_angle(processed)
    footer_source = _rotate_image(raw_gray, best_angle)
    footer_hints = _ocr_footer_date_hints(footer_source)
    merged = "\n".join(part for part in [base_text, footer_hints] if (part or "").strip()).strip()
    if merged:
        signature_date = extract_fecha_documento(merged)
        if signature_date:
            return signature_date

    # 2) Fallback secundario: primera pagina, ancla "Fecha de Entrega".
    first_page = _get_pdf_page(doc, 0, str(path))
    if first_page is not None:
        try:
            boxed_date = _extract_first_page_delivery_box_date(first_page)
        except Exception as exc:
            logger.debug("First-page boxed date OCR failed for %s: %s", file_path, exc)
            boxed_date = None
    else:
        boxed_date = None
    if boxed_date:
        return boxed_date

    pil_first = _render_page_to_pil(
        first_page,
        scale=2.6,
        file_path=str(path),
        page_index=0,
        reader_cache=reader_cache,
    )
    first_bgr = cv2.cvtColor(np.array(pil_first), cv2.COLOR_RGB2BGR)
    first_gray = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2GRAY)
    first_processed = preprocess_pdf_page_for_ocr(pil_first)

    for base_img in (first_gray, first_processed):
        img = base_img  # primera pagina normalmente viene en orientacion correcta
        data = pytesseract.image_to_data(img, lang="spa+eng", config="--oem 3 --psm 11", output_type=Output.DICT)
        n = len(data["text"])
        for i in range(n):
            token = (data["text"][i] or "").strip()
            if not _is_fecha_anchor_token(token):
                continue

            # a) intento por tokens en la misma linea
            line_candidate = _extract_date_from_line_tokens(data, i)
            if line_candidate:
                return line_candidate

            # b) intento por ROI a la derecha del anchor
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            x1 = min(img.shape[1] - 1, x + w + 2)
            x2 = min(img.shape[1], x + int(img.shape[1] * 0.55))
            y1 = max(0, y - int(h * 1.0))
            y2 = min(img.shape[0], y + int(h * 2.2))
            roi = img[y1:y2, x1:x2]
            roi_candidate = _extract_date_from_roi(roi)
            if roi_candidate:
                return roi_candidate

    # 3) Ultimo fallback en primera pagina: zona superior derecha.
    h, w = first_gray.shape[:2]
    top_right = first_gray[0:int(h * 0.35), int(w * 0.45):w]
    top_right_candidate = _extract_date_from_roi(top_right)
    if top_right_candidate:
        return top_right_candidate

    return None


def extract_text_from_pdf_ocr(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not configure_tesseract_cmd():
        logger.warning(
            "Tesseract binary was not resolved for PDF OCR on %s. Set TESSERACT_CMD in .env or add tesseract to PATH.",
            file_path,
        )

    pdf_document_cls = _require_pdf_document_class()
    doc = pdf_document_cls(str(path))
    chunks: list[str] = []
    reader_cache: dict = {}

    page_count = len(doc)
    if page_count > MAX_OCR_PAGES:
        logger.warning(
            "PDF has %d pages; only the first %d will be OCR-processed.",
            page_count, MAX_OCR_PAGES,
        )
        page_count = MAX_OCR_PAGES

    for idx in range(page_count):
        page = _get_pdf_page(doc, idx, str(path))
        pil_image = _render_page_to_pil(
            page,
            scale=2.0,
            file_path=str(path),
            page_index=idx,
            reader_cache=reader_cache,
        )
        raw_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        processed = preprocess_pdf_page_for_ocr(pil_image)

        text, best_angle = _best_oriented_ocr_with_angle(processed)
        text = (text or "").strip()

        # Rescate OCR del pie de pagina para mejorar lectura de fecha de firma.
        # Se limita a ultimas paginas para no degradar rendimiento global.
        if idx == page_count - 1:
            footer_source = _rotate_image(raw_gray, best_angle)
            footer_hints = _ocr_footer_date_hints(footer_source)
            if footer_hints:
                text = f"{text}\n{footer_hints}".strip() if text else footer_hints
            del footer_source

        if text:
            chunks.append(text)

        # Release page images to avoid accumulating memory.
        del pil_image, raw_bgr, raw_gray, processed
        if idx > 0 and idx % 10 == 0:
            gc.collect()

    return "\n\n".join(chunks).strip()


# ---------------------------------------------------------------------------
# Unified single-pass OCR: opens the PDF once, renders each page once,
# and extracts full text + RFC hint + fecha hint in a single traversal.
# ---------------------------------------------------------------------------

@dataclass
class UnifiedOcrResult:
    full_text: str
    rfc_hint: str | None
    fecha_hint: str | None


def unified_extract_from_pdf_ocr(file_path: str) -> UnifiedOcrResult:
    """Extract full text, RFC hint, and fecha hint from a scanned PDF in one pass."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not configure_tesseract_cmd():
        logger.warning(
            "Tesseract binary was not resolved for PDF OCR on %s. Set TESSERACT_CMD in .env or add tesseract to PATH.",
            file_path,
        )

    pdf_document_cls = _require_pdf_document_class()
    doc = pdf_document_cls(str(path))
    page_count = len(doc)
    if page_count == 0:
        return UnifiedOcrResult(full_text="", rfc_hint=None, fecha_hint=None)

    if page_count > MAX_OCR_PAGES:
        logger.warning(
            "PDF has %d pages; only the first %d will be OCR-processed.",
            page_count, MAX_OCR_PAGES,
        )
        page_count = MAX_OCR_PAGES

    from app.services.parsing.fields import extract_fecha_documento

    chunks: list[str] = []
    rfc_hint: str | None = None
    fecha_hint: str | None = None
    reader_cache: dict = {}

    # ------------------------------------------------------------------
    # Cache for first-page images so they can be reused later
    # ------------------------------------------------------------------
    first_page_raw_gray: "cv2.typing.MatLike | None" = None
    first_page_processed: "cv2.typing.MatLike | None" = None
    first_page_best_angle: int = 0
    first_page_text: str = ""
    first_page_delivery_hint: str | None = None

    for idx in range(page_count):
        page = _get_pdf_page(doc, idx, str(path))
        is_first = idx == 0
        is_last = idx == page_count - 1

        # ---- Render ------------------------------------------------------
        # First page uses 2.5 scale (higher res needed for RFC extraction).
        # Other pages use 2.0.
        if is_first:
            scale = 2.5
        else:
            scale = 2.0

        pil_image = _render_page_to_pil(
            page,
            scale=scale,
            file_path=str(path),
            page_index=idx,
            reader_cache=reader_cache,
        )
        raw_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
        processed = preprocess_pdf_page_for_ocr(pil_image)

        # ---- Orientation detection (one OCR call per page) ---------------
        text, best_angle = _best_oriented_ocr_with_angle(processed)
        text = (text or "").strip()

        # ---- First page: extract RFC hint --------------------------------
        if is_first:
            first_page_raw_gray = raw_gray
            first_page_processed = processed
            first_page_best_angle = best_angle
            first_page_text = text

            rfc_hint = _unified_extract_rfc(
                raw_gray, processed, text, best_angle, pil_image,
            )
            try:
                if page is not None:
                    first_page_delivery_hint = _extract_first_page_delivery_box_date(page)
                else:
                    first_page_delivery_hint = None
            except Exception as exc:
                logger.debug("First-page boxed date OCR failed for %s: %s", file_path, exc)
                first_page_delivery_hint = None
            if first_page_delivery_hint and fecha_hint is None:
                fecha_hint = first_page_delivery_hint

        # ---- Last page: extract fecha hint from footer -------------------
        if is_last and fecha_hint is None:
            footer_source = _rotate_image(raw_gray, best_angle)
            footer_hints = _ocr_footer_date_hints(footer_source)
            merged = "\n".join(part for part in [text, footer_hints] if (part or "").strip()).strip()
            if merged:
                fecha_hint = extract_fecha_documento(merged)

            # Append footer hints to text for richness (matches existing behaviour)
            if footer_hints:
                text = f"{text}\n{footer_hints}".strip() if text else footer_hints
            del footer_source

        if text:
            chunks.append(text)

        # ---- Release page images to avoid accumulating memory ------------
        del pil_image, raw_bgr
        if not is_first:
            del raw_gray, processed
        if idx > 0 and idx % 10 == 0:
            gc.collect()

    # ------------------------------------------------------------------
    # Fecha fallback: if not found on the last page, try first page
    # anchor detection (reuse cached images).
    # ------------------------------------------------------------------
    if fecha_hint is None and first_page_delivery_hint is not None:
        fecha_hint = first_page_delivery_hint

    if fecha_hint is None and first_page_raw_gray is not None:
        fecha_hint = _unified_extract_fecha_from_first_page(
            first_page_raw_gray, first_page_processed, extract_fecha_documento,
        )

    # Release first-page cached images now that fecha extraction is done.
    del first_page_raw_gray, first_page_processed

    full_text = "\n\n".join(chunks).strip()
    return UnifiedOcrResult(full_text=full_text, rfc_hint=rfc_hint, fecha_hint=fecha_hint)


# ---------------------------------------------------------------------------
# Internal helpers for the unified function
# ---------------------------------------------------------------------------

def _unified_extract_rfc(
    raw_gray: "cv2.typing.MatLike",
    processed: "cv2.typing.MatLike",
    best_text: str,
    best_angle: int,
    pil_image,
) -> str | None:
    """RFC extraction logic: tries anchor-based token search, then ROI,
    then regex fallback.  Mirrors extract_rfc_hint_from_pdf_ocr but reuses
    the already-oriented image/text."""

    best_candidate: str | None = None

    for base_img in (raw_gray, processed):
        for angle in (0, 180, 90, 270):
            img = _rotate_image(base_img, angle)
            data = pytesseract.image_to_data(
                img, lang="spa+eng", config="--oem 3 --psm 11", output_type=Output.DICT,
            )
            n = len(data["text"])
            for i in range(n):
                token = (data["text"][i] or "").strip().upper()
                if not token:
                    continue
                if token in {"RFC", "RFC:", "R.F.C", "R.F.C."} or token.startswith("RFC"):
                    # 1) Token-based line search
                    line_candidate = _extract_rfc_from_line_tokens(data, i)
                    if line_candidate:
                        best_candidate = line_candidate
                        break

                    # 2) ROI to the right of the anchor
                    x = data["left"][i]
                    y = data["top"][i]
                    w = data["width"][i]
                    h = data["height"][i]

                    x1 = min(img.shape[1] - 1, x + w + 2)
                    x2 = min(img.shape[1], x + int(img.shape[1] * 0.60))
                    y1 = max(0, y - int(h * 1.0))
                    y2 = min(img.shape[0], y + int(h * 3.0))
                    roi = img[y1:y2, x1:x2]
                    candidate = _candidate_from_roi(roi)
                    if candidate:
                        best_candidate = candidate
                        break
            if best_candidate:
                break
        if best_candidate:
            break

    # Regex fallback on the best OCR text
    if not best_candidate:
        m_text = re.search(
            r"(?i)\bRFC\b[^A-Z0-9\u00D1&]{0,10}([A-Z\u00D1&]{3,4}[0-9OIS]{6}[A-Z0-9]{2,3})",
            best_text,
        )
        if m_text:
            normalized = _normalize_rfc_ocr(m_text.group(1))
            if _is_acceptable_rfc_hint(normalized):
                best_candidate = normalized

    return best_candidate


def _unified_extract_fecha_from_first_page(
    first_gray: "cv2.typing.MatLike",
    first_processed: "cv2.typing.MatLike | None",
    extract_fecha_documento,
) -> str | None:
    """Fecha fallback: anchor detection on the first page (reuses cached images)."""

    bases = [first_gray]
    if first_processed is not None:
        bases.append(first_processed)

    for base_img in bases:
        img = base_img  # first page normally in correct orientation
        data = pytesseract.image_to_data(
            img, lang="spa+eng", config="--oem 3 --psm 11", output_type=Output.DICT,
        )
        n = len(data["text"])
        for i in range(n):
            token = (data["text"][i] or "").strip()
            if not _is_fecha_anchor_token(token):
                continue

            # a) tokens on the same line
            line_candidate = _extract_date_from_line_tokens(data, i)
            if line_candidate:
                return line_candidate

            # b) ROI to the right of the anchor
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]

            x1 = min(img.shape[1] - 1, x + w + 2)
            x2 = min(img.shape[1], x + int(img.shape[1] * 0.55))
            y1 = max(0, y - int(h * 1.0))
            y2 = min(img.shape[0], y + int(h * 2.2))
            roi = img[y1:y2, x1:x2]
            roi_candidate = _extract_date_from_roi(roi)
            if roi_candidate:
                return roi_candidate

    # Last fallback: top-right region of first page
    h, w = first_gray.shape[:2]
    top_right = first_gray[0:int(h * 0.35), int(w * 0.45):w]
    top_right_candidate = _extract_date_from_roi(top_right)
    if top_right_candidate:
        return top_right_candidate

    return None
