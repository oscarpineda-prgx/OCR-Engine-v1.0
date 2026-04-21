import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

# Use shared RFC module when available for normalization/validation
try:
    from app.core.rfc import normalize_rfc_ocr as _shared_normalize_rfc
    from app.core.rfc import is_valid_rfc_date as _shared_is_valid_rfc_date
    _HAS_SHARED_RFC = True
except ImportError:
    _HAS_SHARED_RFC = False


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _clean_encoding_noise(text: str) -> str:
    return (
        (text or "")
        .replace("ÃƒÂ³", "o")
        .replace("ÃƒÂ¡", "a")
        .replace("ÃƒÂ©", "e")
        .replace("ÃƒÂ­", "i")
        .replace("ÃƒÂº", "u")
        .replace("ÃƒÂ±", "n")
    )


def _strip_accents(value: str) -> str:
    table = str.maketrans("áéíóúÁÉÍÓÚñÑ", "aeiouAEIOUnN")
    return (value or "").translate(table)


def _normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _normalize_for_doc_type(text: str) -> str:
    cleaned = _clean_encoding_noise(text or "")
    cleaned = _strip_accents(cleaned).lower()
    cleaned = re.sub(r"[^a-z0-9\s/&\-_]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# ---------------------------------------------------------------------------
# RFC extraction
# ---------------------------------------------------------------------------

MEX_RFC_PATTERN = re.compile(r"\b([A-Z\u00D1&]{3,4}\d{6}[A-Z0-9]{3})\b", re.IGNORECASE)
MEX_RFC_GROUPED_PATTERN = re.compile(
    r"\b([A-Z\u00D1&]{3,4})[\s\-\._]*([0-9OIS]{2})[\s\-\._]*([0-9OIS]{2})[\s\-\._]*([0-9OIS]{2})[\s\-\._]*([A-Z0-9]{2,3})\b",
    re.IGNORECASE,
)
RFC_LABEL_PATTERN = re.compile(
    r"(?i)\b(?:RFC|R\.?\s*F\.?\s*C\.?|REGISTRO\s+FEDERAL\s+DE\s+CONTRIBUYENTES|F\.?\s*C)\b"
)
RFC_FIELD_FALLBACK_CONTEXT_PATTERN = re.compile(
    r"(?i)\b(?:RAZ\w*\s+SOCI\w*|DOMICILIO\s+FISCAL|NOMBRE\s+DEL\s+PROVEEDOR|PROVEEDOR|TIPO\s+DE\s+PROVEEDOR)\b"
)
FOREIGN_TAX_ID_PATTERN = re.compile(
    r"(?i)\b(?:US\s+FEDERAL\s+TAX\s+ID|FEDERAL\s+TAX\s+ID|TAX\s+ID|TIN|EIN)\b[^0-9]{0,12}([0-9]{2}\s*[\-\.]?\s*[0-9]{7})\b"
)
IGNORED_CLIENT_RFCS = {"CCO8605231N4"}


def _normalize_mex_rfc(token: str) -> str:
    if _HAS_SHARED_RFC:
        return _shared_normalize_rfc(token)
    # Fallback: original logic
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
    return "".join(chars)


def _is_valid_mex_rfc_date(token: str) -> bool:
    if _HAS_SHARED_RFC:
        return _shared_is_valid_rfc_date(token)
    # Fallback: original logic
    up = token.upper()
    if len(up) == 12:
        part = up[3:9]
    elif len(up) == 13:
        part = up[4:10]
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


def _score_rfc_context(full_upper: str, start: int, end: int, base: int) -> int:
    window = full_upper[max(0, start - 90): min(len(full_upper), end + 90)]
    score = base
    if "RFC" in window or "REGISTRO FEDERAL DE CONTRIBUYENTES" in window:
        score += 25
    if "EL PROVEEDOR" in window or "PROVEEDOR" in window:
        score += 30
    if "EL CLIENTE" in window or "CADENA COMERCIAL OXXO" in window or "OXXO" in window:
        score -= 20
    return score


def _normalize_foreign_tax_id(token: str) -> str:
    digits = re.sub(r"\D", "", token or "")
    if len(digits) != 9:
        return ""
    return f"{digits[:2]}-{digits[2:]}"


def _extract_soft_labeled_rfc(lines: list[str]) -> str | None:
    """Fallback for OCR forms with an explicit RFC field but malformed date digits.

    Some scanned supplier forms contain an RFC typed in a dedicated labeled block,
    yet OCR or source typos make the YYMMDD portion fail strict date validation.
    We only accept that fallback when the local neighborhood looks like a supplier
    identity block, so a bare invalid RFC line is still rejected.
    """
    for i, line in enumerate(lines):
        if not RFC_LABEL_PATTERN.search(line):
            continue

        scope = "\n".join(lines[max(0, i - 2): min(len(lines), i + 3)])
        if not RFC_FIELD_FALLBACK_CONTEXT_PATTERN.search(scope):
            continue

        upper_line = line.upper()
        for pattern in (MEX_RFC_PATTERN, MEX_RFC_GROUPED_PATTERN):
            for match in pattern.finditer(upper_line):
                token = match.group(1) if pattern is MEX_RFC_PATTERN else "".join(match.groups())
                candidate = _normalize_mex_rfc(token)
                if candidate in IGNORED_CLIENT_RFCS:
                    continue
                return candidate

    return None


def extract_rfc(text: str) -> str | None:
    if not text:
        return None

    source = _clean_encoding_noise(text)
    upper = source.upper().replace("_", " ")
    candidates: list[tuple[int, int, str]] = []

    lines = upper.splitlines()
    running_pos = 0
    for i, line in enumerate(lines):
        line_start = running_pos
        running_pos += len(line) + 1
        if "CURP" in line:
            continue

        scope = line
        if i + 1 < len(lines):
            scope = f"{line} {lines[i + 1]}"
        base = 70 if RFC_LABEL_PATTERN.search(line) else 35

        for m in MEX_RFC_PATTERN.finditer(scope):
            cand = _normalize_mex_rfc(m.group(1))
            if cand in IGNORED_CLIENT_RFCS:
                continue
            if _is_valid_mex_rfc_date(cand):
                s = line_start + m.start(1)
                e = line_start + m.end(1)
                candidates.append((_score_rfc_context(upper, s, e, base), s, cand))

        for m in MEX_RFC_GROUPED_PATTERN.finditer(scope):
            cand = _normalize_mex_rfc("".join(m.groups()))
            if cand in IGNORED_CLIENT_RFCS:
                continue
            if _is_valid_mex_rfc_date(cand):
                s = line_start + m.start(1)
                e = line_start + m.end(4)
                candidates.append((_score_rfc_context(upper, s, e, base - 3), s, cand))

    if candidates:
        best_by_value: dict[str, tuple[int, int, str]] = {}
        for item in candidates:
            score, pos, value = item
            cur = best_by_value.get(value)
            if cur is None or score > cur[0] or (score == cur[0] and pos < cur[1]):
                best_by_value[value] = item
        ranked = sorted(best_by_value.values(), key=lambda x: (-x[0], x[1]))
        result = ranked[0][2]
        logger.debug("RFC extracted: %s", result)
        return result

    soft_fallback = _extract_soft_labeled_rfc(lines)
    if soft_fallback:
        logger.debug("RFC extracted via labeled fallback: %s", soft_fallback)
        return soft_fallback

    # Foreign IDs (e.g. US Federal Tax ID)
    m = FOREIGN_TAX_ID_PATTERN.search(source)
    if m:
        foreign = _normalize_foreign_tax_id(m.group(1))
        if foreign:
            logger.debug("RFC extracted (foreign tax ID): %s", foreign)
            return foreign

    # Contextual foreign fallback
    m2 = re.search(
        r"(?i)\bRFC\b.{0,50}\b(?:US\s+FEDERAL\s+TAX\s+ID|TAX\s+ID|TIN|EIN)\b[^0-9]{0,12}([0-9]{2}\s*[\-\.]?\s*[0-9]{7})\b",
        source,
    )
    if m2:
        foreign = _normalize_foreign_tax_id(m2.group(1))
        if foreign:
            logger.debug("RFC extracted (contextual foreign): %s", foreign)
            return foreign

    logger.debug("No RFC match found in text")
    return None


# ---------------------------------------------------------------------------
# Date extraction
# ---------------------------------------------------------------------------

MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}

MONTH_ABBREVS: dict[str, int] = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4,
    "may": 5, "jun": 6, "jul": 7, "ago": 8,
    "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}

DOCUMENT_DATE_MIN = datetime(year=2000, month=1, day=1)
DOCUMENT_SPLIT_YEAR_MIN = 2000
DOCUMENT_SPLIT_YEAR_MAX = 2100


def _match_month_name(raw: str) -> int | None:
    """Match a Spanish month name, handling OCR-truncated variants."""
    key = _strip_accents(raw.strip().lower())
    key = re.sub(r"[^a-z]", "", key)
    if not key or len(key) < 3:
        return None
    if key in MONTHS:
        return MONTHS[key]
    if key in MONTH_ABBREVS:
        return MONTH_ABBREVS[key]
    for full_name, num in MONTHS.items():
        if full_name.startswith(key) or key.startswith(full_name):
            return num
    for abbr, num in MONTH_ABBREVS.items():
        if abbr.startswith(key) or (key.startswith(abbr) and len(key) - len(abbr) <= 1):
            return num
    for alias, num in MONTH_ABBREVS.items():
        if len(key) <= 4 and _is_single_edit_month_match(key, alias):
            return num
    return None


def _is_single_edit_month_match(value: str, target: str) -> bool:
    if abs(len(value) - len(target)) > 1:
        return False

    if value == target:
        return True

    # Fast path for same-length OCR substitutions.
    if len(value) == len(target):
        mismatches = sum(1 for left, right in zip(value, target) if left != right)
        return mismatches <= 1

    # Single insertion/deletion tolerance.
    if len(value) > len(target):
        value, target = target, value

    i = j = edits = 0
    while i < len(value) and j < len(target):
        if value[i] == target[j]:
            i += 1
            j += 1
            continue
        edits += 1
        if edits > 1:
            return False
        j += 1

    return True

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b"),
    re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2})\b"),
    re.compile(r"\b(\d{1,2}-\d{1,2}-\d{4})\b"),
    re.compile(r"\b(\d{1,2}-\d{1,2}-\d{2})\b"),
    re.compile(r"\b(\d{4}-\d{1,2}-\d{1,2})\b"),
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b"),
    re.compile(r"\b(\d{1,2}\.\d{1,2}\.\d{2})\b"),
]
TEXTUAL_DATE_PATTERN = re.compile(
    r"(?i)\b(?:siendo\s+el\s+d[ií]a\s+)?(\d{1,2})\s*(?:de)?\s+([a-zA-Z\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]+)\s*(?:de)?\s+(\d{4})\b"
)
TEXTUAL_DATE_OCR_PATTERN = re.compile(
    r"(?i)\b(?:siendo\s+el\s+d[ií]a\s+)?([0-9OILS]{1,2})\s*(?:de)?\s*([a-zA-Z\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]{3,15})\s*(?:de)?\s*([0-9OILS]{4})\b"
)
NUMERIC_TEXTUAL_DATE_PATTERN = re.compile(
    r"(?i)\b(?:siendo\s+el\s+d[ií]a\s+)?([0-9OILS]{1,2})\s*de\s*([0-9OILS]{1,2})\s*(?:de|del)\s*([0-9OILS]{4})\b"
)


TEXTUAL_DATE_SHORT_YEAR_PATTERN = re.compile(
    r"(?i)\b(?:siendo\s+el\s+d[iÃ­]a\s+)?(\d{1,2})\s*(?:de)?\s+([a-zA-Z\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]+)\s*(?:de)?\s+(\d{2})\b"
)
TEXTUAL_DATE_OCR_SHORT_YEAR_PATTERN = re.compile(
    r"(?i)\b(?:siendo\s+el\s+d[iÃ­]a\s+)?([0-9OILS]{1,2})\s*(?:de)?\s*([a-zA-Z0-9\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]{3,15})\s*(?:de)?\s*([0-9OILS]{2})\b"
)
NUMERIC_TEXTUAL_DATE_SHORT_YEAR_PATTERN = re.compile(
    r"(?i)\b(?:siendo\s+el\s+d[iÃ­]a\s+)?([0-9OILS]{1,2})\s*de\s*([0-9OILS]{1,2})\s*(?:de|del)\s*([0-9OILS]{2})\b"
)
SLASH_TEXTUAL_DATE_PATTERN = re.compile(
    r"(?i)\b([0-9OILS]{1,2})\s*[/\-\._]?\s*([a-zA-Z0-9\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]{3,10})\s*[/\-\._]?\s*([0-9OILS]{2,4})\b"
)
MONTH_FIRST_TEXTUAL_DATE_PATTERN = re.compile(
    r"(?i)\b([a-zA-Z\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]{3,15})\s+([0-9OILS]{1,2})\s*,?\s+([0-9OILS]{4})\b"
)
MONTH_FIRST_TEXTUAL_DATE_SHORT_YEAR_PATTERN = re.compile(
    r"(?i)\b([a-zA-Z0-9\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]{3,15})\s+([0-9OILS]{1,2})\s*,?\s+([0-9OILS]{2})\b"
)
NOISY_OCR_TEXTUAL_DATE_PATTERN = re.compile(
    r"(?i)\b([0-9OILS]{1,2})\s*([a-zA-Z]{3,6})[0-9OILS]?\s*([0-9OILS]{2,4})\b"
)
SIGNATURE_FORMULA_DATE_PATTERN = re.compile(
    "(?i)\\b(?:a\\s+los?|los?)\\s+([0-9OILS]{1,2})\\s+"
    "d[i\u00ED]as?\\s+del\\s+mes\\s+de\\s+"
    "([a-zA-Z0-9\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]{3,15})\\s+"
    "(?:de|del)\\s+([0-9OILS]{2,4})\\b"
)
NOISY_SPLIT_YEAR_TEXTUAL_DATE_PATTERN = re.compile(
    "(?i)\\b(?:siendo\\s+el\\s+d[i\u00ED]a\\s*)?"
    "([0-9OILS]{1,2})\\s*[\\|\\[\\]!¡,.;:_-]*\\s*(?:de)?\\s+"
    "([a-zA-Z0-9\u00E1\u00E9\u00ED\u00F3\u00FA\u00F1]{3,15})\\s*"
    "[\\|\\[\\]!¡,.;:_-]*\\s*(?:de|del)?\\s*"
    "([0-9OILS]{2}\\s*[/\\|\\[\\]!¡,.;:_-]+\\s*[0-9OILS]{2})\\b"
)
HISTORICAL_LEGAL_DATE_CONTEXT_PATTERN = re.compile(
    r"\b("
    r"ley|leyes|ley federal|leyes vigentes|gobierno|gubernamental|"
    r"estados unidos mexicanos|diario oficial|federacion|secretaria|"
    r"escritura publica|instrumento publico|fe del notario|notario publico|notarial|"
    r"registro publico|registro publico de la propiedad|propiedad y del comercio|"
    r"decreto|codigo civil|codigo de comercio|articulo|constitucion"
    r")\b"
)
SIGNATURE_CONTEXT_TERMS = (
    "lo firman",
    "firman",
    "firma",
    "firmado",
    "suscriben",
    "suscribe",
    "presente contrato",
    "contrato",
    "partes",
    "testigos",
    "representante legal",
)
SIGNATURE_HEAVY_DOCUMENT_TYPES = {
    "suministro de productos de linea en cedis",
    "convenio entrega local",
}


def normalize_date(value: str) -> str | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%d.%m.%Y", "%d.%m.%y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _is_before_document_min_date(normalized_date: str) -> bool:
    try:
        parsed = datetime.strptime(normalized_date, "%Y-%m-%d")
    except ValueError:
        return False
    return parsed < DOCUMENT_DATE_MIN


def _normalize_ocr_digits(value: str) -> str:
    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5"})
    return (value or "").translate(table)


SPLIT_YEAR_FRAGMENT_PATTERN = re.compile(r"\s*([0-9]{2})\s*[^0-9]+\s*([0-9]{2})\s*")


def _normalize_split_year_fragment(value: str) -> int | None:
    normalized = _normalize_ocr_digits(value)
    match = SPLIT_YEAR_FRAGMENT_PATTERN.fullmatch(normalized)
    if not match:
        return None

    year = int(f"{match.group(1)}{match.group(2)}")
    if DOCUMENT_SPLIT_YEAR_MIN <= year <= DOCUMENT_SPLIT_YEAR_MAX:
        return year
    return None


def _normalize_year_fragment(value: str) -> int | None:
    normalized = _normalize_ocr_digits(value)
    split_year = _normalize_split_year_fragment(value)
    if split_year is not None:
        return split_year
    if SPLIT_YEAR_FRAGMENT_PATTERN.fullmatch(normalized):
        return None

    digits = re.sub(r"\D", "", normalized)
    if len(digits) == 2:
        yy = int(digits)
        return 1900 + yy if yy >= 30 else 2000 + yy
    if len(digits) == 4:
        return int(digits)
    return None


def _has_split_year_tail(source: str, year_end: int) -> bool:
    tail = source[year_end:year_end + 12]
    return re.match(r"\s*[/\|\[\]!¡,.;:_-]+\s*[0-9OILS]{2}\b", tail, flags=re.IGNORECASE) is not None


def _parse_textual_date_flexible(day_raw: str, month_raw: str, year_raw: str) -> str | None:
    day_raw = _normalize_ocr_digits(day_raw)
    try:
        day = int(day_raw)
    except ValueError:
        return None

    year = _normalize_year_fragment(year_raw)
    if year is None:
        return None

    month = _match_month_name(month_raw)
    if not month:
        return None

    try:
        return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_numeric_textual_date_flexible(day_raw: str, month_raw: str, year_raw: str) -> str | None:
    day_raw = _normalize_ocr_digits(day_raw)
    month_raw = _normalize_ocr_digits(month_raw)
    try:
        day = int(day_raw)
        month = int(month_raw)
    except ValueError:
        return None

    year = _normalize_year_fragment(year_raw)
    if year is None:
        return None

    try:
        return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_month_first_textual_date_flexible(month_raw: str, day_raw: str, year_raw: str) -> str | None:
    day_raw = _normalize_ocr_digits(day_raw)
    try:
        day = int(day_raw)
    except ValueError:
        return None

    year = _normalize_year_fragment(year_raw)
    if year is None:
        return None

    month = _match_month_name(month_raw)
    if not month:
        return None

    try:
        return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_textual_date(day_raw: str, month_raw: str, year_raw: str) -> str | None:
    day_raw = _normalize_ocr_digits(day_raw)
    year_raw = _normalize_ocr_digits(year_raw)
    try:
        day = int(day_raw)
        year = int(year_raw)
    except ValueError:
        return None

    month = _match_month_name(month_raw)
    if not month:
        return None

    try:
        return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _parse_numeric_textual_date(day_raw: str, month_raw: str, year_raw: str) -> str | None:
    day_raw = _normalize_ocr_digits(day_raw)
    month_raw = _normalize_ocr_digits(month_raw)
    year_raw = _normalize_ocr_digits(year_raw)
    try:
        day = int(day_raw)
        month = int(month_raw)
        year = int(year_raw)
    except ValueError:
        return None

    try:
        return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except ValueError:
        return None


def _date_context(source: str, start: int, end: int, *, before: int = 160, after: int = 160) -> str:
    raw = source[max(0, start - before): min(len(source), end + after)]
    return _strip_accents(raw.lower())


def _is_signature_heavy_document_type(tipo_documento: str | None) -> bool:
    normalized = _normalize_for_doc_type(tipo_documento or "")
    return normalized in SIGNATURE_HEAVY_DOCUMENT_TYPES


def _score_date_context(ctx: str, base: int, tipo_documento: str | None = None) -> int:
    score = base
    if "siendo el dia" in ctx:
        score += 8
    if "fecha del documento" in ctx or "fecha de documento" in ctx or "fecha de emision" in ctx:
        score += 6
    if "fecha de entrega" in ctx:
        score += 12
    if "anexo al convenio" in ctx:
        score += 3
    if "fecha" in ctx:
        score += 2

    signature_hits = sum(1 for term in SIGNATURE_CONTEXT_TERMS if term in ctx)
    if signature_hits:
        score += min(18, signature_hits * 5)
        if _is_signature_heavy_document_type(tipo_documento):
            score += 8

    return score


def _append_fecha_candidate(
    candidates: list[tuple[int, int, str]],
    source: str,
    start: int,
    end: int,
    normalized: str,
    base: int,
    tipo_documento: str | None = None,
) -> None:
    if _is_before_document_min_date(normalized):
        logger.debug("Fecha skipped before %s: %s", DOCUMENT_DATE_MIN.date(), normalized)
        return

    ctx = _date_context(source, start, end)
    if HISTORICAL_LEGAL_DATE_CONTEXT_PATTERN.search(ctx):
        logger.debug("Fecha skipped by legal/government context: %s", normalized)
        return

    candidates.append((_score_date_context(ctx, base, tipo_documento), start, normalized))


def _extract_noisy_ocr_textual_dates(
    source: str,
    tipo_documento: str | None = None,
) -> list[tuple[int, int, str]]:
    candidates: list[tuple[int, int, str]] = []

    for m in NOISY_OCR_TEXTUAL_DATE_PATTERN.finditer(source or ""):
        normalized = _parse_textual_date_flexible(m.group(1), m.group(2), m.group(3))
        if not normalized:
            continue
        _append_fecha_candidate(
            candidates,
            source,
            m.start(),
            m.end(),
            normalized,
            7,
            tipo_documento,
        )

    return candidates


def extract_fecha_documento(text: str, tipo_documento: str | None = None) -> str | None:
    if not text:
        return None

    source = _clean_encoding_noise(text).replace("_", "")
    candidates: list[tuple[int, int, str]] = []

    for pattern in DATE_PATTERNS:
        for m in pattern.finditer(source):
            normalized = normalize_date(m.group(1))
            if not normalized:
                continue
            _append_fecha_candidate(
                candidates,
                source,
                m.start(1),
                m.end(1),
                normalized,
                2,
                tipo_documento,
            )

    for pattern, parser, base in [
        (TEXTUAL_DATE_PATTERN, _parse_textual_date, 4),
        (TEXTUAL_DATE_OCR_PATTERN, _parse_textual_date, 3),
        (NUMERIC_TEXTUAL_DATE_PATTERN, _parse_numeric_textual_date, 5),
        (TEXTUAL_DATE_SHORT_YEAR_PATTERN, _parse_textual_date_flexible, 7),
        (TEXTUAL_DATE_OCR_SHORT_YEAR_PATTERN, _parse_textual_date_flexible, 7),
        (NUMERIC_TEXTUAL_DATE_SHORT_YEAR_PATTERN, _parse_numeric_textual_date_flexible, 7),
        (SLASH_TEXTUAL_DATE_PATTERN, _parse_textual_date_flexible, 8),
        (MONTH_FIRST_TEXTUAL_DATE_PATTERN, _parse_month_first_textual_date_flexible, 8),
        (MONTH_FIRST_TEXTUAL_DATE_SHORT_YEAR_PATTERN, _parse_month_first_textual_date_flexible, 8),
        (SIGNATURE_FORMULA_DATE_PATTERN, _parse_textual_date_flexible, 14),
        (NOISY_SPLIT_YEAR_TEXTUAL_DATE_PATTERN, _parse_textual_date_flexible, 12),
    ]:
        for m in pattern.finditer(source):
            if _has_split_year_tail(source, m.end(3)):
                continue
            normalized = parser(m.group(1), m.group(2), m.group(3))
            if not normalized:
                continue
            _append_fecha_candidate(
                candidates,
                source,
                m.start(),
                m.end(),
                normalized,
                base,
                tipo_documento,
            )

    candidates.extend(_extract_noisy_ocr_textual_dates(source, tipo_documento))

    if not candidates:
        logger.debug("No fecha match found in text")
        return None

    candidates.sort(key=lambda x: (-x[0], x[1]))
    result = candidates[0][2]
    logger.debug("Fecha extracted: %s", result)
    return result


# ---------------------------------------------------------------------------
# Document type extraction
# ---------------------------------------------------------------------------

_BUILTIN_DOCUMENT_TYPE_RULES = [
    {"name": "Carta Anexo a Convenio Entrega a Cedis", "any_terms": ("carta anexo b convenio entrega en cedis", "carta anexo convenio entrega en cedis", "carta anexo convenio en cedis")},
    {"name": "Carta Intercedis", "any_terms": ("carta intercedis", "intercedis"), "all_terms": ("carta",)},
    {"name": "Excepciones a Condiciones Logisticas", "any_terms": ("formato excepciones a condiciones logisticas", "excepciones a condiciones logisticas")},
    {"name": "Adendums Soporte", "any_terms": ("adendum", "adendums", "cartas de acuerdos de operaciones logisticas", "acuerdos de operaciones logisticas", "incrementando su costo", "incremento de costo")},
    {"name": "Carta Soporte Descuentos", "all_terms": ("carta soporte", "descuento")},
    {"name": "Carta Soporte Costos", "all_terms": ("carta soporte",), "any_terms": ("costo", "costos"), "none_terms": ("descuento", "descuentos")},
    {"name": "Solicitud cambio Descuento", "all_terms": ("solicitud de cambio", "descuento")},
    {"name": "Solicitud Cambio Costo", "all_terms": ("solicitud de cambio",), "any_terms": ("costo", "costos"), "none_terms": ("descuento", "descuentos")},
    {"name": "Convenio Modificatorio Cedis", "all_terms": ("convenio modificatorio", "cedis")},
    {"name": "Convenio Modificatorio Local", "any_terms": ("carta anexo b convenio entrega en tienda", "convenio modificatorio"), "all_terms": ("tienda",)},
    {"name": "Comida Rapida", "any_terms": ("contrato comida rapida", "comida rapida")},
    {"name": "Convenio de Manufactura y Suministros", "any_terms": ("convenio de manufactura y suministros", "convenio de manufactura y suministro de productos")},
    {"name": "Convenio Entrega Cedis", "any_terms": ("convenio entrega en cedis", "convenio entrega cedis"), "none_terms": ("carta anexo",)},
    {"name": "Convenio Entrega Local", "any_terms": ("convenio proveedores locales", "proveedores locales")},
    {"name": "Convenio Factoraje", "any_terms": ("convenio relativo a la mecanica operativa del sistema de factoraje", "factoraje"), "all_terms": ("convenio",)},
    {"name": "Convenio Genericos", "any_terms": ("convenio oxxo genericos", "oxxo genericos")},
    {"name": "Convenio Logistico", "any_terms": ("convenio logistico",)},
    {"name": "Convenio Condiciones Comerciales", "any_terms": ("convenio condiciones comerciales",), "all_terms": ("condiciones comerciales",)},
    {"name": "Importaciones Marca Propia", "any_terms": ("contrato importaciones marcas propias", "importaciones marcas propias")},
    {"name": "Importations Contract", "any_terms": ("importations contract of store brand", "importations contract")},
    {"name": "Logistico MP", "any_terms": ("contrato logistico marca propia personas morales", "logistico marca propia personas morales")},
    {"name": "Marca Propia CEDIS", "any_terms": ("contrato entrega en cedis marca propia personas fisicas", "entrega en cedis marca propia personas fisicas")},
    {"name": "Marca Propia Tiendas", "any_terms": ("contrato marca propia personas morales entrega en tienda", "marca propia personas morales entrega en tienda")},
    {"name": "Portal de Proveedores", "any_terms": ("convenio para acceso a portal de proveedores", "convenio portal de proveedores", "portal de proveedores")},
    {"name": "Product Supply Agreement", "any_terms": ("product supply agreement",)},
    {"name": "Proveedores Nacionales y Regionales Cedis", "any_terms": ("proveedores nacionales y regionales a cedis",)},
    {"name": "Suministro de Materias Primas", "any_terms": ("convenio de suministro de materias primas personas morales", "carta anexo a convenio de suministro de materias primas a personas morales")},
    {"name": "Suministro de Productos de Linea en CEDIS", "any_terms": ("contrato de suministro de productos de linea en cedis", "suministro de productos de linea en cedis")},
]
_BUILTIN_DEFAULT_TYPE = "Documentos Varios"


@lru_cache(maxsize=1)
def _load_document_type_config() -> tuple[list[dict], str]:
    """Load document type rules from YAML, falling back to built-in rules."""
    if _yaml is None:
        return _BUILTIN_DOCUMENT_TYPE_RULES, _BUILTIN_DEFAULT_TYPE
    rules_path = Path(__file__).resolve().parents[3] / "docs" / "rules" / "document_type_rules.yaml"
    if not rules_path.exists():
        return _BUILTIN_DOCUMENT_TYPE_RULES, _BUILTIN_DEFAULT_TYPE
    with open(rules_path, "r", encoding="utf-8") as f:
        data = _yaml.safe_load(f)
    rules = data.get("rules", _BUILTIN_DOCUMENT_TYPE_RULES)
    default_type = data.get("default_type", _BUILTIN_DEFAULT_TYPE)
    return rules, default_type


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _contains_all(text: str, terms: tuple[str, ...]) -> bool:
    return all(term in text for term in terms)


def _score_document_type_rule(rule: dict, first_page: str, full_text: str) -> int | None:
    any_terms = tuple(rule.get("any_terms", ()))
    all_terms = tuple(rule.get("all_terms", ()))
    none_terms = tuple(rule.get("none_terms", ()))

    if any_terms and not _contains_any(full_text, any_terms):
        return None
    if all_terms and not _contains_all(full_text, all_terms):
        return None
    if none_terms and _contains_any(full_text, none_terms):
        return None

    score = 10
    if any_terms:
        score += sum(8 for t in any_terms if t in full_text)
    if all_terms:
        score += 10 * len(all_terms)

    first_terms = any_terms or all_terms
    if first_terms and _contains_any(first_page, first_terms):
        score += 35
    if first_terms and _contains_any(first_page[:1200], first_terms):
        score += 15
    return score


def extract_tipo_documento(text: str) -> str | None:
    if not text:
        return None
    normalized = _normalize_for_doc_type(text)
    if not normalized:
        return None

    rules, default_type = _load_document_type_config()
    first_page_window = normalized[:7000]
    best: tuple[int, int, str] | None = None
    for idx, rule in enumerate(rules):
        score = _score_document_type_rule(rule, first_page_window, normalized)
        if score is None:
            continue
        cur = (score, -idx, rule["name"])
        if best is None or cur > best:
            best = cur
    result = best[2] if best else default_type
    logger.debug("Tipo documento extracted: %s", result)
    return result


# ---------------------------------------------------------------------------
# Supplier name extraction
# ---------------------------------------------------------------------------

PROVEEDOR_CONTEXT_PATTERNS = [
    re.compile(
        r"(?is)\bconfirma\s+su\s+acuerdo\s+con\s+(.+?)\s+a\s+quien\s+en\s+lo\s+sucesivo(?:\s+se\s+le)?(?:\s+denom\w+)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?is)\by\s+(.+?)\s+con\s+RFC\b.*?(?:denomina|denominara|denominado|en\s+delante\s+se\s+denomina)\s+[\*\"'“”]?\s*EL\s+PROVEEDOR",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?is)\bEL\s+CLIENTE.{0,280}?\by\s+(.+?)\s+con\s+RFC\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?is)\by\s+(.+?)\s+con\s+RFC\s+(?:US\s+FEDERAL\s+TAX\s+ID|TAX\s+ID|TIN|EIN)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?is)\bpor\s+otra\s+parte\s+la\s+sociedad\s+mercantil\s+(.+?)(?:\s+representad[ao]|\s+a\s+quien\b|,)",
        re.IGNORECASE,
    ),
]
PROVEEDOR_LINE_PATTERNS = [
    re.compile(
        r"(?im)^\s*(?:RAZ[\w?]*\s+SOCI[\w?]*(?:\s+PROVEEDOR)?|PROVEEDOR|EMISOR)\s*[:\-]\s*(.+?)\s*$"
    ),
]
LEGAL_ENTITY_PATTERN = re.compile(
    r"(?i)\bS\.?\s*A\.?\s*(DE\s*C\.?\s*V\.?)?\b|\bLLC\b|\bINC\b|\bLTD\b|\bCOMPANY\b"
)
COMPANY_END_PATTERN = re.compile(r"(?i)\b(.+?\bS\.?\s*A\.?\s*DE\s*C\.?\s*V\.?)\b")
TRAILING_NOISE_PATTERN = re.compile(
    r"(?i)\b(NO\.?\s*CUENTA|NUM\.?\s*CUENTA|CODIGO\s*DE\s*CLIENTE|CTA\.?|CURP|RFC)\b.*$"
)
BLOCKED_PROVIDER_PATTERN = re.compile(r"(?i)(CADENA\s+COMERCIAL\s+OXXO|FEMSA|EL\s+CLIENTE)")
PLACEHOLDER_PROVIDER_PATTERN = re.compile(
    r"(?i)\b(?:NOMBRE\s+DE\s+RAZ\w*\s+SOCI\w*|NOMBRE\s+DEL\s+PROVEEDOR|NOMBRE\s+PROVEEDOR)\b"
)


def _looks_like_company_name(value: str) -> bool:
    clean = _normalize_spaces(value).strip(" .:-|")
    if len(clean) < 4:
        return False
    if BLOCKED_PROVIDER_PATTERN.search(clean):
        return False
    if PLACEHOLDER_PROVIDER_PATTERN.search(clean):
        return False

    letters = [c for c in clean if c.isalpha()]
    if not letters:
        return False
    words = [w for w in re.split(r"\s+", clean) if w]
    return bool(LEGAL_ENTITY_PATTERN.search(clean)) or len(words) >= 2


def _looks_like_labeled_supplier_name(value: str) -> bool:
    clean = _normalize_spaces(value).strip(" .:-|")
    if len(clean) < 4:
        return False
    if BLOCKED_PROVIDER_PATTERN.search(clean):
        return False
    if PLACEHOLDER_PROVIDER_PATTERN.search(clean):
        return False
    return _looks_like_company_name(clean)


def _normalize_supplier_name_fragments(value: str) -> str:
    v = _normalize_spaces((value or "").replace("?", " ")).strip(" .:-|")
    if not v or LEGAL_ENTITY_PATTERN.search(v):
        return v

    fragment_pattern = re.compile(r"\b([A-Z][a-z]{2,4})\s+([a-z]{1,4})\b")
    while True:
        new_v = fragment_pattern.sub(
            lambda m: m.group(1) + m.group(2)
            if len(m.group(1)) + len(m.group(2)) <= 10
            else m.group(0),
            v,
        )
        if new_v == v:
            break
        v = new_v
    return _normalize_spaces(v)


def normalize_company_name(value: str) -> str:
    v = _normalize_spaces(value).strip(" .:-|*'\"”“’`_")
    v = re.sub(r"(?i)^y\s+", "", v).strip(" .:-|*'\"”“’`_")
    v = re.sub(r"(?i)^.*?\bACUERDO\s+CON\s+", "", v).strip(" .:-|")
    v = re.sub(r"(?i)\s+CON\s+RFC\b.*$", "", v).strip(" .:-|")
    v = re.sub(r"(?i)\s+a\s+quien\b.*$", "", v).strip(" .:-|")
    v = re.sub(r"(?i)\s+representad[ao]\s+por\b.*$", "", v).strip(" .:-|")
    v = TRAILING_NOISE_PATTERN.sub("", v).strip(" .:-|")
    v = _normalize_supplier_name_fragments(v)
    m = COMPANY_END_PATTERN.search(v)
    if m:
        return m.group(1).strip(" .:-|")
    return v


def extract_nombre_proveedor(text: str) -> str | None:
    if not text:
        return None

    cleaned = _clean_encoding_noise(text)
    search_text = _strip_accents(cleaned)

    for pattern in PROVEEDOR_LINE_PATTERNS:
        for m in pattern.finditer(search_text):
            candidate = normalize_company_name(m.group(1))
            if candidate and _looks_like_labeled_supplier_name(candidate):
                logger.debug("Nombre proveedor extracted: %s", candidate)
                return candidate

    for pattern in PROVEEDOR_CONTEXT_PATTERNS:
        m = pattern.search(search_text)
        if not m:
            continue
        candidate = normalize_company_name(m.group(1))
        if candidate and _looks_like_company_name(candidate):
            logger.debug("Nombre proveedor extracted: %s", candidate)
            return candidate

    for line in search_text.splitlines():
        line = line.strip(" .:-|")
        if not line:
            continue
        if " ACUERDO CON " in line.upper():
            candidate = normalize_company_name(line)
            if candidate and _looks_like_company_name(candidate):
                logger.debug("Nombre proveedor extracted: %s", candidate)
                return candidate

    logger.debug("No nombre proveedor match found in text")
    return None
