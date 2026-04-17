"""Unified RFC normalization and validation.

Consolidates duplicated RFC logic previously scattered across three modules:

- app/services/extraction/ocr_pdf.py   (_normalize_rfc_ocr, _is_valid_rfc_date,
                                        _has_valid_rfc_check_digit, _is_acceptable_rfc_hint)
- app/services/parsing/fields.py       (_normalize_mex_rfc, _is_valid_mex_rfc_date)
- app/services/vendor_master/matcher.py (normalize_rfc_for_match, canonicalize_rfc)

This module is the single source of truth for RFC handling.  It depends only
on the Python standard library (``re`` and ``datetime``).
"""

from __future__ import annotations

import re
from itertools import product
from datetime import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RFC_SHAPE_PATTERN: re.Pattern[str] = re.compile(
    r"^[A-Z\u00D1&]{3,4}[0-9]{6}[A-Z0-9]{2,3}$"
)
"""Anchored pattern that matches a well-formed Mexican RFC."""

RFC_INLINE_PATTERN: re.Pattern[str] = re.compile(
    r"[A-Z\u00D1&]{3,4}[0-9OISBGZD]{6}[A-Z0-9]{2,3}"
)
"""Un-anchored pattern for finding RFC-like tokens inside free text,
tolerating common OCR misreads in the numeric portion."""

FOREIGN_TAX_FORMAT: re.Pattern[str] = re.compile(
    r"^(\d{2})-(\d{7})$"
)
"""Pattern for a US-style foreign tax ID already in XX-XXXXXXX form."""

RFC_CHAR_VALUES: dict[str, int] = {
    **{str(i): i for i in range(10)},
    "A": 10, "B": 11, "C": 12, "D": 13, "E": 14,
    "F": 15, "G": 16, "H": 17, "I": 18, "J": 19,
    "K": 20, "L": 21, "M": 22, "N": 23, "&": 24,
    "O": 25, "P": 26, "Q": 27, "R": 28, "S": 29,
    "T": 30, "U": 31, "V": 32, "W": 33, "X": 34,
    "Y": 35, "Z": 36, "\u00D1": 37,
}
"""Character weights used by the SAT modulo-11 check-digit algorithm."""

OCR_NUMERIC_CORRECTIONS: dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "G": "6",
    "Z": "2",
    "D": "0",
    "l": "1",  # lowercase L
}
"""Expanded OCR confusion table for characters that appear inside the
six-digit date portion of an RFC where only digits are expected."""

PERSONA_FISICA_SKIP_GIVEN_NAMES: set[str] = {"JOSE", "JO", "J", "MARIA", "MA", "MA."}
PERSONA_FISICA_LEGAL_ENTITY_PATTERN: re.Pattern[str] = re.compile(
    r"(?i)\bS\.?\s*A\.?\s*(DE\s+C\.?\s*V\.?)?\b|\bLLC\b|\bINC\b|\bLTD\b|\bCOMPANY\b"
)
RFC_TAIL_OCR_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "0": ("0", "O", "Q", "D"),
    "O": ("O", "0", "Q", "D"),
    "Q": ("Q", "0", "O"),
    "1": ("1", "I", "L"),
    "I": ("I", "1", "L"),
    "L": ("L", "1", "I"),
    ")": ("L", "1", "I", "9"),
    "9": ("9", "L", "1", "I"),
    "5": ("5", "S"),
    "S": ("S", "5", "8"),
    "8": ("8", "B"),
    "B": ("B", "8"),
    "6": ("6", "G"),
    "G": ("G", "6"),
    "2": ("2", "Z"),
    "Z": ("Z", "2"),
    "7": ("7", "T"),
    "T": ("T", "7"),
}

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_rfc_ocr(token: str) -> str:
    """OCR-aware RFC normalization.

    1. Strip everything except ``[A-Z0-9\u00D1&]`` and uppercase.
    2. In the six-digit *date portion* (positions 3-8 for 12-char RFCs,
       4-9 for 13-char RFCs), replace visually ambiguous letters with the
       digit they most likely represent using :data:`OCR_NUMERIC_CORRECTIONS`.

    Returns the cleaned token (may still be invalid).
    """
    # Preserve lowercase 'l' before uppercasing so the table can catch it.
    raw_preserved = re.sub(r"[^A-Za-z0-9\u00D1&]", "", token or "")
    chars = list(raw_preserved)

    if len(chars) == 12:
        num_range = range(3, 9)
    elif len(chars) == 13:
        num_range = range(4, 10)
    else:
        # Length doesn't match a valid RFC; just clean and return.
        return re.sub(r"[^A-Z0-9\u00D1&]", "", (token or "").upper())

    # Apply OCR corrections *before* uppercasing so lowercase 'l' works.
    for i in num_range:
        replacement = OCR_NUMERIC_CORRECTIONS.get(chars[i])
        if replacement is not None:
            chars[i] = replacement

    result = "".join(chars).upper()
    return re.sub(r"[^A-Z0-9\u00D1&]", "", result)


def normalize_rfc_for_match(value: str | None) -> str:
    """Simple strip to uppercase alphanumeric for database / comparison matching.

    No OCR correction is applied—just remove anything that is not ``[A-Z0-9]``
    and uppercase the result.
    """
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def canonicalize_rfc(value: str | None) -> str | None:
    """Return the RFC in its standard canonical form.

    * Mexican RFCs are returned as uppercase alphanumeric strings.
    * Foreign tax IDs consisting of exactly 9 digits are reformatted to
      ``XX-XXXXXXX``.
    * Returns ``None`` for empty / whitespace-only input.
    """
    normalized = normalize_rfc_for_match(value)
    if not normalized:
        return None
    if normalized.isdigit() and len(normalized) == 9:
        return f"{normalized[:2]}-{normalized[2:]}"
    return normalized


def _strip_accents(value: str) -> str:
    table = str.maketrans("ÁÉÍÓÚáéíóúÑñ", "AEIOUaeiouNn")
    return (value or "").translate(table)


def derive_persona_fisica_rfc_prefix(full_name: str | None) -> str | None:
    """Best-effort 4-letter RFC prefix for natural-person names.

    Assumes the common order: given names first, then paternal surname,
    then maternal surname. Returns ``None`` for legal-entity-like names or
    when there are not enough tokens to infer the prefix.
    """
    normalized = _strip_accents((full_name or "").upper())
    if PERSONA_FISICA_LEGAL_ENTITY_PATTERN.search(normalized):
        return None

    words = [w for w in re.split(r"[^A-Z]+", normalized) if w]
    if len(words) < 3:
        return None

    paternal = words[-2]
    maternal = words[-1]
    given_names = words[:-2]
    given = next((w for w in given_names if w not in PERSONA_FISICA_SKIP_GIVEN_NAMES), given_names[0])

    internal_vowel = next((ch for ch in paternal[1:] if ch in "AEIOU"), "X")
    return f"{paternal[0]}{internal_vowel}{maternal[0]}{given[0]}"


def repair_persona_fisica_rfc_ocr(raw_candidate: str | None, full_name: str | None) -> str | None:
    """Repair an OCR-damaged RFC using a natural-person supplier name.

    This is intentionally conservative: it only returns a value when the
    supplier name yields a plausible physical-person prefix, the OCR string
    still contains a valid YYMMDD fragment, and one of the small confusion-set
    variants produces a shape-valid RFC with a valid SAT check digit.
    """
    prefix = derive_persona_fisica_rfc_prefix(full_name)
    if not prefix or not raw_candidate:
        return None

    scope = _strip_accents(raw_candidate).upper()
    label_match = re.search(r"\bRFC\b[^\n]{0,80}", scope)
    if label_match:
        scope = label_match.group(0)

    scope = re.sub(r"[^A-Z0-9\u00D1&\)\(\s]", " ", scope)
    match = re.search(r"([0-9OISBGZD]{6})([A-Z0-9\)\(\s]{2,8})", scope)
    if not match:
        return None

    date_digits = "".join(OCR_NUMERIC_CORRECTIONS.get(ch, ch) for ch in match.group(1)).upper()
    if len(date_digits) != 6 or not date_digits.isdigit():
        return None

    observed_tail = re.sub(r"[^A-Z0-9\)\(]", "", match.group(2).upper())
    if len(observed_tail) < 3:
        return None
    observed_tail = observed_tail[:3]

    choices = [RFC_TAIL_OCR_CONFUSIONS.get(ch, (ch,)) for ch in observed_tail]
    best: tuple[int, str] | None = None

    for candidate_tail in product(*choices):
        tail = "".join(candidate_tail)
        candidate = prefix + date_digits + tail
        if not is_acceptable_rfc(candidate):
            continue
        if not has_valid_check_digit(candidate):
            continue

        cost = sum(0 if src == dst else 1 for src, dst in zip(observed_tail, tail))
        if best is None or cost < best[0]:
            best = (cost, candidate)

    return best[1] if best else None

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def is_valid_rfc_date(token: str) -> bool:
    """Check that the embedded YYMMDD in *token* is a valid calendar date.

    The date sits at positions 3-8 (12-char RFC) or 4-9 (13-char RFC).
    Years ``>= 30`` are mapped to 19xx; years ``< 30`` to 20xx.
    """
    up = (token or "").upper()
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


def is_acceptable_rfc(token: str) -> bool:
    """Shape check plus date validation.

    Returns ``True`` when *token* matches ``^[A-ZÑ&]{3,4}[0-9]{6}[A-Z0-9]{2,3}$``
    **and** the embedded date is a valid calendar date.
    """
    if not RFC_SHAPE_PATTERN.fullmatch((token or "").upper()):
        return False
    return is_valid_rfc_date(token)


def has_valid_check_digit(token: str) -> bool:
    """Verify the RFC check digit using the SAT modulo-11 algorithm.

    The last character of a 12- or 13-character RFC is a check digit
    computed from the weighted sum of character values defined in
    :data:`RFC_CHAR_VALUES`.  This function returns ``True`` when the
    stated digit matches the computed one.
    """
    up = (token or "").upper()
    if len(up) not in (12, 13):
        return False

    base = up[:-1]
    expected = up[-1]

    try:
        values = [RFC_CHAR_VALUES[ch] for ch in base]
    except KeyError:
        return False

    start_weight = 13 if len(up) == 13 else 12
    total = sum(value * (start_weight - idx) for idx, value in enumerate(values))

    mod = total % 11
    calc = 11 - mod

    if calc == 11:
        check = "0"
    elif calc == 10:
        check = "A"
    else:
        check = str(calc)

    return expected == check

# ---------------------------------------------------------------------------
# Candidate selection
# ---------------------------------------------------------------------------


def pick_best_rfc(candidates: list[str]) -> str | None:
    """Given multiple RFC candidates choose the best one.

    Priority (highest to lowest):

    1. Valid check digit (and acceptable shape/date).
    2. Acceptable shape and valid embedded date (no check-digit match).
    3. First candidate in the list that is non-empty.

    Returns ``None`` when *candidates* is empty or contains only empty strings.
    """
    if not candidates:
        return None

    best_check: str | None = None
    best_date: str | None = None
    first_nonempty: str | None = None

    for raw in candidates:
        cleaned = normalize_rfc_ocr(raw)
        if not cleaned:
            continue

        if first_nonempty is None:
            first_nonempty = cleaned

        if is_acceptable_rfc(cleaned):
            if best_date is None:
                best_date = cleaned
            if has_valid_check_digit(cleaned):
                best_check = cleaned
                break  # can't do better

    if best_check is not None:
        return best_check
    if best_date is not None:
        return best_date
    return first_nonempty
