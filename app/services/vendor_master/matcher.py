from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

from sqlalchemy.orm import Session

from app.db.models import VendorMaster
from app.services.parsing.fields import normalize_company_name

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

# ---------------------------------------------------------------------------
# Configuration loading from YAML (with hardcoded fallbacks)
# ---------------------------------------------------------------------------

_BUILTIN_MATCH_CONFIG = {
    "thresholds": {
        "short_name": 0.85,   # names with <= 2 tokens
        "long_name": 0.82,    # names with 3+ tokens
    },
    "weights": {
        "ratio": 0.30,
        "token_sort": 0.30,
        "token_set": 0.20,
        "contains": 0.10,
        "same_first": 0.10,
    },
}


@lru_cache(maxsize=1)
def _load_match_config() -> dict:
    if _yaml is None:
        logger.debug("YAML not available, using built-in vendor matching config")
        return _BUILTIN_MATCH_CONFIG
    config_path = Path(__file__).resolve().parents[3] / "docs" / "rules" / "vendor_matching.yaml"
    if not config_path.exists():
        logger.debug("YAML config not found at %s, using built-in fallback", config_path)
        return _BUILTIN_MATCH_CONFIG
    logger.info("Loading vendor matching config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = _yaml.safe_load(f)
    merged = {**_BUILTIN_MATCH_CONFIG}
    for key in merged:
        if key in data and isinstance(data[key], dict):
            merged[key] = {**merged[key], **data[key]}
        elif key in data:
            merged[key] = data[key]
    return merged

# Use shared RFC module if available, fallback to local
try:
    from app.core.rfc import canonicalize_rfc, normalize_rfc_for_match
except ImportError:
    def normalize_rfc_for_match(value: str | None) -> str:
        return re.sub(r"[^A-Z0-9]", "", (value or "").upper())

    def canonicalize_rfc(value: str | None) -> str | None:
        normalized = normalize_rfc_for_match(value)
        if not normalized:
            return None
        if normalized.isdigit() and len(normalized) == 9:
            return f"{normalized[:2]}-{normalized[2:]}"
        return normalized


LEGAL_ENTITY_TOKENS = {
    "SA", "S", "A", "DE", "CV", "C", "V",
    "SACV", "SADECV",
    "SOCIEDAD", "ANONIMA", "ANONIMO",
    "COMPANIA", "COMPANY", "CIA",
    "LIMITADA", "LTDA", "LTD", "LLC",
    "INC", "CORP", "CORPORATION",
    "THE", "AND", "Y",
}

LEADING_NOISE_PATTERNS = (
    re.compile(r"(?i)^\s*confirma\s+su\s+acuerdo\s+con\s+"),
    re.compile(r"(?i)^\s*acuerdo\s+con\s+"),
    re.compile(r"(?i)^\s*por\s+otra\s+parte\s+la\s+sociedad\s+mercantil\s+"),
    re.compile(r"(?i)^\s*(raz[oó]n\s+social|nombre\s+del\s+proveedor|proveedor)\s*[:\-]\s*"),
    re.compile(r"(?i)^\s*y\s+"),
)

TRAILING_NOISE_PATTERNS = (
    re.compile(r"(?i)\s+con\s+rfc\b.*$"),
    re.compile(r"(?i)\s+a\s+quien\b.*$"),
    re.compile(r"(?i)\s+representad[oa]\s+por\b.*$"),
    re.compile(r"(?i)\b(curp|cuenta|clabe|telefono|tel[eé]fono)\b.*$"),
)


def _strip_accents(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_vendor_name_for_match(value: str | None) -> str:
    if not value:
        return ""

    candidate = normalize_company_name(value)
    candidate = _strip_accents(candidate).upper()
    candidate = re.sub(r"[_\.\,\;\:\-\(\)\[\]\{\}/\\\|]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()

    for pattern in LEADING_NOISE_PATTERNS:
        candidate = pattern.sub("", candidate).strip()
    for pattern in TRAILING_NOISE_PATTERNS:
        candidate = pattern.sub("", candidate).strip()

    return re.sub(r"\s+", " ", candidate).strip()


def core_vendor_name(normalized_name: str) -> str:
    if not normalized_name:
        return ""
    tokens = [tok for tok in normalized_name.split() if tok and tok not in LEGAL_ENTITY_TOKENS]
    if not tokens:
        tokens = [tok for tok in normalized_name.split() if tok]
    return " ".join(tokens)


def _token_overlap_score(a: str, b: str) -> float:
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return inter / union if union else 0.0


def _similarity_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    if _HAS_RAPIDFUZZ:
        # rapidfuzz: ~10x faster, better with transpositions
        w = _load_match_config()["weights"]
        ratio = _fuzz.ratio(a, b) / 100.0
        token_sort = _fuzz.token_sort_ratio(a, b) / 100.0
        token_set = _fuzz.token_set_ratio(a, b) / 100.0
        contains = 1.0 if a in b or b in a else 0.0
        same_first = 1.0 if a.split()[0] == b.split()[0] else 0.0
        return (
            ratio * w["ratio"]
            + token_sort * w["token_sort"]
            + token_set * w["token_set"]
            + contains * w["contains"]
            + same_first * w["same_first"]
        )

    # Fallback to stdlib SequenceMatcher
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, a, b).ratio()
    overlap = _token_overlap_score(a, b)
    contains = 1.0 if a in b or b in a else 0.0
    same_first = 1.0 if a.split()[0] == b.split()[0] else 0.0
    return (ratio * 0.55) + (overlap * 0.30) + (contains * 0.10) + (same_first * 0.05)


@dataclass(frozen=True)
class VendorMasterEntry:
    vendor_name: str
    rfc: str | None
    vendor_name_normalized: str
    vendor_name_core: str


@dataclass(frozen=True)
class VendorMasterMatch:
    entry: VendorMasterEntry
    score: float
    strategy: str


class VendorMasterResolver:
    def __init__(self, entries: Iterable[VendorMasterEntry]):
        self.entries = [entry for entry in entries if entry.vendor_name_normalized or entry.rfc]
        self.by_rfc: dict[str, VendorMasterEntry] = {}
        self.by_normalized_name: dict[str, VendorMasterEntry] = {}
        self.by_core_name: dict[str, VendorMasterEntry] = {}
        self.by_first_core_token: dict[str, list[VendorMasterEntry]] = {}

        for entry in self.entries:
            if entry.rfc:
                key = normalize_rfc_for_match(entry.rfc)
                if key and key not in self.by_rfc:
                    self.by_rfc[key] = entry

            if entry.vendor_name_normalized and entry.vendor_name_normalized not in self.by_normalized_name:
                self.by_normalized_name[entry.vendor_name_normalized] = entry

            if entry.vendor_name_core and entry.vendor_name_core not in self.by_core_name:
                self.by_core_name[entry.vendor_name_core] = entry
                first = entry.vendor_name_core.split()[0]
                self.by_first_core_token.setdefault(first, []).append(entry)

    @classmethod
    def from_db(cls, db: Session) -> VendorMasterResolver:
        records = db.query(VendorMaster).all()
        entries = [
            VendorMasterEntry(
                vendor_name=(record.vendor_name or "").strip(),
                rfc=canonicalize_rfc(record.rfc or record.rfc_normalized),
                vendor_name_normalized=record.vendor_name_normalized or "",
                vendor_name_core=record.vendor_name_core or "",
            )
            for record in records
        ]
        return cls(entries)

    def _name_candidates(self, core_name: str) -> list[VendorMasterEntry]:
        if not core_name:
            return []
        first = core_name.split()[0]
        candidates = self.by_first_core_token.get(first, [])
        if candidates:
            return candidates
        return self.entries

    def find_best_by_name(self, provider_name: str | None) -> VendorMasterMatch | None:
        logger.debug("Vendor match attempt for: %s", provider_name)
        normalized_name = normalize_vendor_name_for_match(provider_name)
        if not normalized_name:
            return None

        core_name = core_vendor_name(normalized_name)
        if not core_name:
            return None

        exact_name = self.by_normalized_name.get(normalized_name)
        if exact_name:
            logger.debug("Exact normalized name match: %s (score=1.0)", exact_name.vendor_name)
            return VendorMasterMatch(entry=exact_name, score=1.0, strategy="exact_normalized_name")

        exact_core = self.by_core_name.get(core_name)
        if exact_core:
            logger.debug("Exact core name match: %s (score=0.98)", exact_core.vendor_name)
            return VendorMasterMatch(entry=exact_core, score=0.98, strategy="exact_core_name")

        candidates = self._name_candidates(core_name)
        if not candidates:
            return None

        best_entry: VendorMasterEntry | None = None
        best_score = 0.0
        for candidate in candidates:
            score = _similarity_score(core_name, candidate.vendor_name_core or candidate.vendor_name_normalized)
            if score > best_score:
                best_entry = candidate
                best_score = score

        if not best_entry:
            return None

        token_count = len(core_name.split())
        cfg_thresholds = _load_match_config()["thresholds"]
        threshold = cfg_thresholds["short_name"] if token_count <= 2 else cfg_thresholds["long_name"]
        if best_score < threshold:
            logger.debug(
                "Best fuzzy match below threshold: %s (score=%.4f, threshold=%.2f)",
                best_entry.vendor_name, best_score, threshold,
            )
            return None

        logger.debug(
            "Best match found: %s (score=%.4f, strategy=fuzzy_core_name)",
            best_entry.vendor_name, best_score,
        )
        return VendorMasterMatch(entry=best_entry, score=round(best_score, 4), strategy="fuzzy_core_name")

    def fill_missing_fields(
        self,
        rfc: str | None,
        nombre_proveedor: str | None,
    ) -> tuple[str | None, str | None]:
        current_rfc = canonicalize_rfc(rfc)
        current_name = (nombre_proveedor or "").strip() or None

        if current_rfc and not current_name:
            record = self.by_rfc.get(normalize_rfc_for_match(current_rfc))
            if record and record.vendor_name:
                return current_rfc, record.vendor_name
            return current_rfc, current_name

        if current_name and not current_rfc:
            best_match = self.find_best_by_name(current_name)
            if best_match and best_match.entry.rfc:
                return best_match.entry.rfc, current_name

        return current_rfc, current_name
