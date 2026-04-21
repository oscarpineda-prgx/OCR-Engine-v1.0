from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.services.parsing.fields import extract_rfc
from app.services.vendor_master.matcher import (
    VendorMasterResolver,
    canonicalize_rfc,
    core_vendor_name,
    normalize_vendor_name_for_match,
)

PATH_SPLIT_PATTERN = re.compile(r"[\\/]+")
FILE_EXTENSION_PATTERN = re.compile(r"\.[A-Za-z0-9]{1,8}$")
COUNTER_SUFFIX_PATTERN = re.compile(r"\(\s*\d+\s*\)")
LEGAL_ENTITY_HINT_PATTERN = re.compile(
    r"(?i)\b(?:"
    r"S\.?\s*A\.?|SAPI|S\.?\s*DE\s*R\.?\s*L\.?|DE\s*C\.?\s*V\.?|C\.?\s*V\.?|"
    r"LLC|INC|LTD|COMPANY|COMPANIA|CIA|CORP|CORPORATION"
    r")\b"
)

GENERIC_NORMALIZED_SEGMENTS = {
    "AMER PRGX COM",
    "AMER PRGX",
    "PRGX",
    "IMAGES",
    "OXXOMEX",
    "OXXO MEX",
    "LINK",
    "LINK 2026",
    "PROCESO ALTAS",
    "PROCESO ALTA",
    "ALTAS",
    "ALTA",
    "SOPORTES",
    "SOPORTE",
    "CONTRATOS",
    "CONTRATO",
    "CONVENIOS",
    "CONVENIO",
    "CEDIS",
    "PROVEEDORES",
    "PROVEEDOR",
    "INCOMING",
    "EXPORTS",
    "DATA",
}

GENERIC_CORE_SEGMENTS = {
    "AMER PRGX COM",
    "AMER PRGX",
    "PRGX",
    "IMAGES",
    "OXXOMEX",
    "OXXO MEX",
    "LINK",
    "PROCESO ALTAS",
    "PROCESO ALTA",
    "ALTAS",
    "ALTA",
    "SOPORTES",
    "SOPORTE",
    "CONTRATOS",
    "CONTRATO",
    "CONVENIOS",
    "CONVENIO",
    "CEDIS",
    "PROVEEDORES",
    "PROVEEDOR",
    "INCOMING",
    "EXPORTS",
    "DATA",
}


@dataclass(frozen=True)
class VendorPathFallback:
    rfc: str | None
    nombre_proveedor: str | None
    source_path: str
    candidate: str
    strategy: str
    score: float | None = None

    def to_log_dict(self) -> dict:
        return asdict(self)


def _split_path_parts(path_value: str | None) -> list[str]:
    if not path_value:
        return []
    return [part.strip() for part in PATH_SPLIT_PATTERN.split(str(path_value)) if part.strip()]


def _strip_file_extension(value: str) -> str:
    return FILE_EXTENSION_PATTERN.sub("", value or "")


def _clean_path_segment(value: str) -> str:
    candidate = _strip_file_extension(value)
    candidate = COUNTER_SUFFIX_PATTERN.sub(" ", candidate)
    candidate = re.sub(r"[_\.,;:\-\[\]\{\}\|]+", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip(" .:-|")
    return candidate


def _ordered_path_segments(path_value: str | None) -> list[str]:
    parts = _split_path_parts(path_value)
    if not parts:
        return []

    folders = parts[:-1]
    filename = parts[-1]
    ordered = list(reversed(folders))
    if filename:
        ordered.append(_strip_file_extension(filename))
    return ordered


def _looks_like_vendor_path_segment(candidate: str) -> bool:
    normalized = normalize_vendor_name_for_match(candidate)
    if not normalized:
        return False

    if normalized in GENERIC_NORMALIZED_SEGMENTS:
        return False

    core = core_vendor_name(normalized)
    if not core or core in GENERIC_CORE_SEGMENTS:
        return False

    if re.fullmatch(r"\d{2,8}", normalized):
        return False

    alpha_count = sum(1 for char in normalized if char.isalpha())
    if alpha_count < 4:
        return False

    if LEGAL_ENTITY_HINT_PATTERN.search(normalized):
        return True

    return len(core.split()) >= 2


def vendor_name_candidates_from_path(path_value: str | None) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    for segment in _ordered_path_segments(path_value):
        candidate = _clean_path_segment(segment)
        if not _looks_like_vendor_path_segment(candidate):
            continue

        normalized = normalize_vendor_name_for_match(candidate)
        if normalized in seen:
            continue

        seen.add(normalized)
        candidates.append(candidate)

    return candidates


def resolve_vendor_from_path(
    path_value: str | None,
    resolver: VendorMasterResolver,
) -> VendorPathFallback | None:
    source_path = str(path_value or "").strip()
    if not source_path:
        return None

    for segment in _ordered_path_segments(source_path):
        candidate = _clean_path_segment(segment)
        candidate_rfc = canonicalize_rfc(extract_rfc(candidate))
        if not candidate_rfc:
            continue

        rfc, provider_name = resolver.fill_missing_fields(candidate_rfc, None)
        return VendorPathFallback(
            rfc=rfc,
            nombre_proveedor=provider_name,
            source_path=source_path,
            candidate=candidate_rfc,
            strategy="path_rfc",
        )

    for candidate in vendor_name_candidates_from_path(source_path):
        match = resolver.find_best_by_name(candidate)
        if not match:
            continue

        provider_name = match.entry.vendor_name or candidate
        return VendorPathFallback(
            rfc=match.entry.rfc,
            nombre_proveedor=provider_name,
            source_path=source_path,
            candidate=candidate,
            strategy=f"path_name:{match.strategy}",
            score=match.score,
        )

    return None
