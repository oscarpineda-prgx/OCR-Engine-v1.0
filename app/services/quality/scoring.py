import logging
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import yaml as _yaml
except ImportError:
    _yaml = None

try:
    from app.core.rfc import has_valid_check_digit, is_valid_rfc_date
    _HAS_RFC_MODULE = True
except ImportError:
    _HAS_RFC_MODULE = False

# ---------------------------------------------------------------------------
# Configuration loading from YAML (with hardcoded fallbacks)
# ---------------------------------------------------------------------------

_BUILTIN_CONFIG = {
    "field_weights": {"rfc": 35, "fecha_documento": 30, "tipo_documento": 20, "nombre_proveedor": 15},
    "thresholds": {"green_min": 85, "yellow_min": 65},
    "validation": {
        "rfc_format": r"^[A-Z\u00D1&]{3,4}\d{6}[A-Z0-9]{3}$",
        "foreign_tax_format": r"^\d{2}-\d{7}$",
        "date_format": r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$",
        "fecha_min_year": 1990,
        "fecha_max_year_offset": 1,
        "nombre_min_length": 3,
    },
    "confidence_rules": {
        "rfc_valid": 1.0, "rfc_foreign": 0.9, "rfc_bad_format": 0.3, "rfc_bad_date": 0.5,
        "fecha_valid": 1.0, "fecha_future": 0.3, "fecha_too_old": 0.4, "fecha_bad_format": 0.2,
        "tipo_classified": 1.0, "tipo_fallback": 0.3,
        "nombre_valid": 1.0, "nombre_too_short": 0.2,
    },
}


@lru_cache(maxsize=1)
def _load_config() -> dict:
    if _yaml is None:
        logger.debug("YAML not available, using built-in scoring config")
        return _BUILTIN_CONFIG
    config_path = Path(__file__).resolve().parents[3] / "docs" / "rules" / "quality_thresholds.yaml"
    if not config_path.exists():
        logger.debug("YAML config not found at %s, using built-in fallback", config_path)
        return _BUILTIN_CONFIG
    logger.info("Loading scoring config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = _yaml.safe_load(f)
    merged = {**_BUILTIN_CONFIG}
    for key in merged:
        if key in data and isinstance(data[key], dict):
            merged[key] = {**merged[key], **data[key]}
        elif key in data:
            merged[key] = data[key]
    return merged


# ---------------------------------------------------------------------------
# Field validators — return (confidence 0.0-1.0, reason_if_invalid | None)
# ---------------------------------------------------------------------------


def _validate_rfc(rfc: str | None) -> tuple[float, str | None]:
    cfg = _load_config()
    conf = cfg["confidence_rules"]
    val = cfg["validation"]

    if not rfc:
        return 0.0, "missing_rfc"

    foreign_re = re.compile(val["foreign_tax_format"])
    if foreign_re.match(rfc):
        return conf["rfc_foreign"], None

    rfc_re = re.compile(val["rfc_format"])
    if not rfc_re.match(rfc):
        return conf["rfc_bad_format"], "invalid_rfc_format"

    # Validate embedded date
    n = len(rfc)
    date_part = rfc[3:9] if n == 12 else rfc[4:10] if n == 13 else ""
    if not date_part or not date_part.isdigit():
        return conf["rfc_bad_date"], "invalid_rfc_date"
    try:
        yy, mm, dd = int(date_part[:2]), int(date_part[2:4]), int(date_part[4:6])
        year = 1900 + yy if yy >= 30 else 2000 + yy
        datetime(year=year, month=mm, day=dd)
    except (ValueError, IndexError):
        return conf["rfc_bad_date"], "invalid_rfc_date"

    return conf["rfc_valid"], None


def _validate_fecha(fecha: str | None) -> tuple[float, str | None]:
    cfg = _load_config()
    conf = cfg["confidence_rules"]
    val = cfg["validation"]

    if not fecha:
        return 0.0, "missing_fecha_documento"

    date_re = re.compile(val["date_format"])
    if not date_re.match(fecha):
        return conf["fecha_bad_format"], "invalid_fecha_format"
    try:
        dt = datetime.strptime(fecha, "%Y-%m-%d")
        max_year = datetime.now().year + val["fecha_max_year_offset"]
        if dt.year > max_year:
            return conf["fecha_future"], "fecha_future"
        if dt.year < val["fecha_min_year"]:
            return conf["fecha_too_old"], "fecha_too_old"
        return conf["fecha_valid"], None
    except ValueError:
        return conf["fecha_bad_format"], "invalid_fecha_value"


def _validate_tipo_documento(tipo: str | None) -> tuple[float, str | None]:
    conf = _load_config()["confidence_rules"]
    if not tipo:
        return 0.0, "missing_tipo_documento"
    if tipo == "Documentos Varios":
        return conf["tipo_fallback"], "tipo_documento_fallback"
    return conf["tipo_classified"], None


def _validate_nombre_proveedor(nombre: str | None) -> tuple[float, str | None]:
    cfg = _load_config()
    conf = cfg["confidence_rules"]
    min_len = cfg["validation"]["nombre_min_length"]

    if not nombre:
        return 0.0, "missing_nombre_proveedor"
    clean = nombre.strip()
    if len(clean) < min_len:
        return conf["nombre_too_short"], "nombre_proveedor_too_short"
    return conf["nombre_valid"], None


# ---------------------------------------------------------------------------
# Weighted scoring with per-field confidence
# ---------------------------------------------------------------------------

# Keep as module-level alias for backward compatibility
FIELD_WEIGHTS = _BUILTIN_CONFIG["field_weights"]


def score_document_fields(
    rfc: str | None,
    fecha_documento: str | None,
    tipo_documento: str | None,
    nombre_proveedor: str | None,
) -> dict[str, Any]:
    cfg = _load_config()
    weights = cfg["field_weights"]
    thresholds = cfg["thresholds"]

    reasons: list[str] = []

    rfc_conf, rfc_reason = _validate_rfc(rfc)
    fecha_conf, fecha_reason = _validate_fecha(fecha_documento)
    tipo_conf, tipo_reason = _validate_tipo_documento(tipo_documento)
    nombre_conf, nombre_reason = _validate_nombre_proveedor(nombre_proveedor)

    if rfc_reason:
        reasons.append(rfc_reason)
    if fecha_reason:
        reasons.append(fecha_reason)
    if tipo_reason:
        reasons.append(tipo_reason)
    if nombre_reason:
        reasons.append(nombre_reason)

    score = round(
        weights["rfc"] * rfc_conf
        + weights["fecha_documento"] * fecha_conf
        + weights["tipo_documento"] * tipo_conf
        + weights["nombre_proveedor"] * nombre_conf
    )

    if score >= thresholds["green_min"]:
        traffic_light = "verde"
    elif score >= thresholds["yellow_min"]:
        traffic_light = "amarillo"
    else:
        traffic_light = "rojo"

    logger.debug(
        "Scored document: score=%d, traffic_light=%s, reasons=%s",
        score, traffic_light, reasons,
    )

    return {
        "score": score,
        "traffic_light": traffic_light,
        "reasons": reasons,
        "field_confidence": {
            "rfc": round(rfc_conf, 2),
            "fecha_documento": round(fecha_conf, 2),
            "tipo_documento": round(tipo_conf, 2),
            "nombre_proveedor": round(nombre_conf, 2),
        },
    }
